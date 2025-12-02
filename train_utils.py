
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F



def collate_fn(prompts, labels):

    max_seq_len = max(prompt.size(1) for prompt in prompts)
    batch_size = len(prompts)
    dtype = prompts[0].dtype
    device = prompts[0].device

    # CHANGE 1: Remove the '1' in the shape definition. Shape is now [Batch, Max_Seq_Len]
    prompts_padded = torch.zeros(batch_size, max_seq_len, dtype=dtype, device=device) # <-- Removed the '1'

    # Pad each prompt to the maximum sequence length
    for i, prompt in enumerate(prompts):
        seq_len = prompt.size(1)
        
        # CHANGE 2: Indexing also removes the '1' dimension.
        # Assuming prompt shape is [1, L] (typical tokenizer output)
        prompts_padded[i, :seq_len] = prompt.squeeze(0) # <-- Squeeze the input prompt

    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return prompts_padded, labels


def get_last_non_padded_token_rep(hidden_states, attention_mask):
    """
    Get the last non-padded token's representation for each sequence in the batch.
    """
    # Find the length of each sequence by summing the attention mask (1 for real tokens, 0 for padding)
    lengths = attention_mask.squeeze().sum(dim=1).long()

    # Index the last non-padded token for each sequence
    batch_size, max_seq_len, hidden_size = hidden_states.size()
    last_token_reps = torch.stack([hidden_states[i, lengths[i]-1, :] for i in range(batch_size)])

    return last_token_reps

def get_ex_data(model, prompts, labels, batch_size, centroids, sinkhorn, 
                num_selected_data, cls_dist, args):

    model.eval()

    all_embeddings = []
    all_labels = []
    num_samples = len(prompts)

    with torch.no_grad():
        for batch_start in tqdm(range(0, num_samples, batch_size)):

            # ---- Prepare batch ----
            batch_prompts = prompts[batch_start: batch_start + batch_size]
            batch_labels = labels[batch_start: batch_start + batch_size]
            batch_prompts, batch_labels = collate_fn(batch_prompts, batch_labels)

            # Move to GPU
            batch_prompts = batch_prompts.cuda(non_blocking=True)
            batch_labels = batch_labels.cuda(non_blocking=True)
            attention_mask = (batch_prompts != 0).to(batch_prompts.device)

            all_labels.append(batch_labels.cpu().numpy())

            # ---- IMPORTANT: disable SDPA to avoid repeat_kv unpack errors ----
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False
            ):
                with autocast(dtype=torch.float16):
                    output = model(
                    input_ids=batch_prompts.squeeze(1), 
                    attention_mask=attention_mask.squeeze(1),
                        output_hidden_states=True,   # need hidden_states[-1]
                        use_cache=False              # disable KV cache
                    )

            # ---- Last layer hidden state only ----
            last_hidden = output.hidden_states[-1]  # [B, L, H]

            # ---- Extract last non padded token ----
            last_token_rep = get_last_non_padded_token_rep(
                last_hidden, attention_mask
            )  # [B, H]

            # Store on CPU to save vRAM
            all_embeddings.append(last_token_rep.float().cpu())

        # ---- Combine & move back to GPU ----
        all_embeddings = torch.cat(all_embeddings, dim=0).cuda()  # [N, H]

        # Normalize (FP32 for stability)
        all_embeddings = F.normalize(all_embeddings, p=2, dim=-1)

        # Ensure centroids are FP32 too
        centroids = centroids.float().cuda()

        # ---- Sinkhorn pseudo-labels ----
        pseudo_label = sinkhorn(all_embeddings, centroids)  # FP32

        # ---- Entropy-based selection ----
        selected_indices = compute_entropy(
            all_embeddings, centroids, pseudo_label,
            num_selected_data, cls_dist, args
        )

        selected_labels_soft = pseudo_label[selected_indices]

    return selected_indices, selected_labels_soft


def compute_ot_loss_cos(last_token_rep, centroids, pseudo_label, args):
    
    # make sure both tensors are float32 for matmul
    last_token_rep = F.normalize(last_token_rep.float(), p=2, dim=-1)
    centroids = F.normalize(centroids.float(), p=2, dim=-1)

    similarities = torch.matmul(last_token_rep, centroids.T)

    similarities = similarities / args.cos_temp

    pt = torch.softmax(similarities, dim=-1)

    ot_loss = -torch.sum(pseudo_label.float() * torch.log(pt + 1e-8)) / pseudo_label.shape[0]

    return ot_loss, similarities


def compute_entropy(last_token_rep, centroids, pseudo_label, k, cls_dist, args):
    

    last_token_rep = F.normalize(last_token_rep, p=2, dim=-1)
    
    centroids = F.normalize(centroids, p=2, dim=-1)

    similarities = torch.matmul(last_token_rep, centroids.T)  

    similarities = similarities / args.cos_temp
    
    pt = F.softmax(similarities, dim=-1)  
    
    ce = - (pseudo_label * torch.log(pt + 1e-8))
    
    pseudo_label_hard = torch.argmax(pt,dim=1) 
    
    # * Added for preventing severe cases
    # Class-wise data selection: Select pseudo-labeled unlabeled data in proportion to the class distribution of the exemplar set. 
    
    cls0_num = k*cls_dist[0]
    cls1_num = k*cls_dist[1]
    
    cls_0_indices = (pseudo_label_hard == 0).nonzero(as_tuple=True)[0]
    cls_1_indices = (pseudo_label_hard == 1).nonzero(as_tuple=True)[0]

    ce = torch.sum(ce, dim=1)
    
    ce_class_0 = ce[cls_0_indices]
    ce_class_1 = ce[cls_1_indices]
    
    if len(ce_class_0) < cls0_num or len(ce_class_1) < cls1_num: # Fallback to top-k across all classes
        
        _, top_k_indices = torch.topk(ce, k, largest=False, sorted=True)
        
    else:
        
        top_0_indices = cls_0_indices[torch.topk(ce_class_0, int(cls0_num), largest=False, sorted=True).indices]  
        top_1_indices = cls_1_indices[torch.topk(ce_class_1, int(cls1_num), largest=False, sorted=True).indices]  
        top_k_indices = torch.cat((top_0_indices, top_1_indices))
        
    return top_k_indices  

def compute_ot_and_repulsion_loss(last_token_rep, centroids, pseudo_label, args):
    """
    Calculates the combined Optimal Transport (OT) loss and the Prototype Repulsion loss.

    L = OT_loss + lambda_rep * (mu_T^T * mu_H)^2

    Args:
        last_token_rep (torch.Tensor): Embeddings of the last non-padded tokens [B, H].
        centroids (torch.Tensor): Centroids/Prototypes [2, H]. Assumes centroids[0] is mu_T
                                   and centroids[1] is mu_H (or vice versa).
        pseudo_label (torch.Tensor): Soft pseudo-labels [B, 2].
        args: Arguments object containing cos_temp and lam (lambda_rep).

    Returns:
        torch.Tensor: The total combined loss (scalar).
        torch.Tensor: The similarities matrix [B, 2].
    """
    
    # Ensure both tensors are float32 and normalized for stable cosine similarity
    last_token_rep_norm = F.normalize(last_token_rep.float(), p=2, dim=-1)
    centroids_norm = F.normalize(centroids.float(), p=2, dim=-1)

    # Cosine Similarity: [B, H] @ [H, 2] -> [B, 2]
    similarities = torch.matmul(last_token_rep_norm, centroids_norm.T)

    # Apply temperature (tau)
    similarities = similarities / args.cos_temp

    # Compute softmax probabilities (p_t or q in your notation)
    pt = torch.softmax(similarities, dim=-1)

    # Calculate cross-entropy loss (OT loss)
    # L_OT = - sum(pseudo_label * log(pt)) / N
    ot_loss = -torch.sum(pseudo_label.float() * torch.log(pt + 1e-8)) / pseudo_label.shape[0]
    
    # Extract mu_T and mu_H (assumed to be centroids[0] and centroids[1])
    mu_T = centroids_norm[0]
    mu_H = centroids_norm[1]
    
    # Calculate the dot product (cosine similarity, since they are normalized)
    # This is mu_T^T * mu_H
    dot_product = torch.dot(mu_T, mu_H)
    
    # Calculate the squared dot product: (mu_T^T * mu_H)^2
    squared_dot_product = dot_product.pow(2)

    repulsion_loss = args.lambda_repulsion * squared_dot_product
    total_loss = ot_loss + repulsion_loss

    return total_loss, similarities


def update_centroids_ema(centroids, last_token_rep, pseudo_label, args):

    # Ensure consistent dtype (float32)
    last_token_rep_norm = F.normalize(last_token_rep.float(), p=2, dim=1)
    pseudo_label = pseudo_label.float()

    centroids = F.normalize(centroids.float(), p=2, dim=1)

    # Weighted sum
    weighted_sum = torch.matmul(pseudo_label.T, last_token_rep_norm)

    # Normalize the weighted sums to get new centroids
    pseudo_label_sum = pseudo_label.sum(dim=0).unsqueeze(1) + 1e-8
    new_centroids_batch = weighted_sum / pseudo_label_sum

    # EMA update
    updated_centroids = args.ema_decay * centroids + (1 - args.ema_decay) * new_centroids_batch
    updated_centroids = F.normalize(updated_centroids, p=2, dim=1)

    return updated_centroids

def update_centroids_ema_hard(centroids, last_token_rep, pseudo_label, args):

    # Ensure consistent dtype (float32)
    last_token_rep_norm = F.normalize(last_token_rep.float(), p=2, dim=1)
    pseudo_label = pseudo_label.float()
    centroids = F.normalize(centroids.float(), p=2, dim=1)

    # Convert soft pseudo labels to discrete one-hot
    max_indices = torch.argmax(pseudo_label, dim=1)
    discrete_labels = torch.zeros_like(pseudo_label)
    discrete_labels[torch.arange(pseudo_label.size(0)), max_indices] = 1

    # Weighted sum
    weighted_sum = torch.matmul(discrete_labels.T, last_token_rep_norm)

    # Normalize
    pseudo_label_sum = discrete_labels.sum(dim=0).unsqueeze(1) + 1e-8
    new_centroids_batch = weighted_sum / pseudo_label_sum

    # EMA update
    updated_centroids = args.ema_decay * centroids + (1 - args.ema_decay) * new_centroids_batch
    updated_centroids = F.normalize(updated_centroids, p=2, dim=1)

    return updated_centroids
