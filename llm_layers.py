import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
from cache_utils import Cache
from transformers.activations import ACT2FN

class AttentionWrapper(nn.Module):
    """
    A generic wrapper for self-attention modules to apply TSV intervention.
    This wrapper correctly handles all the arguments passed by the parent
    Decoder Layer (like attention_mask, past_key_value, etc.) and ensures
    the original attention module is called correctly before applying the TSV.
    """
    def __init__(self, original_attn, tsv_layer):
        super().__init__()
        self.original_attn = original_attn
        self.tsv_layer = tsv_layer
        
        # CRITICAL: Copy all necessary attributes from the original layer
        for attr_name, attr_value in original_attn.__dict__.items():
            if attr_name not in self.__dict__ and not callable(attr_value):
                setattr(self, attr_name, attr_value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # Llama/Qwen specific arguments are passed as **kwargs
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    )-> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        

        if hasattr(self.original_attn, 'forward'):
            
            # Capture all arguments passed to the wrapper
            attn_kwargs = {
                'hidden_states': hidden_states,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'past_key_value': past_key_value,
                'output_attentions': output_attentions,
                'use_cache': use_cache,
                'cache_position': cache_position,
                **kwargs, # Includes position_embeddings for Llama if present
            }
            
            # A more robust fix relies on knowing the model type:
            if position_embeddings is not None and 'position_embeddings' not in attn_kwargs:
                attn_kwargs['position_embeddings'] = position_embeddings
            # If it wasn't passed, we must explicitly add it as None if the target module expects it.
            # Since the traceback shows Qwen2Attention requires it, we add it back if missing.
            elif 'position_embeddings' not in attn_kwargs:
                attn_kwargs['position_embeddings'] = None
            
            # The common signature for `self_attn` in HuggingFace Llama/Qwen models is usually
            # output_tuple = self.original_attn(hidden_states=..., attention_mask=..., ...)
            
            # We will try the safest call, which is the full set of kwargs:
            output_tuple = self.original_attn(**attn_kwargs)
            
        else:
            # Fallback for simpler attention module (e.g. if it was just a sequential layer before)
            # This case is unlikely for HF models, but keeps the code safe.
            output_tuple = self.original_attn(hidden_states)
        
        
        # 2. Get the new hidden states (always the first element in the output tuple)
        attn_output = output_tuple[0]
        
        # 3. Apply the TSV intervention (residual injection)
        intervened_attn_output = self.tsv_layer(attn_output)

        # 4. Return the intervened hidden states plus the rest of the original output tuple
        return (intervened_attn_output,) + output_tuple[1:]

class LlamaDecoderLayerWrapper(nn.Module):
    def __init__(self, decoder_layer, tsv_layer, model_name):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.tsv_layer = tsv_layer
        self.model_name = model_name

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs
    ):

        # Save original dtype/device
        device = hidden_states.device

        # Normalize input
        residual = hidden_states
        hidden_states = self.decoder_layer.input_layernorm(hidden_states)

        # ----------------------------------------------------
        # Different attention behavior for Qwen2.5 vs LLaMA
        # ----------------------------------------------------
        if "qwen" in self.model_name:
            attn_out = self.decoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            # Qwen2 returns (hidden_states, attn_weights?) — no present_kv
            hidden_states = attn_out[0]
            self_attn_weights = attn_out[1] if output_attentions else None
            present_key_value = None

        else:
            hidden_states, self_attn_weights, present_key_value = \
                self.decoder_layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs
                )

        # Residual add
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # TSV injection
        hidden_states = self.tsv_layer(hidden_states)

        # ----------------------------------------------------
        # Qwen2: return only hidden_states (no tuple allowed)
        # ----------------------------------------------------
        if "qwen" in self.model_name:
            return hidden_states

        # LLaMA-style tuple return
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
        
class TSVLayer(nn.Module):
    """
    Numerically-stable TSV layer supporting FP32 steering and 
    projection into model hidden dimension.
    """
    def __init__(self, tsv_vec, lam, hidden_size):
        super().__init__()
        self.lam = lam

        # Convert TSV to float32 for stability
        self.register_buffer("tsv_vec", tsv_vec.float())

        # Projection from (orig_dim) → hidden_size
        self.tsv_proj = nn.Linear(tsv_vec.shape[-1], hidden_size, bias=False)

    def forward(self, x):
        if self.tsv_vec is None:
            return x

        B, T, H = x.shape
        device = x.device

        # Move TSV vec to correct device
        tsv = self.tsv_vec.to(device)

        # Project only in FP32
        tsv_h = self.tsv_proj(tsv).unsqueeze(0).unsqueeze(1)  # (1,1,H)

        # Expand to batch and sequence
        y = tsv_h.expand(B, T, H) * self.lam[0]

        # FP32 addition to avoid NaNs
        out = (x.float() + y.float()).to(x.dtype)

        return out
        

def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)


def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def get_embedding_layer(model: PreTrainedModel):

    keywords = ["emb", "wte"]
    return find_module(model, keywords)


def get_lm_head(model: PreTrainedModel):
    keywords = ["lm_head", "embed_out"]
    return find_module(model, keywords)


def get_lm_pipeline(model: PreTrainedModel):
    model_class = model.__class__.__name__

    if model_class == "LlamaForCausalLM":
        return nn.Sequential(model.model.norm, model.lm_head)
    elif model_class == "RWForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoXForCausalLM":
        return nn.Sequential(model.gpt_neox.final_layer_norm, model.embed_out)

    # TODO: make the default case more robust
    return get_lm_head(model)


def get_layers_path(model: PreTrainedModel):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path


def get_layers(model: PreTrainedModel):
    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def add_tsv_layers(model: PreTrainedModel, tsv: Tensor, alpha: list, args):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    attn_keywords = ["self_attn"]
    
    assert len(tsv) == len(layers)
    if args.component == 'mlp':
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                original_mlp = find_module(layer, mlp_keywords)
                layer.mlp = nn.Sequential(original_mlp, TSVLayer(tsv[i], alpha)) 

    elif args.component == 'attn':
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                original_attn = find_module(layer, attn_keywords)
                layer.self_attn = nn.Sequential(original_attn, TSVLayer(tsv[i], alpha)) 
                
    elif args.component == 'res':
        
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                decoder_layer = layers[i]
                layers[i] = LlamaDecoderLayerWrapper(decoder_layer, TSVLayer(tsv[i], alpha), args.model_name)
