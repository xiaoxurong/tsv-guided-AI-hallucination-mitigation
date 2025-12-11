import torch
from torch.nn import functional as F
from torch import nn
from transformers import PreTrainedModel
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
from cache_utils import Cache
from transformers.activations import ACT2FN

class LlamaAttentionWrapper(nn.Module):
    """
    Custom wrapper for the LlamaAttention module to inject TSV.
    This wrapper maintains the original function signature by accepting *args and **kwargs,
    passing them to the original module, and applying the TSV to the resulting hidden_states
    (the first element of the attention output tuple).
    """
    def __init__(self, original_attn, tsv_layer):
        super().__init__()
        self.original_attn = original_attn
        self.tsv = tsv_layer

    def forward(self, *args, **kwargs):
        # 1. Call original attention block with all arguments
        # This preserves the required keyword arguments (like hidden_states, attention_mask, etc.)
        attn_out = self.original_attn(*args, **kwargs)

        # 2. attn_out is a tuple: (hidden_states, past_key_value, ...)
        # Apply TSV only to the hidden_states (the first element, index 0)
        original_hidden_states = attn_out[0]
        new_hidden_states = self.tsv(original_hidden_states)
        
        # 3. Reconstruct the output tuple with the modified hidden states
        new_attn_out = list(attn_out)
        new_attn_out[0] = new_hidden_states
        
        return tuple(new_attn_out)

class LlamaDecoderLayerWrapper(nn.Module):
    def __init__(self, llama_decoder_layer, tsv_layer):
        super().__init__()
        self.inner = llama_decoder_layer
        self.tsv = tsv_layer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        # ---- Call original layer exactly as HF expects ----
        out = self.inner(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # HF always returns BaseModelOutputWithPast
        # which behaves like a tuple: (hidden_states, ...)
        hidden_states = out[0]

        # ---- add TSV ----
        hidden_states = self.tsv(hidden_states)

        # ---- reconstruct same structure ----
        # out is a BaseModelOutputWithPast / tuple-like object
        new_out = list(out)
        new_out[0] = hidden_states      # replace only the hidden states

        if isinstance(out, tuple):
            return tuple(new_out)
        else:
            return out.__class__(*new_out)

class QwenAttentionWrapper(nn.Module):
    """Wraps Qwen2 attention and adds TSV shift to output."""
    def __init__(self, original_attn, tsv_layer):
        super().__init__()
        self.original_attn = original_attn
        self.tsv_layer = tsv_layer

    def forward(self, hidden_states, **kwargs):
        x = self.original_attn(hidden_states, **kwargs)
        tsv = self.tsv_layer.tsv.to(x.device).repeat(1, x.shape[1], 1)
        return x + tsv * self.tsv_layer.lam[0]


class QwenDecoderLayerWrapper(nn.Module):
    """Wraps entire Qwen2 decoder layer for residual-based TSV injection."""
    def __init__(self, original_layer, tsv_layer):
        super().__init__()
        self.original_layer = original_layer
        self.tsv_layer = tsv_layer

    def forward(self, hidden_states, **kwargs):
        x = self.original_layer(hidden_states, **kwargs)
        tsv = self.tsv_layer.tsv.to(x.device).repeat(1, x.shape[1], 1)
        return x + tsv * self.tsv_layer.lam[0]
        
class TSVLayer(nn.Module):

    def __init__(self, tsv, lam):
        super(TSVLayer, self).__init__()
        self.tsv = tsv
        self.lam = lam

    def forward(self, x):
        if self.tsv is not None:

            x = x.half()
            y = self.lam[0] * self.tsv.repeat(1,x.shape[1],1)
            y = y.to(x.device)
            x = x.half() + y
            
            return x.half()
        
        else:
            
            return x.half()
        

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
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "layers"):
        return model.layers
    else:
        raise ValueError("Cannot find decoder layers in model.")

def get_mlp_layers(model: PreTrainedModel):
    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

def add_tsv_layers(model: PreTrainedModel, tsv: torch.Tensor, alpha: list, args):
    layers = get_layers(model)

    assert len(tsv) == len(layers), \
        f"TSV has {len(tsv)} vectors but model has {len(layers)} layers."

    # Import model-specific classes
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    except:
        Qwen2DecoderLayer, Qwen2Attention = None, None

    # Llama imports (may or may not exist for your model)
    try:
        from llm_layers import LlamaAttentionWrapper, LlamaDecoderLayerWrapper
    except:
        LlamaAttentionWrapper, LlamaDecoderLayerWrapper = None, None

    # -------------------------
    # Choose component type
    # -------------------------

    # -------------------------
    # 1. MLP INJECTION (works for Qwen and Llama)
    # -------------------------
    if args.component == "mlp":
        for i, layer in enumerate(layers):
            if i == args.str_layer:
                original_mlp = layer.mlp
                layer.mlp = nn.Sequential(
                    original_mlp,
                    TSVLayer(tsv[i], alpha)
                )
        return model

    # -------------------------
    # 2. ATTENTION INJECTION
    # -------------------------
    elif args.component == "attn":
        for i, layer in enumerate(layers):
            if i == args.str_layer:

                # Qwen-style attention
                if Qwen2Attention and isinstance(layer.self_attn, Qwen2Attention):
                    layer.self_attn = QwenAttentionWrapper(
                        layer.self_attn,
                        TSVLayer(tsv[i], alpha)
                    )

                # Llama-style attention
                elif LlamaAttentionWrapper and hasattr(layer, "self_attn"):
                    layer.self_attn = LlamaAttentionWrapper(
                        layer.self_attn,
                        TSVLayer(tsv[i], alpha)
                    )

                else:
                    print(f"[Warning] Cannot wrap attention of layer {i} (type={type(layer.self_attn)})")

        return model

    # -------------------------
    # 3. RESIDUAL INJECTION
    # -------------------------
    elif args.component == "res":
        for i, layer in enumerate(layers):
            if i == args.str_layer:

                # Qwen residual wrapper
                if Qwen2DecoderLayer and isinstance(layer, Qwen2DecoderLayer):
                    layers[i] = QwenDecoderLayerWrapper(
                        layer,
                        TSVLayer(tsv[i], alpha)
                    )

                # Llama residual wrapper
                elif LlamaDecoderLayerWrapper and isinstance(layer, LlamaDecoderLayerWrapper):
                    layers[i] = LlamaDecoderLayerWrapper(
                        layer,
                        TSVLayer(tsv[i], alpha)
                    )

                else:
                    # Fallback: inject after MLP (original Llama TSV trick)
                    orig_mlp = layer.mlp
                    layer.mlp = nn.Sequential(
                        orig_mlp,
                        TSVLayer(tsv[i], alpha)
                    )
                    print(f"[Warning] True residual wrapper not available for layer {i}. Applied post-MLP TSV instead.")

        return model

    else:
        raise ValueError(f"Unknown component type: {args.component}")
