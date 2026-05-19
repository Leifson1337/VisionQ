import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext

@register_attention("flash")
class FlashAttention(AttentionBackend):
    """
    Flash Attention Backend using PyTorch SDPA.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.attn_drop_p = attn_drop

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: AttentionContext) -> torch.Tensor:
        # SDPA expects (B, H, N, D)
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False
        )
