import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext

@register_attention("flash")
class FlashAttention(AttentionBackend):
    """
    Flash Attention Backend using PyTorch Scaled Dot Product Attention.
    Optimized for IO-aware execution on GPUs.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.attn_drop_p = attn_drop
        # projection remains part of backend as per previous design,
        # but forward now takes q,k,v.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: AttentionContext) -> torch.Tensor:
        """
        Forward pass using SDPA.
        Expects q, k, v in shape (B, H, N, D) or (B, N, C) that can be reshaped.
        """
        B, N, C = q.shape if q.dim() == 3 else (q.shape[0], q.shape[2], q.shape[1] * q.shape[3])

        if q.dim() == 3:
            # Reshape to (B, H, N, D)
            q = q.reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = k.reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = v.reshape(B, N, self.num_heads, -1).transpose(1, 2)

        # PyTorch native SDPA
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False
        )

        # Back to (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
