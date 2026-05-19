import torch
import torch.nn as nn
from typing import Optional
from .base import AttentionBackend
from .registry import register_attention, get_attention_backend
from ..core.context import AttentionContext

@register_attention("spatiotemporal_hybrid")
class SpatioTemporalHybridAttention(AttentionBackend):
    """
    Hybrid factorized spatio-temporal attention.
    Factorized into Spatial and Temporal attention steps.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        # Use underlying backends as pure compute ops (without redundant projections if possible)
        # In this architecture, backends themselves might have projections.
        # To fix this properly, backends should ideally be functional or we should use them carefully.
        self.spatial_op = get_attention_backend("spatial_neighborhood")(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.temporal_op = get_attention_backend("temporal_neighborhood")(dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 32,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 1. Spatial Step
        # Expects (B, H, N, D)
        out = self.spatial_op(q, k, v, context)

        # 2. Temporal Step
        # Spatial backend might have reshaped or flattened.
        # Temporal op expects (B, T, H, N, D) if coming from structured backbone.
        # For factorized attention, we typically do Spatial -> Temporal

        # Ensure 'out' is formatted correctly for temporal op
        # Assuming q was (B, H, N_total, D)
        B, H, N_total, D = q.shape
        T = context.temporal_dim or 1
        N = N_total // T

        # (B, H, N_total, D) -> (B, T, H, N, D)
        out = out.transpose(1, 2).reshape(B, T, N, H, D).transpose(2, 3)

        out = self.temporal_op(out, out, out, context)

        # Back to (B, H, N_total, D)
        out = out.transpose(2, 3).reshape(B, N_total, H, D).transpose(1, 2)
        return out
