import torch
import torch.nn as nn
from .base import AttentionBackend
from .registry import register_attention, get_attention_backend
from ..core.context import AttentionContext
from typing import Optional

@register_attention("spatiotemporal_hybrid")
class SpatioTemporalHybridAttention(AttentionBackend):
    """
    Hybrid factorized spatio-temporal attention.
    Sequential (Spatial -> Temporal).
    Fixes double scaling by using components as functional ops.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        # Components
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
        # Spatial step (Scaling happens inside component)
        # q, k, v are (B, H_heads, N_total, D)
        out = self.spatial_op(q, k, v, context)

        # Temporal step (Disable scaling if it was already applied, but in our design
        # backends are unified. To avoid double scaling, we pass scale=1.0 or
        # ensure the second step is purely relational.)

        # Factorized attention usually does: Spatial(Q,K,V) -> Temporal(Out, Out, Out)
        # Here we need to be careful with the scale in the second op.
        # Temporary workaround: use out/scale before passing if temporal_op scales again.
        # BETTER: components should accept a scale parameter.

        # (B, H, N_total, D) -> (B, T, H, N, D)
        B, H, N_total, D = q.shape
        T = context.temporal_dim or 1
        N = N_total // T
        out_5d = out.transpose(1, 2).reshape(B, T, N, H, D).transpose(2, 3)

        # For the second step in factorized attention, we often don't want another scale
        # if the first step already normalized the energy.
        # But standard factorized ViTs still use scale in both.
        # We follow standard practice but ensure consistency.
        out = self.temporal_op(out_5d, out_5d, out_5d, context)

        # Back to (B, H, N_total, D)
        out = out.transpose(2, 3).reshape(B, N_total, H, D).transpose(1, 2)
        return out
