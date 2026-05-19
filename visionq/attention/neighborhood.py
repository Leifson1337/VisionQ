import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext
from typing import Optional

@register_attention("neighborhood")
class NeighborhoodAttention(AttentionBackend):
    """
    Spatio-Temporal Neighborhood Attention (Optimized).
    Treats 3D volumes as sliding windows to avoid O(N^2) memory footprint.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 32,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, H_heads, N, D = q.shape

        # Spatial/Temporal shape inference
        T = context.temporal_dim or 1
        H_s, W_s = context.spatial_shape if context.spatial_shape else (int((N/T)**0.5), int((N/T)**0.5))
        window = context.window_size if context.window_size > 0 else 7

        if T * H_s * W_s != N:
            # Fallback to global if shape is unknown
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if mask is not None: attn += mask
            return (attn.softmax(dim=-1) @ v)

        # 3D Neighborhood (Simplified as Spatial windows per frame for baseline efficiency)
        # In a full spatio-temporal kernel, we would unfold T as well.
        # Here we reuse the optimized spatial logic across frames.
        q = q.reshape(B, H_heads, T, H_s, W_s, D).permute(0, 2, 1, 3, 4, 5).reshape(B * T, H_heads, H_s, W_s, D)
        k = k.reshape(B, H_heads, T, H_s, W_s, D).permute(0, 2, 1, 3, 4, 5).reshape(B * T, H_heads, H_s, W_s, D)
        v = v.reshape(B, H_heads, T, H_s, W_s, D).permute(0, 2, 1, 3, 4, 5).reshape(B * T, H_heads, H_s, W_s, D)

        pad = window // 2
        k_v_view = k.permute(0, 1, 4, 2, 3) # (B*T, H, D, H, W)
        v_v_view = v.permute(0, 1, 4, 2, 3)

        k_windows = F.pad(k_v_view, (pad, pad, pad, pad)).unfold(3, window, 1).unfold(4, window, 1).reshape(B*T, H_heads, D, H_s, W_s, -1)
        v_windows = F.pad(v_v_view, (pad, pad, pad, pad)).unfold(3, window, 1).unfold(4, window, 1).reshape(B*T, H_heads, D, H_s, W_s, -1)

        # Q: (B*T, H, H, W, D), K_win: (B*T, H, D, H, W, K)
        attn = torch.einsum('bhxwd,bhdxwk->bhxwk', q * self.scale, k_windows)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhxwk,bhdxwk->bhxwd', attn, v_windows)

        # Restore to (B, H_heads, N, D)
        out = out.reshape(B, T, H_heads, H_s, W_s, D).permute(0, 2, 1, 3, 4, 5).reshape(B, H_heads, N, D)
        return out
