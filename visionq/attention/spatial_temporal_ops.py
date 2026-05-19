import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext
from typing import Optional

@register_attention("spatial_neighborhood")
class SpatialNeighborhoodAttention(AttentionBackend):
    """
    Computes spatial-only neighborhood attention efficiently using unfolding.
    Avoids O(N^2) memory bottlenecks by using sliding window logic.
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
        """
        Efficient Neighborhood Attention using unfolding.
        Expects Q, K, V projected.
        """
        # q, k, v are (B, T, H_heads, N_spatial, D) or (B, H_heads, N_spatial, D)
        orig_shape = q.shape
        if q.dim() == 5:
            B, T, H_heads, N_spatial, D = q.shape
            q = q.reshape(B * T, H_heads, N_spatial, D)
            k = k.reshape(B * T, H_heads, N_spatial, D)
            v = v.reshape(B * T, H_heads, N_spatial, D)

        B_p, H_heads, N_spatial, D = q.shape

        # Spatial shape inference
        H_s, W_s = context.spatial_shape if context.spatial_shape else (int(N_spatial**0.5), int(N_spatial**0.5))
        window = context.spatial_window[0] if (context.spatial_window and context.spatial_window[0] > 0) else 7

        if H_s * W_s != N_spatial:
            # Fallback for irregular shapes (e.g. sequence without spatial context)
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if mask is not None: attn += mask
            attn = attn.softmax(dim=-1)
            out = attn @ v
        else:
            # Standard Neighborhood Attention (Local Sliding Window)
            # 1. Reshape to image format
            q = q.reshape(B_p, H_heads, H_s, W_s, D)
            k = k.reshape(B_p, H_heads, H_s, W_s, D)
            v = v.reshape(B_p, H_heads, H_s, W_s, D)

            # 2. Extract windows for K, V
            # (B, H, H_s, W_s, D) -> (B, H, D, H_s, W_s)
            k = k.permute(0, 1, 4, 2, 3)
            v = v.permute(0, 1, 4, 2, 3)

            # Padding for boundaries
            pad = window // 2
            k_pad = F.pad(k, (pad, pad, pad, pad))
            v_pad = F.pad(v, (pad, pad, pad, pad))

            # Unfold to get local windows: (B, H, D, H_s, W_s, W_size, W_size)
            k_windows = k_pad.unfold(3, window, 1).unfold(4, window, 1)
            v_windows = v_pad.unfold(3, window, 1).unfold(4, window, 1)

            # 3. Compute Attention
            # Q: (B, H, H_s, W_s, D)
            # K_win: (B, H, D, H_s, W_s, W_size*W_size)
            k_windows = k_windows.reshape(B_p, H_heads, D, H_s, W_s, -1)
            v_windows = v_windows.reshape(B_p, H_heads, D, H_s, W_s, -1)

            # Attention scores: (B, H, H_s, W_s, W_size*W_size)
            # q * scale @ k_win
            attn = torch.einsum('bhxwd,bhdxwk->bhxwk', q * self.scale, k_windows)

            if mask is not None:
                # Local masking would be applied here if needed
                pass

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # 4. Accumulate values
            # (B, H, H_s, W_s, W_size*W_size) @ (B, H, H_s, W_s, W_size*W_size, D)
            out = torch.einsum('bhxwk,bhdxwk->bhxwd', attn, v_windows)
            out = out.reshape(B_p, H_heads, N_spatial, D)

        if len(orig_shape) == 5:
            out = out.reshape(orig_shape)
        return out

@register_attention("temporal_neighborhood")
class TemporalNeighborhoodAttention(AttentionBackend):
    """
    Computes temporal-only neighborhood attention across the T dimension.
    Input must be 5D: (B, T, H, N, D)
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
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
        if q.dim() != 5:
            return q

        B, T, H, N, D = q.shape
        # Permute to (B, N, H, T, D) to attend over T
        q_p = q.permute(0, 3, 2, 1, 4).reshape(B * N, H, T, D)
        k_p = k.permute(0, 3, 2, 1, 4).reshape(B * N, H, T, D)
        v_p = v.permute(0, 3, 2, 1, 4).reshape(B * N, H, T, D)

        q_p = q_p * self.scale
        attn = (q_p @ k_p.transpose(-2, -1)) # (B*N, H, T, T)

        if context.temporal_window > 0:
            indices = torch.arange(T, device=q.device)
            dist = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))
            t_mask = torch.full((T, T), float('-inf'), device=q.device)
            t_mask[dist <= context.temporal_window // 2] = 0
            attn = attn + t_mask

        if mask is not None:
             attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v_p).reshape(B, N, H, T, D).permute(0, 3, 2, 1, 4)
        return out
