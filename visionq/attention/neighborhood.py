import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext
from typing import Optional, Tuple

@register_attention("neighborhood")
class NeighborhoodAttention(AttentionBackend):
    """
    Spatio-Temporal Neighborhood Attention.
    Implements localized attention for 2D images and 3D videos.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._mask_cache = {}

    def _get_3d_neighborhood_mask(self, T: int, H: int, W: int, window_size: int, device: torch.device):
        cache_key = (T, H, W, window_size, device)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        N = T * H * W
        mask = torch.full((N, N), float('-inf'), device=device)

        coords_t = torch.arange(T, device=device)
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        grid_t, grid_h, grid_w = torch.meshgrid(coords_t, coords_h, coords_w, indexing='ij')
        grid = torch.stack([grid_t, grid_h, grid_w], dim=-1).reshape(N, 3)

        dist = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0))
        in_window = (dist[:, :, 0] <= window_size // 2) & \
                    (dist[:, :, 1] <= window_size // 2) & \
                    (dist[:, :, 2] <= window_size // 2)

        mask[in_window] = 0
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: AttentionContext) -> torch.Tensor:
        B, N, C = q.shape if q.dim() == 3 else (q.shape[0], q.shape[2], q.shape[1] * q.shape[3])

        if q.dim() == 4: # (B, H, N, D)
            q = q.transpose(1, 2).reshape(B, N, -1)
            k = k.transpose(1, 2).reshape(B, N, -1)
            v = v.transpose(1, 2).reshape(B, N, -1)

        q_heads = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k_heads = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v_heads = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        q_heads = q_heads * self.scale
        attn = (q_heads @ k_heads.transpose(-2, -1))

        window_size = context.window_size if context.window_size > 0 else 7

        if context.modality == "video" and context.temporal_dim and context.spatial_shape:
            T, (H, W) = context.temporal_dim, context.spatial_shape
            mask = self._get_3d_neighborhood_mask(T, H, W, window_size, q.device)
            attn = attn + mask
        elif context.spatial_shape:
            H, W = context.spatial_shape
            mask = self._get_3d_neighborhood_mask(1, H, W, window_size, q.device)
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v_heads).transpose(1, 2).reshape(B, N, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
