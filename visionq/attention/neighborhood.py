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
    Supports local attention for 2D images and 3D videos.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._mask_cache = {}

    def _get_3d_neighborhood_mask(self, T: int, H: int, W: int, window_size: int, device: torch.device):
        """
        Generates a 3D (Spatio-Temporal) neighborhood mask.
        """
        cache_key = (T, H, W, window_size, device)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        N = T * H * W
        mask = torch.full((N, N), float('-inf'), device=device)

        # Grid coordinates
        coords_t = torch.arange(T, device=device)
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        grid_t, grid_h, grid_w = torch.meshgrid(coords_t, coords_h, coords_w, indexing='ij')
        grid = torch.stack([grid_t, grid_h, grid_w], dim=-1).reshape(N, 3) # (N, 3)

        # Distance calculation
        # Spatial window and Temporal window (here using same window_size for all)
        dist = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0)) # (N, N, 3)
        in_window = (dist[:, :, 0] <= window_size // 2) & \
                    (dist[:, :, 1] <= window_size // 2) & \
                    (dist[:, :, 2] <= window_size // 2)

        mask[in_window] = 0
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, x, context: AttentionContext):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if context.window_size > 0:
            if context.modality == "video" and context.temporal_dim and context.spatial_shape:
                T = context.temporal_dim
                H, W = context.spatial_shape
                if T * H * W == N:
                    mask = self._get_3d_neighborhood_mask(T, H, W, context.window_size, x.device)
                    attn = attn + mask
            elif context.spatial_shape:
                H, W = context.spatial_shape
                if H * W == N:
                    # Reusing the 2D logic (implicitly via 3D with T=1)
                    mask = self._get_3d_neighborhood_mask(1, H, W, context.window_size, x.device)
                    attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
