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
    Neighborhood Attention (NA) implementation.
    Restricts attention to a local neighborhood around each token.
    This version includes better support for 2D spatial structures.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._mask_cache = {}

    def _get_2d_neighborhood_mask(self, H: int, W: int, window_size: int, device: torch.device):
        """
        Generates a 2D neighborhood mask efficiently with caching.
        """
        cache_key = (H, W, window_size, device)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        N = H * W
        mask = torch.full((N, N), float('-inf'), device=device)

        # Grid coordinates
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing='ij')
        grid = torch.stack([grid_h, grid_w], dim=-1).reshape(N, 2) # (N, 2)

        # Distance calculation
        dist = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0)) # (N, N, 2)
        in_window = (dist[:, :, 0] <= window_size // 2) & (dist[:, :, 1] <= window_size // 2)

        mask[in_window] = 0
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, x, context: AttentionContext):
        """
        Forward pass for Neighborhood Attention.

        Args:
            x (torch.Tensor): Input tensor (B, N, C).
            context (AttentionContext): Context containing spatial_shape and window_size.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if context.window_size > 0:
            if context.spatial_shape:
                H, W = context.spatial_shape
                if H * W == N:
                    mask = self._get_2d_neighborhood_mask(H, W, context.window_size, x.device)
                    attn = attn + mask
                else:
                    # Fallback to 1D if shape doesn't match
                    mask = self._get_1d_neighborhood_mask(N, context.window_size, x.device)
                    attn = attn + mask
            else:
                mask = self._get_1d_neighborhood_mask(N, context.window_size, x.device)
                attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _get_1d_neighborhood_mask(self, N: int, window_size: int, device: torch.device):
        indices = torch.arange(N, device=device)
        dist = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))
        mask = torch.full((N, N), float('-inf'), device=device)
        mask[dist <= window_size // 2] = 0
        return mask
