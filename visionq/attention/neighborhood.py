import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext

@register_attention("neighborhood")
class NeighborhoodAttention(AttentionBackend):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self._mask_cache = {}

    def _get_3d_mask(self, T, H, W, window_size, device):
        key = (T, H, W, window_size, device)
        if key in self._mask_cache: return self._mask_cache[key]
        N = T * H * W
        mask = torch.full((N, N), float('-inf'), device=device)
        grid_t, grid_h, grid_w = torch.meshgrid(torch.arange(T, device=device), torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid = torch.stack([grid_t, grid_h, grid_w], dim=-1).reshape(N, 3)
        dist = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0))
        in_window = (dist <= window_size // 2).all(dim=-1)
        mask[in_window] = 0
        self._mask_cache[key] = mask
        return mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: AttentionContext) -> torch.Tensor:
        B, H, N, D = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        window = context.window_size if context.window_size > 0 else 7
        if context.modality == "video" and context.temporal_dim and context.spatial_shape:
            T, (H_s, W_s) = context.temporal_dim, context.spatial_shape
            if T * H_s * W_s == N:
                attn = attn + self._get_3d_mask(T, H_s, W_s, window, q.device)
        elif context.spatial_shape:
            H_s, W_s = context.spatial_shape
            if H_s * W_s == N:
                attn = attn + self._get_3d_mask(1, H_s, W_s, window, q.device)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v)
        return out
