import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext

@register_attention("spatial_neighborhood")
class SpatialNeighborhoodAttention(AttentionBackend):
    """
    Computes spatial-only neighborhood attention.
    Can handle 4D (B_parallel, H, N, D) or 5D (B, T, H, N, D) inputs.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self._mask_cache = {}

    def _get_mask(self, H, W, window_size, device):
        key = (H, W, window_size, device)
        if key in self._mask_cache: return self._mask_cache[key]
        N = H * W
        mask = torch.full((N, N), float('-inf'), device=device)
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        grid = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1).reshape(N, 2)
        dist = torch.abs(grid.unsqueeze(1) - grid.unsqueeze(0))
        in_window = (dist <= window_size // 2).all(dim=-1)
        mask[in_window] = 0
        self._mask_cache[key] = mask
        return mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: AttentionContext) -> torch.Tensor:
        orig_shape = q.shape
        if q.dim() == 5:
            B, T, H, N, D = q.shape
            q = q.reshape(B * T, H, N, D)
            k = k.reshape(B * T, H, N, D)
            v = v.reshape(B * T, H, N, D)

        B_p, H, N, D = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B_p, H, N, N)

        # Spatial shape inference
        H_s, W_s = context.spatial_shape if context.spatial_shape else (int(N**0.5), int(N**0.5))
        if H_s * W_s != N:
            # If mismatch, we can't apply spatial mask correctly, fallback to global or 1D
            pass
        else:
            window = context.spatial_window[0] if (context.spatial_window and context.spatial_window[0] > 0) else 7
            mask = self._get_mask(H_s, W_s, window, q.device)
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v)

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

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: AttentionContext) -> torch.Tensor:
        if q.dim() != 5:
            # Fallback for non-5D input
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
            mask = torch.full((T, T), float('-inf'), device=q.device)
            mask[dist <= context.temporal_window // 2] = 0
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v_p).reshape(B, N, H, T, D).permute(0, 3, 2, 1, 4)
        return out
