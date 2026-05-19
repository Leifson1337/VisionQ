from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.context import AttentionContext
from .base import AttentionBackend
from .registry import AttentionBackendName, register_attention


def _local_attention_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    spatial_shape: tuple[int, int],
    window: tuple[int, int],
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    bp, heads, tokens, dim = q.shape
    height, width = spatial_shape
    wh, ww = window
    if height * width != tokens:
        raise ValueError(f"spatial_shape {spatial_shape} does not match token count {tokens}")
    q_img = q.reshape(bp, heads, height, width, dim)
    k_img = k.reshape(bp, heads, height, width, dim).permute(0, 1, 4, 2, 3)
    v_img = v.reshape(bp, heads, height, width, dim).permute(0, 1, 4, 2, 3)
    pad_h, pad_w = wh // 2, ww // 2
    k_windows = F.pad(k_img, (pad_w, pad_w, pad_h, pad_h)).unfold(3, wh, 1).unfold(4, ww, 1)
    v_windows = F.pad(v_img, (pad_w, pad_w, pad_h, pad_h)).unfold(3, wh, 1).unfold(4, ww, 1)
    k_windows = k_windows.reshape(bp, heads, dim, height, width, wh * ww)
    v_windows = v_windows.reshape(bp, heads, dim, height, width, wh * ww)
    scores = torch.einsum("bhxyd,bhdxyk->bhxyk", q_img * (dim**-0.5), k_windows)
    attn = torch.softmax(scores, dim=-1)
    attn = F.dropout(attn, p=dropout_p, training=training)
    out = torch.einsum("bhxyk,bhdxyk->bhxyd", attn, v_windows)
    return out.reshape(bp, heads, tokens, dim)


@register_attention(AttentionBackendName.SPATIAL_NEIGHBORHOOD)
class SpatialNeighborhoodAttention(AttentionBackend):
    """Reference spatial sliding-window attention for image tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        del qkv_bias, proj_drop
        super().__init__(dim, num_heads=num_heads, attn_drop=attn_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 32,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del block_size
        if mask is not None:
            raise ValueError("SpatialNeighborhoodAttention does not support attn_mask")
        self.validate_qkv(q, k, v, allow_5d=True)
        window = context.spatial_window or (context.window_size or 3, context.window_size or 3)
        if q.dim() == 5:
            b, t, heads, tokens, dim = q.shape
            if context.spatial_shape is None:
                raise ValueError("spatial_shape is required for 5D SpatialNeighborhoodAttention")
            q_flat = q.reshape(b * t, heads, tokens, dim)
            k_flat = k.reshape(b * t, heads, tokens, dim)
            v_flat = v.reshape(b * t, heads, tokens, dim)
            out = _local_attention_2d(
                q_flat,
                k_flat,
                v_flat,
                context.spatial_shape,
                window,
                self.attn_drop_p,
                self.training,
            )
            return out.reshape(b, t, heads, tokens, dim)
        if context.spatial_shape is None:
            raise ValueError("spatial_shape is required for SpatialNeighborhoodAttention")
        return _local_attention_2d(
            q, k, v, context.spatial_shape, window, self.attn_drop_p, self.training
        )


@register_attention(AttentionBackendName.TEMPORAL_NEIGHBORHOOD)
class TemporalNeighborhoodAttention(AttentionBackend):
    """Reference temporal sliding-window attention for video tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        del qkv_bias, proj_drop
        super().__init__(dim, num_heads=num_heads, attn_drop=attn_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 32,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del block_size
        self.validate_qkv(q, k, v, allow_5d=True)
        if q.dim() != 5:
            raise ValueError(
                "TemporalNeighborhoodAttention expects (B, T, heads, spatial_tokens, D)"
            )
        b, t, heads, tokens, dim = q.shape
        q_t = q.permute(0, 3, 2, 1, 4).reshape(b * tokens, heads, t, dim)
        k_t = k.permute(0, 3, 2, 1, 4).reshape(b * tokens, heads, t, dim)
        v_t = v.permute(0, 3, 2, 1, 4).reshape(b * tokens, heads, t, dim)
        attn_mask = mask
        if context.temporal_window:
            idx = torch.arange(t, device=q.device)
            allowed = (idx[:, None] - idx[None, :]).abs() <= context.temporal_window // 2
            local_mask = torch.zeros((t, t), device=q.device, dtype=q.dtype).masked_fill(
                ~allowed, float("-inf")
            )
            attn_mask = local_mask if attn_mask is None else attn_mask + local_mask
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=context.causal,
        )
        return out.reshape(b, tokens, heads, t, dim).permute(0, 3, 2, 1, 4)
