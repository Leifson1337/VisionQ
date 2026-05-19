from __future__ import annotations

from typing import Any, cast

import torch

from ..core.context import AttentionContext
from .base import AttentionBackend
from .registry import AttentionBackendName, get_attention_backend, register_attention


@register_attention(AttentionBackendName.SPATIOTEMPORAL_HYBRID)
class SpatioTemporalHybridAttention(AttentionBackend):
    """Factorized spatial attention followed by temporal attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__(dim, num_heads=num_heads, attn_drop=attn_drop)
        spatial_cls = cast(Any, get_attention_backend(AttentionBackendName.SPATIAL_NEIGHBORHOOD))
        temporal_cls = cast(Any, get_attention_backend(AttentionBackendName.TEMPORAL_NEIGHBORHOOD))
        self.spatial_op = spatial_cls(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.temporal_op = temporal_cls(dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 32,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise ValueError("SpatioTemporalHybridAttention does not support attn_mask")
        self.validate_qkv(q, k, v)
        if context.temporal_dim is None or context.spatial_shape is None:
            raise ValueError("temporal_dim and spatial_shape are required for hybrid attention")
        b, heads, total_tokens, dim = q.shape
        t = context.temporal_dim
        spatial_tokens = context.spatial_shape[0] * context.spatial_shape[1]
        if total_tokens != t * spatial_tokens:
            raise ValueError(
                f"expected {t * spatial_tokens} tokens from context, got {total_tokens}"
            )
        q5 = q.reshape(b, heads, t, spatial_tokens, dim).permute(0, 2, 1, 3, 4)
        k5 = k.reshape(b, heads, t, spatial_tokens, dim).permute(0, 2, 1, 3, 4)
        v5 = v.reshape(b, heads, t, spatial_tokens, dim).permute(0, 2, 1, 3, 4)
        spatial = self.spatial_op(q5, k5, v5, context, block_size=block_size)
        temporal = self.temporal_op(spatial, spatial, spatial, context, block_size=block_size)
        return temporal.permute(0, 2, 1, 3, 4).reshape(b, heads, total_tokens, dim)
