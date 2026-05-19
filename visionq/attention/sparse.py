from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.context import AttentionContext
from .base import AttentionBackend
from .registry import AttentionBackendName, register_attention


@register_attention(AttentionBackendName.SPARSE)
class SparseAttention(AttentionBackend):
    """Reference strided key/value attention controlled by ``context.dilation``."""

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
        self.validate_qkv(q, k, v)
        stride = context.dilation
        k_sparse = k[:, :, ::stride, :]
        v_sparse = v[:, :, ::stride, :]
        sparse_mask = None if mask is None else mask[..., ::stride]
        out = F.scaled_dot_product_attention(
            q,
            k_sparse,
            v_sparse,
            attn_mask=sparse_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )
        return out
