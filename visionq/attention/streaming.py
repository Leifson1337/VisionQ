from __future__ import annotations

import torch

from ..core.context import AttentionContext
from ..kernels.triton.attention_kernel import ReferenceBlockwiseAttentionKernel
from .base import AttentionBackend
from .registry import AttentionBackendName, register_attention


@register_attention(AttentionBackendName.CHUNKED_STREAMING)
class ChunkedStreamingAttention(AttentionBackend):
    """Reference blockwise exact attention with chunked query/key processing."""

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
        self.kernel = ReferenceBlockwiseAttentionKernel()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 32,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.validate_qkv(q, k, v)
        if self.training and self.attn_drop_p:
            raise ValueError("ChunkedStreamingAttention does not support dropout")
        return self.kernel.forward(q, k, v, context, block_size=block_size, mask=mask)
