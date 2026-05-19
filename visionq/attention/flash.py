from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.context import AttentionContext
from .base import AttentionBackend
from .registry import AttentionBackendName, register_attention


@register_attention(AttentionBackendName.FLASH)
class FlashAttention(AttentionBackend):
    """PyTorch SDPA backend. Kernel selection is delegated to PyTorch."""

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
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=context.causal,
        )
