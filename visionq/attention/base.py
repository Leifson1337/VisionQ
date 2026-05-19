from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..core.context import AttentionContext
from ..core.validation import validate_same_qkv


class AttentionBackend(nn.Module, ABC):
    """Base class for attention backends using projected Q/K/V tensors."""

    def __init__(self, dim: int, num_heads: int = 1, attn_drop: float = 0.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop_p = float(attn_drop)

    @staticmethod
    def validate_qkv(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, allow_5d: bool = False
    ) -> None:
        validate_same_qkv(q, k, v, allow_5d=allow_5d)

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 32,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention for projected Q/K/V tensors."""
        raise NotImplementedError


__all__ = ["AttentionBackend"]
