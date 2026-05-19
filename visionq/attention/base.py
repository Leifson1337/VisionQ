import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional
from ..core.context import AttentionContext

class AttentionBackend(nn.Module, ABC):
    """
    Abstract base class for all attention backends.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 32,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Industrial-grade Attention Kernel Interface.

        Args:
            q: Query tensor (B, H, N, D)
            k: Key tensor (B, H, N, D)
            v: Value tensor (B, H, N, D)
            context: Metadata governing the execution
            block_size: Tiling/blocking parameter for GPU execution
            mask: Optional attention mask
        """
        pass
