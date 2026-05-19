import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ..core.context import AttentionContext

class AttentionBackend(nn.Module, ABC):
    """
    Abstract base class for all attention backends.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: AttentionContext) -> torch.Tensor:
        """
        Computes attention.

        Args:
            q: Query tensor (B, H, N, D) or (B, N, C)
            k: Key tensor
            v: Value tensor
            context: AttentionContext containing metadata
        """
        pass
