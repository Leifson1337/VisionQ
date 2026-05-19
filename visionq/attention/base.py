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
    def forward(self, x, context: AttentionContext):
        """
        Computes attention.

        Args:
            x: Input tensor (B, N, C)
            context: AttentionContext containing metadata
        """
        pass
