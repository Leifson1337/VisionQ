import torch
from abc import ABC, abstractmethod
from typing import Optional, Literal
from ...core.context import AttentionContext

MaskType = Literal["none", "causal", "neighborhood", "block_sparse"]

class AttentionKernel(ABC):
    """
    Unified GPU Attention Kernel Abstraction.
    Treats attention as a memory-bandwidth limited streaming operation.
    """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 64,
        mask_type: MaskType = "none"
    ) -> torch.Tensor:
        """
        Executes the block-based attention pipeline.

        Pipeline:
        1. Tile Loading (HBM -> SRAM)
        2. Partial QK^T Score Computation
        3. Online Softmax Update (Numerically Stable)
        4. Incremental V Accumulation
        """
        pass

    @staticmethod
    def get_optimal_block_size(device: torch.device, head_dim: int) -> int:
        """Heuristic for SRAM-aware tiling based on GPU architecture."""
        if device.type == "cuda":
            # 64-128 is typical for modern GPUs to fit in shared memory
            return 128 if head_dim <= 64 else 64
        return 32
