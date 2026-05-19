import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext

@register_attention("sparse")
class SparseAttention(AttentionBackend):
    """
    Block-Sparse Attention Abstraction.
    Correctly handles (B, H, N, D) inputs.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.attn_drop_p = attn_drop

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
        block_size: int = 32,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Input: (B, H, N, D)
        B, H, N, D = q.shape
        stride = max(1, context.dilation)

        # Sparse subset of keys/values
        k_sparse = k[:, :, ::stride, :]
        v_sparse = v[:, :, ::stride, :]

        scale = D ** -0.5
        attn = (q * scale) @ k_sparse.transpose(-2, -1)

        if mask is not None:
            # Mask must be sliced to match sparse dimensions
            m_sparse = mask[:, :, :, ::stride]
            attn += m_sparse

        attn = attn.softmax(dim=-1)
        out = attn @ v_sparse
        return out
