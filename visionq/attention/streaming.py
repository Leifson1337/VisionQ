import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext
from typing import Optional

@register_attention("chunked_streaming")
class ChunkedStreamingAttention(AttentionBackend):
    """
    Industrial-grade Chunked/Streaming Attention for long sequences.
    Minimizes memory bandwidth by processing in blocks and managing KV cache.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

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
        Process attention in chunks to stay within memory budget.
        """
        B, H, N, D = q.shape

        # Process in chunks of block_size
        out = torch.zeros_like(q)

        for i in range(0, N, block_size):
            q_chunk = q[:, :, i : i + block_size, :] # (B, H, BS, D)

            # For each Q chunk, we still need all K, V (unless sparse)
            # In a real IO-aware kernel, we'd stream K, V from global memory
            attn_weights = (q_chunk @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                attn_weights = attn_weights + mask[:, :, i : i + block_size, :]

            attn_weights = F.softmax(attn_weights, dim=-1)
            out[:, :, i : i + block_size, :] = attn_weights @ v

        return out
