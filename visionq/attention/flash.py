import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext

@register_attention("flash")
class FlashAttention(AttentionBackend):
    """
    Industrial-grade Flash Attention backend.
    Uses PyTorch's scaled_dot_product_attention for optimal performance.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_p = attn_drop

    def forward(self, x, context: AttentionContext):
        """
        Forward pass using Flash Attention (SDPA).

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            context (AttentionContext): Execution context.
        """
        B, N, C = x.shape
        # (B, N, 3, H, D) -> (3, B, H, N, D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention (handles Flash Attention, Memory Efficient, etc. internally)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False
        )

        # (B, H, N, D) -> (B, N, H*D)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
