import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext

@register_attention("sparse")
class SparseAttention(AttentionBackend):
    """
    Sparse Attention Backend.
    Provides a framework for block-sparse or strided attention implementations.
    Currently implements a strided/dilated global attention as a functional baseline.
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context: AttentionContext):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Strided/Sparse logic using dilation if specified
        dilation = context.dilation
        if dilation > 1:
            # Simple strided attention for long sequences
            # In a real industrial sparse kernel, this would be a fused op.
            q = q * self.scale
            # Here we just apply the dilation as a mask or subset for demonstration
            # while keeping it 'real' and functional.
            indices = torch.arange(0, N, dilation, device=x.device)
            k_sparse = k[:, :, indices, :]
            v_sparse = v[:, :, indices, :]

            attn = (q @ k_sparse.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v_sparse).transpose(1, 2).reshape(B, N, C)
        else:
            # Fallback to standard scaled dot product
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
