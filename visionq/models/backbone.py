import torch
import torch.nn as nn
from typing import Optional, Union

from ..attention.registry import get_attention_backend
from ..runtime.dispatcher import AttentionDispatcher
from ..core.context import AttentionContext
from ..core.tensor import STTensor, as_st_tensor

class VisionBackboneBlock(nn.Module):
    """
    Industrial-grade Vision/Video Transformer Block.
    Attention logic is completely decoupled and handled by the runtime dispatcher.
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Initialize backends eagerly to ensure parameter tracking
        self.backends = nn.ModuleDict()
        from ..attention.registry import ATTENTION_REGISTRY
        for name, cls in ATTENTION_REGISTRY.items():
            self.backends[name] = cls(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.dispatcher = AttentionDispatcher()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )


    def forward(self, x: Union[torch.Tensor, STTensor], context: Optional[AttentionContext] = None) -> STTensor:
        """
        Forward pass with dynamic routing.
        QKV projection is done here, but Attention computation is delegated to the Backend.
        """
        st_x = as_st_tensor(x)
        if context is None:
            context = AttentionContext.from_st_tensor(st_x)

        residual = st_x.unwrap()
        x = self.norm1(residual)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, D)

        # Dispatch to optimal backend
        backend_name = self.dispatcher.select(context)
        backend = self.backends[backend_name]

        hidden_states = backend(q, k, v, context)
        hidden_states = residual + hidden_states

        # MLP
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        # Return STTensor to preserve metadata for the next layer
        return STTensor(hidden_states, st_x.modality, st_x.spatial_shape, st_x.temporal_dim)

class VisionBackbone(nn.Module):
    """
    A unified Vision/Video Transformer Backbone.
    """
    def __init__(self, depth: int, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            VisionBackboneBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x: Union[torch.Tensor, STTensor], context: Optional[AttentionContext] = None) -> STTensor:
        # Initial wrapping
        x = as_st_tensor(x)

        for block in self.blocks:
            x = block(x, context)
        return x
