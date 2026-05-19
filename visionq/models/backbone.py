import torch
import torch.nn as nn
from typing import Optional, Union

from ..attention.registry import get_attention_backend
from ..runtime.dispatcher import AttentionDispatcher
from ..core.context import AttentionContext
from ..core.tensor import STTensor, as_st_tensor

class VisionBackboneBlock(nn.Module):
    """
    Industrial-grade Vision/Video Transformer Block with dynamic attention routing.
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        # We store backends in a ModuleDict.
        # For industrial-grade production, we only instantiate what we might need.
        self.backends = nn.ModuleDict()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias

        self.dispatcher = AttentionDispatcher()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def _get_backend(self, name: str) -> nn.Module:
        """Lazily instantiates backends to save memory."""
        if name not in self.backends:
            backend_cls = get_attention_backend(name)
            backend = backend_cls(self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias)

            # Ensure the new backend is on the correct device and dtype
            # We use a parameter from this module as a reference.
            reference_param = next(self.parameters(), None)
            if reference_param is not None:
                backend.to(device=reference_param.device, dtype=reference_param.dtype)

            self.backends[name] = backend
        return self.backends[name]

    def forward(self, x: Union[torch.Tensor, STTensor], context: Optional[AttentionContext] = None) -> torch.Tensor:
        """
        Forward pass with dynamic routing.

        Args:
            x: Input tensor or STTensor.
            context: Optional AttentionContext. If missing, it's derived from STTensor.
        """
        st_x = as_st_tensor(x)
        if context is None:
            context = AttentionContext.from_st_tensor(st_x)

        residual = st_x.unwrap()
        x = self.norm1(residual)

        # Dispatch
        backend_name = self.dispatcher.select(context)
        backend = self._get_backend(backend_name)

        x = backend(x, context)
        x = residual + x

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

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

    def forward(self, x: Union[torch.Tensor, STTensor], context: Optional[AttentionContext] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, context)
        return x
