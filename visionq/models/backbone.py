import torch
import torch.nn as nn
from typing import Optional, Union, Dict
from ..core.tensor import SpatioTemporalTensor, as_st_tensor
from ..core.context import AttentionContext
from ..runtime.dispatcher import AttentionDispatcher
from ..attention.registry import get_attention_backend

class VisionBackboneBlock(nn.Module):
    """
    Industrial-grade Vision/Video Transformer Block.
    Shares weights across backends and lazily initializes compute ops.
    """

    # Shared instance pool for compute-only backends to save memory across blocks
    # if they don't hold parameters.
    _backend_pool: Dict[str, nn.Module] = {}

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.dispatcher = AttentionDispatcher()

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

        # Local backend cache for this block (only contains used ones)
        self.backends = nn.ModuleDict()

    def _get_backend(self, name: str) -> nn.Module:
        """Lazily instantiates the backend."""
        if name not in self.backends:
            cls = get_attention_backend(name)
            # Backend is compute-only, no parameters needed
            self.backends[name] = cls(self.dim, num_heads=self.num_heads)

            # Ensure correct device/dtype
            ref = next(self.parameters())
            self.backends[name].to(device=ref.device, dtype=ref.dtype)

        return self.backends[name]

    def forward(self, x: Union[torch.Tensor, SpatioTemporalTensor], context: Optional[AttentionContext] = None) -> SpatioTemporalTensor:
        st_x = as_st_tensor(x)
        if context is None:
            context = AttentionContext.from_st_tensor(st_x)

        residual = st_x.flatten_all()
        x_norm = self.norm1(residual)

        B, N_total, C = x_norm.shape
        qkv = self.qkv(x_norm).reshape(B, N_total, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Dispatch to optimal backend
        backend_name = self.dispatcher.select(context)
        backend = self._get_backend(backend_name)

        out = backend(q, k, v, context)

        # Standardize output
        if out.dim() == 5: # (B, T, H, N, D)
            out = out.transpose(2, 3).reshape(B, N_total, C)
        elif out.dim() == 4: # (B, H, N, D)
            out = out.transpose(1, 2).reshape(B, N_total, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        hidden_states = residual + out
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return SpatioTemporalTensor(
            hidden_states,
            st_x.modality,
            st_x.spatial_shape,
            st_x.temporal_dim
        )

class VisionBackbone(nn.Module):
    def __init__(self, depth: int, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            VisionBackboneBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x: Union[torch.Tensor, SpatioTemporalTensor], context: Optional[AttentionContext] = None) -> SpatioTemporalTensor:
        st_x = as_st_tensor(x)
        for block in self.blocks:
            st_x = block(st_x, context)
        return st_x
