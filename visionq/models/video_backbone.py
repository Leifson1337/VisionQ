import torch
import torch.nn as nn
from typing import Optional, Union
from ..core.tensor import SpatioTemporalTensor, as_st_tensor
from ..core.context import AttentionContext
from .backbone import VisionBackboneBlock

class VideoBackboneBlock(VisionBackboneBlock):
    """
    Native Video Transformer Block.
    Optimized for structured 3D tensors.
    """
    def forward(self, x: Union[torch.Tensor, SpatioTemporalTensor], context: Optional[AttentionContext] = None) -> SpatioTemporalTensor:
        st_x = as_st_tensor(x)
        if context is None:
            context = AttentionContext.from_st_tensor(st_x)

        residual = st_x.flatten_all()
        x = self.norm1(residual)

        B, N_total, C = x.shape
        # (B, N, 3*C) -> (3, B, H, N, D)
        qkv = self.qkv(x).reshape(B, N_total, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        backend_name = self.dispatcher.select(context)
        backend = self.backends[backend_name]

        # Dispatch
        out = backend(q, k, v, context)

        # Standardize output to (B, N, C)
        if out.dim() == 5: # (B, T, H, N, D)
            out = out.transpose(2, 3).reshape(B, N_total, C)
        elif out.dim() == 4: # (B, H, N, D)
            out = out.transpose(1, 2).reshape(B, N_total, C)

        out = self.proj(out)

        hidden_states = residual + out
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        return SpatioTemporalTensor(
            hidden_states,
            st_x.modality,
            st_x.spatial_shape,
            st_x.temporal_dim
        )

class VideoBackbone(nn.Module):
    def __init__(self, depth: int, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            VideoBackboneBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

    def forward(self, x: Union[torch.Tensor, SpatioTemporalTensor], context: Optional[AttentionContext] = None) -> SpatioTemporalTensor:
        st_x = as_st_tensor(x)
        for block in self.blocks:
            st_x = block(st_x, context)
        return st_x
