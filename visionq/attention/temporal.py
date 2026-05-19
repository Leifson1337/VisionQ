import torch
import torch.nn as nn
from .base import AttentionBackend
from .registry import register_attention
from ..core.context import AttentionContext

@register_attention("temporal_neighborhood")
class TemporalNeighborhoodAttention(AttentionBackend):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__(dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: AttentionContext) -> torch.Tensor:
        # q shape (B, T, H, N, D)
        B, T, H, N, D = q.shape
        q_p = q.permute(0, 3, 2, 1, 4).reshape(B * N, H, T, D)
        k_p = k.permute(0, 3, 2, 1, 4).reshape(B * N, H, T, D)
        v_p = v.permute(0, 3, 2, 1, 4).reshape(B * N, H, T, D)

        q_p = q_p * self.scale
        attn = (q_p @ k_p.transpose(-2, -1))

        if context.temporal_window > 0:
            window = context.temporal_window
            indices = torch.arange(T, device=q.device)
            dist = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))
            mask = torch.full((T, T), float('-inf'), device=q.device)
            mask[dist <= window // 2] = 0
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v_p).reshape(B, N, H, T, D).permute(0, 3, 2, 1, 4)
        return out
