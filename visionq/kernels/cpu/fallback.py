import torch
import torch.nn.functional as F
from typing import Optional

def cpu_tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 32,
    mask: Optional[torch.Tensor] = None,
    scale: float = 1.0
) -> torch.Tensor:
    """
    CPU Fallback for tiled attention computation.
    Simulates IO-aware block processing.
    """
    B, H, N, D = q.shape
    out = torch.zeros_like(q)

    # Process in blocks to maintain memory locality
    for i in range(0, N, block_size):
        q_tile = q[:, :, i : i + block_size, :]

        # Accumulators for this tile
        row_sum = torch.zeros(B, H, q_tile.shape[2], 1, device=q.device)
        row_max = torch.full((B, H, q_tile.shape[2], 1), float('-inf'), device=q.device)
        tile_out = torch.zeros(B, H, q_tile.shape[2], D, device=q.device)

        for j in range(0, N, block_size):
            k_tile = k[:, :, j : j + block_size, :]
            v_tile = v[:, :, j : j + block_size, :]

            # (B, H, BS, BS)
            attn_tile = (q_tile @ k_tile.transpose(-2, -1)) * scale

            if mask is not None:
                attn_tile = attn_tile + mask[:, :, i:i+block_size, j:j+block_size]

            # Online softmax logic (FlashAttention style)
            curr_max = torch.max(attn_tile, dim=-1, keepdim=True)[0]
            new_max = torch.max(row_max, curr_max)

            exp_attn = torch.exp(attn_tile - new_max)
            exp_row_max = torch.exp(row_max - new_max)

            tile_out = tile_out * exp_row_max + exp_attn @ v_tile
            row_sum = row_sum * exp_row_max + torch.sum(exp_attn, dim=-1, keepdim=True)
            row_max = new_max

        out[:, :, i : i + block_size, :] = tile_out / row_sum

    return out
