from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..ops.online_softmax import OnlineSoftmax


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


class ReferenceBlockwiseAttentionKernel:
    """CPU/GPU PyTorch reference for exact blockwise attention."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Any,
        block_size: int = 64,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context
        if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
            raise ValueError("q, k and v must have shape (B, heads, N, D)")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        batch, heads, tokens, dim = q.shape
        out = torch.empty_like(q)
        scale = dim**-0.5

        for start in range(0, tokens, block_size):
            stop = min(start + block_size, tokens)
            q_tile = q[:, :, start:stop, :]
            running_max = torch.full(
                (batch, heads, stop - start, 1),
                float("-inf"),
                device=q.device,
                dtype=q.dtype,
            )
            running_sum = torch.zeros_like(running_max)
            acc = torch.zeros_like(q_tile)

            for key_start in range(0, tokens, block_size):
                key_stop = min(key_start + block_size, tokens)
                k_tile = k[:, :, key_start:key_stop, :]
                v_tile = v[:, :, key_start:key_stop, :]
                scores = (q_tile @ k_tile.transpose(-2, -1)) * scale
                if mask is not None:
                    scores = scores + mask[..., start:stop, key_start:key_stop]
                new_max, new_sum, exp_scores = OnlineSoftmax.update(
                    running_max, running_sum, scores
                )
                acc = acc * torch.exp(running_max - new_max) + exp_scores @ v_tile
                running_max = new_max
                running_sum = new_sum

            out[:, :, start:stop, :] = acc / running_sum.clamp_min(torch.finfo(q.dtype).tiny)
        return out


@dataclass(frozen=True, slots=True)
class TritonKernelLimits:
    block_m: int = 16
    block_n: int = 32


class TritonAttentionKernel:
    """Optional Triton dense attention kernel for CUDA inference experiments.

    Limitations are explicit: CUDA and Triton are required, mask/dropout/causal
    modes are unsupported, and head dimensions must be powers of two up to 128.
    """

    def __init__(self, limits: TritonKernelLimits | None = None) -> None:
        self.limits = limits or TritonKernelLimits()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: Any,
        block_size: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del block_size
        if mask is not None:
            raise ValueError("TritonAttentionKernel does not support masks")
        is_causal = getattr(context, "causal", False)
        if not triton_available():
            raise RuntimeError("Triton is not installed; install visionq[triton] on Linux CUDA")
        if not q.is_cuda:
            raise RuntimeError("TritonAttentionKernel requires CUDA tensors")
        if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
            raise ValueError("q, k and v must have shape (B, heads, N, D)")
        if k.stride() != q.stride() or v.stride() != q.stride():
            raise ValueError("q, k and v must have matching strides for the Triton kernel")
        if q.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("TritonAttentionKernel supports float16, bfloat16 and float32")
        head_dim = q.shape[-1]
        if head_dim > 128 or head_dim & (head_dim - 1):
            raise ValueError("head dimension must be a power of two up to 128")

        import triton

        out = torch.empty_like(q)
        batch, heads, tokens, _ = q.shape
        grid = (triton.cdiv(tokens, self.limits.block_m), batch * heads)
        scale = head_dim**-0.5
        _dense_attention_kernel[grid](
            q,
            k,
            v,
            out,
            tokens,
            head_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            heads,
            scale,
            IS_CAUSAL=is_causal,
            BLOCK_M=self.limits.block_m,
            BLOCK_N=self.limits.block_n,
            BLOCK_D=head_dim,
        )
        return out


if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _dense_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        tokens: tl.constexpr,
        head_dim: tl.constexpr,
        stride_b: tl.constexpr,
        stride_h: tl.constexpr,
        stride_n: tl.constexpr,
        stride_d: tl.constexpr,
        heads: tl.constexpr,
        scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        batch_idx = pid_bh // heads
        head_idx = pid_bh - batch_idx * heads
        base = batch_idx * stride_b + head_idx * stride_h

        q = tl.load(
            q_ptr + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
            mask=(offs_m[:, None] < tokens) & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in range(0, tokens, BLOCK_N):
            cols = start_n + offs_n
            k = tl.load(
                k_ptr + base + cols[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=(cols[:, None] < tokens) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(cols[None, :] < tokens, scores, -float("inf"))
            if IS_CAUSAL:
                scores = tl.where(offs_m[:, None] >= cols[None, :], scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p, axis=1)
            v_block = tl.load(
                v_ptr + base + cols[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=(cols[:, None] < tokens) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
            m_i = m_new
            l_i = l_new

        out = acc / l_i[:, None]
        tl.store(
            out_ptr + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
            out,
            mask=(offs_m[:, None] < tokens) & (offs_d[None, :] < head_dim),
        )
else:
    _dense_attention_kernel = None
