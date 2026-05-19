from __future__ import annotations

import torch

from ..exceptions import ShapeError


def validate_same_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    allow_5d: bool = False,
) -> None:
    expected_dims = (4, 5) if allow_5d else (4,)
    if q.dim() not in expected_dims:
        raise ShapeError(f"q must be {expected_dims}D, got shape {tuple(q.shape)}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ShapeError(
            "q, k and v must have identical shapes; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if not (q.device == k.device == v.device):
        raise ShapeError("q, k and v must be on the same device")
    if not (q.dtype == k.dtype == v.dtype):
        raise ShapeError("q, k and v must have the same dtype")


def validate_spatial_tokens(tokens: int, spatial_shape: tuple[int, int]) -> None:
    expected = spatial_shape[0] * spatial_shape[1]
    if tokens != expected:
        raise ShapeError(f"spatial_shape {spatial_shape} expects {expected} tokens, got {tokens}")


__all__ = ["validate_same_qkv", "validate_spatial_tokens"]
