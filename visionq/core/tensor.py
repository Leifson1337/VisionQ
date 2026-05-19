from __future__ import annotations

from typing import Any

import torch

from .types import ModalityType


class SpatioTemporalTensor:
    """Tensor wrapper with canonical layout ``(B, T, H, W, C)``."""

    def __init__(
        self,
        x: torch.Tensor,
        modality: ModalityType,
        spatial_shape: tuple[int, int] | None = None,
        temporal_dim: int | None = None,
    ) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)!r}")
        if modality not in {"image", "video", "sequence"}:
            raise ValueError("modality must be one of 'image', 'video' or 'sequence'")
        self._modality = modality
        self._x = self._canonicalize(x, modality, spatial_shape, temporal_dim)

    @staticmethod
    def _canonicalize(
        x: torch.Tensor,
        modality: ModalityType,
        spatial_shape: tuple[int, int] | None,
        temporal_dim: int | None,
    ) -> torch.Tensor:
        if x.dim() == 5:
            if spatial_shape is not None and spatial_shape != (x.shape[2], x.shape[3]):
                raise ValueError("spatial_shape does not match 5D input")
            if temporal_dim is not None and temporal_dim != x.shape[1]:
                raise ValueError("temporal_dim does not match 5D input")
            return x
        if x.dim() == 4:
            if modality == "video":
                raise ValueError("video tensors must be 5D (B, T, H, W, C)")
            if spatial_shape is not None:
                if spatial_shape == (x.shape[1], x.shape[2]):
                    return x.unsqueeze(1)
                if spatial_shape == (x.shape[2], x.shape[3]):
                    return x.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
                raise ValueError("spatial_shape does not match 4D input")
            if x.shape[1] <= 4:
                return x.permute(0, 2, 3, 1).unsqueeze(1).contiguous()
            return x.unsqueeze(1)
        if x.dim() == 3:
            b, n, c = x.shape
            t = temporal_dim or 1
            if spatial_shape is None:
                if modality != "sequence":
                    raise ValueError("3D image/video input requires spatial_shape")
                return x.reshape(b, n, 1, 1, c)
            h, w = spatial_shape
            if t * h * w != n:
                raise ValueError(f"3D input has {n} tokens but temporal_dim*H*W is {t * h * w}")
            return x.reshape(b, t, h, w, c)
        raise ValueError(f"Unsupported tensor rank {x.dim()}; expected 3D, 4D or 5D")

    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def modality(self) -> ModalityType:
        return self._modality

    @property
    def shape(self) -> torch.Size:
        return self._x.shape

    @property
    def B(self) -> int:
        return self._x.shape[0]

    @property
    def T(self) -> int:
        return self._x.shape[1]

    @property
    def H(self) -> int:
        return self._x.shape[2]

    @property
    def W(self) -> int:
        return self._x.shape[3]

    @property
    def C(self) -> int:
        return self._x.shape[4]

    @property
    def dtype(self) -> torch.dtype:
        return self._x.dtype

    @property
    def device(self) -> torch.device:
        return self._x.device

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return (self.H, self.W)

    @property
    def temporal_dim(self) -> int:
        return self.T

    def flatten_all(self) -> torch.Tensor:
        return self._x.reshape(self.B, self.T * self.H * self.W, self.C)

    def to_tiles(self, tile_size: tuple[int, int]) -> torch.Tensor:
        th, tw = tile_size
        if th <= 0 or tw <= 0:
            raise ValueError("tile dimensions must be positive")
        b, t, h, w, c = self.shape
        if h % th != 0 or w % tw != 0:
            raise ValueError(f"tile_size {tile_size} must evenly divide spatial shape {(h, w)}")
        x = self._x.reshape(b, t, h // th, th, w // tw, tw, c)
        return x.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, t, -1, th * tw, c)

    def unwrap(self) -> torch.Tensor:
        return self._x

    def to(self, *args: Any, **kwargs: Any) -> SpatioTemporalTensor:
        return SpatioTemporalTensor(self._x.to(*args, **kwargs), self._modality)

    def __repr__(self) -> str:
        return f"SpatioTemporalTensor(modality={self._modality!r}, shape={tuple(self.shape)})"


def as_st_tensor(x: Any, **kwargs: Any) -> SpatioTemporalTensor:
    if isinstance(x, SpatioTemporalTensor):
        return x
    if "modality" not in kwargs:
        raise ValueError("modality is required when converting a raw tensor")
    return SpatioTemporalTensor(x, **kwargs)
