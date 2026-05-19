import torch
from typing import Optional, Tuple, Union, Any
from .types import ModalityType, ShapeType

class STTensor:
    """
    Spatio-Temporal Tensor Abstraction.
    Unified representation for Image, Video, and Sequence data.

    Shapes:
    - Image:  (B, H*W, C) or (B, C, H, W)
    - Video:  (B, T*H*W, C) or (B, C, T, H, W)
    - Sequence: (B, L, C)
    """
    def __init__(
        self,
        x: torch.Tensor,
        modality: ModalityType,
        spatial_shape: Optional[Tuple[int, int]] = None,
        temporal_dim: Optional[int] = None
    ):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"STTensor expects torch.Tensor, got {type(x)}")

        self._x = x
        self._modality = modality
        self._spatial_shape = spatial_shape
        self._temporal_dim = temporal_dim

    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def modality(self) -> ModalityType:
        return self._modality

    @property
    def spatial_shape(self) -> Optional[Tuple[int, int]]:
        return self._spatial_shape

    @property
    def temporal_dim(self) -> Optional[int]:
        return self._temporal_dim

    @property
    def shape(self):
        return self._x.shape

    @property
    def device(self):
        return self._x.device

    @property
    def dtype(self):
        return self._x.dtype

    def to(self, *args, **kwargs) -> 'STTensor':
        return STTensor(self._x.to(*args, **kwargs), self._modality, self._spatial_shape, self._temporal_dim)

    def unwrap(self) -> torch.Tensor:
        return self._x

    def __repr__(self):
        return (f"STTensor(modality={self._modality}, shape={list(self.shape)}, "
                f"spatial={self._spatial_shape}, temporal={self._temporal_dim})")

def as_st_tensor(x: Any, modality: Optional[ModalityType] = None, **kwargs) -> STTensor:
    if isinstance(x, STTensor):
        if modality is not None:
            # Update modality if provided
            return STTensor(x.unwrap(), modality, x.spatial_shape, x.temporal_dim)
        return x

    if modality is None:
        # Fallback for depth > 1 where we might receive a raw tensor
        # This is a safety net; ideally blocks should return STTensors
        modality = "image"

    return STTensor(x, modality, **kwargs)
