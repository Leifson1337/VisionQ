import torch
from typing import Optional, Tuple, Union, Any
from .types import ModalityType

class SpatioTemporalTensor:
    def __init__(
        self,
        x: torch.Tensor,
        modality: ModalityType,
        spatial_shape: Optional[Tuple[int, int]] = None,
        temporal_dim: Optional[int] = None
    ):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        if x.dim() == 5:
            self._x = x
        elif x.dim() == 4:
            self._x = x.unsqueeze(1)
        elif x.dim() == 3:
            if spatial_shape and temporal_dim:
                self._x = x.reshape(x.shape[0], temporal_dim, *spatial_shape, x.shape[-1])
            elif spatial_shape:
                self._x = x.reshape(x.shape[0], 1, *spatial_shape, x.shape[-1])
            else:
                self._x = x.reshape(x.shape[0], x.shape[1], 1, 1, x.shape[-1])
        else:
            raise ValueError(f"Unsupported dimension: {x.dim()}")

        self._modality = modality

    @property
    def x(self) -> torch.Tensor: return self._x
    @property
    def modality(self) -> ModalityType: return self._modality
    @property
    def shape(self): return self._x.shape
    @property
    def B(self): return self._x.shape[0]
    @property
    def T(self): return self._x.shape[1]
    @property
    def H(self): return self._x.shape[2]
    @property
    def W(self): return self._x.shape[3]
    @property
    def C(self): return self._x.shape[4]
    @property
    def dtype(self): return self._x.dtype
    @property
    def device(self): return self._x.device
    @property
    def spatial_shape(self) -> Tuple[int, int]: return (self.H, self.W)
    @property
    def temporal_dim(self) -> int: return self.T

    def flatten_all(self) -> torch.Tensor:
        return self._x.reshape(self.B, self.T * self.H * self.W, self.C)

    def to_tiles(self, tile_size: Tuple[int, int]) -> torch.Tensor:
        """
        Partitions the spatial dimensions into tiles.
        Returns: (B, T, num_tiles, tile_H * tile_W, C)
        """
        th, tw = tile_size
        B, T, H, W, C = self.shape
        if H % th != 0 or W % tw != 0:
            # Padding would happen here in industrial version
            pass

        # (B, T, H//th, th, W//tw, tw, C) -> (B, T, H//th * W//tw, th * tw, C)
        x = self._x.reshape(B, T, H // th, th, W // tw, tw, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, T, -1, th * tw, C)
        return x

    def unwrap(self) -> torch.Tensor: return self._x

    def to(self, *args, **kwargs) -> 'SpatioTemporalTensor':
        return SpatioTemporalTensor(self._x.to(*args, **kwargs), self._modality)

    def __repr__(self):
        return f"STTensor(modality={self._modality}, shape={list(self.shape)})"

def as_st_tensor(x: Any, **kwargs) -> SpatioTemporalTensor:
    if isinstance(x, SpatioTemporalTensor): return x
    return SpatioTemporalTensor(x, **kwargs)
