import torch
from typing import Optional, Tuple, Union

class STTensor:
    """
    Spatio-Temporal Tensor.
    A wrapper around torch.Tensor that provides semantic information about the modality
    and its spatio-temporal structure.

    Attributes:
        x (torch.Tensor): The underlying PyTorch tensor. Expected shape (B, N, C).
        modality (str): The data modality, e.g., 'image' or 'video'.
        spatial_shape (Tuple[int, int], optional): (H, W) dimensions for spatial data.
        temporal_dim (int, optional): Number of frames for video data.
    """
    def __init__(
        self,
        x: torch.Tensor,
        modality: str,
        spatial_shape: Optional[Tuple[int, int]] = None,
        temporal_dim: Optional[int] = None
    ):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        self.x = x
        self.modality = modality
        self.spatial_shape = spatial_shape
        self.temporal_dim = temporal_dim

    @property
    def shape(self):
        return self.x.shape

    @property
    def device(self):
        return self.x.device

    @property
    def dtype(self):
        return self.x.dtype

    def to(self, *args, **kwargs):
        """Moves the underlying tensor to a new device or dtype."""
        return STTensor(self.x.to(*args, **kwargs), self.modality, self.spatial_shape, self.temporal_dim)

    def unwrap(self) -> torch.Tensor:
        """Returns the underlying torch.Tensor."""
        return self.x

    def __repr__(self):
        return (f"STTensor(shape={list(self.x.shape)}, modality='{self.modality}', "
                f"spatial_shape={self.spatial_shape}, temporal_dim={self.temporal_dim})")

def as_st_tensor(
    x: Union[torch.Tensor, STTensor],
    modality: str = "image",
    spatial_shape: Optional[Tuple[int, int]] = None,
    temporal_dim: Optional[int] = None
) -> STTensor:
    """Utility to ensure input is an STTensor."""
    if isinstance(x, STTensor):
        return x
    return STTensor(x, modality, spatial_shape, temporal_dim)
