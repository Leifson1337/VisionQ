from typing import Optional, Tuple, Any, Dict
import torch

class AttentionContext:
    """
    Contextual information for attention execution.
    Encapsulates metadata required by the dispatcher and kernels to optimize execution.
    """
    def __init__(
        self,
        modality: str,
        spatial_shape: Optional[Tuple[int, int]] = None,
        temporal_dim: Optional[int] = None,
        window_size: int = 0,
        dilation: int = 1,
        precision: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        extra_args: Optional[Dict[str, Any]] = None
    ):
        self.modality = modality
        self.spatial_shape = spatial_shape
        self.temporal_dim = temporal_dim
        self.window_size = window_size
        self.dilation = dilation
        self.precision = precision
        self.device = device
        self.extra_args = extra_args or {}

    @classmethod
    def from_st_tensor(cls, st_tensor: Any, **kwargs) -> 'AttentionContext':
        """Creates a context from an STTensor, allowing overrides."""
        return cls(
            modality=kwargs.get('modality', st_tensor.modality),
            spatial_shape=kwargs.get('spatial_shape', st_tensor.spatial_shape),
            temporal_dim=kwargs.get('temporal_dim', st_tensor.temporal_dim),
            precision=kwargs.get('precision', st_tensor.dtype),
            device=kwargs.get('device', st_tensor.device),
            window_size=kwargs.get('window_size', 0),
            dilation=kwargs.get('dilation', 1),
            extra_args=kwargs.get('extra_args', {})
        )

    def __repr__(self):
        return (f"AttentionContext(modality='{self.modality}', spatial_shape={self.spatial_shape}, "
                f"window_size={self.window_size}, device={self.device})")
