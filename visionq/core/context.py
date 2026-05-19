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
        sequence_length: int = 0,
        spatial_shape: Optional[Tuple[int, int]] = None,
        temporal_dim: Optional[int] = None,
        temporal_flag: bool = False,
        window_size: int = 0,
        dilation: int = 1,
        compute_budget_hint: str = "balanced", # fast | balanced | accuracy
        precision: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        extra_args: Optional[Dict[str, Any]] = None
    ):
        self.modality = modality
        self.sequence_length = sequence_length
        self.spatial_shape = spatial_shape
        self.temporal_dim = temporal_dim
        self.temporal_flag = temporal_flag or (modality == "video")
        self.window_size = window_size
        self.dilation = dilation
        self.compute_budget_hint = compute_budget_hint
        self.precision = precision
        self.device = device
        self.extra_args = extra_args or {}

    @classmethod
    def from_st_tensor(cls, st_tensor: Any, **kwargs) -> 'AttentionContext':
        """Creates a context from an STTensor, allowing overrides."""
        return cls(
            modality=kwargs.get('modality', st_tensor.modality),
            sequence_length=kwargs.get('sequence_length', st_tensor.shape[1]),
            spatial_shape=kwargs.get('spatial_shape', st_tensor.spatial_shape),
            temporal_dim=kwargs.get('temporal_dim', st_tensor.temporal_dim),
            temporal_flag=kwargs.get('temporal_flag', st_tensor.modality == "video"),
            precision=kwargs.get('precision', st_tensor.dtype),
            device=kwargs.get('device', st_tensor.device),
            window_size=kwargs.get('window_size', 0),
            dilation=kwargs.get('dilation', 1),
            compute_budget_hint=kwargs.get('compute_budget_hint', 'balanced'),
            extra_args=kwargs.get('extra_args', {})
        )

    def __repr__(self):
        return (f"AttentionContext(modality='{self.modality}', spatial_shape={self.spatial_shape}, "
                f"window_size={self.window_size}, device={self.device})")
