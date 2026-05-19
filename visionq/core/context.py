from typing import Optional, Tuple, Any, Dict, Literal
import torch

AttentionMode = Literal["spatial_only", "temporal_only", "spatio_temporal"]

class AttentionContext:
    """
    Video-Aware Attention Context.
    Controls the execution of spatial, temporal, and hybrid attention modes.
    """
    def __init__(
        self,
        modality: str,
        sequence_length: int = 0,
        spatial_shape: Optional[Tuple[int, int]] = None,
        temporal_dim: Optional[int] = None,
        temporal_window: int = 0,
        spatial_window: Optional[Tuple[int, int]] = None,
        stride_spatial: int = 1,
        stride_temporal: int = 1,
        attention_mode: AttentionMode = "spatio_temporal",
        compute_budget_hint: str = "balanced",
        precision: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        window_size: int = 0, # Added for backward compatibility
        dilation: int = 1,     # Added for backward compatibility
        extra_args: Optional[Dict[str, Any]] = None
    ):
        self.modality = modality
        self.sequence_length = sequence_length
        self.spatial_shape = spatial_shape
        self.temporal_dim = temporal_dim
        self.temporal_window = temporal_window
        self.spatial_window = spatial_window
        self.stride_spatial = stride_spatial
        self.stride_temporal = stride_temporal
        self.attention_mode = attention_mode
        self.compute_budget_hint = compute_budget_hint
        self.precision = precision
        self.device = device
        self.window_size = window_size
        self.dilation = dilation
        self.extra_args = extra_args or {}

    @classmethod
    def from_st_tensor(cls, st_tensor: Any, **kwargs) -> 'AttentionContext':
        T = getattr(st_tensor, 'T', 1) or 1
        H = getattr(st_tensor, 'H', 1) or 1
        W = getattr(st_tensor, 'W', 1) or 1

        return cls(
            modality=kwargs.get('modality', st_tensor.modality),
            sequence_length=T * H * W,
            spatial_shape=st_tensor.spatial_shape,
            temporal_dim=st_tensor.temporal_dim,
            temporal_window=kwargs.get('temporal_window', 0),
            spatial_window=kwargs.get('spatial_window', None),
            stride_spatial=kwargs.get('stride_spatial', 1),
            stride_temporal=kwargs.get('stride_temporal', 1),
            attention_mode=kwargs.get('attention_mode', "spatio_temporal"),
            precision=kwargs.get('precision', st_tensor.dtype),
            device=kwargs.get('device', st_tensor.device),
            window_size=kwargs.get('window_size', 0),
            dilation=kwargs.get('dilation', 1),
            compute_budget_hint=kwargs.get('compute_budget_hint', 'balanced'),
            extra_args=kwargs.get('extra_args', {})
        )

    def __repr__(self):
        return (f"AttentionContext(modality='{self.modality}', mode='{self.attention_mode}', "
                f"spatial_window={self.spatial_window}, temporal_window={self.temporal_window})")
