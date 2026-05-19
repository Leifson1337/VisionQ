from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

AttentionMode = Literal["spatial_only", "temporal_only", "spatio_temporal"]
Modality = Literal["image", "video", "sequence"]


@dataclass(slots=True)
class AttentionContext:
    """Validated metadata used by router and attention backends."""

    modality: Modality
    sequence_length: int = 0
    spatial_shape: tuple[int, int] | None = None
    temporal_dim: int | None = None
    temporal_window: int = 0
    spatial_window: tuple[int, int] | None = None
    stride_spatial: int = 1
    stride_temporal: int = 1
    attention_mode: AttentionMode = "spatio_temporal"
    compute_budget_hint: str = "balanced"
    precision: torch.dtype | None = None
    device: torch.device | None = None
    window_size: int = 0
    dilation: int = 1
    causal: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in {"image", "video", "sequence"}:
            raise ValueError("modality must be one of 'image', 'video' or 'sequence'")
        if self.sequence_length < 0:
            raise ValueError("sequence_length must be non-negative")
        if self.spatial_shape is not None:
            h, w = self.spatial_shape
            if h <= 0 or w <= 0:
                raise ValueError(f"spatial_shape values must be positive, got {self.spatial_shape}")
        if self.temporal_dim is not None and self.temporal_dim <= 0:
            raise ValueError(f"temporal_dim must be positive, got {self.temporal_dim}")
        if self.temporal_window < 0 or self.window_size < 0:
            raise ValueError("window sizes must be non-negative")
        if self.spatial_window is not None:
            sh, sw = self.spatial_window
            if sh <= 0 or sw <= 0:
                raise ValueError(
                    f"spatial_window values must be positive, got {self.spatial_window}"
                )
        if self.stride_spatial <= 0 or self.stride_temporal <= 0 or self.dilation <= 0:
            raise ValueError("stride_spatial, stride_temporal and dilation must be positive")
        if self.device is not None and not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)
        if self.sequence_length and self.spatial_shape is not None:
            expected = self.spatial_shape[0] * self.spatial_shape[1] * (self.temporal_dim or 1)
            if expected != self.sequence_length:
                raise ValueError(
                    "sequence_length is inconsistent with spatial_shape/temporal_dim: "
                    f"expected {expected}, got {self.sequence_length}"
                )

    @classmethod
    def from_st_tensor(cls, st_tensor: Any, **kwargs: Any) -> AttentionContext:
        return cls(
            modality=kwargs.pop("modality", st_tensor.modality),
            sequence_length=kwargs.pop("sequence_length", st_tensor.T * st_tensor.H * st_tensor.W),
            spatial_shape=kwargs.pop("spatial_shape", st_tensor.spatial_shape),
            temporal_dim=kwargs.pop("temporal_dim", st_tensor.temporal_dim),
            precision=kwargs.pop("precision", st_tensor.dtype),
            device=kwargs.pop("device", st_tensor.device),
            **kwargs,
        )
