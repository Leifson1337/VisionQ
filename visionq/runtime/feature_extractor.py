from __future__ import annotations

from typing import Any

import torch

from ..core.context import AttentionContext


class FeatureExtractor:
    """Extract routing features from validated attention context."""

    @staticmethod
    def extract(context: AttentionContext) -> dict[str, Any]:
        device = context.device or torch.device("cpu")
        spatial_complexity = (
            context.spatial_shape[0] * context.spatial_shape[1]
            if context.spatial_shape is not None
            else 0
        )
        features: dict[str, Any] = {
            "sequence_length": context.sequence_length,
            "modality": context.modality,
            "spatial_complexity": spatial_complexity,
            "temporal_complexity": context.temporal_dim or 1,
            "sparsity_estimate": 1.0 / context.dilation,
            "device_type": device.type,
            "compute_capability": (0, 0),
            "vram_total": 0.0,
            "vram_free": 0.0,
            "memory_pressure": False,
        }
        if device.type == "cuda" and torch.cuda.is_available():
            idx = device.index if device.index is not None else torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(idx)
            features["compute_capability"] = torch.cuda.get_device_capability(idx)
            features["vram_total"] = total / (1024**3)
            features["vram_free"] = free / (1024**3)
            features["memory_pressure"] = total > 0 and free / total < 0.2
        return features
