import torch
from typing import Dict, Any
from ..core.context import AttentionContext

class FeatureExtractor:
    """
    Industrial Feature Extractor for Attention Routing.
    Gathers metrics from input tensors and hardware state.
    """

    @staticmethod
    def extract(context: AttentionContext) -> Dict[str, Any]:
        """
        Extracts a feature dictionary from execution context.
        """
        features = {
            "sequence_length": context.sequence_length,
            "modality": context.modality,
            "spatial_complexity": (context.spatial_shape[0] * context.spatial_shape[1]) if context.spatial_shape else 0,
            "temporal_complexity": context.temporal_dim or 1,
            "sparsity_estimate": 1.0 / max(1, context.dilation),
            "device_type": context.device.type if context.device else "cpu",
        }

        # Hardware-specific features (simplified industrial version)
        if features["device_type"] == "cuda":
            features["compute_capability"] = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            features["vram_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
            features["vram_free"] = (torch.cuda.mem_get_info()[0] / (1024**3)) if torch.cuda.is_available() else 0
        else:
            features["compute_capability"] = (0, 0)
            features["vram_total"] = 0
            features["vram_free"] = 0

        return features
