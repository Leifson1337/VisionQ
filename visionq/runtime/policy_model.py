from __future__ import annotations

from typing import Any

from ..attention.registry import AttentionBackendName


class HeuristicPolicy:
    """Deterministic routing heuristic; no learned weights are used."""

    version = "heuristic-v1"

    def score_backends(self, features: dict[str, Any]) -> dict[str, float]:
        n = int(features["sequence_length"])
        modality = features["modality"]
        spatial = int(features["spatial_complexity"])
        temporal = int(features["temporal_complexity"])
        memory_pressure = bool(features["memory_pressure"])

        scores = {name.value: 0.0 for name in AttentionBackendName}
        scores[AttentionBackendName.FLASH.value] = 100.0
        if n > 4096:
            scores[AttentionBackendName.FLASH.value] -= 30.0
        if modality == "image" and spatial > 0:
            scores[AttentionBackendName.SPATIAL_NEIGHBORHOOD.value] = 120.0
        if modality == "video" and temporal > 1:
            scores[AttentionBackendName.SPATIOTEMPORAL_HYBRID.value] = 140.0
            scores[AttentionBackendName.TEMPORAL_NEIGHBORHOOD.value] = 80.0
        if n >= 8192 or memory_pressure:
            scores[AttentionBackendName.SPARSE.value] = 115.0
        if n >= 16384:
            scores[AttentionBackendName.CHUNKED_STREAMING.value] = 130.0
        return scores

    @staticmethod
    def get_best_backend(scores: dict[str, float]) -> str:
        if not scores:
            raise RuntimeError("cannot select a backend from empty scores")
        return max(scores, key=lambda name: scores[name])


PolicyModel = HeuristicPolicy
