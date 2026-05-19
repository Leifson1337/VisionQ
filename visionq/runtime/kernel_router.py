from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.context import AttentionContext
from .feature_extractor import FeatureExtractor
from .policy_model import HeuristicPolicy


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    policy_version: str
    backend: str
    scores: dict[str, float]
    parameters: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "backend": self.backend,
            "scores": dict(self.scores),
            "parameters": dict(self.parameters),
            "reason": self.reason,
        }


class KernelRouter:
    """Deterministic execution planner based on documented heuristics."""

    def __init__(self) -> None:
        self.feature_extractor = FeatureExtractor()
        self.policy = HeuristicPolicy()

    def plan_execution(self, context: AttentionContext) -> RoutingDecision:
        features = self.feature_extractor.extract(context)
        scores = self.policy.score_backends(features)
        backend = self.policy.get_best_backend(scores)
        params = self.parameters_for(backend, features)
        reason = f"selected {backend} with score {scores[backend]:.1f}"
        return RoutingDecision(
            policy_version=self.policy.version,
            backend=backend,
            scores=scores,
            parameters=params,
            reason=reason,
        )

    @staticmethod
    def parameters_for(backend: str, features: dict[str, Any]) -> dict[str, Any]:
        n = int(features["sequence_length"])
        block_size = 128 if n > 8192 else 64 if n > 2048 else 32
        if features["device_type"] == "cuda" and features["compute_capability"][0] >= 8:
            block_size = max(block_size, 128)
        return {"block_size": block_size, "window_size": 3, "stride": 1}
