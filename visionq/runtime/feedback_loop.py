from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FeedbackRecord:
    backend: str
    latency_ms: float
    memory_mb: float | None = None


class FeedbackLoop:
    """In-memory telemetry collector for offline router analysis."""

    def __init__(self) -> None:
        self.records: list[FeedbackRecord] = []

    def record(self, backend: str, latency_ms: float, memory_mb: float | None = None) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        self.records.append(FeedbackRecord(backend, latency_ms, memory_mb))

    def summary(self) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for record in self.records:
            totals.setdefault(record.backend, []).append(record.latency_ms)
        return {backend: sum(values) / len(values) for backend, values in totals.items()}
