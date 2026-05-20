from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


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

    def export_json(self, path: str | Path) -> None:
        """Serializes all recorded telemetry to a JSON file."""
        data = [asdict(r) for r in self.records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def import_json(self, path: str | Path) -> None:
        """Loads telemetry records from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self.records.append(FeedbackRecord(**item))
