# Architecture

VisionQ separates tensor metadata, backend implementations, routing and model
integration.

- `visionq.core` owns canonical `(B, T, H, W, C)` tensor metadata.
- `visionq.attention` contains projected Q/K/V attention backends.
- `visionq.runtime` selects a backend using deterministic heuristics.
- `visionq.models` provides a small transformer-style backbone.
- `visionq.compiler` provides an experimental graph IR used for validation and
  planning annotations only.
- `visionq.experimental` exposes APIs that are intentionally outside the stable
  compatibility promise, including the optional Triton kernel.
