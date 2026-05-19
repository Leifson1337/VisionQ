# Changelog

## 0.1.0

- Added validated core tensor and attention context types.
- Added deterministic attention backend registry and heuristic router.
- Added CPU-compatible reference attention backends and blockwise online-softmax kernel.
- Added compiler IR validation, graph serialization, optimizer and fusion annotations.
- Added tests, CI configuration, documentation and benchmarks.
- Added optional experimental Triton dense attention kernel with GPU/Triton skip tests.
- Added benchmark environment metadata and CSV/JSON output support.
