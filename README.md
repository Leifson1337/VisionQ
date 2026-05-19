# VisionQ

VisionQ is a typed Python reference library for experimenting with hardware-aware
spatio-temporal attention backend selection in vision and video transformers.

The current implementation is intentionally honest: CPU-compatible reference
backends are provided for correctness and testing, PyTorch SDPA is used for the
`flash` backend, and the optional Triton kernel is guarded as an experimental
CUDA-only low-level path. The router is deterministic and heuristic-based, not
learned.

## Installation

```bash
pip install visionq
```

For development:

```bash
pip install -e ".[dev]"
```

VisionQ supports Python 3.10-3.12 and PyTorch 2.1 or newer. CPU-only installs are
supported. The optional `triton` extra enables the experimental low-level Triton
kernel on Linux CUDA systems and is not required by the reference backends.

## Quickstart

```python
import torch
from visionq.core import SpatioTemporalTensor
from visionq.models import VisionBackbone

x = torch.randn(1, 8, 16)
model = VisionBackbone(depth=1, dim=16, num_heads=4, backend_override="flash")
out = model(x)
print(out.flatten_all().shape)
```

Image tokens:

```python
x = torch.randn(1, 16, 32)
st_x = SpatioTemporalTensor(x, modality="image", spatial_shape=(4, 4))
model = VisionBackbone(depth=1, dim=32, num_heads=4, backend_override="spatial_neighborhood")
out = model(st_x)
```

Video input:

```python
x = torch.randn(1, 2, 4, 4, 32)  # (B, T, H, W, C)
model = VisionBackbone(depth=1, dim=32, num_heads=4, backend_override="spatiotemporal_hybrid")
out = model(x)
```

## Backends

Registered backend keys:

- `flash`: PyTorch `torch.nn.functional.scaled_dot_product_attention`.
- `sparse`: exact attention against strided key/value samples controlled by
  `AttentionContext.dilation`.
- `spatial_neighborhood`: 2D sliding-window reference attention.
- `temporal_neighborhood`: temporal sliding-window reference attention for video tensors.
- `spatiotemporal_hybrid`: spatial neighborhood attention followed by temporal attention.
- `chunked_streaming`: exact blockwise attention using online softmax.

Unsupported masks are rejected with `ValueError` instead of silently ignored.
The blockwise kernel is a PyTorch reference implementation. A separate optional
`TritonAttentionKernel` exists for CUDA inference experiments and explicitly
rejects unsupported masks, causal mode and CPU tensors.

## Backend Selection

`AttentionDispatcher` uses a deterministic `HeuristicPolicy`. Decisions are based
on modality, sequence length, spatial shape, temporal length and available CUDA
memory metadata. The last routing decision is available as
`dispatcher.last_decision`.

## API Overview

- `visionq.core`: `SpatioTemporalTensor`, `AttentionContext`.
- `visionq.attention`: backend classes, registry and backend-name enum.
- `visionq.runtime`: dispatcher and heuristic router.
- `visionq.models`: `VisionBackbone`.
- `visionq.experimental`: unstable low-level APIs such as the optional Triton
  kernel.
- `visionq.compiler`: small serializable graph IR with conservative optimizer and
  fusion-pattern annotations.

## Tests and Quality Checks

```bash
python -m compileall visionq tests
python -m pytest
ruff check .
ruff format --check .
mypy visionq
python -m build
twine check dist/*
```

## Benchmarks

Benchmarks are not part of the default test suite:

```bash
python benchmarks/attention_benchmark.py --device cpu --seq-lens 128 512 1024
python benchmarks/attention_benchmark.py --device cpu --output results/cpu.csv --metadata-output results/cpu.json
```

Benchmark output reports measurements from the current machine and does not
encode fixed performance claims.

## Known Limitations

- The optional Triton JIT kernel is experimental and limited to dense non-causal
  CUDA attention without masks or dropout.
- Neighborhood attention is a reference PyTorch implementation and is optimized
  for clarity and testability, not peak throughput.
- `sparse` is strided key/value sampling, not block-sparse matrix execution.
- The router is heuristic-based and reproducible; it is not trained.
- Some masks are only supported by `flash`, `sparse` and `chunked_streaming`.

## Roadmap

- Add GPU benchmarks with environment metadata.
- Add learned routing only when training data, model serialization and tests are
  present.

## License

MIT. See [LICENSE](LICENSE).
