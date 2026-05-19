# VisionQ: High-Performance Attention Compute Engine

VisionQ is a low-level attention compute engine designed as a hardware-aware replacement for standard PyTorch attention. It treats attention not as a layer, but as a **memory access problem**, optimizing for IO-bandwidth and locality.

## Key Architectural Principles

- **Memory-Limited Optimization**: Optimized for GPU memory hierarchy (HBM/SRAM), minimizing global memory bandwidth via tiling and block-based streaming.
- **Online Softmax Integration**: Implements the FlashAttention principle of streaming normalization for numerically stable, constant-memory attention computation.
- **Geometric 3D Awareness**: Native support for video and image data without forced flattening, preserving spatial and temporal locality.
- **IO-Aware Kernel Abstraction**: A unified execution layer for CUDA, Triton, and CPU-fallback kernels.
- **Efficient Sliding Windows**: Neighborhood attention implemented via `unfold` to maintain constant memory overhead regardless of spatial resolution.
- **Policy-Driven Execution Planning**: Uses a learned `KernelRouter` and `PolicyModel` to dynamically decide the best attention algorithm, block structure, and sparsity strategy in real-time.
- **Hardware-Aware Autotuning**: Automatically tunes parameters like `block_size` and `window_size` based on GPU compute capability and VRAM pressure.
- **Intelligent Dispatching**: Dynamically routes execution to optimal kernels (Flash, Neighborhood, Block-Sparse, Streaming) based on input modality and sequence complexity.

## Core Components

- `visionq.core`: Unified `SpatioTemporalTensor` and metadata-rich `AttentionContext`.
- `visionq.attention`: Industrial compute backends including factorized Spatio-Temporal Hybrid Attention.
- `visionq.runtime`: High-level Kernel Dispatcher for adaptive routing.
- `visionq.kernels`: Low-level Abstraction Layer for hardware-specific implementations (CPU/CUDA/Triton).

## Advanced Routing Rules

- **Small Sequences (< 1024 tokens)**: Routes to Fused IO-aware kernels (FlashAttention).
- **Long Videos (> 16 frames)**: Routes to Block-Sparse Temporal attention to avoid T² complexity.
- **High-Resolution Images**: Routes to Spatial Neighborhood (local window) kernels.
- **Massive Sequences (> 4096 tokens)**: Routes to Chunked Streaming execution.

## Getting Started

```python
import torch
from visionq.core import SpatioTemporalTensor
from visionq.models import VisionBackbone

# Native 3D Video Input: (B, T, H, W, C)
x = torch.randn(1, 8, 16, 16, 128)
st_x = SpatioTemporalTensor(x, modality="video")

# Industrial-grade Backbone
model = VisionBackbone(depth=6, dim=128, num_heads=8)
output = model(st_x) # Automatically dispatches to optimal Spatio-Temporal kernels
```

## Performance Targets

- Reduce complexity from $O(N^2)$ to $O(N \cdot k)$ via locality-aware kernels.
- Support video-scale processing ($T \times H \times W$) through factorized execution.
- Maintain a constant memory footprint for massive sequences via chunked streaming.
