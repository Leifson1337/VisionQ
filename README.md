# VisionQ: Spatio-Temporal Attention Runtime

VisionQ is a modular, high-performance runtime for Vision and Video Transformers. It provides a unified interface for different attention backends and dynamically selects the best implementation based on the execution context.

## Core Features

- **Unified Representation**: `STTensor` handles both image and video data seamlessly.
- **Dynamic Routing**: `AttentionDispatcher` selects optimal kernels (e.g., Flash Attention, Neighborhood Attention) at runtime.
- **Pluggable Architecture**: Easily register new attention backends.
- **Industrial-Grade Implementation**: Built with performance and scalability in mind.

## Installation

```bash
pip install .
```

## Quick Start

```python
import torch
from visionq.core import STTensor, AttentionContext
from visionq.models import VisionBackbone

# Initialize model
model = VisionBackbone(depth=6, dim=256, num_heads=8)

# Prepare data
x = torch.randn(1, 196, 256)
st_x = STTensor(x, modality="image", spatial_shape=(14, 14))

# Global Attention
ctx_global = AttentionContext(modality="image")
output = model(st_x, ctx_global)

# Local (Neighborhood) Attention
ctx_local = AttentionContext(modality="image", spatial_shape=(14, 14), window_size=7)
output = model(st_x, ctx_local)
```

## Architecture

- `core/`: Fundamental data types and context management.
- `attention/`: Backend implementations (Flash, Neighborhood, etc.).
- `runtime/`: Execution logic and backend selection.
- `models/`: High-level Transformer components.
