# Backend Support Matrix

| Backend | CPU | CUDA | Masks | Causal | Dropout | Notes |
|---|---:|---:|---:|---:|---:|---|
| `flash` | yes | yes via PyTorch | yes | yes | training only | Uses PyTorch SDPA |
| `sparse` | yes | yes via PyTorch ops | additive | no | training only | Strided KV sampling |
| `spatial_neighborhood` | yes | yes via PyTorch ops | no | no | training only | Reference local 2D attention |
| `temporal_neighborhood` | yes | yes via PyTorch ops | limited | yes via SDPA | training only | 5D video component |
| `spatiotemporal_hybrid` | yes | yes via PyTorch ops | no | no | inherited | Spatial then temporal |
| `chunked_streaming` | yes | yes via PyTorch ops | additive | no | no | Exact blockwise reference |
| `experimental.TritonAttentionKernel` | no | Linux CUDA | no | no | no | Experimental dense inference kernel |

All unsupported combinations should raise explicit errors instead of silently
falling back.
