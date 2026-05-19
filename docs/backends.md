# Backends

All public backend names are defined by `AttentionBackendName`.

`flash` delegates to PyTorch SDPA. `sparse` performs strided key/value sampling.
`spatial_neighborhood` and `temporal_neighborhood` are local reference
implementations. `spatiotemporal_hybrid` composes the spatial and temporal
backends. `chunked_streaming` computes exact dense attention blockwise with
online softmax.

Unsupported masks raise exceptions instead of being ignored.

## Optional Triton Kernel

`visionq.kernels.triton.TritonAttentionKernel` is a low-level experimental CUDA
kernel. It is not registered as a default attention backend because it has a
narrow support envelope: dense non-causal attention only, no masks, no dropout,
CUDA tensors only, and power-of-two head dimensions up to 128. Tests are marked
with `gpu` and `triton` and skip automatically when hardware or dependencies are
not available.
