# API Reference

Stable public namespaces:

- `visionq.core`
  - `SpatioTemporalTensor`
  - `AttentionContext`
- `visionq.attention`
  - `AttentionBackend`
  - `AttentionBackendName`
  - `get_attention_backend`
  - `available_attention_backends`
  - backend classes
- `visionq.runtime`
  - `AttentionDispatcher`
  - `KernelRouter`
  - `RoutingDecision`
- `visionq.models`
  - `PatchEmbed`
  - `VisionBackbone`
  - `VisionBackboneBlock`
  - `VisionTransformer`
- `visionq.exceptions`
  - `VisionQError`
  - `BackendNotAvailableError`
  - `ShapeError`
  - `UnsupportedFeatureError`
- `visionq.compiler`
  - graph IR and conservative optimizer/fusion annotations

Experimental APIs:

- `visionq.experimental.triton`
  - `TritonAttentionKernel`
  - `TritonKernelLimits`
  - `triton_available`

Experimental APIs may change in minor releases. Stable APIs follow semantic
versioning once the project reaches `1.0.0`.
