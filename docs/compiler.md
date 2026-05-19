# Compiler

The compiler package is a conservative planning layer, not a GPU code generator.
It includes:

- a serializable directed acyclic graph IR,
- topological ordering and cycle detection,
- duplicate projection elimination with input rewiring,
- fusion-pattern annotations for SDPA-like graph fragments.

It does not emit CUDA, Triton or fused binary kernels.
