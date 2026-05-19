# Backend Development

New attention backends must:

1. Inherit `AttentionBackend`.
2. Use a key from `AttentionBackendName` or add a new enum value with tests.
3. Validate Q/K/V shape, dtype and device.
4. Document mask, causal and dropout support.
5. Provide CPU tests and skip-safe GPU tests if hardware-specific.
6. Compare numerics against PyTorch SDPA or an equivalent dense reference.
7. Avoid silent fallbacks.

Minimal shape contract:

```python
q.shape == k.shape == v.shape == (batch, heads, tokens, head_dim)
```

Video-specific internal components may use `(batch, time, heads, spatial_tokens,
head_dim)`, but public backends should document any deviation.
