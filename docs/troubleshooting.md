# Troubleshooting

## Triton Is Not Available

Install `visionq[triton]` on a Linux CUDA environment. Windows CPU environments
should use the reference backends and will skip Triton tests.

## CUDA Requested But Not Available

Confirm `torch.cuda.is_available()` returns `True`. The benchmark exits early
when `--device cuda` is requested without CUDA.

## Backend Rejects Masks

Some reference backends intentionally reject masks rather than silently ignoring
them. Use `flash`, `sparse` or `chunked_streaming` for supported additive masks.

## Shape Validation Errors

`SpatioTemporalTensor` uses canonical `(B, T, H, W, C)` layout internally. Token
inputs require `spatial_shape` when used as image or video data.
