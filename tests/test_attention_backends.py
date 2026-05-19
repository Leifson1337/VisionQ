import torch
import torch.nn.functional as F

from visionq.attention import (
    ChunkedStreamingAttention,
    FlashAttention,
    SparseAttention,
    SpatialNeighborhoodAttention,
    SpatioTemporalHybridAttention,
    TemporalNeighborhoodAttention,
)
from visionq.core import AttentionContext
from visionq.kernels.ops.online_softmax import OnlineSoftmax
from visionq.kernels.triton.attention_kernel import (
    ReferenceBlockwiseAttentionKernel,
    TritonAttentionKernel,
    triton_available,
)


def qkv(tokens=8, dim=4, heads=2):
    torch.manual_seed(0)
    return tuple(torch.randn(1, heads, tokens, dim, requires_grad=True) for _ in range(3))


def test_flash_matches_sdpa_and_backpropagates():
    q, k, v = qkv()
    ctx = AttentionContext(modality="sequence", sequence_length=8)
    out = FlashAttention(dim=8, num_heads=2)(q, k, v, ctx)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v))
    out.sum().backward()
    assert q.grad is not None


def test_flash_handles_additive_mask_and_extreme_logits():
    q, k, v = qkv(tokens=4, dim=4)
    q = q * 100
    k = k * 100
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., -1] = float("-inf")
    ctx = AttentionContext(modality="sequence", sequence_length=4)
    out = FlashAttention(dim=8, num_heads=2)(q, k, v, ctx, mask=mask)
    assert torch.isfinite(out).all()


def test_sparse_dilation_one_matches_dense():
    q, k, v = qkv()
    ctx = AttentionContext(modality="sequence", sequence_length=8, dilation=1)
    out = SparseAttention(dim=8, num_heads=2)(q, k, v, ctx)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v))


def test_spatial_neighborhood_shape_rectangular():
    q, k, v = qkv(tokens=12)
    ctx = AttentionContext(
        modality="image", sequence_length=12, spatial_shape=(3, 4), spatial_window=(3, 3)
    )
    out = SpatialNeighborhoodAttention(dim=8, num_heads=2)(q, k, v, ctx)
    assert out.shape == q.shape


def test_temporal_neighborhood_shape_and_error():
    q, k, v = qkv(tokens=8)
    ctx = AttentionContext(
        modality="video", sequence_length=8, spatial_shape=(4, 1), temporal_dim=2
    )
    q5 = q.reshape(1, 2, 2, 4, 4)
    out = TemporalNeighborhoodAttention(dim=8, num_heads=2)(q5, q5, q5, ctx)
    assert out.shape == q5.shape


def test_hybrid_video_shape():
    q, k, v = qkv(tokens=8)
    ctx = AttentionContext(
        modality="video",
        sequence_length=8,
        spatial_shape=(2, 2),
        temporal_dim=2,
        spatial_window=(3, 3),
        temporal_window=3,
    )
    out = SpatioTemporalHybridAttention(dim=8, num_heads=2)(q, k, v, ctx)
    assert out.shape == q.shape


def test_online_softmax_and_blockwise_match_dense():
    q, k, v = qkv(tokens=16, dim=8)
    ctx = AttentionContext(modality="sequence", sequence_length=16)
    out = ReferenceBlockwiseAttentionKernel().forward(q, k, v, ctx, block_size=4)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-5)


def test_chunked_streaming_matches_dense():
    q, k, v = qkv(tokens=16, dim=8)
    ctx = AttentionContext(modality="sequence", sequence_length=16)
    out = ChunkedStreamingAttention(dim=16, num_heads=2)(q, k, v, ctx, block_size=4)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-5)


def test_online_softmax_update_is_finite():
    scores = torch.randn(1, 2, 4, 4)
    prev_max = torch.full((1, 2, 4, 1), float("-inf"))
    prev_sum = torch.zeros((1, 2, 4, 1))
    new_max, new_sum, exp_scores = OnlineSoftmax.update(prev_max, prev_sum, scores)
    assert torch.isfinite(new_max).all()
    assert torch.isfinite(new_sum).all()
    assert torch.isfinite(exp_scores).all()


def test_triton_kernel_requires_triton_or_cuda():
    q, k, v = qkv(tokens=8, dim=8)
    ctx = AttentionContext(modality="sequence", sequence_length=8)
    kernel = TritonAttentionKernel()
    if not triton_available():
        try:
            kernel.forward(q, k, v, ctx)
        except RuntimeError as exc:
            assert "Triton is not installed" in str(exc)
        else:
            raise AssertionError("TritonAttentionKernel should reject missing Triton")
    elif not torch.cuda.is_available():
        try:
            kernel.forward(q, k, v, ctx)
        except RuntimeError as exc:
            assert "requires CUDA" in str(exc)
        else:
            raise AssertionError("TritonAttentionKernel should reject CPU tensors")
