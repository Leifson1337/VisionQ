import pytest
import torch

from visionq.core import AttentionContext, SpatioTemporalTensor


def test_st_tensor_from_tokens_image():
    x = torch.randn(2, 12, 8)
    st = SpatioTemporalTensor(x, modality="image", spatial_shape=(3, 4))
    assert st.shape == (2, 1, 3, 4, 8)
    assert st.flatten_all().shape == (2, 12, 8)


def test_st_tensor_from_bchw_image():
    x = torch.randn(2, 3, 4, 5)
    st = SpatioTemporalTensor(x, modality="image")
    assert st.shape == (2, 1, 4, 5, 3)


def test_context_rejects_inconsistent_shape():
    with pytest.raises(ValueError, match="inconsistent"):
        AttentionContext(modality="video", sequence_length=10, spatial_shape=(2, 2), temporal_dim=2)


def test_tiles_require_divisible_shape():
    st = SpatioTemporalTensor(torch.randn(1, 1, 5, 4, 3), modality="image")
    with pytest.raises(ValueError, match="evenly divide"):
        st.to_tiles((3, 2))


def test_supported_shape_matrix_round_trips():
    cases = [
        (torch.randn(1, 2, 3), "sequence", None, None, (1, 2, 1, 1, 3)),
        (torch.randn(1, 6, 3), "image", (2, 3), None, (1, 1, 2, 3, 3)),
        (torch.randn(1, 2, 2, 3, 4), "video", None, None, (1, 2, 2, 3, 4)),
        (torch.randn(1, 3, 4, 5), "image", None, None, (1, 1, 4, 5, 3)),
    ]
    for tensor, modality, spatial_shape, temporal_dim, expected in cases:
        st = SpatioTemporalTensor(
            tensor,
            modality=modality,
            spatial_shape=spatial_shape,
            temporal_dim=temporal_dim,
        )
        assert st.shape == expected
        assert st.flatten_all().shape[-1] == expected[-1]
