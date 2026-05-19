import pytest
import torch

from visionq.core import AttentionContext, SpatioTemporalTensor
from visionq.models import PatchEmbed, VisionBackbone, VisionTransformer


def test_backbone_sequence_forward_and_grad():
    model = VisionBackbone(depth=1, dim=16, num_heads=4, backend_override="flash")
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 5, 1, 1, 16)
    out.flatten_all().sum().backward()
    assert x.grad is not None


def test_backbone_image_forward():
    model = VisionBackbone(depth=1, dim=16, num_heads=4, backend_override="spatial_neighborhood")
    x = SpatioTemporalTensor(torch.randn(1, 16, 16), modality="image", spatial_shape=(4, 4))
    ctx = AttentionContext(
        modality="image", sequence_length=16, spatial_shape=(4, 4), spatial_window=(3, 3)
    )
    out = model(x, ctx)
    assert out.shape == (1, 1, 4, 4, 16)


def test_backbone_video_forward():
    model = VisionBackbone(depth=1, dim=16, num_heads=4, backend_override="spatiotemporal_hybrid")
    x = torch.randn(1, 2, 2, 2, 16)
    out = model(x)
    assert out.shape == (1, 2, 2, 2, 16)


def test_invalid_heads_raise():
    with pytest.raises(ValueError, match="divisible"):
        VisionBackbone(depth=1, dim=10, num_heads=4)


def test_patch_embed_and_vit_forward():
    patch = PatchEmbed(image_size=(8, 8), patch_size=(4, 4), in_channels=3, embed_dim=16)
    x = torch.randn(2, 3, 8, 8)
    assert patch(x).shape == (2, 4, 16)
    model = VisionTransformer(
        image_size=(8, 8),
        patch_size=(4, 4),
        in_channels=3,
        num_classes=5,
        depth=1,
        dim=16,
        num_heads=4,
        backend_override="flash",
    )
    logits = model(x)
    assert logits.shape == (2, 5)
