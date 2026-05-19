import torch
import unittest
from visionq.core import STTensor, AttentionContext
from visionq.attention.registry import ATTENTION_REGISTRY
from visionq.runtime.dispatcher import AttentionDispatcher
from visionq.models.backbone import VisionBackbone

class TestVisionQ(unittest.TestCase):

    def test_st_tensor(self):
        x = torch.randn(1, 196, 128)
        st_tensor = STTensor(x, modality="image", spatial_shape=(14, 14))
        self.assertEqual(st_tensor.shape, (1, 196, 128))
        self.assertEqual(st_tensor.modality, "image")
        self.assertEqual(st_tensor.spatial_shape, (14, 14))

    def test_attention_registry(self):
        self.assertIn("neighborhood", ATTENTION_REGISTRY)
        self.assertIn("flash", ATTENTION_REGISTRY)

    def test_dispatcher(self):
        dispatcher = AttentionDispatcher()

        ctx_image = AttentionContext(modality="image")
        backend_name = dispatcher.select(ctx_image)
        self.assertEqual(backend_name, "flash")

        ctx_neighborhood = AttentionContext(modality="image", window_size=7)
        backend_name = dispatcher.select(ctx_neighborhood)
        self.assertEqual(backend_name, "neighborhood")

    def test_backbone_forward_with_st_tensor(self):
        dim = 64
        model = VisionBackbone(depth=1, dim=dim, num_heads=4)
        x = torch.randn(1, 100, dim)
        st_x = STTensor(x, modality="image", spatial_shape=(10, 10))

        # Test default context from STTensor
        out = model(st_x)
        self.assertEqual(out.shape, (1, 100, dim))

        # Test Neighborhood Attention
        ctx_local = AttentionContext(modality="image", spatial_shape=(10, 10), window_size=3)
        out_local = model(st_x, ctx_local)
        self.assertEqual(out_local.shape, (1, 100, dim))

    def test_neighborhood_2d_mask(self):
        from visionq.attention.neighborhood import NeighborhoodAttention
        na = NeighborhoodAttention(dim=32, num_heads=2)
        ctx = AttentionContext(modality="image", spatial_shape=(4, 4), window_size=3)
        x = torch.randn(1, 16, 32)
        out = na(x, ctx)
        self.assertEqual(out.shape, (1, 16, 32))

if __name__ == '__main__':
    unittest.main()
