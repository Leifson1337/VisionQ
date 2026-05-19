import torch
import unittest
from visionq.core import SpatioTemporalTensor, AttentionContext
from visionq.models.backbone import VisionBackbone

class TestVisionQ(unittest.TestCase):

    def test_st_tensor(self):
        x = torch.randn(1, 196, 128)
        st_tensor = SpatioTemporalTensor(x, modality="image", spatial_shape=(14, 14))
        self.assertEqual(st_tensor.shape, (1, 1, 14, 14, 128))
        self.assertEqual(st_tensor.modality, "image")

    def test_dispatcher(self):
        from visionq.runtime.dispatcher import AttentionDispatcher
        dispatcher = AttentionDispatcher()

        ctx_image = AttentionContext(modality="image", sequence_length=2048)
        backend_name = dispatcher.select(ctx_image)
        # For large image, routing should be neighborhood
        self.assertEqual(backend_name, "neighborhood")

    def test_backbone_forward_with_st_tensor(self):
        dim = 64
        model = VisionBackbone(depth=1, dim=dim, num_heads=4)
        x = torch.randn(1, 100, dim)
        st_x = SpatioTemporalTensor(x, modality="image", spatial_shape=(10, 10))

        out = model(st_x)
        self.assertEqual(out.unwrap().shape, (1, 1, 10, 10, dim))

    def test_depth_two(self):
        dim = 32
        model = VisionBackbone(depth=2, dim=dim, num_heads=2)
        x = torch.randn(1, 1, 4, 4, dim)
        st_x = SpatioTemporalTensor(x, modality="image")
        out = model(st_x)
        self.assertEqual(out.shape, (1, 1, 4, 4, dim))

if __name__ == '__main__':
    unittest.main()
