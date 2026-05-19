import torch
import unittest
from visionq.core import STTensor, AttentionContext
from visionq.attention.registry import ATTENTION_REGISTRY
from visionq.runtime.dispatcher import AttentionDispatcher
from visionq.models.backbone import VisionBackbone

class TestVisionQIndustrial(unittest.TestCase):

    def setUp(self):
        self.dim = 64
        self.num_heads = 4
        self.model = VisionBackbone(depth=2, dim=self.dim, num_heads=self.num_heads)

    def test_image_flow(self):
        x = torch.randn(1, 100, self.dim)
        st_x = STTensor(x, modality="image", spatial_shape=(10, 10))
        ctx = AttentionContext(modality="image", spatial_shape=(10, 10), window_size=3)

        output = self.model(st_x, ctx)
        self.assertIsInstance(output, STTensor)
        self.assertEqual(output.shape, (1, 100, self.dim))

    def test_video_flow_3d_neighborhood(self):
        T, H, W = 2, 4, 4
        N = T * H * W
        x = torch.randn(1, N, self.dim)
        st_x = STTensor(x, modality="video", spatial_shape=(H, W), temporal_dim=T)
        ctx = AttentionContext.from_st_tensor(st_x, window_size=3)

        output = self.model(st_x, ctx)
        self.assertIsInstance(output, STTensor)
        self.assertEqual(output.shape, (1, N, self.dim))

    def test_depth_greater_than_one(self):
        # Testing the depth > 1 fix
        x = torch.randn(1, 16, self.dim)
        st_x = STTensor(x, modality="image", spatial_shape=(4, 4))

        # Should not crash
        output = self.model(st_x)
        self.assertEqual(output.shape, (1, 16, self.dim))

    def test_dispatcher_routing_rules(self):
        dispatcher = AttentionDispatcher()

        # Rule: image -> neighborhood default
        ctx_img = AttentionContext(modality="image")
        self.assertEqual(dispatcher.select(ctx_img), "neighborhood")

        # Rule: video -> neighborhood default
        ctx_vid = AttentionContext(modality="video")
        self.assertEqual(dispatcher.select(ctx_vid), "neighborhood")

        # Rule: sequence -> sparse
        ctx_seq = AttentionContext(modality="sequence")
        self.assertEqual(dispatcher.select(ctx_seq), "sparse")

if __name__ == '__main__':
    unittest.main()
