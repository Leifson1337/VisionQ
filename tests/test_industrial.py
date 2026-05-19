import torch
import unittest
from visionq.core import SpatioTemporalTensor, AttentionContext
from visionq.attention.registry import ATTENTION_REGISTRY
from visionq.runtime.dispatcher import AttentionDispatcher
from visionq.models.backbone import VisionBackbone

class TestVisionQIndustrial(unittest.TestCase):

    def setUp(self):
        self.dim = 64
        self.num_heads = 4
        self.model = VisionBackbone(depth=1, dim=self.dim, num_heads=self.num_heads)

    def test_image_flow(self):
        x = torch.randn(1, 100, self.dim)
        st_x = SpatioTemporalTensor(x, modality="image", spatial_shape=(10, 10))
        ctx = AttentionContext(modality="image", spatial_shape=(10, 10), spatial_window=(3, 3), sequence_length=2048)

        output = self.model(st_x, ctx)
        self.assertIsInstance(output, SpatioTemporalTensor)
        self.assertEqual(output.shape, (1, 1, 10, 10, self.dim))

    def test_video_flow_3d_neighborhood(self):
        T, H, W = 2, 4, 4
        N = T * H * W
        x = torch.randn(1, T, H, W, self.dim)
        st_x = SpatioTemporalTensor(x, modality="video")
        ctx = AttentionContext.from_st_tensor(st_x, attention_mode="spatio_temporal", spatial_window=(3, 3), temporal_window=3, sequence_length=2048)

        output = self.model(st_x, ctx)
        self.assertIsInstance(output, SpatioTemporalTensor)
        self.assertEqual(output.shape, (1, T, H, W, self.dim))

    def test_depth_greater_than_one(self):
        x = torch.randn(1, 4, 4, self.dim)
        st_x = SpatioTemporalTensor(x, modality="image")
        model = VisionBackbone(depth=2, dim=self.dim, num_heads=self.num_heads)
        output = model(st_x)
        self.assertEqual(output.shape, (1, 1, 4, 4, self.dim))

    def test_dispatcher_routing_rules(self):
        dispatcher = AttentionDispatcher()

        # Rule: large image -> neighborhood default
        ctx_img = AttentionContext(modality="image", sequence_length=2048)
        self.assertEqual(dispatcher.select(ctx_img), "neighborhood")

        # Rule: video -> spatio_temporal hybrid default
        ctx_vid = AttentionContext(modality="video", attention_mode="spatio_temporal", sequence_length=2048)
        self.assertEqual(dispatcher.select(ctx_vid), "spatiotemporal_hybrid")

        # Rule: massive sequence -> streaming
        ctx_seq = AttentionContext(modality="sequence", sequence_length=5000)
        self.assertEqual(dispatcher.select(ctx_seq), "chunked_streaming")

if __name__ == '__main__':
    unittest.main()
