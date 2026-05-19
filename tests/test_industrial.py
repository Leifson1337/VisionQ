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
        # Sequence length matters for the router
        x = torch.randn(1, 1024, self.dim)
        st_x = SpatioTemporalTensor(x, modality="image", spatial_shape=(32, 32))
        ctx = AttentionContext(modality="image", spatial_shape=(32, 32), sequence_length=1024)

        output = self.model(st_x, ctx)
        self.assertIsInstance(output, SpatioTemporalTensor)

    def test_video_flow_3d_neighborhood(self):
        T, H, W = 2, 8, 8
        N = T * H * W
        x = torch.randn(1, T, H, W, self.dim)
        st_x = SpatioTemporalTensor(x, modality="video")
        ctx = AttentionContext.from_st_tensor(st_x, attention_mode="spatio_temporal", sequence_length=N)

        output = self.model(st_x, ctx)
        self.assertIsInstance(output, SpatioTemporalTensor)

    def test_dispatcher_routing_rules(self):
        dispatcher = AttentionDispatcher()

        # Rule: small sequence -> flash
        ctx_small = AttentionContext(modality="sequence", sequence_length=512)
        self.assertEqual(dispatcher.select(ctx_small), "flash")

        # Rule: large image -> neighborhood
        ctx_img = AttentionContext(modality="image", sequence_length=4096)
        self.assertEqual(dispatcher.select(ctx_img), "neighborhood")

        # Rule: long video -> spatiotemporal or sparse
        ctx_vid = AttentionContext(modality="video", sequence_length=4096)
        self.assertEqual(dispatcher.select(ctx_vid), "spatiotemporal_hybrid")

        # Rule: massive sequence -> chunked_streaming or sparse
        ctx_seq = AttentionContext(modality="sequence", sequence_length=20000)
        # Sequence > 16384 in PolicyModel defaults to chunked_streaming or sparse depending on threshold
        self.assertIn(dispatcher.select(ctx_seq), ["chunked_streaming", "sparse"])

if __name__ == '__main__':
    unittest.main()
