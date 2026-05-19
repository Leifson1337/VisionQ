import torch
import unittest
from visionq.core import SpatioTemporalTensor
from visionq.core.context import AttentionContext
from visionq.models.video_backbone import VideoBackbone

class TestVisionQVideo(unittest.TestCase):

    def setUp(self):
        self.dim = 64
        self.num_heads = 4
        self.model = VideoBackbone(depth=1, dim=self.dim, num_heads=self.num_heads)

    def test_video_native_forward(self):
        B, T, H, W, C = 1, 4, 8, 8, self.dim
        x = torch.randn(B, T, H, W, C)
        st_x = SpatioTemporalTensor(x, modality="video")

        ctx = AttentionContext.from_st_tensor(st_x, attention_mode="spatio_temporal", spatial_window=(3, 3), temporal_window=3)

        output = self.model(st_x, ctx)
        self.assertIsInstance(output, SpatioTemporalTensor)
        self.assertEqual(output.unwrap().shape, (B, T, H, W, C))

if __name__ == '__main__':
    unittest.main()
