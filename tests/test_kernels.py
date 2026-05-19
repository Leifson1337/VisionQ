import torch
import unittest
from visionq.core import AttentionContext
from visionq.kernels.triton.attention_kernel import TritonAttentionKernel
from visionq.kernels.ops.online_softmax import OnlineSoftmax

class TestKernelEngine(unittest.TestCase):

    def test_online_softmax_stability(self):
        B, H, N, D = 1, 1, 32, 1
        prev_max = torch.full((B, H, N, 1), -10.0)
        prev_sum = torch.ones((B, H, N, 1))
        scores = torch.randn(B, H, N, 16)

        new_max, new_sum, exp_scores = OnlineSoftmax.update(prev_max, prev_sum, scores)

        # Values should be numerically stable
        self.assertFalse(torch.isnan(new_max).any())
        self.assertFalse(torch.isinf(new_sum).any())

    def test_tiled_attention_numerical_parity(self):
        # Compare Triton-style tiled forward with standard PyTorch
        B, H, N, D = 1, 1, 128, 32
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)

        context = AttentionContext(modality="sequence", sequence_length=N)
        kernel = TritonAttentionKernel()

        # Tiled forward (block_size=32)
        out_tiled = kernel.forward(q, k, v, context, block_size=32)

        # Standard dense forward
        scale = D ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out_dense = attn @ v

        # Parity check
        torch.testing.assert_close(out_tiled, out_dense, atol=1e-5, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
