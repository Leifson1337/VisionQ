import pytest
import torch
import torch.nn.functional as F

from visionq.core import AttentionContext
from visionq.experimental import TritonAttentionKernel as ExperimentalTritonAttentionKernel
from visionq.kernels.triton import TritonAttentionKernel, triton_available

pytestmark = [pytest.mark.gpu, pytest.mark.triton]


def test_experimental_triton_reexport():
    assert ExperimentalTritonAttentionKernel is TritonAttentionKernel


@pytest.mark.skipif(not triton_available(), reason="Triton is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_triton_attention_matches_sdpa_on_cuda():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 16, 16, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ctx = AttentionContext(modality="sequence", sequence_length=16, device=q.device)
    out = TritonAttentionKernel().forward(q, k, v, ctx)
    expected = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
