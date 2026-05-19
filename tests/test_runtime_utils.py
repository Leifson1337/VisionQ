import torch
import torch.nn as nn
import torch.nn.functional as F

from visionq.compiler.graph_ir.ir import AttentionGraph, QKVProjectionNode
from visionq.compiler.scheduler.scheduler import ExecutionScheduler
from visionq.core import AttentionContext
from visionq.kernels.cpu.attention import neighborhood_mask_cpu
from visionq.kernels.cpu.fallback import cpu_tiled_attention
from visionq.runtime.dispatcher import AttentionDispatcher
from visionq.runtime.feedback_loop import FeedbackLoop
from visionq.utils.helpers import count_parameters, get_device


def test_cpu_helpers_match_dense_and_create_mask():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 8, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = cpu_tiled_attention(q, k, v, block_size=4, scale=4**-0.5)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-5)
    mask = neighborhood_mask_cpu(1, 2, 2, 3, q.device)
    assert mask.shape == (4, 4)
    assert torch.isfinite(mask).all()


def test_feedback_and_misc_helpers():
    loop = FeedbackLoop()
    loop.record("flash", 1.0)
    loop.record("flash", 3.0)
    assert loop.summary()["flash"] == 2.0
    assert get_device().type in {"cpu", "cuda"}
    assert count_parameters(nn.Linear(2, 3)) == 9


def test_dispatcher_compile_and_dispatch_kernel():
    dispatcher = AttentionDispatcher()
    graph = dispatcher.compile_graph({"dim": 8})
    assert graph.topological_order() == ["qkv", "scores", "softmax"]
    q = torch.randn(1, 2, 4, 4)
    ctx = AttentionContext(modality="sequence", sequence_length=4)
    out = dispatcher.dispatch_kernel(q, q, q, ctx)
    assert out.shape == q.shape


def test_scheduler_reference_plan():
    graph = AttentionGraph()
    graph.add_node(QKVProjectionNode(id="qkv", dim=8))
    scheduler = ExecutionScheduler(graph)
    assert scheduler.plan({}) == ["qkv"]
    assert scheduler.allocate_memory(8) == {"qkv": "buffer_0"}
