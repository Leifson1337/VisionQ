import pytest

from visionq.compiler.fusion_engine.fusion import FusionEngine
from visionq.compiler.graph_ir.ir import AttentionGraph, MatMulNode, QKVProjectionNode, SoftmaxNode
from visionq.compiler.optimizer.optimizer import GraphOptimizer


def test_graph_topological_order_and_serialization():
    graph = AttentionGraph()
    graph.add_node(QKVProjectionNode(id="qkv", dim=128))
    graph.add_node(MatMulNode(id="scores", inputs=["qkv"]))
    assert graph.topological_order() == ["qkv", "scores"]
    assert graph.to_dict()["nodes"][0]["id"] == "qkv"


def test_graph_rejects_missing_input():
    graph = AttentionGraph()
    with pytest.raises(ValueError, match="missing"):
        graph.add_node(MatMulNode(id="scores", inputs=["qkv"]))


def test_optimizer_removes_duplicate_projection_and_rewires():
    graph = AttentionGraph()
    graph.add_node(QKVProjectionNode(id="qkv1", dim=64))
    graph.add_node(QKVProjectionNode(id="qkv2", dim=64))
    graph.add_node(MatMulNode(id="scores", inputs=["qkv2"]))
    optimized = GraphOptimizer(graph).optimize()
    assert "qkv2" not in optimized.nodes
    assert optimized.nodes["scores"].inputs == ["qkv1"]


def test_fusion_annotates_sdpa_pattern():
    graph = AttentionGraph()
    graph.add_node(QKVProjectionNode(id="qkv", dim=64))
    graph.add_node(MatMulNode(id="scores", inputs=["qkv"]))
    graph.add_node(SoftmaxNode(id="softmax", inputs=["scores"]))
    fused = FusionEngine(graph).fuse()
    assert fused.nodes["scores"].metadata["sdpa_pattern"] is True
