import os
import json
import pytest
import torch
from pathlib import Path
from visionq.runtime.feedback_loop import FeedbackLoop, FeedbackRecord
from visionq.compiler.graph_ir.ir import AttentionGraph, QKVProjectionNode, MatMulNode, SoftmaxNode
from visionq.compiler.optimizer.optimizer import GraphOptimizer
from visionq.compiler.fusion_engine.fusion import FusionEngine

def test_feedback_loop_persistence(tmp_path):
    loop = FeedbackLoop()
    loop.record("flash", 10.5, 1024.0)
    loop.record("sparse", 15.2, 512.0)

    path = tmp_path / "telemetry.json"
    loop.export_json(path)

    assert path.exists()
    with open(path, "r") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["backend"] == "flash"

    new_loop = FeedbackLoop()
    new_loop.import_json(path)
    assert len(new_loop.records) == 2
    assert new_loop.records[0].latency_ms == 10.5
    assert new_loop.summary()["flash"] == 10.5

def test_dead_code_elimination_complex():
    graph = AttentionGraph()
    # Path 1: Useful
    graph.add_node(QKVProjectionNode(id="qkv_useful", dim=64, outputs=["t1"]))
    graph.add_node(MatMulNode(id="matmul_useful", inputs=["qkv_useful"], outputs=["t2"]))
    graph.add_node(SoftmaxNode(id="softmax_leaf", inputs=["matmul_useful"], outputs=["final"]))

    # Path 2: Dead
    graph.add_node(QKVProjectionNode(id="qkv_dead", dim=64, outputs=["t3"]))
    graph.add_node(MatMulNode(id="matmul_dead", inputs=["qkv_dead"], outputs=[])) # No named output, not used

    optimizer = GraphOptimizer(graph)
    optimized = optimizer.optimize()

    assert "qkv_useful" in optimized.nodes
    assert "softmax_leaf" in optimized.nodes
    assert "qkv_dead" not in optimized.nodes
    assert "matmul_dead" not in optimized.nodes

def test_fusion_engine_sdpa_annotation():
    graph = AttentionGraph()
    graph.add_node(QKVProjectionNode(id="qkv", dim=64, outputs=["t1"]))
    graph.add_node(MatMulNode(id="matmul", inputs=["qkv"], outputs=["t2"]))
    graph.add_node(SoftmaxNode(id="softmax", inputs=["matmul"], outputs=["t3"]))

    fusion = FusionEngine(graph)
    fused = fusion.fuse()

    assert fused.nodes["matmul"].metadata["sdpa_pattern"] is True
    assert fused.nodes["softmax"].metadata["fused_into"] == "matmul"
