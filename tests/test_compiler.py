import unittest
from visionq.compiler.graph_ir.ir import AttentionGraph, QKVProjectionNode
from visionq.compiler.optimizer.optimizer import GraphOptimizer
from visionq.compiler.fusion_engine.fusion import FusionEngine

class TestAttentionCompiler(unittest.TestCase):

    def test_graph_creation(self):
        graph = AttentionGraph()
        graph.add_node(QKVProjectionNode(id="qkv", dim=128))
        self.assertIn("qkv", graph.nodes)

    def test_optimization_pass(self):
        graph = AttentionGraph()
        graph.add_node(QKVProjectionNode(id="qkv", dim=128))
        optimizer = GraphOptimizer(graph)
        opt_graph = optimizer.optimize()
        self.assertEqual(len(opt_graph.nodes), 1)

    def test_fusion_pass(self):
        graph = AttentionGraph()
        graph.add_node(QKVProjectionNode(id="qkv", dim=128))
        fusion = FusionEngine(graph)
        fused_graph = fusion.fuse()
        self.assertEqual(len(fused_graph.nodes), 1)

if __name__ == '__main__':
    unittest.main()
