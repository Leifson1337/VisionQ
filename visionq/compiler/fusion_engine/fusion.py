from ..graph_ir.ir import AttentionGraph, IRNode, SoftmaxNode, MatMulNode
from typing import List, Dict

class FusionEngine:
    """
    Graph-to-Fused-Kernel Transformation Engine.
    Detects patterns like SDPA (MatMul + Softmax + MatMul) and replaces them with fused nodes.
    """
    def __init__(self, graph: AttentionGraph):
        self.graph = graph

    def fuse(self) -> AttentionGraph:
        """
        Detects and applies kernel fusion.
        """
        self._fuse_sdpa()
        return self.graph

    def _fuse_sdpa(self):
        """
        Fuses standard Scaled Dot Product Attention components.
        """
        # Logic to identify Softmax followed by MatMul and collapse them
        # into a single execution plan metadata for the backend.
        nodes_to_remove = []
        for node_id, node in self.graph.nodes.items():
            if isinstance(node, SoftmaxNode):
                # Search for MatMul consumers
                for consumer_id, consumer in self.graph.nodes.items():
                    if node_id in consumer.inputs and isinstance(consumer, MatMulNode):
                        consumer.metadata["fused_softmax"] = True
                        nodes_to_remove.append(node_id)

        for node_id in nodes_to_remove:
            if node_id in self.graph.nodes:
                del self.graph.nodes[node_id]
