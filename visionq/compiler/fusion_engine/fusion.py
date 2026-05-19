from __future__ import annotations

from ..graph_ir.ir import AttentionGraph, MatMulNode, SoftmaxNode


class FusionEngine:
    """Annotates fusable SDPA patterns without generating GPU code."""

    def __init__(self, graph: AttentionGraph) -> None:
        self.graph = graph

    def fuse(self) -> AttentionGraph:
        for node_id in self.graph.topological_order():
            node = self.graph.nodes[node_id]
            if isinstance(node, MatMulNode):
                consumers = [
                    candidate
                    for candidate in self.graph.nodes.values()
                    if node_id in candidate.inputs and isinstance(candidate, SoftmaxNode)
                ]
                if consumers:
                    node.metadata["sdpa_pattern"] = True
        return self.graph
