from __future__ import annotations

from ..graph_ir.ir import AttentionGraph, MatMulNode, SoftmaxNode


class FusionEngine:
    """Annotates fusable SDPA patterns without generating GPU code."""

    def __init__(self, graph: AttentionGraph) -> None:
        self.graph = graph

    def fuse(self) -> AttentionGraph:
        self._fuse_sdpa_patterns()
        self._fuse_linear_bias_patterns()
        return self.graph

    def _fuse_sdpa_patterns(self) -> None:
        """Annotates MatMul + Softmax patterns for SDPA fusion."""
        for node_id in self.graph.topological_order():
            node = self.graph.nodes[node_id]
            if isinstance(node, MatMulNode):
                # Check if this MatMul is followed by a Softmax
                softmax_consumers = [
                    candidate
                    for candidate in self.graph.nodes.values()
                    if node_id in candidate.inputs and isinstance(candidate, SoftmaxNode)
                ]
                if softmax_consumers:
                    node.metadata["sdpa_pattern"] = True
                    for sm in softmax_consumers:
                        sm.metadata["fused_into"] = node_id

    def _fuse_linear_bias_patterns(self) -> None:
        """Example of future expansion: fusing MatMul and subsequent Add for bias."""
        # This is a placeholder for industrial-grade fusion logic
        # In a real compiler, we would look for patterns like:
        # MatMul -> Add (Bias) -> Activation
        pass
