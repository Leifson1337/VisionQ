from __future__ import annotations

from ..graph_ir.ir import AttentionGraph, QKVProjectionNode


class GraphOptimizer:
    """Small semantics-preserving optimizer for the reference attention IR."""

    def __init__(self, graph: AttentionGraph) -> None:
        self.graph = graph

    def optimize(self) -> AttentionGraph:
        self._eliminate_duplicate_projections()
        self.graph.topological_order()
        return self.graph

    def _eliminate_duplicate_projections(self) -> None:
        seen: dict[tuple[tuple[str, ...], int, int], str] = {}
        replacements: dict[str, str] = {}
        for node_id in self.graph.topological_order():
            node = self.graph.nodes[node_id]
            if isinstance(node, QKVProjectionNode):
                key = (tuple(node.inputs), node.dim, node.num_heads)
                if key in seen:
                    replacements[node_id] = seen[key]
                else:
                    seen[key] = node_id
        for old, new in replacements.items():
            for node in self.graph.nodes.values():
                node.inputs = [new if inp == old else inp for inp in node.inputs]
            self.graph.remove_node(old)
