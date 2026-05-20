from __future__ import annotations

from ..graph_ir.ir import AttentionGraph, QKVProjectionNode


class GraphOptimizer:
    """Small semantics-preserving optimizer for the reference attention IR."""

    def __init__(self, graph: AttentionGraph) -> None:
        self.graph = graph

    def optimize(self) -> AttentionGraph:
        self._eliminate_duplicate_projections()
        self._dead_code_elimination()
        self.graph.topological_order()
        return self.graph

    def _dead_code_elimination(self) -> None:
        """Removes nodes that do not contribute to any output."""
        changed = True
        while changed:
            changed = False
            used_ids = set()
            for node in self.graph.nodes.values():
                used_ids.update(node.inputs)

            to_remove = []
            for node_id in self.graph.nodes:
                # If a node is not used by any other node AND it doesn't have explicit outputs
                # marked in its metadata (or list), it's dead.
                # However, our current IR uses 'outputs' as a list of tensor names it produces.
                # If 'outputs' is empty, it means it doesn't produce any named tensor that could
                # be an external output of the graph.
                if node_id not in used_ids and not self.graph.nodes[node_id].outputs:
                    # Special case: don't remove if it's the only node (very unlikely)
                    # or if we want to preserve it for some reason.
                    # For this reference impl, we'll be aggressive.
                    to_remove.append(node_id)

            if to_remove:
                for node_id in to_remove:
                    self.graph.remove_node(node_id)
                changed = True

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
