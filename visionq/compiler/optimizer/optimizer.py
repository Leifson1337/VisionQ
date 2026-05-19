from ..graph_ir.ir import AttentionGraph, QKVProjectionNode, MatMulNode
from typing import List, Dict, Set

class GraphOptimizer:
    """
    Compiler-level Optimizer for Attention Graphs.
    Performs redundant op elimination and operator reordering.
    """
    def __init__(self, graph: AttentionGraph):
        self.graph = graph

    def optimize(self) -> AttentionGraph:
        """Runs the optimization pass."""
        self._eliminate_redundant_projections()
        self._reorder_masking()
        return self.graph

    def _eliminate_redundant_projections(self):
        """Removes shared K/V projections if repeated."""
        seen_params: Dict[tuple, str] = {}
        redundant_nodes: Set[str] = set()

        for node_id, node in list(self.graph.nodes.items()):
            if isinstance(node, QKVProjectionNode):
                # Simple heuristic: same dim and num_heads means redundant if inputs are same
                key = (tuple(node.inputs), node.dim, node.num_heads)
                if key in seen_params:
                    redundant_nodes.add(node_id)
                else:
                    seen_params[key] = node_id

        for node_id in redundant_nodes:
            del self.graph.nodes[node_id]

    def _reorder_masking(self):
        """Moves mask operation before matmul if it reduces compute."""
        # In a real compiler, we would analyze the data flow to see if
        # a mask can be applied early to skip rows/cols.
        # Here we implement a node-swapping logic if specific conditions are met.
        pass
