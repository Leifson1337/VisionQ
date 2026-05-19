from __future__ import annotations

from ..graph_ir.ir import AttentionGraph


class CodeGenerator:
    """Produces a human-readable reference execution plan."""

    def __init__(self, graph: AttentionGraph) -> None:
        self.graph = graph

    def generate(self) -> str:
        nodes = ", ".join(self.graph.topological_order())
        return f"Reference attention execution plan: {nodes}"
