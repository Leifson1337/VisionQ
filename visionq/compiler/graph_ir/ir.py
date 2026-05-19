from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class IRNode:
    id: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QKVProjectionNode(IRNode):
    dim: int = 0
    num_heads: int = 8


@dataclass
class MatMulNode(IRNode):
    transpose_b: bool = False


@dataclass
class SoftmaxNode(IRNode):
    dim: int = -1


@dataclass
class MaskNode(IRNode):
    mask_type: str = "none"


@dataclass
class WeightedSumNode(IRNode):
    value_input: str = "v"


class AttentionGraph:
    """Serializable DAG representation of an attention execution plan."""

    def __init__(self) -> None:
        self.nodes: dict[str, IRNode] = {}

    def add_node(self, node: IRNode) -> None:
        if not node.id:
            raise ValueError("node id must be non-empty")
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id '{node.id}'")
        missing = [inp for inp in node.inputs if inp not in self.nodes]
        if missing:
            raise ValueError(f"node '{node.id}' references missing inputs {missing}")
        self.nodes[node.id] = node
        self.topological_order()

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        for node in self.nodes.values():
            node.inputs = [inp for inp in node.inputs if inp != node_id]
            node.outputs = [out for out in node.outputs if out != node_id]
        del self.nodes[node_id]

    def topological_order(self) -> list[str]:
        temporary: set[str] = set()
        permanent: set[str] = set()
        order: list[str] = []

        def visit(node_id: str) -> None:
            if node_id in permanent:
                return
            if node_id in temporary:
                raise ValueError("attention graph contains a cycle")
            temporary.add(node_id)
            node = self.nodes[node_id]
            for inp in node.inputs:
                if inp not in self.nodes:
                    raise ValueError(f"node '{node_id}' references missing input '{inp}'")
                visit(inp)
            temporary.remove(node_id)
            permanent.add(node_id)
            order.append(node_id)

        for node_id in self.nodes:
            visit(node_id)
        return order

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [asdict(self.nodes[node_id]) for node_id in self.topological_order()]}

    def __repr__(self) -> str:
        return f"AttentionGraph(nodes={self.topological_order()})"
