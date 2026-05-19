from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import torch

@dataclass
class IRNode:
    """Base node for Attention Graph IR."""
    id: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QKVProjectionNode(IRNode):
    """Node for linear projection to Q, K, V."""
    dim: int = 0
    num_heads: int = 8

@dataclass
class MatMulNode(IRNode):
    """Node for matrix multiplication (e.g., QK^T)."""
    transpose_b: bool = False

@dataclass
class SoftmaxNode(IRNode):
    """Node for softmax normalization."""
    dim: int = -1

@dataclass
class MaskNode(IRNode):
    """Node for attention masking."""
    mask_type: str = "none"

@dataclass
class WeightedSumNode(IRNode):
    """Node for Softmax(QK^T) * V."""
    pass

class AttentionGraph:
    """
    Intermediate Representation of an Attention workflow.
    Allows for cross-operation optimization and fusion.
    """
    def __init__(self):
        self.nodes: Dict[str, IRNode] = {}
        self.entry_nodes: List[str] = []

    def add_node(self, node: IRNode):
        self.nodes[node.id] = node

    def __repr__(self):
        return f"AttentionGraph(nodes={list(self.nodes.keys())})"
