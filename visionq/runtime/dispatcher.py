from __future__ import annotations

import logging
from typing import Any

import torch

from ..attention.registry import ATTENTION_REGISTRY, get_attention_backend
from ..compiler.fusion_engine.fusion import FusionEngine
from ..compiler.graph_ir.ir import AttentionGraph, MatMulNode, QKVProjectionNode, SoftmaxNode
from ..compiler.optimizer.optimizer import GraphOptimizer
from ..core.context import AttentionContext
from .kernel_router import KernelRouter, RoutingDecision

LOGGER = logging.getLogger(__name__)


class AttentionDispatcher:
    """Transparent backend selector for VisionQ attention modules."""

    def __init__(self) -> None:
        get_attention_backend("flash")
        self.router = KernelRouter()
        self.last_decision: RoutingDecision | None = None

    def compile_graph(self, qkv_params: dict[str, Any]) -> AttentionGraph:
        dim = qkv_params.get("dim")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("compile_graph requires a positive integer 'dim'")
        graph = AttentionGraph()
        graph.add_node(QKVProjectionNode(id="qkv", dim=dim, outputs=["scores"]))
        graph.add_node(
            MatMulNode(id="scores", inputs=["qkv"], outputs=["softmax"], transpose_b=True)
        )
        graph.add_node(SoftmaxNode(id="softmax", inputs=["scores"], outputs=[]))
        graph = GraphOptimizer(graph).optimize()
        return FusionEngine(graph).fuse()

    def plan(self, context: AttentionContext) -> RoutingDecision:
        decision = self.router.plan_execution(context)
        if decision.backend not in ATTENTION_REGISTRY:
            raise RuntimeError(f"Router selected unregistered backend '{decision.backend}'")
        context.extra_args.update(decision.parameters)
        self.last_decision = decision
        LOGGER.debug("Attention backend decision: %s", decision)
        return decision

    def select(self, context: AttentionContext) -> str:
        return self.plan(context).backend

    def dispatch_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        context: AttentionContext,
    ) -> torch.Tensor:
        backend_name = self.select(context)
        backend_cls = get_attention_backend(backend_name)
        backend = backend_cls(q.shape[-1] * q.shape[1], num_heads=q.shape[1])
        return backend(q, k, v, context, block_size=context.extra_args.get("block_size", 32))
