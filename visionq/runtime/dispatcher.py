from ..attention.registry import ATTENTION_REGISTRY
from ..core.context import AttentionContext
from ..kernels.triton.attention_kernel import TritonAttentionKernel
from .kernel_router import KernelRouter
from ..compiler.graph_ir.ir import AttentionGraph, QKVProjectionNode, MatMulNode, SoftmaxNode
from ..compiler.optimizer.optimizer import GraphOptimizer
from ..compiler.fusion_engine.fusion import FusionEngine
from typing import Type, Optional, Dict, Any

class AttentionDispatcher:
    """
    Intelligent Dispatcher using a learned KernelRouter for execution planning.
    """
    _selection_cache = {}

    def __init__(self):
        self.triton_kernel = TritonAttentionKernel()
        self.router = KernelRouter()

    def compile_graph(self, qkv_params: Dict[str, Any]) -> AttentionGraph:
        """
        Translates a request into a compiled attention plan.
        """
        graph = AttentionGraph()
        # 1. Capture IR
        graph.add_node(QKVProjectionNode(id="qkv", dim=qkv_params["dim"]))
        graph.add_node(MatMulNode(id="attn_scores", transpose_b=True))
        graph.add_node(SoftmaxNode(id="softmax"))

        # 2. Optimize
        graph = GraphOptimizer(graph).optimize()

        # 3. Fuse
        graph = FusionEngine(graph).fuse()

        return graph

    def plan(self, context: AttentionContext) -> str:
        """
        Plans the execution using the KernelRouter and Compiler insights.
        Returns the best backend name.
        """
        # Integrate Compiler insights here in the future
        graph = self.compile_graph({"dim": 128})

        selection, autotuned_params = self.router.plan_execution(context)
        context.extra_args.update(autotuned_params)

        if selection not in ATTENTION_REGISTRY:
             selection = "flash" if "flash" in ATTENTION_REGISTRY else list(ATTENTION_REGISTRY.keys())[0]

        return selection

    def select(self, context: AttentionContext) -> str:
        """Compatibility layer for backend selection."""
        return self.plan(context)

    def dispatch_kernel(self, q, k, v, context):
        """Low-level kernel dispatch path."""
        if context.device.type == "cuda":
             # Use the industrial block-based kernel
             return self.triton_kernel.forward(q, k, v, context)

        # Fallback to standard SDPA for CPU
        from ..attention.flash import FlashAttention
        fallback = FlashAttention(q.shape[-1])
        return fallback(q, k, v, context)
