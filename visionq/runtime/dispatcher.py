from ..attention.registry import ATTENTION_REGISTRY
from ..core.context import AttentionContext
from ..kernels.triton.attention_kernel import TritonAttentionKernel
from .kernel_router import KernelRouter
from typing import Type, Optional, Dict, Any

class AttentionDispatcher:
    """
    Intelligent Dispatcher using a learned KernelRouter for execution planning.
    """
    _selection_cache = {}

    def __init__(self):
        self.triton_kernel = TritonAttentionKernel()
        self.router = KernelRouter()

    def plan(self, context: AttentionContext) -> str:
        """
        Plans the execution using the KernelRouter.
        Returns the best backend name.
        """
        selection, autotuned_params = self.router.plan_execution(context)

        # Merge autotuned parameters into context for backends to consume
        context.extra_args.update(autotuned_params)

        # Ensure selection exists in registry, otherwise fallback
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
