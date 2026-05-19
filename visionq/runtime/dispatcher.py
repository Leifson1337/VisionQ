from ..attention.registry import ATTENTION_REGISTRY
from ..core.context import AttentionContext
from ..kernels.triton.attention_kernel import TritonAttentionKernel
from typing import Type, Optional

class AttentionDispatcher:
    """
    Dispatcher for backend selection.
    Decision rules based on context (modality, window sizes, etc.)
    """
    _selection_cache = {}

    def __init__(self):
        self.triton_kernel = TritonAttentionKernel()

    def select(self, context: AttentionContext) -> str:
        cache_key = (
            context.modality,
            context.sequence_length,
            context.spatial_window,
            context.temporal_window,
            context.dilation,
            context.attention_mode,
            context.device.type if context.device else None
        )
        if cache_key in self._selection_cache:
            return self._selection_cache[cache_key]

        # Advanced Kernel Dispatch Logic
        if context.sequence_length < 1024:
            # Small sequences fit in cache, use fused IO-aware kernel
            selection = "flash"
        elif context.modality == "video" and context.temporal_dim and context.temporal_dim > 16:
            # Long videos use block sparse temporal to avoid T^2 complexity
            selection = "sparse"
        elif context.modality == "image":
            # Primary mode for image is neighborhood (local window)
            selection = "neighborhood"
        elif context.modality == "video":
            if context.attention_mode == "spatio_temporal":
                selection = "spatiotemporal_hybrid"
            else:
                selection = "neighborhood"
        elif context.sequence_length > 4096:
            # Massive sequences use streaming chunked execution
            selection = "chunked_streaming"
        else:
            selection = "flash"

        # Fallback logic
        if selection not in ATTENTION_REGISTRY:
            if "neighborhood" in ATTENTION_REGISTRY: selection = "neighborhood"
            elif "flash" in ATTENTION_REGISTRY: selection = "flash"
            else: selection = list(ATTENTION_REGISTRY.keys())[0]

        self._selection_cache[cache_key] = selection
        return selection

    def dispatch_kernel(self, q, k, v, context):
        """Low-level kernel dispatch path."""
        if context.device.type == "cuda":
             # Use the industrial block-based kernel
             return self.triton_kernel.forward(q, k, v, context)

        # Fallback to standard SDPA for CPU
        from ..attention.flash import FlashAttention
        fallback = FlashAttention(q.shape[-1])
        return fallback(q, k, v, context)
