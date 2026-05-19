from ..attention.registry import ATTENTION_REGISTRY
from ..core.context import AttentionContext
from typing import Type

class AttentionDispatcher:
    """
    Dispatcher for backend selection.
    Decision rules based on context (modality, window sizes, etc.)
    """
    _selection_cache = {}

    def select(self, context: AttentionContext) -> str:
        cache_key = (
            context.modality,
            context.spatial_window,
            context.temporal_window,
            context.dilation,
            context.attention_mode,
            context.device.type if context.device else None
        )
        if cache_key in self._selection_cache:
            return self._selection_cache[cache_key]

        if context.modality == "image":
            selection = "neighborhood"
        elif context.modality == "video":
            if context.attention_mode == "spatio_temporal":
                selection = "spatiotemporal_hybrid"
            elif context.attention_mode == "temporal_only":
                selection = "temporal_neighborhood"
            elif context.attention_mode == "spatial_only":
                selection = "spatial_neighborhood"
            else:
                selection = "neighborhood" # Default 3D neighborhood
        elif context.dilation > 1 or context.modality == "sequence":
            selection = "sparse"
        else:
            selection = "flash"

        # Fallback logic
        if selection not in ATTENTION_REGISTRY:
            if "neighborhood" in ATTENTION_REGISTRY: selection = "neighborhood"
            elif "flash" in ATTENTION_REGISTRY: selection = "flash"
            else: selection = list(ATTENTION_REGISTRY.keys())[0]

        self._selection_cache[cache_key] = selection
        return selection
