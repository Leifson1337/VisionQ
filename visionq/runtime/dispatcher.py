from ..attention.registry import ATTENTION_REGISTRY
from ..core.context import AttentionContext
from typing import Type

class AttentionDispatcher:
    """
    The intelligence layer of VisionQ.
    Determines the most efficient attention backend for a given execution context.
    """

    # Cache for selections to avoid repeated logic for identical contexts
    _selection_cache = {}

    def select(self, context: AttentionContext) -> str:
        """
        Selection logic for attention backends.
        Returns the name of the backend to use based on industrial routing rules.
        """
        # Simple cache key
        cache_key = (
            context.modality,
            context.window_size,
            context.dilation,
            context.device.type if context.device else None
        )
        if cache_key in self._selection_cache:
            return self._selection_cache[cache_key]

        # Industrial Routing Rules:
        # 1. image -> neighborhood default
        # 2. video -> neighborhood (spatio-temporal) default
        # 3. long sequence / specific dilation -> sparse
        # 4. global / default -> flash

        if context.modality == "image":
            selection = "neighborhood"
        elif context.modality == "video":
            selection = "neighborhood"
        elif context.dilation > 1 or context.modality == "sequence":
            selection = "sparse"
        else:
            selection = "flash"

        # Ensure selection exists in registry, fallback to flash then any
        if selection not in ATTENTION_REGISTRY:
            selection = "flash" if "flash" in ATTENTION_REGISTRY else list(ATTENTION_REGISTRY.keys())[0]

        self._selection_cache[cache_key] = selection
        return selection
