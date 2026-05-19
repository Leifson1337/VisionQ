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
        Returns the name of the backend to use.
        """
        # Simple cache key
        cache_key = (context.modality, context.window_size > 0, context.device.type if context.device else None)
        if cache_key in self._selection_cache:
            return self._selection_cache[cache_key]

        if context.window_size > 0:
            selection = "neighborhood"
        elif context.modality == "video":
            selection = "flash"
        else:
            selection = "flash"

        # Ensure selection exists in registry
        if selection not in ATTENTION_REGISTRY:
            selection = list(ATTENTION_REGISTRY.keys())[0]

        self._selection_cache[cache_key] = selection
        return selection
