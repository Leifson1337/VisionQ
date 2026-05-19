from typing import Dict, Type

ATTENTION_REGISTRY: Dict[str, Type] = {}

def register_attention(name: str):
    """
    Decorator to register a new attention backend.
    """
    def wrapper(cls):
        if name in ATTENTION_REGISTRY:
            raise ValueError(f"Attention backend '{name}' is already registered.")
        ATTENTION_REGISTRY[name] = cls
        return cls
    return wrapper

def get_attention_backend(name: str):
    """
    Retrieves an attention backend class from the registry.
    """
    if name not in ATTENTION_REGISTRY:
        raise KeyError(f"Attention backend '{name}' not found in registry. "
                        f"Available: {list(ATTENTION_REGISTRY.keys())}")
    return ATTENTION_REGISTRY[name]
