from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from ..exceptions import BackendNotAvailableError
from .base import AttentionBackend


class AttentionBackendName(str, Enum):
    FLASH = "flash"
    SPARSE = "sparse"
    SPATIAL_NEIGHBORHOOD = "spatial_neighborhood"
    TEMPORAL_NEIGHBORHOOD = "temporal_neighborhood"
    SPATIOTEMPORAL_HYBRID = "spatiotemporal_hybrid"
    CHUNKED_STREAMING = "chunked_streaming"


ATTENTION_REGISTRY: dict[str, type[AttentionBackend]] = {}


def register_attention(
    name: str | AttentionBackendName, *, override: bool = False
) -> Callable[[type[AttentionBackend]], type[AttentionBackend]]:
    backend_name = name.value if isinstance(name, AttentionBackendName) else str(name)

    def wrapper(cls: type[AttentionBackend]) -> type[AttentionBackend]:
        if not issubclass(cls, AttentionBackend):
            raise TypeError(f"{cls.__name__} must inherit AttentionBackend")
        if backend_name in ATTENTION_REGISTRY and not override:
            existing = ATTENTION_REGISTRY[backend_name].__name__
            raise ValueError(
                f"Attention backend '{backend_name}' is already registered by {existing}"
            )
        ATTENTION_REGISTRY[backend_name] = cls
        return cls

    return wrapper


def ensure_attention_backends_registered() -> None:
    import visionq.attention  # noqa: F401


def available_attention_backends() -> tuple[str, ...]:
    ensure_attention_backends_registered()
    return tuple(sorted(ATTENTION_REGISTRY))


def get_attention_backend(name: str | AttentionBackendName) -> type[AttentionBackend]:
    ensure_attention_backends_registered()
    backend_name = name.value if isinstance(name, AttentionBackendName) else str(name)
    try:
        return ATTENTION_REGISTRY[backend_name]
    except KeyError as exc:
        available = ", ".join(available_attention_backends()) or "<none>"
        raise BackendNotAvailableError(
            f"Attention backend '{backend_name}' is not registered. "
            f"Available backends: {available}"
        ) from exc


__all__ = [
    "ATTENTION_REGISTRY",
    "AttentionBackendName",
    "available_attention_backends",
    "ensure_attention_backends_registered",
    "get_attention_backend",
    "register_attention",
]
