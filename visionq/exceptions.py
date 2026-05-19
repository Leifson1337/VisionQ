class VisionQError(Exception):
    """Base exception for VisionQ errors."""


class BackendNotAvailableError(RuntimeError, VisionQError):
    """Raised when a requested backend is not registered or unavailable."""


class ShapeError(ValueError, VisionQError):
    """Raised when tensor shapes violate an API contract."""


class UnsupportedFeatureError(ValueError, VisionQError):
    """Raised when a backend explicitly does not support a requested feature."""


__all__ = [
    "BackendNotAvailableError",
    "ShapeError",
    "UnsupportedFeatureError",
    "VisionQError",
]
