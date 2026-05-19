import pytest

from visionq.attention import AttentionBackendName, available_attention_backends
from visionq.attention.base import AttentionBackend
from visionq.attention.registry import get_attention_backend, register_attention
from visionq.core import AttentionContext
from visionq.runtime import RoutingDecision
from visionq.runtime.dispatcher import AttentionDispatcher


def test_expected_backends_are_registered():
    assert set(available_attention_backends()) == {name.value for name in AttentionBackendName}


def test_unknown_backend_raises_clear_error():
    with pytest.raises(RuntimeError, match="not registered"):
        get_attention_backend("missing")


def test_duplicate_registration_is_rejected():
    cls = get_attention_backend("flash")
    with pytest.raises(ValueError, match="already registered"):
        register_attention("flash")(cls)


def test_router_uses_registered_names():
    dispatcher = AttentionDispatcher()
    assert dispatcher.select(AttentionContext(modality="sequence", sequence_length=512)) == "flash"
    assert (
        dispatcher.select(
            AttentionContext(modality="image", sequence_length=1024, spatial_shape=(32, 32))
        )
        == "spatial_neighborhood"
    )
    assert (
        dispatcher.select(
            AttentionContext(
                modality="video", sequence_length=128, spatial_shape=(8, 8), temporal_dim=2
            )
        )
        == "spatiotemporal_hybrid"
    )
    assert isinstance(dispatcher.last_decision, RoutingDecision)
    decision = dispatcher.last_decision.to_dict()
    assert decision["policy_version"] == "heuristic-v1"
    assert decision["backend"] == "spatiotemporal_hybrid"


def test_registered_classes_inherit_base():
    for name in available_attention_backends():
        assert issubclass(get_attention_backend(name), AttentionBackend)
