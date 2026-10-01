"""tests/test_domain_plugin.py — Unit tests for DomainPlugin ABC."""

import pytest
from nv_platform.domain_plugin import DomainPlugin


def test_direct_instantiation_raises():
    """DomainPlugin cannot be instantiated directly."""
    with pytest.raises(TypeError):
        DomainPlugin()  # type: ignore[abstract]


def test_concrete_subclass_instantiates():
    """A minimal concrete subclass can be instantiated."""

    class _MinimalDomain(DomainPlugin):
        @property
        def input_schema(self):
            return {
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "number"}},
            }

        @property
        def output_schema(self):
            return {
                "type": "object",
                "required": ["label"],
                "properties": {"label": {"type": "string"}},
            }

        @property
        def knowledge_base_path(self):
            return "/tmp/test_kb"

        @property
        def threshold_hints(self):
            return {"novelty": 0.5, "uncertainty": 0.6}

    domain = _MinimalDomain()
    assert domain.domain_name() == "_MinimalDomain"


def test_validate_event_valid():
    """validate_event returns True for a conforming event."""

    class _D(DomainPlugin):
        @property
        def input_schema(self):
            return {
                "type": "object",
                "required": ["x"],
                "properties": {"x": {"type": "number"}},
            }

        @property
        def output_schema(self):
            return {}

        @property
        def knowledge_base_path(self):
            return "/tmp"

        @property
        def threshold_hints(self):
            return {}

    assert _D().validate_event({"x": 42.0}) is True


def test_validate_event_invalid():
    """validate_event returns False for a non-conforming event."""

    class _D(DomainPlugin):
        @property
        def input_schema(self):
            return {
                "type": "object",
                "required": ["x"],
                "properties": {"x": {"type": "number"}},
            }

        @property
        def output_schema(self):
            return {}

        @property
        def knowledge_base_path(self):
            return "/tmp"

        @property
        def threshold_hints(self):
            return {}

    # Missing required field 'x'
    assert _D().validate_event({"y": 1}) is False
    # Wrong type
    assert _D().validate_event({"x": "not_a_number"}) is False
