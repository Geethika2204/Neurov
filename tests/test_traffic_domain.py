"""tests/test_traffic_domain.py — Unit tests for TrafficDomain."""

import pytest
import torch
from domains.traffic.domain import TrafficDomain, FEATURE_COLS


@pytest.fixture
def domain():
    return TrafficDomain()


def _valid_event():
    event = {"timestamp": 1727000000.0}
    for col in FEATURE_COLS:
        event[col] = 55.0
    return event


def test_domain_instantiates(domain):
    assert domain is not None


def test_input_schema_has_required_fields(domain):
    schema = domain.input_schema
    for col in FEATURE_COLS:
        assert col in schema["properties"], f"{col} missing from input_schema"
    assert "timestamp" in schema["properties"]


def test_output_schema_has_required_fields(domain):
    schema = domain.output_schema
    assert "label" in schema["properties"]
    assert "confidence" in schema["properties"]


def test_threshold_hints_keys(domain):
    hints = domain.threshold_hints
    for key in ("novelty", "uncertainty", "prediction_error", "memory_confidence"):
        assert key in hints, f"{key} missing from threshold_hints"
        assert 0.0 <= hints[key] <= 1.0


def test_validate_event_valid(domain):
    assert domain.validate_event(_valid_event()) is True


def test_validate_event_missing_feature(domain):
    event = _valid_event()
    del event["lane1_speed"]
    assert domain.validate_event(event) is False


def test_validate_event_wrong_type(domain):
    event = _valid_event()
    event["lane1_speed"] = "fast"
    assert domain.validate_event(event) is False


def test_event_from_row():
    row = {
        "unix_time": 1727000000.0,
        "crash_record": 1,
        "milemarker": 10.5,
        "day": 3,
    }
    for col in FEATURE_COLS:
        row[col] = 60.0

    event = TrafficDomain.event_from_row(row)
    assert event["timestamp"] == 1727000000.0
    assert event["crash_record"] == 1
    for col in FEATURE_COLS:
        assert col in event


def test_event_to_tensor():
    event = _valid_event()
    tensor = TrafficDomain.event_to_tensor(event)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 12)
    assert tensor.dtype == torch.float32


def test_knowledge_base_path_is_string(domain):
    path = domain.knowledge_base_path
    assert isinstance(path, str)
    assert len(path) > 0


def test_domain_name(domain):
    assert domain.domain_name() == "TrafficDomain"
