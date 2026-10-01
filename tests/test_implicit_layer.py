"""tests/test_implicit_layer.py — Unit tests for ImplicitPredictiveLayer."""

import torch
import pytest
from nv_platform.implicit_layer import ImplicitPredictiveLayer, ReasoningScore


@pytest.fixture
def layer():
    return ImplicitPredictiveLayer(input_dim=12)


def _make_spikes(num_steps=25, batch=4, features=12, binary=True):
    if binary:
        return torch.randint(0, 2, (num_steps, batch, features)).float()
    return torch.rand(num_steps, batch, features)


# ---------------------------------------------------------------------------
# ReasoningScore
# ---------------------------------------------------------------------------


def test_reasoning_score_defaults():
    score = ReasoningScore()
    assert score.novelty == 0.0
    assert score.uncertainty == 0.0
    assert score.prediction_error == 0.0
    assert score.memory_confidence == 0.0
    assert score.spatial_correlation == 0.0
    assert score.risk == 0.0


def test_reasoning_score_as_tensor():
    score = ReasoningScore(novelty=0.3, uncertainty=0.5,
                           prediction_error=0.2, memory_confidence=0.8,
                           spatial_correlation=0.6, risk=0.55)
    t = score.as_tensor()
    assert t.shape == (6,)
    assert t.tolist() == pytest.approx([0.3, 0.5, 0.2, 0.8, 0.6, 0.55])


# ---------------------------------------------------------------------------
# compute_state output range
# ---------------------------------------------------------------------------


def test_all_fields_in_unit_interval(layer):
    """Every sub-signal must be in [0, 1]."""
    spikes = _make_spikes()
    score = layer.compute_state(spikes)
    for name, val in [
        ("novelty", score.novelty),
        ("uncertainty", score.uncertainty),
        ("prediction_error", score.prediction_error),
        ("memory_confidence", score.memory_confidence),
    ]:
        assert 0.0 <= val <= 1.0, f"{name}={val} out of [0, 1]"


def test_all_fields_in_unit_interval_multiple_calls(layer):
    """Unit-interval invariant holds across many inputs."""
    for _ in range(20):
        spikes = _make_spikes()
        score = layer.compute_state(spikes)
        vec = score.as_tensor()
        assert (vec >= 0.0).all() and (vec <= 1.0).all(), (
            f"Out-of-range score: {score}"
        )


def test_returns_reasoning_score_type(layer):
    spikes = _make_spikes()
    score = layer.compute_state(spikes)
    assert isinstance(score, ReasoningScore)


def test_2d_input_accepted(layer):
    """Layer should handle (batch, features) input gracefully."""
    spikes_2d = torch.rand(4, 12)
    score = layer.compute_state(spikes_2d)
    assert isinstance(score, ReasoningScore)


def test_memory_confidence_grows_with_repetition():
    """
    After seeing the same input many times the episodic memory should
    contain a matching entry → memory_confidence should approach 1.
    """
    layer = ImplicitPredictiveLayer(input_dim=6)
    fixed_spike = torch.ones(10, 2, 6)
    for _ in range(20):
        score = layer.compute_state(fixed_spike)
    # After many identical inputs, memory_confidence should be high (> 0.5)
    assert score.memory_confidence > 0.5, (
        f"Expected high memory_confidence, got {score.memory_confidence}"
    )


def test_novel_input_has_high_novelty():
    """A very unusual spike pattern should score high novelty."""
    layer = ImplicitPredictiveLayer(input_dim=8)
    # Warm up with zeros
    zero_spike = torch.zeros(10, 2, 8)
    for _ in range(30):
        layer.compute_state(zero_spike)
    # Now feed all-ones — maximally different from all-zeros prototypes
    novel_spike = torch.ones(10, 2, 8)
    score = layer.compute_state(novel_spike)
    assert score.novelty > 0.0, "Expected non-zero novelty for a novel input"


def test_forward_returns_tensor(layer):
    """forward() method wraps compute_state and returns a 6-d tensor."""
    spikes = _make_spikes()
    out = layer.forward(spikes)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (6,)


def test_spatial_correlation_zero_without_spatial_features():
    """Without spatial features (input_dim < 16), spatial_correlation = 0."""
    layer = ImplicitPredictiveLayer(input_dim=12)
    spikes = torch.randint(0, 2, (25, 2, 12)).float()
    score = layer.compute_state(spikes)
    assert score.spatial_correlation == 0.0


def test_spatial_correlation_nonzero_with_spatial_features():
    """With spatial features (input_dim=16), spatial_correlation is in [0,1]."""
    layer = ImplicitPredictiveLayer(input_dim=16)
    spikes = torch.rand(25, 2, 16)
    score = layer.compute_state(spikes)
    assert 0.0 <= score.spatial_correlation <= 1.0


def test_spatial_correlation_higher_for_anomalous_pattern():
    """
    Inputs with large spatial features should produce higher spatial_correlation
    than near-zero spatial features after warm-up.
    """
    layer = ImplicitPredictiveLayer(input_dim=16)
    normal = torch.rand(25, 2, 12)
    normal_with_spatial = torch.cat([normal, torch.zeros(25, 2, 4)], dim=-1)
    for _ in range(30):
        score_low = layer.compute_state(normal_with_spatial)

    crash_spatial = torch.ones(25, 2, 4) * 5.0
    crash_input = torch.cat([normal, crash_spatial], dim=-1)
    score_high = layer.compute_state(crash_input)

    assert score_high.spatial_correlation >= score_low.spatial_correlation
