"""tests/test_baselines.py — Unit tests for gate baselines."""

import pytest
from nv_platform.implicit_layer import ReasoningScore
from baselines.always_threshold_gate import AlwaysThresholdGate
from baselines.always_escalate_gate import AlwaysEscalateGate


# ---------------------------------------------------------------------------
# AlwaysEscalateGate
# ---------------------------------------------------------------------------


def test_always_escalate_returns_true():
    gate = AlwaysEscalateGate()
    for novelty in [0.0, 0.3, 0.99]:
        score = ReasoningScore(novelty=novelty, uncertainty=0.0,
                               prediction_error=0.0, memory_confidence=1.0)
        escalate, conf = gate.should_escalate(score)
        assert escalate is True
        assert conf == 1.0


def test_always_escalate_evaluate():
    gate = AlwaysEscalateGate()
    scores = [ReasoningScore() for _ in range(100)]
    labels = [0] * 90 + [1] * 10
    metrics = gate.evaluate(scores, labels)
    assert metrics["escalation_rate"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AlwaysThresholdGate
# ---------------------------------------------------------------------------


def test_threshold_gate_escalates_above_threshold():
    gate = AlwaysThresholdGate(threshold=0.5)
    high_score = ReasoningScore(novelty=0.9, uncertainty=0.1,
                                prediction_error=0.1, memory_confidence=0.9)
    escalate, conf = gate.should_escalate(high_score)
    assert escalate is True
    assert conf >= 0.5


def test_threshold_gate_no_escalate_below_threshold():
    gate = AlwaysThresholdGate(threshold=0.8)
    low_score = ReasoningScore(novelty=0.1, uncertainty=0.1,
                               prediction_error=0.1, memory_confidence=0.9)
    escalate, conf = gate.should_escalate(low_score)
    assert escalate is False


def test_threshold_gate_low_memory_confidence_escalates():
    """Low memory_confidence should count as high signal → escalate."""
    gate = AlwaysThresholdGate(threshold=0.5)
    score = ReasoningScore(novelty=0.1, uncertainty=0.1,
                           prediction_error=0.1, memory_confidence=0.05)
    # 1 - 0.05 = 0.95 signal → should escalate
    escalate, _ = gate.should_escalate(score)
    assert escalate is True


def test_threshold_gate_invalid_threshold():
    with pytest.raises(ValueError):
        AlwaysThresholdGate(threshold=1.5)
    with pytest.raises(ValueError):
        AlwaysThresholdGate(threshold=-0.1)


def test_threshold_gate_evaluate():
    gate = AlwaysThresholdGate(threshold=0.5)
    # Scores above threshold
    high = ReasoningScore(novelty=0.9, uncertainty=0.9,
                          prediction_error=0.9, memory_confidence=0.1)
    # Scores below threshold
    low = ReasoningScore(novelty=0.1, uncertainty=0.1,
                         prediction_error=0.1, memory_confidence=0.9)

    scores = [high] * 10 + [low] * 90
    labels = [1] * 10 + [0] * 90

    metrics = gate.evaluate(scores, labels)
    assert "escalation_rate" in metrics
    assert "recall" in metrics
    assert "precision" in metrics
    assert "f1" in metrics
    assert 0.0 <= metrics["escalation_rate"] <= 1.0


def test_always_escalate_vs_threshold_reduction():
    """
    AlwaysThresholdGate (threshold=0.5) should escalate fewer events than
    AlwaysEscalateGate on a mostly-normal dataset.
    """
    import random
    random.seed(42)

    always = AlwaysEscalateGate()
    threshold = AlwaysThresholdGate(threshold=0.5)

    # 95% normal events (low scores), 5% anomalies (high scores)
    scores, labels = [], []
    for i in range(200):
        if i < 10:  # anomalies
            scores.append(ReasoningScore(0.8, 0.8, 0.7, 0.1))
            labels.append(1)
        else:  # normal
            scores.append(ReasoningScore(0.1, 0.15, 0.1, 0.9))
            labels.append(0)

    always_metrics = always.evaluate(scores, labels)
    threshold_metrics = threshold.evaluate(scores, labels)

    # Threshold gate should escalate fewer events
    assert threshold_metrics["escalation_rate"] < always_metrics["escalation_rate"], (
        "ThresholdGate should escalate less than AlwaysEscalateGate"
    )
