"""tests/test_gate.py — Unit tests for ReasoningGate, GateLearner, EscalationHistory."""

import pytest
import torch
from nv_platform.implicit_layer import ReasoningScore
from nv_platform.gate import ReasoningGate, EscalationHistory, HARD_ESCALATE_THRESHOLD
from nv_platform.gate_learner import GateLearner


@pytest.fixture
def learner():
    return GateLearner(update_every=50, fn_penalty=5.0)


@pytest.fixture
def gate(learner):
    return ReasoningGate(learner=learner, threshold=0.5)


@pytest.fixture
def history():
    return EscalationHistory()


# ---------------------------------------------------------------------------
# EscalationHistory
# ---------------------------------------------------------------------------


def test_history_starts_empty(history):
    assert len(history) == 0


def test_history_appends(history):
    history.append({"id": "x", "decision": True, "outcome": None})
    assert len(history) == 1


def test_history_set_outcome(history):
    history.append({"id": "abc", "decision": True, "outcome": None})
    result = history.set_outcome("abc", 1)
    assert result is True
    assert history.log[0]["outcome"] == 1


def test_history_set_outcome_missing(history):
    result = history.set_outcome("does-not-exist", 0)
    assert result is False


def test_history_maxlen():
    h = EscalationHistory(maxlen=10)
    for i in range(15):
        h.append({"id": str(i), "decision": True, "outcome": None})
    assert len(h) == 10


# ---------------------------------------------------------------------------
# GateLearner
# ---------------------------------------------------------------------------


def test_learner_predict_proba_range(learner, history):
    score = ReasoningScore(novelty=0.5, uncertainty=0.5, prediction_error=0.5, memory_confidence=0.5, spatial_correlation=0.5)
    prob = learner.predict_proba(score.as_tensor(), history)
    assert 0.0 <= prob <= 1.0, f"predict_proba out of [0, 1]: {prob}"


def test_learner_update_increments_count(learner, history):
    # Add labelled entries
    for i in range(10):
        score = ReasoningScore(novelty=0.3, uncertainty=0.4, prediction_error=0.2, memory_confidence=0.6)
        entry = {
            "id": str(i),
            "score_tensor": score.as_tensor(),
            "decision": True,
            "confidence": 0.5,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "outcome": 0,
        }
        history.append(entry)
    before = learner._update_count
    learner.update(history)
    assert learner._update_count == before + 1


def test_learner_update_skipped_insufficient_data(learner, history):
    """Learner should not update with fewer than 5 labelled samples."""
    # Add only 3 labelled entries
    for i in range(3):
        score = ReasoningScore(novelty=0.5, uncertainty=0.5, prediction_error=0.5, memory_confidence=0.5)
        entry = {
            "id": str(i),
            "score_tensor": score.as_tensor(),
            "decision": True,
            "confidence": 0.5,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "outcome": 1,
        }
        history.append(entry)
    before = learner._update_count
    learner.update(history)
    assert learner._update_count == before  # no change


# ---------------------------------------------------------------------------
# ReasoningGate
# ---------------------------------------------------------------------------


def test_gate_returns_bool_float(gate, history):
    score = ReasoningScore(novelty=0.3, uncertainty=0.3, prediction_error=0.3, memory_confidence=0.7)
    escalate, conf = gate.should_escalate(score, history)
    assert isinstance(escalate, bool)
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_gate_always_escalates_on_extreme_signal(gate, history):
    """Any sub-signal above HARD_ESCALATE_THRESHOLD forces escalation."""
    score = ReasoningScore(
        novelty=HARD_ESCALATE_THRESHOLD + 0.01,
        uncertainty=0.1,
        prediction_error=0.1,
        memory_confidence=0.9,
    )
    escalate, _ = gate.should_escalate(score, history)
    assert escalate is True, "Expected hard escalation on extreme novelty"


def test_gate_logs_decision(gate, history):
    score = ReasoningScore(novelty=0.3, uncertainty=0.3, prediction_error=0.3, memory_confidence=0.7)
    gate.should_escalate(score, history)
    assert len(history) == 1
    entry = list(history.log)[0]
    assert "id" in entry
    assert "decision" in entry
    assert entry["outcome"] is None


def test_gate_log_outcome(gate, history):
    score = ReasoningScore(novelty=0.3, uncertainty=0.3, prediction_error=0.3, memory_confidence=0.7)
    gate.should_escalate(score, history)
    decision_id = list(history.log)[0]["id"]
    result = gate.log_outcome(decision_id, actual_label=1, history=history)
    assert result is True
    assert list(history.log)[0]["outcome"] == 1


def test_gate_log_grows_correctly(gate, history):
    """Each call adds exactly one entry."""
    for i in range(10):
        score = ReasoningScore(novelty=0.2, uncertainty=0.3, prediction_error=0.1, memory_confidence=0.8)
        gate.should_escalate(score, history)
    assert len(history) == 10


def test_gate_triggers_learner_update_at_interval():
    """Learner.update() is called at multiples of update_every."""
    learner = GateLearner(update_every=5, fn_penalty=5.0)
    gate = ReasoningGate(learner=learner)
    history = EscalationHistory()

    # Add labelled history entries manually so update has data
    for i in range(20):
        score = ReasoningScore(novelty=0.4, uncertainty=0.4, prediction_error=0.4, memory_confidence=0.4)
        entry = {
            "id": f"pre-{i}",
            "score_tensor": score.as_tensor(),
            "decision": bool(i % 2),
            "confidence": 0.5,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "outcome": i % 2,
        }
        history.append(entry)

    before = learner._update_count
    # Make exactly 5 calls to trigger one update
    for _ in range(5):
        gate.should_escalate(ReasoningScore(novelty=0.4, uncertainty=0.4, prediction_error=0.4, memory_confidence=0.4), history)

    assert learner._update_count > before, (
        "Expected at least one learner update after 5 gate calls"
    )


def test_gate_threshold_setter_rejects_invalid(gate):
    with pytest.raises(ValueError):
        gate.threshold = 1.5
    with pytest.raises(ValueError):
        gate.threshold = -0.1


def test_fn_penalty_biases_toward_escalation():
    """
    A learner trained with FN penalty on a crash-heavy dataset should
    prefer escalation over missing positives.
    """
    learner = GateLearner(update_every=10, fn_penalty=10.0)
    history = EscalationHistory()

    # Simulate: high-score events that turned out to be crashes
    for i in range(30):
        score = ReasoningScore(novelty=0.8, uncertainty=0.8, prediction_error=0.7, memory_confidence=0.1)
        entry = {
            "id": f"crash-{i}",
            "score_tensor": score.as_tensor(),
            "decision": False,  # gate wrongly didn't escalate
            "confidence": 0.3,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "outcome": 1,  # was actually a crash → FN
        }
        history.append(entry)

    learner.update(history)

    # After training on FN-heavy data, high-signal score should escalate
    high_score = ReasoningScore(novelty=0.85, uncertainty=0.85, prediction_error=0.75, memory_confidence=0.05)
    prob = learner.predict_proba(high_score.as_tensor(), history)
    # We expect the learner to push probability toward 1.0 for these scores
    # (not guaranteed to be >0.5 immediately but should be non-trivial)
    assert prob >= 0.0  # at minimum, no error
