"""tests/test_action_executor.py — mock execution + outcome verification."""

from __future__ import annotations

import torch

from nv_platform.implicit_layer import ReasoningScore
from nv_platform.planner import ActionPlan
from nv_platform.action_memory import DecisionRecord
from nv_platform.action_executor import (
    MockActionExecutor,
    LabelOutcomeVerifier,
    SimulatedOperatorFeedback,
    OutcomeVerification,
    ExecutionResult,
)


# ── MockActionExecutor ─────────────────────────────────────────────────────

def test_mock_executor_runs_all_steps():
    ex = MockActionExecutor()
    plan = ActionPlan(label="crash", steps=["alert", "log", "notify"])
    res = ex.execute(plan, "d1")
    assert isinstance(res, ExecutionResult)
    assert res.status == "executed"
    assert res.executed_steps == ["alert", "log", "notify"]
    assert res.decision_id == "d1"
    assert res.timestamp != ""


def test_mock_executor_injected_failure_is_deterministic():
    ex = MockActionExecutor(seed=1, fail_probability=1.0)  # always fail
    plan = ActionPlan(label="crash", steps=["alert"])
    res = ex.execute(plan, "d1")
    assert res.status == "failed"
    # Same seed → same outcome
    ex2 = MockActionExecutor(seed=1, fail_probability=1.0)
    assert ex2.execute(plan, "d1").status == "failed"


def test_mock_executor_zero_failure_never_fails():
    ex = MockActionExecutor(fail_probability=0.0)
    plan = ActionPlan(steps=["x"])
    for _ in range(50):
        assert ex.execute(plan, "d").status == "executed"


# ── LabelOutcomeVerifier ───────────────────────────────────────────────────

def _record(predicted_label, risk=0.5, score=None):
    s = score or ReasoningScore(risk=risk)
    return DecisionRecord(
        decision_id="d1",
        score=s,
        score_tensor=s.as_tensor(),
        predicted_label=predicted_label,
    )


def test_label_verifier_crash_label():
    v = LabelOutcomeVerifier(label_key="crash_record", crash_value=1)
    rec = _record("crash")
    out = v.verify("d1", rec, context={"crash_record": 1})
    assert isinstance(out, OutcomeVerification)
    assert out.actual_label == 1
    assert out.source == "ground_truth"
    # predicted crash, actual crash → success
    assert out.verified_success is True


def test_label_verifier_normal_label():
    v = LabelOutcomeVerifier(label_key="crash_record", crash_value=1)
    rec = _record("normal")
    out = v.verify("d1", rec, context={"crash_record": 0})
    assert out.actual_label == 0
    # predicted normal, actual normal → success
    assert out.verified_success is True


def test_label_verifier_false_negative_detected():
    """Predicted normal but actual crash → not effective (FN)."""
    v = LabelOutcomeVerifier(label_key="label", crash_value=1)
    rec = _record("normal")
    out = v.verify("d1", rec, context={"label": 1})
    assert out.actual_label == 1
    assert out.verified_success is False


def test_label_verifier_false_positive_detected():
    """Predicted crash but actual normal → not effective (FP)."""
    v = LabelOutcomeVerifier(label_key="label", crash_value=1)
    rec = _record("crash")
    out = v.verify("d1", rec, context={"label": 0})
    assert out.actual_label == 0
    assert out.verified_success is False


def test_label_verifier_uses_record_ground_truth_fallback():
    """When the context lacks the label, fall back to one attached to the record."""
    v = LabelOutcomeVerifier(label_key="missing")
    rec = _record("crash")
    rec.action_plan = {"_ground_truth_label": 1}
    out = v.verify("d1", rec, context={})
    assert out.actual_label == 1


def test_label_verifier_never_uses_prediction_as_truth():
    """Even if the record only has a predicted label, actual comes from context."""
    v = LabelOutcomeVerifier(label_key="label", crash_value=1)
    rec = _record("crash")
    out = v.verify("d1", rec, context={"label": 0})
    # actual is from context (0), NOT the predicted "crash"
    assert out.actual_label == 0


# ── SimulatedOperatorFeedback ──────────────────────────────────────────────

def test_simulated_operator_with_provider():
    """Provider supplies the independent ground truth."""
    v = SimulatedOperatorFeedback(ground_truth_provider=lambda ctx: ctx.get("truth", 0))
    rec = _record("crash")
    out = v.verify("d1", rec, context={"truth": 1})
    assert out.actual_label == 1
    assert out.source == "simulated_operator"
    assert out.verified_success is True


def test_simulated_operator_providerless_uses_risk():
    """Without a provider, the operator judges from risk magnitude (independent
    of the model's predicted_label)."""
    v = SimulatedOperatorFeedback(risk_threshold=0.6)
    rec_high = _record("normal", risk=0.9)   # high risk → operator says crash
    out = v.verify("d1", rec_high)
    assert out.actual_label == 1
    # predicted normal vs actual crash → FN, not effective
    assert out.verified_success is False

    rec_low = _record("normal", risk=0.1)
    out2 = v.verify("d2", rec_low)
    assert out2.actual_label == 0
    assert out2.verified_success is True


def test_simulated_operator_review_noise_flips_success_only():
    """Noise flips the success verdict, not the actual_label."""
    v = SimulatedOperatorFeedback(
        ground_truth_provider=lambda ctx: 1, review_noise=1.0  # always flip
    )
    rec = _record("crash")  # would be success=True with no noise
    out = v.verify("d1", rec, context={})
    assert out.actual_label == 1            # ground truth unchanged
    assert out.verified_success is False    # flipped by noisy review


def test_simulated_operator_never_uses_prediction_as_truth():
    """Provider-less mode derives actual from risk, never from predicted_label."""
    v = SimulatedOperatorFeedback(risk_threshold=0.6)
    # predicted crash but low risk → operator (from risk) says normal
    rec = _record("crash", risk=0.1)
    out = v.verify("d1", rec)
    assert out.actual_label == 0


# ── ExecutionResult / OutcomeVerification round-trip ────────────────────────

def test_execution_result_to_dict():
    r = ExecutionResult(decision_id="d", status="executed",
                        executed_steps=["a"], timestamp="t")
    d = r.to_dict()
    assert d["status"] == "executed" and d["executed_steps"] == ["a"]


def test_outcome_verification_to_dict():
    v = OutcomeVerification(decision_id="d", actual_label=1,
                            verified_success=True, source="ground_truth")
    d = v.to_dict()
    assert d["actual_label"] == 1 and d["verified_success"] is True
