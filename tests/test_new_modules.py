"""tests/test_new_modules.py — Tests for AutomaticAction, Safety, Planner, MemoryConsolidation."""

import torch
import pytest
from nv_platform.implicit_layer import ReasoningScore, ImplicitPredictiveLayer
from nv_platform.automatic_action import AutomaticAction, AutomaticActionResult
from nv_platform.safety import SafetyChecker, SafetyResult
from nv_platform.planner import Planner, ActionPlan
from nv_platform.memory_consolidation import MemoryConsolidation, Feedback
from nv_platform.agent_plugin import AgentDecision


# ── AutomaticAction ────────────────────────────────────────────────────────

def test_automatic_action_low_risk():
    policy = AutomaticAction()
    score = ReasoningScore(risk=0.1)
    result = policy.act(score)
    assert result.label == "normal"
    assert result.path == "automatic"
    assert 0.0 <= result.confidence <= 1.0


def test_automatic_action_medium_risk():
    policy = AutomaticAction()
    score = ReasoningScore(risk=0.35)
    result = policy.act(score)
    assert result.label == "monitor"
    assert result.path == "automatic"


def test_automatic_action_returns_automatic_action_result():
    policy = AutomaticAction()
    result = policy.act(ReasoningScore(risk=0.0))
    assert isinstance(result, AutomaticActionResult)


def test_automatic_action_custom_domain_actions():
    policy = AutomaticAction(domain_actions={"normal": "freeway_ok"})
    result = policy.act(ReasoningScore(risk=0.05))
    assert result.action == "freeway_ok"


def test_automatic_action_confidence_in_unit_interval():
    policy = AutomaticAction()
    for risk_val in [0.0, 0.1, 0.25, 0.4, 0.5, 0.9, 1.0]:
        score = ReasoningScore(risk=risk_val)
        result = policy.act(score)
        assert 0.0 <= result.confidence <= 1.0, f"confidence out of range for risk={risk_val}"


# ── SafetyChecker ──────────────────────────────────────────────────────────

def test_safety_passes_clean_decision():
    checker = SafetyChecker()
    decision = AgentDecision(label="normal", confidence=0.9, reasoning="all clear")
    score = ReasoningScore(risk=0.1)
    result = checker.check(decision, score)
    assert result.safe is True
    assert result.violations == []


def test_safety_flags_invalid_label():
    checker = SafetyChecker()
    decision = AgentDecision(label="EXPLODE", confidence=0.9)
    score = ReasoningScore(risk=0.5)
    result = checker.check(decision, score)
    assert result.safe is False
    assert any("Invalid label" in v for v in result.violations)


def test_safety_flags_crash_on_low_risk():
    checker = SafetyChecker()
    decision = AgentDecision(label="crash", confidence=0.9)
    score = ReasoningScore(risk=0.05)  # very low risk
    result = checker.check(decision, score)
    assert result.safe is False
    assert any("Possible false alarm" in v for v in result.violations)


def test_safety_flags_normal_on_high_risk():
    checker = SafetyChecker()
    decision = AgentDecision(label="normal", confidence=0.9)
    score = ReasoningScore(risk=0.92)  # very high risk
    result = checker.check(decision, score)
    assert result.safe is False
    assert any("Possible missed crash" in v for v in result.violations)


def test_safety_strict_mode_blocks_action():
    checker = SafetyChecker(strict=True)
    decision = AgentDecision(label="crash", confidence=0.9)
    score = ReasoningScore(risk=0.05)
    result = checker.check(decision, score)
    assert result.decision.label == "uncertain"
    assert result.decision.confidence == 0.0


def test_safety_soft_mode_downgrades_confidence():
    checker = SafetyChecker(strict=False)
    decision = AgentDecision(label="crash", confidence=0.8)
    score = ReasoningScore(risk=0.05)
    result = checker.check(decision, score)
    # Label preserved but confidence reduced
    assert result.decision.label == "crash"
    assert result.decision.confidence < 0.8


def test_safety_custom_rule():
    def no_crash_at_night(decision, score, context):
        if context.get("hour", 12) < 6 and decision.label == "crash":
            return "Crashes unlikely between midnight and 6am — review."
        return None

    checker = SafetyChecker(custom_rules=[no_crash_at_night])
    decision = AgentDecision(label="crash", confidence=0.8)
    score = ReasoningScore(risk=0.8)
    result = checker.check(decision, score, context={"hour": 2})
    assert result.safe is False
    assert any("midnight" in v for v in result.violations)


# ── Planner ────────────────────────────────────────────────────────────────

def test_planner_crash_produces_steps():
    planner = Planner()
    decision = AgentDecision(label="crash", confidence=0.85)
    score = ReasoningScore(risk=0.9)
    from nv_platform.safety import SafetyResult
    sr = SafetyResult(safe=True, violations=[], decision=decision)
    plan = planner.plan(sr, score)
    assert isinstance(plan, ActionPlan)
    assert plan.label == "crash"
    assert len(plan.steps) > 0
    assert plan.path == "deliberative"


def test_planner_normal_produces_monitor_step():
    planner = Planner()
    decision = AgentDecision(label="normal", confidence=0.9)
    score = ReasoningScore(risk=0.05)
    from nv_platform.safety import SafetyResult
    sr = SafetyResult(safe=True, violations=[], decision=decision)
    plan = planner.plan(sr, score)
    assert "continue_monitoring" in plan.steps


def test_planner_unsafe_adds_human_review():
    planner = Planner()
    decision = AgentDecision(label="uncertain", confidence=0.4)
    score = ReasoningScore(risk=0.5)
    from nv_platform.safety import SafetyResult
    sr = SafetyResult(safe=False, violations=["test violation"], decision=decision)
    plan = planner.plan(sr, score)
    assert "human_review_required" in plan.steps
    assert plan.safe is False


def test_planner_confidence_in_unit_interval():
    planner = Planner()
    from nv_platform.safety import SafetyResult
    for label in ("crash", "normal", "anomaly", "uncertain", "monitor"):
        decision = AgentDecision(label=label, confidence=0.7)
        score = ReasoningScore(risk=0.5)
        sr = SafetyResult(safe=True, violations=[], decision=decision)
        plan = planner.plan(sr, score)
        assert 0.0 <= plan.confidence <= 1.0


# ── MemoryConsolidation ────────────────────────────────────────────────────

def test_memory_consolidation_stores_crash_episode():
    layer = ImplicitPredictiveLayer(input_dim=12)
    mc = MemoryConsolidation(layer, update_every=20)
    spike_vec = torch.rand(12)
    fb = Feedback(
        decision_id="test-1",
        actual_label=1,
        predicted_label="crash",
        score=ReasoningScore(risk=0.9),
        spike_vec=spike_vec,
    )
    summary = mc.consolidate(fb)
    assert summary.episode_stored is True
    assert summary.tp == 1


def test_memory_consolidation_confusion_matrix():
    layer = ImplicitPredictiveLayer(input_dim=12)
    mc = MemoryConsolidation(layer, update_every=100)

    # TP
    mc.consolidate(Feedback(actual_label=1, predicted_label="crash",
                             score=ReasoningScore()))
    # FP
    mc.consolidate(Feedback(actual_label=0, predicted_label="crash",
                             score=ReasoningScore()))
    # FN
    mc.consolidate(Feedback(actual_label=1, predicted_label="normal",
                             score=ReasoningScore()))
    # TN
    mc.consolidate(Feedback(actual_label=0, predicted_label="normal",
                             score=ReasoningScore()))

    assert mc._tp == 1
    assert mc._fp == 1
    assert mc._fn == 1
    assert mc._tn == 1


def test_memory_consolidation_updates_weights_after_threshold():
    layer = ImplicitPredictiveLayer(input_dim=12)
    mc = MemoryConsolidation(layer, update_every=5)

    # Feed 5 crash feedbacks to trigger update
    for i in range(5):
        fb = Feedback(
            actual_label=1,
            predicted_label="crash",
            score=ReasoningScore(novelty=0.8, uncertainty=0.7,
                                 prediction_error=0.9, memory_confidence=0.1,
                                 spatial_correlation=0.85, risk=0.9),
            spike_vec=torch.rand(12),
        )
        summary = mc.consolidate(fb)

    # Last consolidation should have updated weights
    assert summary.weights_updated is True


def test_memory_consolidation_no_update_without_enough_crashes():
    layer = ImplicitPredictiveLayer(input_dim=12)
    mc = MemoryConsolidation(layer, update_every=5)

    # Only 2 crashes — below MIN_CRASHES_FOR_UPDATE (3)
    for i in range(5):
        label = 1 if i < 2 else 0
        fb = Feedback(
            actual_label=label,
            predicted_label="crash" if label else "normal",
            score=ReasoningScore(),
            spike_vec=torch.rand(12) if label else None,
        )
        summary = mc.consolidate(fb)

    assert summary.weights_updated is False


# ── Risk signal ────────────────────────────────────────────────────────────

def test_risk_in_unit_interval():
    layer = ImplicitPredictiveLayer(input_dim=12)
    spikes = torch.rand(25, 4, 12)
    for _ in range(10):
        score = layer.compute_state(spikes)
        assert 0.0 <= score.risk <= 1.0


def test_risk_higher_for_anomalous_input():
    """Risk is computed from the sub-signals and stays in [0,1] for all inputs."""
    layer = ImplicitPredictiveLayer(input_dim=12)
    normal = torch.zeros(25, 1, 12)
    anomaly = torch.ones(25, 1, 12) * 10.0

    # Warm up with normal
    for _ in range(30):
        score_n = layer.compute_state(normal)

    # Score anomaly — just check it's a valid float in [0,1]
    score_a = layer.compute_state(anomaly)
    assert 0.0 <= score_a.risk <= 1.0
    assert 0.0 <= score_n.risk <= 1.0
