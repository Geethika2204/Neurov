"""tests/test_decision_workflow.py — end-to-end decision/action/memory/feedback loop."""

from __future__ import annotations

from unittest.mock import patch

import torch

from nv_platform.implicit_layer import ImplicitPredictiveLayer, ReasoningScore
from nv_platform.gate import ReasoningGate
from nv_platform.gate_learner import GateLearner
from nv_platform.agent_plugin import LLMAgentPlugin, AgentDecision
from nv_platform.safety import SafetyChecker
from nv_platform.planner import Planner
from nv_platform.automatic_action import AutomaticAction
from nv_platform.memory_consolidation import MemoryConsolidation
from nv_platform.action_memory import ActionMemory
from nv_platform.action_executor import (
    MockActionExecutor,
    LabelOutcomeVerifier,
    SimulatedOperatorFeedback,
)
from nv_platform.decision_workflow import DecisionWorkflow, WorkflowResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_workflow(
    threshold=0.9,
    agent_decision=None,
    outcome_verifier=None,
    with_consolidation=True,
    memory_min_similarity=0.92,
):
    """Build a workflow with a mocked LLM agent so no network is needed.

    ``threshold`` defaults to 0.9 so the untrained GateLearner's cold-start
    bias (sigmoid(+1)≈0.73) does not escalate normal events — crashes still
    escalate via the hard floor (>0.95 on a danger signal).  This isolates the
    two paths cleanly without needing to train the learner.
    """
    implicit = ImplicitPredictiveLayer(input_dim=12)
    learner = GateLearner(update_every=1000)  # avoid online retrain during test
    gate = ReasoningGate(learner=learner, threshold=threshold)
    agent = LLMAgentPlugin()
    safety = SafetyChecker()
    planner = Planner()
    auto = AutomaticAction()
    mem = ActionMemory(min_similarity=memory_min_similarity)
    consolidator = MemoryConsolidation(implicit, update_every=1000) if with_consolidation else None

    wf = DecisionWorkflow(
        gate=gate,
        agent=agent,
        safety=safety,
        planner=planner,
        automatic_action=auto,
        action_memory=mem,
        action_executor=MockActionExecutor(),
        outcome_verifier=outcome_verifier or LabelOutcomeVerifier(label_key="label"),
        memory_consolidation=consolidator,
        implicit_layer=implicit,
    )

    # Mock the agent so we control the deliberative path output.
    if agent_decision is None:
        agent_decision = AgentDecision(label="crash", confidence=0.9,
                                        reasoning="mocked", sources=[])
    patcher = patch.object(agent, "reason", return_value=agent_decision)
    patcher.start()
    return wf, patcher


def _crash_score():
    return ReasoningScore(novelty=0.97, uncertainty=0.96, prediction_error=0.97,
                         memory_confidence=0.02, spatial_correlation=0.97, risk=0.97)


def _normal_score():
    return ReasoningScore(risk=0.02)


def _moderate_crash_score():
    """A crash-like score with all danger signals at 0.9 — below the hard
    floor (0.95) so the ONLY thing that can escalate it is the memory floor."""
    return ReasoningScore(novelty=0.9, uncertainty=0.9, prediction_error=0.9,
                          memory_confidence=0.1, spatial_correlation=0.9, risk=0.9)


# ---------------------------------------------------------------------------
# Decision half-loop
# ---------------------------------------------------------------------------

def test_decide_fast_path_for_normal_event():
    wf, p = _make_workflow(agent_decision=AgentDecision(label="normal", confidence=0.9))
    try:
        res = wf.decide({}, context={"score": _normal_score()})
        assert isinstance(res, WorkflowResult)
        assert res.escalated is False
        assert res.path == "automatic"
        assert res.action_plan.path == "automatic"
        assert len(res.action_plan.steps) >= 1   # default monitoring instruction
        assert res.execution_result.status == "executed"
        # Recorded as pending in Action Memory
        rec = wf.action_memory.get(res.decision_id)
        assert rec is not None
        assert rec.outcome_status == "pending"
    finally:
        p.stop()


def test_decide_deliberative_path_for_crash_event():
    wf, p = _make_workflow()
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        assert res.escalated is True
        assert res.path == "deliberative"
        assert res.action_plan.path == "deliberative"
        assert "alert_traffic_management_centre" in res.action_plan.steps or \
               len(res.action_plan.steps) >= 3
        rec = wf.action_memory.get(res.decision_id)
        assert rec.predicted_label == "crash"
        assert rec.escalated is True
    finally:
        p.stop()


def test_both_paths_produce_an_action_plan():
    """Escalation status must not determine WHETHER a plan is produced."""
    wf, p = _make_workflow()
    try:
        normal = wf.decide({}, context={"score": _normal_score()})
        crash = wf.decide({}, context={"score": _crash_score()})
        assert normal.action_plan.steps      # non-empty
        assert crash.action_plan.steps       # non-empty
        # crash plan is more detailed than the normal default
        assert len(crash.action_plan.steps) >= len(normal.action_plan.steps)
    finally:
        p.stop()


def test_decision_logged_with_signals_and_status():
    wf, p = _make_workflow()
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        rec = wf.action_memory.get(res.decision_id)
        assert rec.predicted_label == "crash"
        assert rec.escalated is True
        assert rec.gate_confidence == res.gate_confidence
        assert rec.action_plan["label"] == "crash"
        assert rec.execution_status == "executed"
        assert rec.outcome_status == "pending"
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# Outcome verification + learning
# ---------------------------------------------------------------------------

def test_verify_outcome_confirms_in_memory():
    wf, p = _make_workflow()
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        ver = wf.verify_outcome(res.decision_id, context={"label": 1})
        assert ver.actual_label == 1
        assert ver.verified_success is True
        rec = wf.action_memory.get(res.decision_id)
        assert rec.outcome_status == "confirmed"
        assert rec.actual_label == 1
    finally:
        p.stop()


def test_verify_outcome_logs_to_gate_history():
    wf, p = _make_workflow()
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        wf.verify_outcome(res.decision_id, context={"label": 1})
        entry = list(wf.history.log)[-1]
        assert entry["id"] == res.decision_id
        assert entry["outcome"] == 1
    finally:
        p.stop()


def test_false_negative_is_identified_and_not_effective():
    """Predicted normal, actual crash → FN, verified_success False."""
    wf, p = _make_workflow(agent_decision=AgentDecision(label="normal", confidence=0.9))
    try:
        # Gate will not escalate on a low score → automatic path predicts normal
        res = wf.decide({}, context={"score": _normal_score()})
        assert res.path == "automatic"
        ver = wf.verify_outcome(res.decision_id, context={"label": 1})
        assert ver.actual_label == 1
        assert ver.verified_success is False
        fns = wf.action_memory.false_negatives()
        assert any(r.decision_id == res.decision_id for r in fns)
    finally:
        p.stop()


def test_false_positive_is_identified():
    """Predicted crash, actual normal → FP."""
    wf, p = _make_workflow()
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        ver = wf.verify_outcome(res.decision_id, context={"label": 0})
        assert ver.actual_label == 0
        assert ver.verified_success is False
        fps = wf.action_memory.false_positives()
        assert any(r.decision_id == res.decision_id for r in fps)
    finally:
        p.stop()


def test_unverified_outcome_does_not_reach_consolidation():
    """Unverified feedback must not update RiskHead weights or episodic store."""
    wf, p = _make_workflow()
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        weights_before = wf.implicit_layer._risk._weights.clone()
        from nv_platform.memory_consolidation import Feedback
        summary = wf.memory_consolidation.consolidate(Feedback(
            decision_id=res.decision_id,
            actual_label=1,
            predicted_label="crash",
            score=_crash_score(),
            verified=False,
        ))
        assert summary.weights_updated is False
        assert summary.recorded_pending is True
        assert summary.action_memory_confirmed is False
        # Weights unchanged
        assert torch.allclose(weights_before, wf.implicit_layer._risk._weights)
        # Record stays pending
        assert wf.action_memory.get(res.decision_id).outcome_status == "pending"
    finally:
        p.stop()


def test_verified_outcome_confirms_in_consolidation():
    wf, p = _make_workflow()
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        # Verified path via the workflow
        wf.verify_outcome(res.decision_id, context={"label": 1})
        rec = wf.action_memory.get(res.decision_id)
        assert rec.outcome_status == "confirmed"
        # The consolidator's confusion matrix saw it
        assert wf.memory_consolidation._tp == 1
    finally:
        p.stop()


def test_confirmed_crash_stored_in_episodic_memory():
    wf, p = _make_workflow()
    try:
        spike_vec = torch.rand(12)
        res = wf.decide({}, context={"score": _crash_score()})
        wf.verify_outcome(res.decision_id, context={"label": 1}, spike_vec=spike_vec)
        # Episodic memory should now hold the confirmed crash pattern
        buf = wf.implicit_layer._memory.get_buffer_tensor()
        assert buf is not None
        assert buf.shape[0] >= 1
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# Memory floor in the gate (reliable confirmed-crash match escalates)
# ---------------------------------------------------------------------------

def test_memory_floor_escalates_confirmed_crash_pattern():
    """A moderate crash (below the hard floor) with a reliable confirmed-crash
    memory match should still escalate — reducing false negatives via
    experience.  With threshold=0.99 the untrained learner (prob≈0.73) and
    the hard floor (>0.95) both stay silent, so ONLY the memory floor can
    escalate."""
    wf, p = _make_workflow(threshold=0.99, memory_min_similarity=0.5)
    try:
        moderate = _moderate_crash_score()
        # Seed memory with a confirmed crash identical to the query.
        seed = wf.decide({}, context={"score": moderate})
        # Seed decision must NOT escalate (no hard floor, no learner, no memory)
        assert seed.escalated is False
        wf.verify_outcome(seed.decision_id, context={"label": 1})
        assert wf.action_memory.get(seed.decision_id).is_crash()

        # Now decide on an identical moderate-crash pattern: only the memory
        # floor can escalate it.
        res = wf.decide({}, context={"score": moderate})
        assert res.escalated is True, "Memory floor failed to escalate a confirmed-crash pattern"
    finally:
        p.stop()


def test_hard_floor_preserved_when_memory_absent():
    """The hard escalation floor still fires even with no memory."""
    wf, p = _make_workflow(threshold=0.99, memory_min_similarity=0.99)
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        # Hard floor (>0.95 on a danger signal) forces escalation
        assert res.escalated is True
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# Simulated operator feedback path
# ---------------------------------------------------------------------------

def test_simulated_operator_closes_loop():
    wf, p = _make_workflow(
        outcome_verifier=SimulatedOperatorFeedback(
            ground_truth_provider=lambda ctx: 1, review_noise=0.0,
        ),
    )
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        ver = wf.verify_outcome(res.decision_id)
        assert ver.source == "simulated_operator"
        assert ver.actual_label == 1
    finally:
        p.stop()


def test_process_with_label_one_call():
    wf, p = _make_workflow()
    try:
        res, ver = wf.process_with_label({}, label=1, context={"score": _crash_score()})
        assert ver.actual_label == 1
        assert wf.action_memory.get(res.decision_id).outcome_status == "confirmed"
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# Continuous learning across many events
# ---------------------------------------------------------------------------

def test_continuous_learning_updates_stats():
    """A batch of verified outcomes accumulates confusion counts and
    confirmed crash patterns in Action Memory."""
    wf, p = _make_workflow()
    try:
        # 3 confirmed crashes + 2 confirmed normals
        for _ in range(3):
            res = wf.decide({}, context={"score": _crash_score()})
            wf.verify_outcome(res.decision_id, context={"label": 1})
        for _ in range(2):
            res = wf.decide({}, context={"score": _normal_score()})
            wf.verify_outcome(res.decision_id, context={"label": 0})
        s = wf.stats()
        assert s["action_memory"]["confirmed"] == 5
        assert s["action_memory"]["confirmed_crashes"] == 3
        assert s["action_memory"]["tp"] == 3   # crashes predicted crash
        assert s["action_memory"]["tn"] == 2   # normals predicted normal
    finally:
        p.stop()


def test_action_effectiveness_evaluated():
    """The workflow tracks which recommended actions were effective."""
    wf, p = _make_workflow()
    try:
        for _ in range(3):
            res = wf.decide({}, context={"score": _crash_score()})
            wf.verify_outcome(res.decision_id, context={"label": 1})
        eff = wf.action_memory.action_effectiveness()
        # The crash plan steps were all effective (predicted crash, actual crash)
        assert any(info["success_rate"] == 1.0 for info in eff.values())
    finally:
        p.stop()


def test_unknown_decision_id_raises():
    wf, p = _make_workflow()
    try:
        try:
            wf.verify_outcome("does-not-exist")
            assert False, "expected KeyError"
        except KeyError:
            pass
    finally:
        p.stop()
