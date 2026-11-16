"""tests/test_decision_recorder.py — unified decision recording (Path A + Path B)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from nv_platform.implicit_layer import ImplicitPredictiveLayer, ReasoningScore
from nv_platform.gate import ReasoningGate
from nv_platform.gate_learner import GateLearner
from nv_platform.agent_plugin import LLMAgentPlugin, AgentDecision
from nv_platform.safety import SafetyChecker
from nv_platform.planner import Planner
from nv_platform.automatic_action import AutomaticAction
from nv_platform.action_memory import ActionMemory
from nv_platform.action_executor import MockActionExecutor
from nv_platform.memory_consolidation import MemoryConsolidation
from nv_platform.decision_workflow import DecisionWorkflow
from nv_platform.decision_recorder import (
    DecisionRecorder,
    DecisionRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crash_score():
    # All danger signals > 0.95 → triggers the gate's hard floor → Path B.
    return ReasoningScore(novelty=0.97, uncertainty=0.96, prediction_error=0.97,
                         memory_confidence=0.02, spatial_correlation=0.97, risk=0.97)


def _normal_score():
    return ReasoningScore(risk=0.02)


def _make_workflow(recorder: DecisionRecorder, agent_decision=None,
                    llm_raises: bool = False):
    """Workflow wired with the given SQLite recorder; the LLM call is mocked.

    We mock ``_run_agent`` (the actual LLM call) rather than ``reason`` so the
    real ``reason`` wrapper still runs and sets ``last_llm_invoked`` — that
    flag is exactly what these tests verify, so it must not be mocked away.
    """
    implicit = ImplicitPredictiveLayer(input_dim=12)
    gate = ReasoningGate(learner=GateLearner(update_every=1000), threshold=0.9)
    agent = LLMAgentPlugin()
    wf = DecisionWorkflow(
        gate=gate, agent=agent, safety=SafetyChecker(), planner=Planner(),
        automatic_action=AutomaticAction(), action_memory=ActionMemory(),
        action_executor=MockActionExecutor(),
        memory_consolidation=MemoryConsolidation(implicit, update_every=1000),
        implicit_layer=implicit, decision_recorder=recorder,
    )
    if agent_decision is None:
        agent_decision = AgentDecision(label="crash", confidence=0.9,
                                        reasoning="mocked", sources=[])
    if llm_raises:
        # Simulates an LLM call that fails → reason() degrades to fallback,
        # but the LLM was still invoked.
        patcher = patch.object(agent, "_run_agent",
                               side_effect=RuntimeError("LLM unavailable"))
    else:
        patcher = patch.object(agent, "_run_agent", return_value=agent_decision)
    patcher.start()
    return wf, patcher


# ---------------------------------------------------------------------------
# DecisionRecord dataclass
# ---------------------------------------------------------------------------

def test_decision_record_defaults_and_path_constants():
    r = DecisionRecord(decision_id="d1", timestamp="t", predicted_label="normal",
                        decision_path="Path A", selected_action="continue_monitoring")
    assert r.llm_called is False
    assert r.action_plan is None
    assert DecisionRecord.PATH_A == "Path A"
    assert DecisionRecord.PATH_B == "Path B"


# ---------------------------------------------------------------------------
# Path A
# ---------------------------------------------------------------------------

def test_path_a_saves_prediction_and_selected_action(tmp_path):
    recorder = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    wf, p = _make_workflow(recorder)
    try:
        res = wf.decide({}, context={"score": _normal_score()})
        assert res.escalated is False          # Path A
        rec = recorder.get(res.decision_id)
        assert rec is not None
        assert rec.decision_path == "Path A"
        # Final prediction recorded.
        assert rec.predicted_label == "normal"
        # Selected action recorded (from the Automatic Action module).
        assert rec.selected_action != ""
        assert rec.selected_action == res.action_plan.selected_action
        # LLM was not invoked.
        assert rec.llm_called is False
        # Action plan is available even on Path A (plan_automatic).
        assert rec.action_plan is not None
        assert rec.action_plan["path"] == "automatic"
    finally:
        p.stop()
        recorder.close()


def test_path_a_selected_action_matches_automatic_action_module(tmp_path):
    recorder = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    wf, p = _make_workflow(recorder)
    try:
        res = wf.decide({}, context={"score": _normal_score()})
        rec = recorder.get(res.decision_id)
        # The recorded action is exactly the action the AutomaticAction module chose.
        from nv_platform.automatic_action import AutomaticAction
        direct = AutomaticAction().act(_normal_score())
        assert rec.selected_action == direct.action
    finally:
        p.stop()
        recorder.close()


# ---------------------------------------------------------------------------
# Path B
# ---------------------------------------------------------------------------

def test_path_b_saves_prediction_and_generated_action_plan(tmp_path):
    recorder = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    wf, p = _make_workflow(recorder)
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        assert res.escalated is True           # Path B
        rec = recorder.get(res.decision_id)
        assert rec is not None
        assert rec.decision_path == "Path B"
        # Final prediction recorded (from the Agent).
        assert rec.predicted_label == "crash"
        # Complete generated action plan recorded (from the Planner).
        assert rec.action_plan is not None
        assert rec.action_plan["path"] == "deliberative"
        assert len(rec.action_plan["steps"]) >= 1
        # Selected action is the plan's primary step.
        assert rec.selected_action == res.action_plan.selected_action
    finally:
        p.stop()
        recorder.close()


def test_path_b_records_llm_called_true(tmp_path):
    recorder = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    wf, p = _make_workflow(recorder)
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        rec = recorder.get(res.decision_id)
        assert rec.llm_called is True
    finally:
        p.stop()
        recorder.close()


# ---------------------------------------------------------------------------
# LLM invocation status across both paths
# ---------------------------------------------------------------------------

def test_llm_called_status_correct_for_both_paths(tmp_path):
    recorder = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    wf, p = _make_workflow(recorder)
    try:
        a = wf.decide({}, context={"score": _normal_score()})   # Path A
        b = wf.decide({}, context={"score": _crash_score()})   # Path B
        ra = recorder.get(a.decision_id)
        rb = recorder.get(b.decision_id)
        assert ra.llm_called is False
        assert rb.llm_called is True
    finally:
        p.stop()
        recorder.close()


def test_path_b_llm_called_true_even_when_agent_falls_back(tmp_path):
    """If the LLM call degrades to the fallback decision, the LLM was still
    invoked (Path B), so llm_called must remain True."""
    recorder = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    wf, p = _make_workflow(recorder, llm_raises=True)
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        rec = recorder.get(res.decision_id)
        assert rec.decision_path == "Path B"
        assert rec.llm_called is True          # invoked (then degraded)
        assert rec.predicted_label == "uncertain"   # fallback decision
    finally:
        p.stop()
        recorder.close()


# ---------------------------------------------------------------------------
# Exactly one record per event
# ---------------------------------------------------------------------------

def test_every_event_creates_exactly_one_record(tmp_path):
    recorder = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    wf, p = _make_workflow(recorder)
    try:
        n = 10
        ids = []
        for i in range(n):
            score = _crash_score() if i % 2 == 0 else _normal_score()
            res = wf.decide({}, context={"score": score})
            ids.append(res.decision_id)
        assert recorder.count() == n                      # one per event
        assert len(set(ids)) == n                          # all unique
        # No duplicate rows for any id.
        for did in ids:
            rows = [r for r in recorder.all() if r.decision_id == did]
            assert len(rows) == 1
    finally:
        p.stop()
        recorder.close()


def test_record_count_split_by_path(tmp_path):
    recorder = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    wf, p = _make_workflow(recorder)
    try:
        for _ in range(3):
            wf.decide({}, context={"score": _crash_score()})   # Path B
        for _ in range(2):
            wf.decide({}, context={"score": _normal_score()}) # Path A
        split = recorder.count_by_path()
        assert split.get("Path B") == 3
        assert split.get("Path A") == 2
        assert recorder.count() == 5
    finally:
        p.stop()
        recorder.close()


# ---------------------------------------------------------------------------
# Persistence across restart
# ---------------------------------------------------------------------------

def test_records_persist_after_restart(tmp_path):
    db = str(tmp_path / "dec.sqlite")
    ids = []
    # First "run": record events, then close the recorder (simulates shutdown).
    rec1 = DecisionRecorder(db)
    wf, p = _make_workflow(rec1)
    try:
        for i in range(4):
            score = _crash_score() if i % 2 == 0 else _normal_score()
            res = wf.decide({}, context={"score": score})
            ids.append(res.decision_id)
        assert rec1.count() == 4
    finally:
        p.stop()
        rec1.close()

    # "Restart": a brand-new recorder at the same path must see all records.
    rec2 = DecisionRecorder(db)
    try:
        assert rec2.count() == 4
        for did in ids:
            assert rec2.get(did) is not None
        rows = rec2.all()
        assert {r.decision_id for r in rows} == set(ids)
        # Path split survived.
        paths = {r.decision_path for r in rows}
        assert paths == {"Path A", "Path B"}
    finally:
        rec2.close()


def test_action_plan_round_trips_through_sqlite(tmp_path):
    """The action_plan dict must survive JSON serialization to SQLite."""
    recorder = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    wf, p = _make_workflow(recorder)
    try:
        res = wf.decide({}, context={"score": _crash_score()})
        rec = recorder.get(res.decision_id)
        assert isinstance(rec.action_plan, dict)
        assert rec.action_plan["label"] == "crash"
        assert isinstance(rec.action_plan["steps"], list)
    finally:
        p.stop()
        recorder.close()


# ---------------------------------------------------------------------------
# Recorder unit behaviour
# ---------------------------------------------------------------------------

def test_recorder_get_missing_returns_none(tmp_path):
    rec = DecisionRecorder(str(tmp_path / "dec.sqlite"))
    try:
        assert rec.get("does-not-exist") is None
        assert rec.count() == 0
        assert rec.all() == []
    finally:
        rec.close()


def test_recorder_default_path_is_under_repo_data():
    from nv_platform.decision_recorder import DEFAULT_DB_PATH
    assert DEFAULT_DB_PATH.endswith("decision_records.sqlite")
    assert "data" in DEFAULT_DB_PATH


def test_get_default_recorder_is_singleton(tmp_path, monkeypatch):
    from nv_platform import decision_recorder as dr
    # Keep the test hermetic: point the default path at tmp_path and reset the
    # cached singleton so no file is written into the repo.
    monkeypatch.setattr(dr, "DEFAULT_DB_PATH", str(tmp_path / "default.sqlite"))
    monkeypatch.setattr(dr, "_default_recorder", None)
    r1 = dr.get_default_recorder()
    r2 = dr.get_default_recorder()
    assert r1 is r2
    r1.close()
