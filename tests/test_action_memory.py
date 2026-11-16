"""tests/test_action_memory.py — Action Memory store/retrieve/verify."""

from __future__ import annotations

import torch

from nv_platform.implicit_layer import ReasoningScore
from nv_platform.action_memory import (
    ActionMemory,
    DecisionRecord,
    MemoryMatch,
)


# ── Recording lifecycle ────────────────────────────────────────────────────

def test_record_decision_starts_pending():
    mem = ActionMemory()
    rec = DecisionRecord(decision_id="d1", score_tensor=ReasoningScore().as_tensor())
    did = mem.record_decision(rec)
    assert did == "d1"
    assert mem.get("d1").outcome_status == "pending"
    assert mem.get("d1").execution_status == "pending"
    assert mem.stats()["pending"] == 1
    assert mem.stats()["confirmed"] == 0


def test_record_decision_assigns_id_when_missing():
    mem = ActionMemory()
    rec = DecisionRecord(score_tensor=ReasoningScore().as_tensor())
    did = mem.record_decision(rec)
    assert did and mem.get(did) is not None


def test_record_action_updates_execution():
    mem = ActionMemory()
    rec = DecisionRecord(decision_id="d1", score_tensor=ReasoningScore().as_tensor())
    mem.record_decision(rec)
    ok = mem.record_action("d1", ["step_a", "step_b"], "executed")
    assert ok is True
    got = mem.get("d1")
    assert got.actions_taken == ["step_a", "step_b"]
    assert got.execution_status == "executed"


def test_confirm_outcome_promotes_to_confirmed():
    mem = ActionMemory()
    mem.record_decision(DecisionRecord(
        decision_id="d1", score_tensor=ReasoningScore().as_tensor(),
        predicted_label="crash",
    ))
    ok = mem.confirm_outcome("d1", actual_label=1, verified_success=True)
    assert ok is True
    assert mem.get("d1").outcome_status == "confirmed"
    assert mem.get("d1").actual_label == 1
    assert mem.get("d1").verified_success is True
    assert mem.stats()["confirmed"] == 1
    assert mem.stats()["pending"] == 0


def test_confirm_outcome_missing_returns_false():
    mem = ActionMemory()
    assert mem.confirm_outcome("nope", actual_label=0) is False


# ── Core invariant: unverified never influences decisions ─────────────────

def test_retrieve_similar_excludes_pending_by_default():
    """Pending records must NOT be returned for decision improvement."""
    mem = ActionMemory()
    q = ReasoningScore(novelty=0.9, uncertainty=0.9, prediction_error=0.9,
                       memory_confidence=0.1, spatial_correlation=0.9, risk=0.9)
    # A pending record identical to the query
    mem.record_decision(DecisionRecord(
        decision_id="pending1", score_tensor=q.as_tensor(),
        predicted_label="crash",
    ))
    # Default confirmed_only=True → must return nothing
    matches = mem.retrieve_similar(q, top_k=5)
    assert matches == [], "Pending records leaked into decision retrieval"


def test_retrieve_similar_includes_pending_when_requested():
    """confirmed_only=False is for observability only."""
    mem = ActionMemory()
    q = ReasoningScore(risk=0.9)
    mem.record_decision(DecisionRecord(
        decision_id="p1", score_tensor=q.as_tensor(), predicted_label="crash",
    ))
    matches = mem.retrieve_similar(q, top_k=5, confirmed_only=False)
    assert len(matches) == 1
    assert matches[0][0].decision_id == "p1"


def test_reliable_match_none_without_confirmed():
    mem = ActionMemory()
    q = ReasoningScore(risk=0.9)
    mem.record_decision(DecisionRecord(
        decision_id="p1", score_tensor=q.as_tensor(), predicted_label="crash",
    ))
    assert mem.reliable_match(q) is None


def test_confirmed_crash_match_is_reliable():
    mem = ActionMemory(min_similarity=0.5)
    crash_score = ReasoningScore(novelty=0.9, uncertainty=0.9,
                                 prediction_error=0.9, memory_confidence=0.1,
                                 spatial_correlation=0.9, risk=0.9)
    mem.record_decision(DecisionRecord(
        decision_id="c1", score_tensor=crash_score.as_tensor(),
        predicted_label="crash",
    ))
    mem.confirm_outcome("c1", actual_label=1, verified_success=True)

    match = mem.reliable_match(crash_score)
    assert isinstance(match, MemoryMatch)
    assert match.was_crash is True
    assert match.was_normal is False
    assert match.similarity >= 0.5


def test_confirmed_normal_match_retrievable():
    mem = ActionMemory(min_similarity=0.5)
    normal_score = ReasoningScore(risk=0.05)
    mem.record_decision(DecisionRecord(
        decision_id="n1", score_tensor=normal_score.as_tensor(),
        predicted_label="normal",
    ))
    mem.confirm_outcome("n1", actual_label=0, verified_success=True)

    match = mem.reliable_match(normal_score)
    assert match is not None
    assert match.was_normal is True
    assert match.was_crash is False


def test_low_similarity_not_reliable():
    mem = ActionMemory(min_similarity=0.99)
    # A crash with high signals across multiple dimensions.
    crash_score = ReasoningScore(novelty=0.9, uncertainty=0.9,
                                 prediction_error=0.9, memory_confidence=0.1,
                                 spatial_correlation=0.9, risk=0.9)
    mem.record_decision(DecisionRecord(
        decision_id="c1", score_tensor=crash_score.as_tensor(),
        predicted_label="crash",
    ))
    mem.confirm_outcome("c1", actual_label=1)
    # A clearly different direction (low across the high dims) — must not match.
    other = ReasoningScore(novelty=0.1, uncertainty=0.1, prediction_error=0.1,
                           memory_confidence=0.9, spatial_correlation=0.1, risk=0.1)
    assert mem.reliable_match(other) is None


# ── Aggregates for continuous learning ─────────────────────────────────────

def test_confusion_matrix_from_confirmed():
    mem = ActionMemory()
    _add(mem, "tp", "crash", 1)
    _add(mem, "fp", "crash", 0)
    _add(mem, "fn", "normal", 1)
    _add(mem, "tn", "normal", 0)
    cm = mem.confusion_matrix()
    assert cm == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


def test_false_negatives_and_positives():
    mem = ActionMemory()
    _add(mem, "fn1", "normal", 1)   # actual crash predicted normal → FN
    _add(mem, "fp1", "crash", 0)    # actual normal predicted crash → FP
    _add(mem, "tp1", "crash", 1)
    _add(mem, "tn1", "normal", 0)
    fns = mem.false_negatives()
    fps = mem.false_positives()
    assert [r.decision_id for r in fns] == ["fn1"]
    assert [r.decision_id for r in fps] == ["fp1"]


def test_confirmed_crash_patterns():
    mem = ActionMemory()
    _add(mem, "c1", "crash", 1)
    _add(mem, "c2", "crash", 1)
    _add(mem, "n1", "normal", 0)
    crashes = mem.confirmed_crash_patterns()
    assert len(crashes) == 2
    assert all(r.actual_label == 1 for r in crashes)


def test_action_effectiveness():
    mem = ActionMemory()
    # step "alert" effective 2/3, step "log" effective 1/1
    _add(mem, "a", "crash", 1, actions=["alert"], success=True)
    _add(mem, "b", "crash", 1, actions=["alert"], success=True)
    _add(mem, "c", "crash", 1, actions=["alert"], success=False)
    _add(mem, "d", "crash", 1, actions=["log"], success=True)
    eff = mem.action_effectiveness()
    assert eff["alert"]["success_rate"] == 0.6667
    assert eff["alert"]["count"] == 3
    assert eff["log"]["success_rate"] == 1.0


def test_effective_actions_threshold():
    mem = ActionMemory()
    # "alert" used 3x all success → effective
    for i in range(3):
        _add(mem, f"a{i}", "crash", 1, actions=["alert"], success=True)
    # "noisy" used 3x, 1 success → not effective
    _add(mem, "n0", "crash", 1, actions=["noisy"], success=True)
    _add(mem, "n1", "crash", 1, actions=["noisy"], success=False)
    _add(mem, "n2", "crash", 1, actions=["noisy"], success=False)
    eff = mem.effective_actions()
    assert "alert" in eff
    assert "noisy" not in eff


def test_stats_summary():
    mem = ActionMemory()
    _add(mem, "c1", "crash", 1)
    _add(mem, "n1", "normal", 0)
    s = mem.stats()
    assert s["confirmed"] == 2
    assert s["confirmed_crashes"] == 1
    assert s["tp"] == 1 and s["tn"] == 1


def test_lru_eviction():
    mem = ActionMemory(max_size=2)
    _add(mem, "a", "normal", 0)
    _add(mem, "b", "normal", 0)
    _add(mem, "c", "normal", 0)
    assert mem.stats()["confirmed"] == 2
    assert mem.get("a") is None  # evicted (oldest)
    assert mem.get("c") is not None


# ── helper ─────────────────────────────────────────────────────────────────

def _add(mem, did, predicted, actual, actions=None, success=None):
    mem.record_decision(DecisionRecord(
        decision_id=did,
        score_tensor=ReasoningScore(risk=0.9 if actual == 1 else 0.1).as_tensor(),
        predicted_label=predicted,
        action_plan={"steps": actions or []},
        actions_taken=actions or [],
    ))
    mem.confirm_outcome(did, actual_label=actual, verified_success=success)
