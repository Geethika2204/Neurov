"""
tests/test_e2e_smoke.py

End-to-end smoke test for the full NeuroVerse pipeline.
Pushes dummy events through all four layers without a real LLM.

Tests
-----
1. Full pipeline runs without error on synthetic spike data.
2. No /platform/ file imports from /domains/ (architecture invariant).
3. Gate escalation rate and summary stats are printed.
4. Pipeline works with TrafficDomain metadata.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nv_platform.snn_plugin import SNNInputPlugin
from nv_platform.implicit_layer import ImplicitPredictiveLayer, ReasoningScore
from nv_platform.gate import ReasoningGate, EscalationHistory
from nv_platform.gate_learner import GateLearner
from nv_platform.agent_plugin import LLMAgentPlugin, AgentDecision
from domains.traffic.domain import TrafficDomain, FEATURE_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_events(n: int = 20):
    """Generate n synthetic traffic sensor events."""
    events = []
    for i in range(n):
        event = {"timestamp": float(1_700_000_000 + i * 300)}
        for col in FEATURE_COLS:
            event[col] = float(torch.rand(1).item() * 100)
        # Every 10th event is labelled as a crash
        event["crash_record"] = 1 if i % 10 == 0 else 0
        events.append(event)
    return events


def _mock_agent_reason(event, context, rag_source):
    """Deterministic mock: 'crash' if gate_score > 0.7, else 'normal'."""
    gate_score = context.get("gate_confidence", 0.5)
    label = "crash" if gate_score > 0.7 else "normal"
    return AgentDecision(
        label=label,
        confidence=float(gate_score),
        reasoning="mocked agent",
        sources=[],
    )


# ---------------------------------------------------------------------------
# Architecture invariant: no /domains/ import in /platform/
# ---------------------------------------------------------------------------


def test_platform_does_not_import_domains():
    """
    Verify that no file under /platform/ contains an import of /domains/.
    Uses ast.walk to inspect import nodes statically.
    """
    platform_dir = REPO_ROOT / "nv_platform"
    violations = []

    for py_file in platform_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("domains"):
                        violations.append((py_file.name, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("domains"):
                    violations.append((py_file.name, module))

    assert not violations, (
        f"/nv_platform/ files must not import from /domains/. Violations:\n"
        + "\n".join(f"  {f}: imports '{m}'" for f, m in violations)
    )


# ---------------------------------------------------------------------------
# Full pipeline smoke test
# ---------------------------------------------------------------------------


def test_full_pipeline_runs():
    """
    Push 20 dummy events through the full four-layer pipeline.
    Assert no exception and that decisions come out the other end.
    """
    domain = TrafficDomain()
    snn = SNNInputPlugin(num_steps=25)
    implicit = ImplicitPredictiveLayer(input_dim=12)
    learner = GateLearner(update_every=50)
    gate = ReasoningGate(learner=learner, threshold=0.5)
    agent = LLMAgentPlugin()
    history = EscalationHistory()

    events = _make_dummy_events(n=20)
    decisions = []
    escalated = 0

    with patch.object(agent, "reason", side_effect=_mock_agent_reason):
        for event in events:
            # 1. Validate event against domain contract
            assert domain.validate_event(event), f"Invalid event: {event}"

            # 2. Layer 1 — SNN
            tensor = TrafficDomain.event_to_tensor(event)  # (1, 1, 12)
            spikes = snn.process_events(tensor)            # (25, 1, 12)

            # 3. Layer 2 — Implicit predictive layer
            score = implicit.compute_state(spikes)
            assert isinstance(score, ReasoningScore)
            assert all(
                0.0 <= v <= 1.0
                for v in score.as_tensor().tolist()
            ), f"Score out of range: {score}"

            # 4. Layer 3 — Gate
            should_esc, conf = gate.should_escalate(score, history)
            assert isinstance(should_esc, bool)
            assert 0.0 <= conf <= 1.0

            # 5. Layer 4 — Agent (mocked)
            if should_esc:
                escalated += 1
                decision = agent.reason(
                    event,
                    {"gate_confidence": conf, "gate_score": score.as_tensor().tolist()},
                    domain.knowledge_base_path,
                )
            else:
                decision = AgentDecision(
                    label="normal",
                    confidence=1.0 - conf,
                    reasoning="gate suppressed",
                )

            assert isinstance(decision, AgentDecision)
            assert decision.label in ("crash", "normal", "uncertain", "anomaly")
            decisions.append(decision)

    total = len(events)
    esc_rate = escalated / total
    avg_conf = sum(d.confidence for d in decisions) / total

    print(
        f"\n[smoke] escalated {escalated}/{total} events "
        f"({esc_rate:.1%}), "
        f"gate_confidence mean={avg_conf:.3f}"
    )

    assert len(decisions) == total, "Expected one decision per event"


def test_pipeline_with_zero_features():
    """All-zero sensor stream should not crash the pipeline."""
    snn = SNNInputPlugin(num_steps=10)
    implicit = ImplicitPredictiveLayer(input_dim=6)
    learner = GateLearner()
    gate = ReasoningGate(learner=learner)
    history = EscalationHistory()

    stream = torch.zeros(1, 1, 6)
    spikes = snn.process_events(stream)
    score = implicit.compute_state(spikes)
    escalate, conf = gate.should_escalate(score, history)

    assert isinstance(escalate, bool)
    assert isinstance(conf, float)


def test_pipeline_with_random_large_batch():
    """Larger batch size should work correctly."""
    snn = SNNInputPlugin(num_steps=25)
    implicit = ImplicitPredictiveLayer(input_dim=12)
    learner = GateLearner()
    gate = ReasoningGate(learner=learner)
    history = EscalationHistory()

    stream = torch.randn(32, 20, 12)  # batch=32
    spikes = snn.process_events(stream)
    assert spikes.shape == (25, 32, 12)

    score = implicit.compute_state(spikes)
    assert all(0.0 <= v <= 1.0 for v in score.as_tensor().tolist())

    escalate, conf = gate.should_escalate(score, history)
    assert isinstance(escalate, bool)


def test_baseline_comparison():
    """
    Verify AlwaysThresholdGate escalates fewer events than AlwaysEscalateGate
    on mostly-normal synthetic data (sanity check for baseline utility).
    """
    from baselines.always_escalate_gate import AlwaysEscalateGate
    from baselines.always_threshold_gate import AlwaysThresholdGate

    always = AlwaysEscalateGate()
    threshold_gate = AlwaysThresholdGate(threshold=0.5)

    # Mostly normal scores
    scores = [
        ReasoningScore(0.1, 0.15, 0.1, 0.9) for _ in range(90)
    ] + [
        ReasoningScore(0.9, 0.85, 0.8, 0.05) for _ in range(10)
    ]
    labels = [0] * 90 + [1] * 10

    always_m = always.evaluate(scores, labels)
    thresh_m = threshold_gate.evaluate(scores, labels)

    print(
        f"\n[baselines] always_escalate: rate={always_m['escalation_rate']:.2f} "
        f"recall={always_m['recall']:.2f}\n"
        f"           threshold_gate:  rate={thresh_m['escalation_rate']:.2f} "
        f"recall={thresh_m['recall']:.2f}"
    )

    assert always_m["escalation_rate"] > thresh_m["escalation_rate"]


# ---------------------------------------------------------------------------
# Full-stack end-to-end decision→action→memory→feedback workflow
# ---------------------------------------------------------------------------


def test_decision_workflow_full_stack_with_domain():
    """
    Run the complete decision/action/memory/feedback workflow end-to-end
    using the real SNN + ImplicitLayer stack and the TrafficDomain, wired
    via an injected tensor_provider (keeping /nv_platform/ domain-free).

    Verifies:
      - Both decision paths produce an ActionPlan.
      - Outcomes are verified from ground-truth labels (never predictions).
      - Pending and confirmed feedback are kept separate.
      - Continuous learning accumulates confusion counts and crash episodes.
    """
    from nv_platform.automatic_action import AutomaticAction
    from nv_platform.safety import SafetyChecker
    from nv_platform.planner import Planner
    from nv_platform.memory_consolidation import MemoryConsolidation
    from nv_platform.action_memory import ActionMemory
    from nv_platform.action_executor import MockActionExecutor
    from nv_platform.decision_workflow import DecisionWorkflow

    domain = TrafficDomain()
    snn = SNNInputPlugin(num_steps=10)
    implicit = ImplicitPredictiveLayer(input_dim=12)
    learner = GateLearner(update_every=1000)
    # High threshold so the cold-start learner bias (≈0.73) does not
    # escalate normal events; crashes still escalate via the hard floor.
    gate = ReasoningGate(learner=learner, threshold=0.9)
    agent = LLMAgentPlugin()
    mem = ActionMemory()
    consolidator = MemoryConsolidation(implicit, update_every=1000)

    wf = DecisionWorkflow(
        gate=gate,
        agent=agent,
        safety=SafetyChecker(),
        planner=Planner(),
        automatic_action=AutomaticAction(),
        action_memory=mem,
        action_executor=MockActionExecutor(),
        memory_consolidation=consolidator,
        implicit_layer=implicit,
        snn=snn,
        tensor_provider=domain.event_to_tensor,   # injected — no /platform/ domain import
        knowledge_base_path=domain.knowledge_base_path,
    )

    events = _make_dummy_events(n=20)

    def _mock_reason(event, context, rag_source):
        # Crash label when the gate is highly confident, else normal.
        return AgentDecision(
            label="crash" if context.get("gate_confidence", 0.0) > 0.5 else "normal",
            confidence=float(context.get("gate_confidence", 0.5)),
            reasoning="mocked agent",
            sources=[],
        )

    escalated = 0
    with patch.object(agent, "reason", side_effect=_mock_reason):
        for event in events:
            res = wf.decide(event)
            # Every decision produces an actionable plan (both paths).
            assert len(res.action_plan.steps) >= 1
            # Verify the outcome from the dataset ground-truth label.
            wf.verify_outcome(
                res.decision_id,
                event=event,
                context={"label": event["crash_record"]},
            )
            if res.escalated:
                escalated += 1

    stats = wf.stats()
    n_crashes = sum(1 for e in events if e["crash_record"] == 1)
    print(
        f"\n[workflow] events={len(events)} escalated={escalated} "
        f"confirmed={stats['action_memory']['confirmed']} "
        f"crashes_in_data={n_crashes} "
        f"confirmed_crashes={stats['action_memory']['confirmed_crashes']} "
        f"tp={stats['action_memory']['tp']} "
        f"fn={stats['action_memory']['fn']}"
    )

    # All events were verified and confirmed.
    assert stats["action_memory"]["confirmed"] == len(events)
    # Confirmed crashes in Action Memory match the dataset crash count.
    assert stats["action_memory"]["confirmed_crashes"] == n_crashes
    # Every decision was recorded (pending → confirmed lifecycle).
    assert stats["action_memory"]["pending"] == 0


def test_decision_workflow_records_decision_log_fields():
    """Decision logging records id, signals, prediction, escalation status,
    action plan and execution status — and ground truth never equals the
    model's own prediction by construction."""
    from nv_platform.automatic_action import AutomaticAction
    from nv_platform.safety import SafetyChecker
    from nv_platform.planner import Planner
    from nv_platform.memory_consolidation import MemoryConsolidation
    from nv_platform.action_memory import ActionMemory
    from nv_platform.action_executor import MockActionExecutor
    from nv_platform.decision_workflow import DecisionWorkflow

    implicit = ImplicitPredictiveLayer(input_dim=12)
    learner = GateLearner(update_every=1000)
    gate = ReasoningGate(learner=learner, threshold=0.9)
    agent = LLMAgentPlugin()
    mem = ActionMemory()
    wf = DecisionWorkflow(
        gate=gate, agent=agent, safety=SafetyChecker(), planner=Planner(),
        automatic_action=AutomaticAction(), action_memory=mem,
        action_executor=MockActionExecutor(),
        memory_consolidation=MemoryConsolidation(implicit, update_every=1000),
        implicit_layer=implicit,
    )

    def _mock_reason(event, context, rag_source):
        return AgentDecision(label="crash", confidence=0.9,
                             reasoning="mocked", sources=[])

    with patch.object(agent, "reason", side_effect=_mock_reason):
        res = wf.decide({}, context={"score": ReasoningScore(
            novelty=0.97, uncertainty=0.96, prediction_error=0.97,
            memory_confidence=0.02, spatial_correlation=0.97, risk=0.97)})

    rec = mem.get(res.decision_id)
    # Decision-log fields present.
    assert rec.decision_id == res.decision_id
    assert rec.score_tensor is not None           # original signals
    assert rec.predicted_label == "crash"          # prediction
    assert rec.escalated is True                    # escalation status
    assert rec.action_plan["label"] == "crash"      # action plan
    assert rec.execution_status == "executed"       # execution status
    # Ground truth not yet known → pending.
    assert rec.outcome_status == "pending"
    assert rec.actual_label is None
