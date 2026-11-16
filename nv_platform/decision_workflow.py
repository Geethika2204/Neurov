"""
/nv_platform/decision_workflow.py

End-to-end Decision → Action → Memory → Feedback workflow
=========================================================
A thin, domain-agnostic orchestrator that wires the existing platform layers
together with Action Memory and simulated execution/verification to form the
complete closed loop:

    Event
      │
      ▼
    SNN → ImplicitLayer → ReasoningScore
      │
      ▼
    Gate.should_escalate(score, history, action_memory)
      │            └─ reliable confirmed-crash matches add escalation
      ▼
    ┌──────── Path A (fast) ────────┐  ┌────── Path B (deliberative) ──────┐
    │ AutomaticAction.act(score)    │  │ Agent.reason(event, ctx, kb)     │
    │  └─ consults Action Memory    │  │ SafetyChecker.check(decision)    │
    │ Planner.plan_automatic(...)   │  │ Planner.plan(safety_result, …)   │
    └───────────────────────────────┘  └──────────────────────────────────┘
      │  both paths → ActionPlan
      ▼
    ActionExecutor.execute(plan)            (mock, replaceable)
      │  ExecutionResult
      ▼
    ActionMemory.record_decision/action     (outcome_status = pending)
      │
      ▼ ── later ──
    OutcomeVerifier.verify(decision_id)     (ground truth OR controlled sim)
      │  OutcomeVerification (never the model's own prediction)
      ▼
    ActionMemory.confirm_outcome(…, actual_label)   → confirmed
    Gate.log_outcome(decision_id, actual_label)       → learner history
    MemoryConsolidation.consolidate(verified Feedback)
      ├─ updates RiskHead signal weights
      └─ stores confirmed crash spike patterns in EpisodicMemory

Domain independence
-------------------
The orchestrator never imports from ``/domains/``.  Event→sensor-tensor
conversion is supplied by the caller via an injected ``tensor_provider``
callable, so any DomainPlugin can drive the workflow without the platform
knowing its feature columns.

The existing layers (SNN, ImplicitLayer, Gate, Agent, Safety, Planner,
AutomaticAction, MemoryConsolidation) are used unchanged — this module only
composes them.  No existing component is replaced or duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import torch

from nv_platform.implicit_layer import ImplicitPredictiveLayer, ReasoningScore
from nv_platform.gate import ReasoningGate, EscalationHistory
from nv_platform.agent_plugin import LLMAgentPlugin, AgentDecision
from nv_platform.safety import SafetyChecker, SafetyResult
from nv_platform.planner import Planner, ActionPlan
from nv_platform.automatic_action import AutomaticAction
from nv_platform.memory_consolidation import MemoryConsolidation, Feedback
from nv_platform.action_memory import ActionMemory, DecisionRecord
from nv_platform.action_executor import (
    ActionExecutor,
    MockActionExecutor,
    OutcomeVerifier,
    LabelOutcomeVerifier,
    OutcomeVerification,
    ExecutionResult,
)
from nv_platform.decision_recorder import (
    DecisionRecorder,
    DecisionRecord as DecisionLogRecord,
)


# ---------------------------------------------------------------------------
# Result of a single decide() call
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResult:
    """
    Everything produced by one pass through the decision→action half-loop.

    Attributes
    ----------
    decision_id      : str   Matches the ActionMemory and gate-history entry.
    score            : ReasoningScore  Signals at decision time.
    escalated        : bool  Whether the gate escalated (Path B).
    gate_confidence  : float Gate escalation probability.
    path             : str   "automatic" | "deliberative".
    action_plan      : ActionPlan  The actionable plan produced (both paths).
    execution_result : ExecutionResult  Result of (mock) execution.
    timestamp        : str
    """
    decision_id: str = ""
    score: ReasoningScore = field(default_factory=ReasoningScore)
    escalated: bool = False
    gate_confidence: float = 0.0
    path: str = "automatic"
    action_plan: ActionPlan = field(default_factory=ActionPlan)
    execution_result: ExecutionResult = field(default_factory=ExecutionResult)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "escalated": self.escalated,
            "gate_confidence": self.gate_confidence,
            "path": self.path,
            "action_plan": self.action_plan.to_dict(),
            "execution_status": self.execution_result.status,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class DecisionWorkflow:
    """
    Domain-agnostic end-to-end decision/action/memory/feedback orchestrator.

    Parameters
    ----------
    gate               : ReasoningGate
    agent              : LLMAgentPlugin
    safety             : SafetyChecker
    planner            : Planner
    automatic_action   : AutomaticAction
    action_memory      : ActionMemory
    action_executor    : ActionExecutor           (default MockActionExecutor)
    outcome_verifier   : OutcomeVerifier          (default LabelOutcomeVerifier)
    memory_consolidation : MemoryConsolidation
    implicit_layer     : ImplicitPredictiveLayer | None  Needed for episodic
                           storage during consolidation; optional if the
                           caller supplies spike vectors on feedback.
    snn                : SNNInputPlugin | None   Optional; when set together
                           with ``tensor_provider`` the workflow computes the
                           ReasoningScore itself.
    tensor_provider    : callable | None         ``event -> sensor_tensor``
                           injected by the caller (keeps /platform/ domain-free).
    history            : EscalationHistory | None  Gate decision log.
    knowledge_base_path: str  RAG path passed to the agent.
    decision_recorder  : DecisionRecorder | None  Optional persistent
                          recorder.  When set, ``decide`` persists exactly
                          one record per event (Path A or Path B) using the
                          unified mechanism.  None = no recording (keeps the
                          workflow hermetic for tests that don't need it).

    Notes
    -----
    The orchestrator is deliberately tolerant of optional pieces so it can be
    used in two modes:

    1. **Full stack** — set ``snn`` + ``tensor_provider`` + ``implicit_layer``
       so ``decide`` computes the score from a raw event.
    2. **Decision-only** — leave them out and pass the ``ReasoningScore`` via
       ``context["score"]`` so the workflow focuses purely on the
       decision/action/memory loop (useful for unit tests and for plugging
       into an existing pipeline that already computes scores).
    """

    def __init__(
        self,
        gate: ReasoningGate,
        agent: LLMAgentPlugin,
        safety: SafetyChecker,
        planner: Planner,
        automatic_action: AutomaticAction,
        action_memory: ActionMemory,
        action_executor: Optional[ActionExecutor] = None,
        outcome_verifier: Optional[OutcomeVerifier] = None,
        memory_consolidation: Optional[MemoryConsolidation] = None,
        implicit_layer: Optional[ImplicitPredictiveLayer] = None,
        snn: Optional[Any] = None,
        tensor_provider: Optional[Callable[[dict], "torch.Tensor"]] = None,
        history: Optional[EscalationHistory] = None,
        knowledge_base_path: str = "",
        decision_recorder: Optional[DecisionRecorder] = None,
    ) -> None:
        self.gate = gate
        self.agent = agent
        self.safety = safety
        self.planner = planner
        self.automatic_action = automatic_action
        self.action_memory = action_memory
        self.action_executor = action_executor or MockActionExecutor()
        self.outcome_verifier = outcome_verifier or LabelOutcomeVerifier()
        self.memory_consolidation = memory_consolidation
        self.implicit_layer = implicit_layer
        self.snn = snn
        self.tensor_provider = tensor_provider
        self.history = history or EscalationHistory()
        self.knowledge_base_path = knowledge_base_path
        self.decision_recorder = decision_recorder

        # Wire Action Memory into the gate and automatic-action so the fast
        # path benefits from reliable confirmed experiences.
        if self.gate.action_memory is None:
            self.gate.set_action_memory(self.action_memory)
        if self.automatic_action.action_memory is None:
            self.automatic_action.action_memory = self.action_memory
        # Wire Action Memory into the consolidator so verified outcomes are
        # confirmed there and unverified ones are tracked as pending.
        if self.memory_consolidation is not None and \
                self.memory_consolidation.action_memory is None:
            self.memory_consolidation.action_memory = self.action_memory

    # ------------------------------------------------------------------
    # Decision + action (records to Action Memory as pending)
    # ------------------------------------------------------------------

    def _compute_score(
        self, event: dict[str, Any], context: dict[str, Any] | None
    ) -> ReasoningScore:
        """Produce a ReasoningScore, computing it if the full stack is wired."""
        ctx = context or {}
        if ctx.get("score") is not None:
            return ctx["score"]
        if self.snn is not None and self.tensor_provider is not None and self.implicit_layer is not None:
            tensor = self.tensor_provider(event)
            spikes = self.snn.process_events(tensor)
            return self.implicit_layer.compute_state(spikes)
        # No way to compute a score — return a neutral score so the pipeline
        # still runs (the gate will not escalate on a flat score).
        return ReasoningScore()

    def decide(
        self,
        event: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """
        Run one event through the full decision→action path and record it
        to Action Memory (pending outcome).
        """
        ctx = dict(context or {})
        score = self._compute_score(event, ctx)

        # Reset the LLM-invocation flag; it is only set True on Path B when
        # agent.reason() actually runs.
        if hasattr(self.agent, "last_llm_invoked"):
            self.agent.last_llm_invoked = False

        # --- Layer 3: Gate ---
        escalate, gate_conf = self.gate.should_escalate(score, self.history)

        # --- Layer 4: Path A or Path B (both produce an ActionPlan) ---
        if escalate:
            decision = self.agent.reason(
                event, ctx, self.knowledge_base_path
            )
            safety_result = self.safety.check(decision, score, ctx)
            action_plan = self.planner.plan(safety_result, score, event, ctx)
            path = "deliberative"
            predicted_label = decision.label
        else:
            auto_result = self.automatic_action.act(score, event, ctx)
            action_plan = self.planner.plan_automatic(auto_result, score, event, ctx)
            path = "automatic"
            predicted_label = auto_result.label

        # --- Execution (mock / replaceable) ---
        # The decision_id is the last gate-history entry — the gate created it.
        decision_id = list(self.history.log)[-1]["id"]
        execution = self.action_executor.execute(action_plan, decision_id, ctx)

        # --- Decision logging: record to Action Memory (pending) ---
        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            score=score,
            score_tensor=score.as_tensor(),
            predicted_label=predicted_label,
            escalated=escalate,
            gate_confidence=float(gate_conf),
            path=path,
            action_plan=action_plan.to_dict(),
            actions_taken=execution.executed_steps,
            execution_status=execution.status,
            outcome_status="pending",
        )
        self.action_memory.record_decision(record)

        # --- Unified decision record (Path A + Path B, one per event) ---
        # Persisted via the same mechanism regardless of path.  llm_called is
        # grounded in whether agent.reason() actually ran (Path B only).
        if self.decision_recorder is not None:
            self.decision_recorder.record(
                DecisionLogRecord(
                    decision_id=decision_id,
                    timestamp=record.timestamp,
                    predicted_label=predicted_label,
                    decision_path=(
                        DecisionLogRecord.PATH_B if escalate
                        else DecisionLogRecord.PATH_A
                    ),
                    selected_action=action_plan.selected_action,
                    action_plan=action_plan.to_dict(),
                    llm_called=bool(
                        escalate
                        and getattr(self.agent, "last_llm_invoked", False)
                    ),
                    gate_confidence=float(gate_conf),
                )
            )

        return WorkflowResult(
            decision_id=decision_id,
            score=score,
            escalated=escalate,
            gate_confidence=float(gate_conf),
            path=path,
            action_plan=action_plan,
            execution_result=execution,
            timestamp=record.timestamp,
        )

    # ------------------------------------------------------------------
    # Outcome verification + continuous learning (closes the loop)
    # ------------------------------------------------------------------

    def verify_outcome(
        self,
        decision_id: str,
        event: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        spike_vec: "torch.Tensor | None" = None,
    ) -> OutcomeVerification:
        """
        Determine the actual outcome from an independent source, confirm it
        in Action Memory, log it to the gate history, and feed *verified*
        feedback to MemoryConsolidation.

        The actual outcome is produced by ``OutcomeVerifier`` — never by the
        model's own prediction.
        """
        record = self.action_memory.get(decision_id)
        if record is None:
            raise KeyError(f"Unknown decision_id: {decision_id!r}")

        ctx = dict(context or {})
        if event is not None:
            ctx.setdefault("event", event)
        verification = self.outcome_verifier.verify(decision_id, record, ctx)

        # --- Promote the record to confirmed ---
        self.action_memory.confirm_outcome(
            decision_id=decision_id,
            actual_label=verification.actual_label,
            verified_success=verification.verified_success,
            feedback_note=verification.feedback_note,
        )
        # Persist the executed steps if not already stored.
        if record.actions_taken:
            self.action_memory.record_action(
                decision_id, list(record.actions_taken), record.execution_status
            )

        # --- Log outcome to the gate so the learner can learn from it ---
        self.gate.log_outcome(decision_id, verification.actual_label, self.history)

        # --- Feed *verified* feedback to MemoryConsolidation ---
        if self.memory_consolidation is not None:
            fb = Feedback(
                decision_id=decision_id,
                actual_label=verification.actual_label,
                predicted_label=record.predicted_label,
                score=record.score,
                spike_vec=spike_vec,
                verified=True,
                actions_taken=list(record.actions_taken),
            )
            self.memory_consolidation.consolidate(fb)
        return verification

    # ------------------------------------------------------------------
    # Convenience: decide + verify in one call using a ground-truth label
    # ------------------------------------------------------------------

    def process_with_label(
        self,
        event: dict[str, Any],
        label: int,
        context: dict[str, Any] | None = None,
        label_key: str = "label",
        spike_vec: "torch.Tensor | None" = None,
    ) -> tuple[WorkflowResult, OutcomeVerification]:
        """
        Run the full loop for one event with a known ground-truth label.

        Uses ``LabelOutcomeVerifier`` semantics: the supplied ``label`` is the
        independent ground truth (0 normal, 1 crash/anomaly), never the model's
        prediction.
        """
        ctx = dict(context or {})
        ctx[label_key] = label
        result = self.decide(event, ctx)
        verification = self.verify_outcome(
            result.decision_id, event=event, context=ctx, spike_vec=spike_vec
        )
        return result, verification

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Aggregate stats across Action Memory and the gate history."""
        mem_stats = self.action_memory.stats()
        return {
            "action_memory": mem_stats,
            "gate_history_len": len(self.history),
            "action_effectiveness": self.action_memory.action_effectiveness(),
        }
