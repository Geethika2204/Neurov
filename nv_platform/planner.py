"""
/nv_platform/planner.py

Planner
=======
Sequences the outputs of RAG, Tools, and Safety into a structured,
executable action plan. Part of the deliberative path:

    RAG ──┐
    Tools─┼──► Planner ──► Action
    Safety┘

The Planner is the final step before producing the system Action on the
high-need path. It receives the Safety-checked AgentDecision and enriches
it with a concrete, sequenced action plan the downstream system can execute.

Interface
---------
    planner = Planner()
    plan    = planner.plan(safety_result, score, event, context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nv_platform.implicit_layer import ReasoningScore
from nv_platform.safety import SafetyResult
from nv_platform.automatic_action import AutomaticActionResult


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ActionPlan:
    """
    Structured, executable action plan produced by the Planner.

    Both decision paths produce an ``ActionPlan`` — the fast SNN path gets a
    short default/monitoring plan, the deliberative LLM path gets a detailed
    context-aware plan — so escalation status never determines *whether* a
    plan is produced, only how detailed it is.

    Attributes
    ----------
    label       : Final classification label.
    confidence  : Final confidence after planning.
    steps       : Ordered list of action steps to execute.
    reasoning   : Full reasoning chain (agent + planner).
    sources     : RAG document IDs that informed this plan.
    risk_score  : Risk score at decision time.
    path        : "deliberative" (LLM) or "automatic" (fast SNN) path.
    safe        : Whether the safety checker approved this plan.
    """
    label: str = "uncertain"
    confidence: float = 0.0
    steps: list[str] = field(default_factory=list)
    reasoning: str = ""
    sources: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    path: str = "deliberative"
    safe: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "steps": self.steps,
            "reasoning": self.reasoning,
            "sources": self.sources,
            "risk_score": self.risk_score,
            "path": self.path,
            "safe": self.safe,
        }

    @property
    def action_steps(self) -> list[str]:
        """Ordered action steps, exposed for Action Memory storage."""
        return list(self.steps)

    @property
    def selected_action(self) -> str:
        """The primary action selected by the system for this event.

        For Path B this is the plan's first step; for Path A ``plan_automatic``
        puts the automatic action first.  Falls back to the label if there are
        no steps.  This gives the unified decision recorder a single,
        path-agnostic notion of "the action selected".
        """
        return self.steps[0] if self.steps else self.label

    def memory_record(self) -> dict[str, Any]:
        """
        Compact representation suitable for Action Memory storage.

        Exposes the label, confidence, steps and path so Memory can later
        evaluate which recommendations were effective.
        """
        return {
            "label": self.label,
            "confidence": self.confidence,
            "steps": self.action_steps,
            "path": self.path,
            "safe": self.safe,
        }

    def __repr__(self) -> str:
        return (
            f"ActionPlan(label={self.label!r}, "
            f"confidence={self.confidence:.3f}, "
            f"steps={len(self.steps)}, safe={self.safe})"
        )


# ---------------------------------------------------------------------------
# Step templates per label
# ---------------------------------------------------------------------------

_STEP_TEMPLATES: dict[str, list[str]] = {
    "crash": [
        "alert_traffic_management_centre",
        "activate_variable_message_signs",
        "notify_emergency_services",
        "increase_upstream_sensor_sampling_rate",
        "log_incident_to_database",
    ],
    "anomaly": [
        "flag_for_human_review",
        "increase_upstream_sensor_sampling_rate",
        "log_anomaly_to_database",
    ],
    "alert": [
        "alert_traffic_management_centre",
        "flag_for_human_review",
        "log_alert_to_database",
    ],
    "monitor": [
        "increase_sampling_rate",
        "log_monitor_event",
    ],
    "normal": [
        "continue_monitoring",
    ],
    "uncertain": [
        "flag_for_human_review",
        "log_uncertain_event",
    ],
}


class Planner:
    """
    Converts a Safety-checked AgentDecision into a structured ActionPlan.

    The Planner:
    1. Accepts the Safety-checked decision (including any violations/downgrades).
    2. Maps the label to a concrete sequence of action steps.
    3. Scales confidence by risk score to avoid over-confident low-risk plans.
    4. Adds domain-specific steps if provided.

    Parameters
    ----------
    extra_steps      : dict[str, list[str]]  Additional steps per label.
    confidence_floor : float  Minimum confidence for "crash" decisions (default 0.4).
    """

    def __init__(
        self,
        extra_steps: dict[str, list[str]] | None = None,
        confidence_floor: float = 0.4,
    ) -> None:
        self._steps = {
            label: list(steps)
            for label, steps in _STEP_TEMPLATES.items()
        }
        if extra_steps:
            for label, steps in extra_steps.items():
                self._steps.setdefault(label, []).extend(steps)
        self._confidence_floor = confidence_floor

    def plan(
        self,
        safety_result: SafetyResult,
        score: ReasoningScore,
        event: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActionPlan:
        """
        Build an ActionPlan from a SafetyResult.

        Parameters
        ----------
        safety_result : SafetyResult  Output from SafetyChecker.check().
        score         : ReasoningScore  Current risk context.
        event         : dict  Optional raw sensor event.
        context       : dict  Optional additional context.

        Returns
        -------
        ActionPlan
        """
        decision = safety_result.decision
        label    = decision.label
        risk     = score.risk

        # Adjust confidence: blend agent confidence with risk signal
        # This prevents high-confidence "normal" on high-risk or vice-versa
        if label in ("crash", "anomaly", "alert"):
            # Scale up with risk — high risk reinforces crash label
            confidence = decision.confidence * 0.6 + risk * 0.4
        elif label == "normal":
            # Scale down if risk is non-trivial
            confidence = decision.confidence * (1.0 - risk * 0.3)
        else:
            confidence = decision.confidence

        # Apply floor for crash decisions (never report crash with near-zero conf)
        if label in ("crash", "anomaly", "alert"):
            confidence = max(confidence, self._confidence_floor)

        confidence = float(max(0.0, min(1.0, confidence)))

        # Build action steps
        steps = list(self._steps.get(label, ["log_event"]))

        # Add contextual steps based on risk magnitude
        if risk > 0.7 and "log_incident_to_database" not in steps:
            steps.append("log_high_risk_event")
        if not safety_result.safe:
            steps.insert(0, "human_review_required")

        return ActionPlan(
            label=label,
            confidence=confidence,
            steps=steps,
            reasoning=decision.reasoning,
            sources=decision.sources,
            risk_score=risk,
            path="deliberative",
            safe=safety_result.safe,
        )

    # ------------------------------------------------------------------
    # Fast path (Path A) — default plans for non-escalated events
    # ------------------------------------------------------------------

    # Default monitoring instructions for normal/borderline events.
    _DEFAULT_NORMAL_STEPS: list[str] = ["continue_monitoring"]
    _DEFAULT_MONITOR_STEPS: list[str] = [
        "increase_sampling_rate",
        "log_monitor_event",
    ]

    def plan_automatic(
        self,
        automatic_result: AutomaticActionResult,
        score: ReasoningScore,
        event: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ActionPlan:
        """
        Build a default ActionPlan from a fast-path AutomaticActionResult.

        Normal events get a small monitoring instruction; borderline events
        get an increased-sampling plan.  When the fast path was informed by
        a reliable confirmed memory match that was a crash, the plan flags
        the event for human review even though it was not escalated — a
        safety net for confirmed-similar crash patterns the gate let through.

        This is the Path A counterpart of ``plan``: escalation status never
        determines *whether* a plan is produced, only its complexity.
        """
        label = automatic_result.label
        risk = score.risk

        # Scale confidence with the fast-path confidence but keep it bounded
        # by risk so a high-risk "normal" never looks over-confident.
        confidence = automatic_result.confidence * (1.0 - risk * 0.2)
        confidence = float(max(0.0, min(1.0, confidence)))

        if label == "monitor":
            steps = list(self._steps.get("monitor", self._DEFAULT_MONITOR_STEPS))
        else:
            steps = list(self._steps.get("normal", self._DEFAULT_NORMAL_STEPS))

        # Use the exact action string the automatic path recommended so the
        # downstream executor and memory see what was actually advised.
        if automatic_result.action and automatic_result.action not in steps:
            steps = [automatic_result.action] + steps

        # Safety net: if the automatic path was memory-informed by a confirmed
        # crash, flag for human review (the fast path cannot escalate, but it
        # should not silently pass a known crash signature through).
        safe = True
        diag = automatic_result.memory_match
        if diag and diag.get("was_crash"):
            steps = ["human_review_required"] + steps
            safe = False

        reasoning = automatic_result.action
        if diag:
            reasoning += (
                f" (memory-informed: sim={diag.get('similarity')}, "
                f"was_crash={diag.get('was_crash')})"
            )

        return ActionPlan(
            label=label,
            confidence=confidence,
            steps=steps,
            reasoning=reasoning,
            sources=[],
            risk_score=risk,
            path="automatic",
            safe=safe,
        )
