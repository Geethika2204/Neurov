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


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ActionPlan:
    """
    Structured, executable action plan produced by the Planner.

    Attributes
    ----------
    label       : Final classification label.
    confidence  : Final confidence after planning.
    steps       : Ordered list of action steps to execute.
    reasoning   : Full reasoning chain (agent + planner).
    sources     : RAG document IDs that informed this plan.
    risk_score  : Risk score at decision time.
    path        : "deliberative" — identifies this as the LLM path.
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
