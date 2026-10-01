"""
/nv_platform/automatic_action.py

Automatic Action — Fast SNN/Policy Path
=========================================
When the Reasoning Gate determines LOW NEED (risk below threshold),
the pipeline takes this fast path instead of invoking the LLM agent.

This module implements a lightweight policy that produces an action
directly from the ReasoningScore without any LLM call. It runs in
microseconds vs. seconds for the deliberative path.

Architecture position
---------------------
    Gate ──(low need)──► AutomaticAction ──► Action
    Gate ──(high need)─► Deliberative Agent (LLM)

Interface
---------
    policy = AutomaticAction()
    action = policy.act(score, event, context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nv_platform.implicit_layer import ReasoningScore


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AutomaticActionResult:
    """
    Fast-path action produced without LLM involvement.

    Attributes
    ----------
    label      : Predicted label ("normal" for low-risk, "monitor" for borderline).
    confidence : Confidence derived directly from risk score.
    action     : Recommended action string.
    risk_score : The risk score that triggered this path.
    path       : Always "automatic" — identifies this as the fast path.
    """
    label: str = "normal"
    confidence: float = 1.0
    action: str = "continue_monitoring"
    risk_score: float = 0.0
    path: str = "automatic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "action": self.action,
            "risk_score": self.risk_score,
            "path": self.path,
        }

    def __repr__(self) -> str:
        return (
            f"AutomaticActionResult(label={self.label!r}, "
            f"confidence={self.confidence:.3f}, "
            f"action={self.action!r})"
        )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

# Risk thresholds for the automatic path action tiers
_RISK_LOW    = 0.25   # clearly normal → no action
_RISK_MEDIUM = 0.45   # borderline → increase monitoring frequency


class AutomaticAction:
    """
    Fast SNN/Policy path for low-need gate decisions.

    Takes a ReasoningScore and produces an action immediately without
    calling the LLM agent. Three tiers:

    risk < 0.25  → "normal"  / "continue_monitoring"
    risk < 0.45  → "monitor" / "increase_sampling_rate"
    risk ≥ 0.45  → should not reach this path (gate should have escalated)
                   returns "monitor" with low confidence as a safety net.

    Parameters
    ----------
    domain_actions : dict   Optional mapping of label → action string,
                            allowing domain-specific action vocabulary.
    """

    DEFAULT_ACTIONS = {
        "normal":  "continue_monitoring",
        "monitor": "increase_sampling_rate",
    }

    def __init__(
        self,
        domain_actions: dict[str, str] | None = None,
    ) -> None:
        self._actions = {**self.DEFAULT_ACTIONS, **(domain_actions or {})}

    def act(
        self,
        score: ReasoningScore,
        event: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> AutomaticActionResult:
        """
        Produce a fast-path action from the ReasoningScore.

        Parameters
        ----------
        score   : ReasoningScore from the Implicit Predictive Layer.
        event   : Raw sensor event (optional — for logging/context).
        context : Additional context dict (optional).

        Returns
        -------
        AutomaticActionResult
        """
        risk = score.risk

        if risk < _RISK_LOW:
            label = "normal"
            confidence = 1.0 - risk
        elif risk < _RISK_MEDIUM:
            label = "monitor"
            confidence = 0.6 - risk * 0.5
        else:
            # Borderline — gate probably should have escalated but didn't
            label = "monitor"
            confidence = max(0.1, 0.5 - risk * 0.3)

        action = self._actions.get(label, "continue_monitoring")
        confidence = float(max(0.0, min(1.0, confidence)))

        return AutomaticActionResult(
            label=label,
            confidence=confidence,
            action=action,
            risk_score=risk,
            path="automatic",
        )
