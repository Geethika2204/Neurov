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
    memory_informed : True when a reliable ActionMemory match adjusted this result.
    memory_match    : Diagnostic dict describing the confirmed experience used,
                     or None when no reliable match was found.
    """
    label: str = "normal"
    confidence: float = 1.0
    action: str = "continue_monitoring"
    risk_score: float = 0.0
    path: str = "automatic"
    memory_informed: bool = False
    memory_match: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "action": self.action,
            "risk_score": self.risk_score,
            "path": self.path,
            "memory_informed": self.memory_informed,
            "memory_match": self.memory_match,
        }

    @property
    def decision_fields(self) -> dict[str, Any]:
        """Fields the unified decision recorder needs from Path A.

        Exposes the final prediction and the automatic action selected by
        this module so the recorder can build a Path A record without
        reaching into the result's internals.
        """
        return {"predicted_label": self.label, "selected_action": self.action}

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

    Before applying the default risk rules, the policy consults Action
    Memory for *reliable confirmed* past experiences with a similar signal
    profile.  A reliable confirmed match can refine the label, action and
    confidence — but only confirmed outcomes are used (pending records
    never influence the fast path).

    Parameters
    ----------
    domain_actions : dict   Optional mapping of label → action string,
                            allowing domain-specific action vocabulary.
    action_memory  : ActionMemory | None  Optional long-term memory of past
                            decisions and verified outcomes.
    min_similarity : float   Minimum cosine similarity for a memory match to
                            be treated as reliable (default 0.92).
    """

    DEFAULT_ACTIONS = {
        "normal":  "continue_monitoring",
        "monitor": "increase_sampling_rate",
    }

    def __init__(
        self,
        domain_actions: dict[str, str] | None = None,
        action_memory: "ActionMemory | None" = None,
        min_similarity: float = 0.92,
    ) -> None:
        self._actions = {**self.DEFAULT_ACTIONS, **(domain_actions or {})}
        # Imported lazily to keep the module importable in isolation and to
        # avoid a hard dependency at construction time.
        self._memory = action_memory
        self._min_similarity = float(min_similarity)

    @property
    def action_memory(self) -> "ActionMemory | None":
        return self._memory

    @action_memory.setter
    def action_memory(self, value: "ActionMemory | None") -> None:
        self._memory = value

    def _consult_memory(
        self,
        score: ReasoningScore,
        risk: float,
    ) -> tuple[str, str, float, dict[str, Any] | None]:
        """
        Look up a reliable confirmed experience.

        Returns (label, action, confidence, match_diag).  When no reliable
        match is found, returns the *unmodified* risk-tier decision so the
        default rules remain the baseline.
        """
        # Base risk-tier decision
        if risk < _RISK_LOW:
            base_label = "normal"
            base_conf = 1.0 - risk
        elif risk < _RISK_MEDIUM:
            base_label = "monitor"
            base_conf = 0.6 - risk * 0.5
        else:
            base_label = "monitor"
            base_conf = max(0.1, 0.5 - risk * 0.3)
        base_conf = float(max(0.0, min(1.0, base_conf)))

        if self._memory is None:
            return base_label, self._actions.get(base_label, "continue_monitoring"), base_conf, None

        match = self._memory.reliable_match(score, min_similarity=self._min_similarity)
        if match is None:
            return base_label, self._actions.get(base_label, "continue_monitoring"), base_conf, None

        diag: dict[str, Any] = {
            "decision_id": match.record.decision_id,
            "similarity": round(match.similarity, 4),
            "was_crash": match.was_crash,
            "was_normal": match.was_normal,
            "previous_action": match.action,
            "previous_effective": match.effective,
        }

        # A reliable confirmed crash with a similar profile is a strong hint
        # that this event should be watched even though risk is below the
        # escalation threshold.  We never *suppress* monitoring on memory —
        # memory only ever increases the response level here.
        label = base_label
        confidence = base_conf
        if match.was_crash:
            label = "monitor"
            # The automatic path cannot escalate, but a confirmed-similar
            # crash raises monitoring confidence and flags the action.
            confidence = max(base_conf, 0.55)
            if match.effective is True and match.action:
                # Reuse the action that previously handled this pattern well.
                return label, match.action, confidence, diag
        elif match.was_normal:
            # A confirmed-similar normal event reinforces "normal" and lifts
            # confidence, but only when the risk tier already says normal.
            if base_label == "normal":
                confidence = min(1.0, base_conf + 0.1)
        action = self._actions.get(label, "continue_monitoring")
        return label, action, confidence, diag

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
        label, action, confidence, diag = self._consult_memory(score, risk)
        confidence = float(max(0.0, min(1.0, confidence)))

        return AutomaticActionResult(
            label=label,
            confidence=confidence,
            action=action,
            risk_score=risk,
            path="automatic",
            memory_informed=diag is not None,
            memory_match=diag,
        )
