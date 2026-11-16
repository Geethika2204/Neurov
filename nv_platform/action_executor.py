"""
/nv_platform/action_executor.py

Simulated Action Execution + Outcome Verification
===================================================
Because there are no real users yet, this module provides *replaceable*
simulated execution and outcome-verification components.  Every concrete
implementation subclasses a small abstract interface so that real user
interaction and external execution systems can drop in later without
touching the workflow.

Two concerns, two interfaces
---------------------------
1. ``ActionExecutor``      — simulates *running* an action plan and reports
                              which steps executed and whether execution
                              succeeded.  Default impl: ``MockActionExecutor``.
2. ``OutcomeVerifier``     — determines the *actual* outcome of a decision
                              from an independent source (never the model's own
                              prediction).  Two default impls:
                                - ``LabelOutcomeVerifier``        (dataset GT)
                                - ``SimulatedOperatorFeedback``  (controlled sim)

Why outcome sources are separated from the model
-------------------------------------------------
"Never use the model's own predictions as ground truth."  Both verifiers
pull the actual label from an independent channel (a dataset label or a
controlled simulation), so a model that is confidently wrong is still
scored as wrong.

Domain independence
-------------------
No imports from ``/domains/``.  The ground-truth label key is configurable
so any dataset's label column can drive ``LabelOutcomeVerifier``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from nv_platform.planner import ActionPlan


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """
    Outcome of executing an ActionPlan.

    Attributes
    ----------
    decision_id    : str   The decision this execution belongs to.
    status         : str   "executed" | "failed" | "skipped".
    executed_steps : list  Steps the executor reports as having run.
    timestamp      : str   ISO-8601.
    notes          : str   Free-text executor note.
    """
    decision_id: str = ""
    status: str = "skipped"
    executed_steps: list[str] = field(default_factory=list)
    timestamp: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "status": self.status,
            "executed_steps": list(self.executed_steps),
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Action executor interface + mock implementation
# ---------------------------------------------------------------------------


class ActionExecutor(ABC):
    """Interface for executing an ActionPlan.  Replaceable with a real system."""

    @abstractmethod
    def execute(
        self,
        plan: ActionPlan,
        decision_id: str,
        context: Optional[dict] = None,
    ) -> ExecutionResult:
        """Run *plan* and report what happened."""


class MockActionExecutor(ActionExecutor):
    """
    Deterministic simulated executor.

    Records every step of the plan as "executed" and reports overall success
    unless a controlled failure is injected.  Determinism comes from an
    optional seed so test outcomes are reproducible.

    Parameters
    ----------
    seed            : int   RNG seed for injected failures.
    fail_probability: float  Probability an execution is reported as failed.
                             0.0 by default — the simulation assumes actions
                             run as commanded unless explicitly degraded.

    Replaceability
    --------------
    A real implementation would dispatch each step to an external control
    system (variable message signs, sampling-rate controller, …) and report
    back actual step statuses.  The interface above is all the workflow sees.
    """

    def __init__(self, seed: int = 42, fail_probability: float = 0.0) -> None:
        self._rng = random.Random(seed)
        self._fail_probability = float(fail_probability)

    def execute(
        self,
        plan: ActionPlan,
        decision_id: str,
        context: Optional[dict] = None,
    ) -> ExecutionResult:
        steps = list(plan.steps)
        # Controlled failure injection (deterministic given the seed).
        failed = (
            self._fail_probability > 0.0
            and self._rng.random() < self._fail_probability
        )
        status = "failed" if failed else "executed"
        notes = "simulated execution" + (" (injected failure)" if failed else "")
        return ExecutionResult(
            decision_id=decision_id,
            status=status,
            executed_steps=steps,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Outcome verification
# ---------------------------------------------------------------------------


@dataclass
class OutcomeVerification:
    """
    The independent actual outcome of a past decision.

    Attributes
    ----------
    decision_id      : str
    actual_label     : int    Ground-truth: 0 normal, 1 crash/anomaly.
    verified_success : bool   Was the recommendation effective?  By default
                              this is "prediction matched actual outcome" —
                              a domain-agnostic notion of correctness.  A
                              verifier may override it (e.g. an operator can
                              mark a correct-but-harmful action as failed).
    source           : str    "ground_truth" | "simulated_operator".
    feedback_note    : str
    """
    decision_id: str = ""
    actual_label: int = 0
    verified_success: bool = False
    source: str = "ground_truth"
    feedback_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "actual_label": self.actual_label,
            "verified_success": self.verified_success,
            "source": self.source,
            "feedback_note": self.feedback_note,
        }


# Set of predicted labels that count as "predicted a crash/anomaly".
_POSITIVE_PREDICTIONS = {
    "crash", "anomaly", "alert", "inner_race", "outer_race", "ball",
}


def _predicted_positive(predicted_label: str) -> bool:
    return predicted_label in _POSITIVE_PREDICTIONS


def _success_from_match(predicted_label: str, actual_label: int) -> bool:
    """Domain-agnostic effectiveness: prediction agreed with the truth."""
    predicted_pos = _predicted_positive(predicted_label)
    actual_pos = actual_label == 1
    return predicted_pos == actual_pos


class OutcomeVerifier(ABC):
    """Interface for determining the actual outcome of a decision."""

    @abstractmethod
    def verify(
        self,
        decision_id: str,
        record: "object",
        context: Optional[dict] = None,
    ) -> OutcomeVerification:
        """
        Determine the actual outcome for *record*.

        ``record`` is a ``DecisionRecord`` (typed loosely to avoid a circular
        import with action_memory at module top — it is imported lazily where
        needed).  Implementations must NOT use ``record.predicted_label`` as
        the actual outcome.
        """


class LabelOutcomeVerifier(OutcomeVerifier):
    """
    Determines the actual outcome from a dataset ground-truth label.

    The label is read from either the supplied event/context dict (option 1:
    dataset ground truth) or a caller-provided lookup.  This is the verifier
    used when labelled evaluation data is available.

    Parameters
    ----------
    label_key  : str   Key holding the ground-truth label in the event dict.
    crash_value : int  Label value that means "crash/anomaly" (default 1).
    """

    def __init__(self, label_key: str = "label", crash_value: int = 1) -> None:
        self._label_key = label_key
        self._crash_value = int(crash_value)

    def verify(
        self,
        decision_id: str,
        record: "object",
        context: Optional[dict] = None,
    ) -> OutcomeVerification:
        ctx = context or {}
        actual_raw = ctx.get(self._label_key)
        if actual_raw is None:
            # Fall back to a record-attached event label if the verifier was
            # given one (e.g. attached to the DecisionRecord's action_plan).
            actual_raw = (record.action_plan or {}).get("_ground_truth_label")
        actual_label = 1 if actual_raw == self._crash_value else 0
        predicted = getattr(record, "predicted_label", "uncertain")
        success = _success_from_match(predicted, actual_label)
        return OutcomeVerification(
            decision_id=decision_id,
            actual_label=actual_label,
            verified_success=success,
            source="ground_truth",
            feedback_note=f"label={actual_raw!r} via {self._label_key!r}",
        )


class SimulatedOperatorFeedback(OutcomeVerifier):
    """
    Controlled simulation of an operator reviewing an action.

    Two modes, both *independent of the model's prediction*:

    1. ``ground_truth_provider`` set — the operator "knows" the real outcome
       (e.g. an after-the-fact incident report) and returns it.  This is a
       controlled simulation driven by an external truth channel.
    2. No provider — the operator derives the actual outcome from observable
       signal magnitudes in the record (a deterministic rule), not from the
       predicted label.

    Parameters
    ----------
    ground_truth_provider : callable | None  ``event/record -> int label``.
    seed                  : int               RNG seed for noisy reviews.
    review_noise          : float             Probability the simulated
                                              operator's success verdict is
                                              flipped (0 = perfect operator).
    risk_threshold        : float             In provider-less mode, a record
                                              whose risk >= this is judged a
                                              crash; else normal.

    Replaceability
    --------------
    A real implementation would surface the plan to a human operator and wait
    for their confirmation.  The interface is unchanged.
    """

    def __init__(
        self,
        ground_truth_provider: Optional[Callable[[Any], int]] = None,
        seed: int = 42,
        review_noise: float = 0.0,
        risk_threshold: float = 0.6,
    ) -> None:
        self._provider = ground_truth_provider
        self._rng = random.Random(seed)
        self._review_noise = float(review_noise)
        self._risk_threshold = float(risk_threshold)

    def verify(
        self,
        decision_id: str,
        record: "object",
        context: Optional[dict] = None,
    ) -> OutcomeVerification:
        # Determine the ACTUAL outcome from an independent source — never the
        # model's own prediction.
        if self._provider is not None:
            actual_label = 1 if self._provider(context or record) else 0
            note = "simulated_operator (provider)"
        else:
            # Provider-less controlled simulation: judge from signal magnitude.
            score = getattr(record, "score", None)
            risk = getattr(score, "risk", 0.0) if score is not None else 0.0
            actual_label = 1 if risk >= self._risk_threshold else 0
            note = f"simulated_operator (risk={float(risk):.3f})"

        predicted = getattr(record, "predicted_label", "uncertain")
        success = _success_from_match(predicted, actual_label)

        # Optionally inject reviewer noise on the *success* verdict only
        # (the actual_label stays the independent ground truth).
        if self._review_noise > 0.0 and self._rng.random() < self._review_noise:
            success = not success
            note += " (noisy review)"

        return OutcomeVerification(
            decision_id=decision_id,
            actual_label=actual_label,
            verified_success=success,
            source="simulated_operator",
            feedback_note=note,
        )
