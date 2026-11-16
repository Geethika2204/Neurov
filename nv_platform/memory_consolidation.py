"""
/nv_platform/memory_consolidation.py

Memory Consolidation — Feedback Loop
======================================
Closes the feedback loop: Action → Feedback → Memory Consolidation → ImplicitLayer.

Architecture position
---------------------
    Action ──► Feedback ──► MemoryConsolidation ──► ImplicitLayer

After the system executes an action and receives feedback (ground truth label,
outcome quality, etc.), Memory Consolidation:

  1. Logs the outcome against the decision that produced it.
  2. Updates the ImplicitLayer's RiskHead weights based on which signals
     correctly predicted the outcome.
  3. Stores high-value episodes (actual crashes) in a long-term memory
     so the EpisodicMemory can retrieve them later.
  4. Returns a consolidation summary for observability.

Interface
---------
    consolidator = MemoryConsolidation(implicit_layer)
    summary = consolidator.consolidate(feedback)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

from nv_platform.implicit_layer import ImplicitPredictiveLayer, ReasoningScore

if TYPE_CHECKING:
    from nv_platform.action_memory import ActionMemory


# ---------------------------------------------------------------------------
# Feedback dataclass — input to consolidation
# ---------------------------------------------------------------------------

@dataclass
class Feedback:
    """
    Outcome feedback for a completed action.

    Attributes
    ----------
    decision_id   : UUID of the gate decision that triggered the action.
    actual_label  : Ground-truth label (0=normal, 1=crash/anomaly).
    predicted_label : What the system predicted ("crash", "normal", etc.).
    score         : The ReasoningScore at the time of the decision.
    spike_vec     : Optional — the mean-pooled spike vector, used to update
                    long-term episodic memory for confirmed crashes.
    outcome_quality : Float in [0,1] — how well the action worked (optional).
    verified      : bool — whether the outcome has been independently verified
                    (ground-truth label or controlled simulation).  Only
                    verified feedback drives learning; unverified feedback is
                    recorded to Action Memory as pending but never touches the
                    RiskHead weights or episodic store.  Defaults to True so
                    existing callers (which always supply verified feedback)
                    keep their prior behaviour.
    actions_taken : list[str] — steps actually executed (for action-effectiveness
                    evaluation).  Optional.
    """
    decision_id: str = ""
    actual_label: int = 0
    predicted_label: str = "normal"
    score: ReasoningScore = field(default_factory=ReasoningScore)
    spike_vec: torch.Tensor | None = None
    outcome_quality: float = 1.0
    verified: bool = True
    actions_taken: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Consolidation summary — output
# ---------------------------------------------------------------------------

@dataclass
class ConsolidationSummary:
    """
    Summary of one consolidation step.

    Attributes
    ----------
    weights_updated   : Whether RiskHead weights were updated.
    episode_stored    : Whether a new long-term episode was stored.
    new_risk_weights  : Updated RiskHead weights (if changed).
    tp / fp / fn / tn : Running confusion matrix counts.
    recorded_pending  : True when an *unverified* feedback was recorded to
                        Action Memory as pending (no learning happened).
    action_memory_confirmed : True when a verified outcome was confirmed in
                              Action Memory.
    """
    weights_updated: bool = False
    episode_stored: bool = False
    new_risk_weights: list[float] = field(default_factory=list)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    recorded_pending: bool = False
    action_memory_confirmed: bool = False

    def __repr__(self) -> str:
        return (
            f"ConsolidationSummary(weights_updated={self.weights_updated}, "
            f"episode_stored={self.episode_stored}, "
            f"tp={self.tp}, fp={self.fp}, fn={self.fn}, tn={self.tn}, "
            f"recorded_pending={self.recorded_pending}, "
            f"action_memory_confirmed={self.action_memory_confirmed})"
        )


# ---------------------------------------------------------------------------
# Memory Consolidation
# ---------------------------------------------------------------------------

# How many recent feedbacks to accumulate before reweighting
_UPDATE_EVERY = 20
# Minimum confirmed crashes needed before weight update
_MIN_CRASHES_FOR_UPDATE = 3


class MemoryConsolidation:
    """
    Closes the feedback loop by updating the ImplicitPredictiveLayer
    from action outcomes.

    Parameters
    ----------
    implicit_layer  : ImplicitPredictiveLayer  The layer to update.
    update_every    : int  Consolidate after this many feedbacks (default 20).
    fn_weight       : float  How much to amplify signals that missed crashes (default 2.0).
    action_memory   : ActionMemory | None  When set, verified outcomes are
                      confirmed in Action Memory so they can inform future
                      decisions; unverified feedback is recorded as pending.

    Verified vs pending
    -------------------
    Only feedback with ``verified=True`` reaches the RiskHead weight update
    and the episodic store.  Unverified feedback is recorded to Action Memory
    as pending (if attached) and otherwise ignored — it never influences
    learning.  This is the invariant "only verified outcomes should reach
    MemoryConsolidation and confirmed episodic/action memory."
    """

    def __init__(
        self,
        implicit_layer: ImplicitPredictiveLayer,
        update_every: int = _UPDATE_EVERY,
        fn_weight: float = 2.0,
        action_memory: "ActionMemory | None" = None,
    ) -> None:
        self._layer = implicit_layer
        self._update_every = update_every
        self._fn_weight = fn_weight
        self._memory = action_memory
        self._buffer: list[Feedback] = []

        # Running confusion matrix
        self._tp = self._fp = self._fn = self._tn = 0

    @property
    def action_memory(self) -> "ActionMemory | None":
        return self._memory

    @action_memory.setter
    def action_memory(self, value: "ActionMemory | None") -> None:
        self._memory = value

    def consolidate(self, feedback: Feedback) -> ConsolidationSummary:
        """
        Process one feedback event.

        Verified feedback drives learning (confusion matrix, episodic
        store, RiskHead weight update) and is confirmed in Action Memory.
        Unverified feedback is recorded to Action Memory as pending (when
        attached) and skipped for learning.
        """
        # --- Unverified path: record/track as pending, do NOT learn ---
        if not feedback.verified:
            recorded_pending = False
            if self._memory is not None and feedback.decision_id:
                from nv_platform.action_memory import DecisionRecord
                # If the decision was already recorded (e.g. by the workflow),
                # it already sits as pending — we just acknowledge that it is
                # tracked without being learned from.  Otherwise create a
                # pending record so the outcome is preserved until an
                # independent verifier promotes it to confirmed.
                if self._memory.get(feedback.decision_id) is None:
                    self._memory.record_decision(
                        DecisionRecord(
                            decision_id=feedback.decision_id,
                            score=feedback.score,
                            score_tensor=feedback.score.as_tensor(),
                            predicted_label=feedback.predicted_label,
                            actual_label=None,
                            outcome_status="pending",
                            actions_taken=list(feedback.actions_taken),
                        )
                    )
                recorded_pending = True
            return ConsolidationSummary(
                tp=self._tp, fp=self._fp, fn=self._fn, tn=self._tn,
                recorded_pending=recorded_pending,
                action_memory_confirmed=False,
            )

        # --- Verified path: this outcome is allowed to influence learning ---
        self._buffer.append(feedback)
        self._update_confusion(feedback)

        weights_updated = False
        episode_stored  = False
        new_weights: list[float] = []
        action_memory_confirmed = False

        # Confirm the verified outcome in Action Memory (decision → confirmed).
        if self._memory is not None and feedback.decision_id:
            confirmed = self._memory.confirm_outcome(
                decision_id=feedback.decision_id,
                actual_label=feedback.actual_label,
                verified_success=self._effective(feedback),
                feedback_note=f"consolidated verified outcome (label={feedback.actual_label})",
            )
            action_memory_confirmed = confirmed
            # Attach the executed steps so action-effectiveness can be scored.
            if confirmed and feedback.actions_taken:
                rec = self._memory.get(feedback.decision_id)
                if rec is not None and not rec.actions_taken:
                    self._memory.record_action(
                        feedback.decision_id,
                        list(feedback.actions_taken),
                        "executed",
                    )

        # Store confirmed crash episodes in episodic memory
        if feedback.actual_label == 1 and feedback.spike_vec is not None:
            self._layer._memory.store(feedback.spike_vec)
            episode_stored = True

        # Periodic weight update
        if len(self._buffer) >= self._update_every:
            new_weights_tensor = self._reweight()
            if new_weights_tensor is not None:
                self._layer._risk.update_weights(new_weights_tensor)
                new_weights = new_weights_tensor.tolist()
                weights_updated = True
            self._buffer.clear()

        return ConsolidationSummary(
            weights_updated=weights_updated,
            episode_stored=episode_stored,
            new_risk_weights=new_weights,
            tp=self._tp, fp=self._fp,
            fn=self._fn, tn=self._tn,
            action_memory_confirmed=action_memory_confirmed,
        )

    @staticmethod
    def _effective(feedback: Feedback) -> bool:
        """
        Domain-agnostic effectiveness: the prediction agreed with the truth.
        Uses the *actual* label (independent of the model) plus the model's
        own predicted_label only to judge agreement — never as a substitute
        for the ground truth.
        """
        predicted_positive = feedback.predicted_label in (
            "crash", "anomaly", "alert", "inner_race", "outer_race", "ball",
        )
        actual_positive = feedback.actual_label == 1
        return predicted_positive == actual_positive

    def _update_confusion(self, fb: Feedback) -> None:
        predicted_positive = fb.predicted_label in ("crash", "anomaly", "alert")
        actual_positive    = fb.actual_label == 1
        if predicted_positive and actual_positive:
            self._tp += 1
        elif predicted_positive and not actual_positive:
            self._fp += 1
        elif not predicted_positive and actual_positive:
            self._fn += 1
        else:
            self._tn += 1

    def _reweight(self) -> torch.Tensor | None:
        """
        Compute new RiskHead signal weights from buffered feedback.

        Strategy: for each false negative (missed crash), identify which
        signals were LOW when they should have been HIGH. Amplify those
        signals' weights. For false positives, slightly reduce weights
        of signals that were erroneously high.
        """
        crashes  = [fb for fb in self._buffer if fb.actual_label == 1]
        normals  = [fb for fb in self._buffer if fb.actual_label == 0]

        if len(crashes) < _MIN_CRASHES_FOR_UPDATE:
            return None  # not enough crash data to reweight

        # Mean signal values for crashes vs normals
        def mean_signals(items: list[Feedback]) -> torch.Tensor:
            if not items:
                return torch.zeros(5)
            tensors = []
            for fb in items:
                s = fb.score
                tensors.append(torch.tensor([
                    s.novelty, s.uncertainty, s.prediction_error,
                    1.0 - s.memory_confidence, s.spatial_correlation
                ]))
            return torch.stack(tensors).mean(0)

        crash_signals  = mean_signals(crashes)
        normal_signals = mean_signals(normals) if normals else torch.zeros(5)

        # Discriminability: how much higher is each signal during crashes?
        discriminability = (crash_signals - normal_signals).clamp(min=0.0)

        # New weights proportional to discriminability + regularisation toward prior
        prior = self._layer._risk.DEFAULT_WEIGHTS
        alpha = 0.7  # blend factor — how much to trust new evidence
        new_w = alpha * discriminability + (1 - alpha) * prior
        new_w = new_w / (new_w.sum() + 1e-9)  # normalise

        # Apply FN amplification: if we missed crashes, boost all crash signals
        fn_in_buffer = sum(
            1 for fb in self._buffer
            if fb.actual_label == 1
            and fb.predicted_label not in ("crash", "anomaly", "alert")
        )
        if fn_in_buffer > 0:
            fn_ratio = fn_in_buffer / max(len(crashes), 1)
            boost = 1.0 + self._fn_weight * fn_ratio
            new_w = (new_w * boost).clamp(0.0, 1.0)
            new_w = new_w / (new_w.sum() + 1e-9)

        return new_w
