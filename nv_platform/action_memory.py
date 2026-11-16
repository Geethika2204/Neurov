"""
/nv_platform/action_memory.py

Action Memory — long-term record of decisions, actions and *verified* outcomes
================================================================================
Closes the experience half of the feedback loop.  It is the single source of
truth for "what did we decide, what did we do, and what actually happened".

Position in the architecture
----------------------------
    Decision (gate + agent/automatic-action)
        │  record_decision()        → pending
        ▼
    Action execution
        │  record_action()          → pending (execution status)
        ▼
    Outcome verification            (ground-truth label OR controlled simulation)
        │  confirm_outcome()        → confirmed
        ▼
    MemoryConsolidation             (only confirmed outcomes reach here)

Why a separate memory
---------------------
The ImplicitLayer's ``_EpisodicMemory`` stores *spike vectors* to score
novelty/confidence at signal-generation time.  Action Memory stores
*decisions and outcomes* to improve routing and action selection at
decision time.  They are different stores for different purposes and must
not be merged (merging would duplicate memory systems, which the spec
forbids).

Core invariant
--------------
Unverified (pending) records are stored but never used to inform future
decisions or learning.  Only confirmed records participate in retrieval
for decision improvement and in continuous learning.

Domain independence
-------------------
Action Memory is fully domain-agnostic: it keys on the 6-dim
``ReasoningScore`` tensor and generic label strings.  It never imports from
``/domains/``.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import torch
from torch import Tensor

from nv_platform.implicit_layer import ReasoningScore


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """
    A single decision and everything attached to it over its lifecycle.

    Lifecycle
    ---------
    1. ``record_decision``      → ``outcome_status == "pending"``,
                                  ``execution_status == "pending"``.
    2. ``record_action``        → ``execution_status`` → "executed"/"failed".
    3. ``confirm_outcome``      → ``outcome_status == "confirmed"``,
                                  ``actual_label`` / ``verified_success`` set.

    Attributes
    ----------
    decision_id       : str   Unique id (matches the gate's history entry id).
    timestamp        : str   ISO-8601 of when the decision was made.
    score            : ReasoningScore  Signals at decision time.
    score_tensor     : Tensor (6,)    Score used for similarity retrieval.
    predicted_label  : str   What the system predicted.
    escalated        : bool  Whether the gate escalated (Path B vs Path A).
    gate_confidence  : float Gate's escalation probability.
    path             : str   "automatic" (fast) or "deliberative" (LLM).
    action_plan      : dict  Structured plan (``ActionPlan.to_dict()``).
    actions_taken    : list  Steps actually executed.
    execution_status : str   "pending" | "executed" | "failed" | "skipped".
    outcome_status   : str   "pending" | "confirmed".
    actual_label     : int | None   Ground-truth (0 normal, 1 crash/anomaly).
    verified_success : bool | None Was the recommendation effective?
    feedback_note    : str | None  Free-text operator/verifier note.
    """

    decision_id: str = ""
    timestamp: str = ""
    score: ReasoningScore = field(default_factory=ReasoningScore)
    score_tensor: Optional[Tensor] = None
    predicted_label: str = "uncertain"
    escalated: bool = False
    gate_confidence: float = 0.0
    path: str = "automatic"
    action_plan: dict[str, Any] = field(default_factory=dict)
    actions_taken: list[str] = field(default_factory=list)
    execution_status: str = "pending"
    outcome_status: str = "pending"
    actual_label: Optional[int] = None
    verified_success: Optional[bool] = None
    feedback_note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "predicted_label": self.predicted_label,
            "escalated": self.escalated,
            "gate_confidence": self.gate_confidence,
            "path": self.path,
            "action_plan": self.action_plan,
            "actions_taken": list(self.actions_taken),
            "execution_status": self.execution_status,
            "outcome_status": self.outcome_status,
            "actual_label": self.actual_label,
            "verified_success": self.verified_success,
            "feedback_note": self.feedback_note,
        }

    def is_crash(self) -> bool:
        """True if the *verified* actual outcome was a crash/anomaly."""
        return self.outcome_status == "confirmed" and self.actual_label == 1

    def is_normal(self) -> bool:
        """True if the *verified* actual outcome was normal."""
        return self.outcome_status == "confirmed" and self.actual_label == 0


# ---------------------------------------------------------------------------
# Match result returned to callers (gate / automatic-action)
# ---------------------------------------------------------------------------


@dataclass
class MemoryMatch:
    """
    A reliable confirmed experience similar to the current score.

    Attributes
    ----------
    record     : DecisionRecord  The matched confirmed record.
    similarity : float           Cosine similarity to the query score ∈ [0, 1].
    was_crash  : bool            Whether the matched event was a confirmed crash.
    was_normal : bool            Whether the matched event was confirmed normal.
    action     : str             The action that was recommended last time.
    effective  : bool | None     Whether that recommendation was verified effective.
    """
    record: DecisionRecord
    similarity: float = 0.0
    was_crash: bool = False
    was_normal: bool = False
    action: str = ""
    effective: Optional[bool] = None


# ---------------------------------------------------------------------------
# Action Memory
# ---------------------------------------------------------------------------


# Default reliability thresholds — a match must be this similar to be trusted.
# Kept deliberately high so memory only nudges decisions when the evidence is
# genuinely comparable.
DEFAULT_MIN_SIMILARITY = 0.92


class ActionMemory:
    """
    Store and retrieve decisions, actions and *verified* outcomes.

    Parameters
    ----------
    min_similarity : float  Cosine similarity required for a "reliable" match
                            (default 0.92).
    max_size       : int    Soft cap on confirmed records (LRU eviction when
                            exceeded).  Pending records are never evicted until
                            confirmed or pruned.

    Examples
    --------
    >>> mem = ActionMemory()
    >>> rec = DecisionRecord(decision_id="d1", score_tensor=torch.zeros(6))
    >>> mem.record_decision(rec)
    >>> mem.confirm_outcome("d1", actual_label=1, verified_success=True)
    True
    >>> match = mem.reliable_match(ReasoningScore())   # no similar → None
    """

    def __init__(
        self,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        max_size: int = 10_000,
    ) -> None:
        self.min_similarity = float(min_similarity)
        self.max_size = int(max_size)
        self._pending: dict[str, DecisionRecord] = {}
        self._confirmed: dict[str, DecisionRecord] = {}
        # Insertion order for LRU eviction of confirmed records.
        self._confirmed_order: list[str] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_decision(self, record: DecisionRecord) -> str:
        """
        Log a freshly-made decision as pending.

        Assigns a ``decision_id`` if none is set and returns it.
        """
        if not record.decision_id:
            record.decision_id = str(uuid.uuid4())
        if not record.timestamp:
            record.timestamp = datetime.now(timezone.utc).isoformat()
        self._pending[record.decision_id] = record
        return record.decision_id

    def record_action(
        self,
        decision_id: str,
        actions_taken: list[str],
        execution_status: str,
    ) -> bool:
        """
        Attach the executed steps / execution status to a pending decision.

        Returns ``True`` if the decision was found, ``False`` otherwise.
        """
        rec = self._pending.get(decision_id) or self._confirmed.get(decision_id)
        if rec is None:
            return False
        rec.actions_taken = list(actions_taken)
        rec.execution_status = execution_status
        return True

    def confirm_outcome(
        self,
        decision_id: str,
        actual_label: int,
        verified_success: Optional[bool] = None,
        feedback_note: Optional[str] = None,
    ) -> bool:
        """
        Move a pending decision to confirmed with the *verified* outcome.

        This is the ONLY path that promotes a record to confirmed.  Once
        confirmed, the record may inform future decisions and learning.

        Returns ``True`` if the decision was found, ``False`` otherwise.
        """
        rec = self._pending.pop(decision_id, None) or self._confirmed.get(
            decision_id
        )
        if rec is None:
            return False
        rec.actual_label = int(actual_label)
        rec.verified_success = verified_success
        rec.outcome_status = "confirmed"
        if feedback_note is not None:
            rec.feedback_note = feedback_note
        if decision_id not in self._confirmed:
            self._confirmed[decision_id] = rec
            self._confirmed_order.append(decision_id)
            self._evict_if_needed()
        return True

    def _evict_if_needed(self) -> None:
        while len(self._confirmed_order) > self.max_size:
            oldest = self._confirmed_order.pop(0)
            self._confirmed.pop(oldest, None)

    # ------------------------------------------------------------------
    # Retrieval (decision improvement)
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: Tensor, b: Tensor) -> float:
        na = a / (a.norm() + 1e-9)
        nb = b / (b.norm() + 1e-9)
        return float(torch.clamp((na * nb).sum(), -1.0, 1.0))

    def _query_tensor(self, score: ReasoningScore | Tensor) -> Tensor:
        if isinstance(score, Tensor):
            return score.float()
        return score.as_tensor().float()

    def retrieve_similar(
        self,
        score: ReasoningScore | Tensor,
        top_k: int = 5,
        confirmed_only: bool = True,
        min_similarity: float = 0.0,
    ) -> list[tuple[DecisionRecord, float]]:
        """
        Return the most similar stored records.

        Parameters
        ----------
        score          : ReasoningScore or (6,) tensor  Query.
        top_k          : int   Max records to return.
        confirmed_only : bool  If True (default), only confirmed records are
                               returned — pending records never influence
                               decisions.  Set False only for observability.
        min_similarity : float  Floor below which matches are dropped.

        Returns
        -------
        list of (DecisionRecord, similarity) sorted by descending similarity.
        """
        q = self._query_tensor(score)
        pool = self._confirmed if confirmed_only else {**self._pending, **self._confirmed}
        scored: list[tuple[DecisionRecord, float]] = []
        for rec in pool.values():
            if rec.score_tensor is None:
                continue
            sim = self._cosine(q, rec.score_tensor)
            if sim >= min_similarity:
                scored.append((rec, sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def reliable_match(
        self,
        score: ReasoningScore | Tensor,
        min_similarity: Optional[float] = None,
    ) -> Optional[MemoryMatch]:
        """
        Return the closest *confirmed* record if it is reliable enough.

        "Reliable" means: confirmed outcome AND cosine similarity at least
        ``min_similarity`` (defaults to the store's threshold).

        Returns ``None`` when there is no reliable confirmed match, so
        callers can fall back to their default behaviour.  Pending records
        are deliberately never returned here.
        """
        thresh = self.min_similarity if min_similarity is None else float(min_similarity)
        matches = self.retrieve_similar(
            score, top_k=1, confirmed_only=True, min_similarity=thresh
        )
        if not matches:
            return None
        rec, sim = matches[0]
        plan = rec.action_plan or {}
        action = plan.get("action") or (
            plan.get("steps", [""])[0] if plan.get("steps") else ""
        )
        return MemoryMatch(
            record=rec,
            similarity=sim,
            was_crash=rec.is_crash(),
            was_normal=rec.is_normal(),
            action=action,
            effective=rec.verified_success,
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get(self, decision_id: str) -> Optional[DecisionRecord]:
        return self._pending.get(decision_id) or self._confirmed.get(decision_id)

    def pending(self) -> list[DecisionRecord]:
        return list(self._pending.values())

    def confirmed(self) -> list[DecisionRecord]:
        return list(self._confirmed.values())

    # ------------------------------------------------------------------
    # Aggregates for continuous learning
    # ------------------------------------------------------------------

    def confusion_matrix(self) -> dict[str, int]:
        """
        TP/FP/FN/TN over confirmed records, using the *verified* outcome.

        "Predicted positive" = predicted_label in crash/anomaly/alert set.
        """
        tp = fp = fn = tn = 0
        pos = {"crash", "anomaly", "alert", "inner_race", "outer_race", "ball"}
        for rec in self._confirmed.values():
            pred_pos = rec.predicted_label in pos
            actual_pos = rec.actual_label == 1
            if pred_pos and actual_pos:
                tp += 1
            elif pred_pos and not actual_pos:
                fp += 1
            elif not pred_pos and actual_pos:
                fn += 1
            else:
                tn += 1
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def false_negatives(self) -> list[DecisionRecord]:
        """Confirmed crashes the system predicted as normal/uncertain."""
        return [r for r in self._confirmed.values()
                if r.actual_label == 1
                and r.predicted_label not in
                ("crash", "anomaly", "alert", "inner_race", "outer_race", "ball")]

    def false_positives(self) -> list[DecisionRecord]:
        """Confirmed normals the system predicted as a crash/anomaly."""
        return [r for r in self._confirmed.values()
                if r.actual_label == 0
                and r.predicted_label in
                ("crash", "anomaly", "alert", "inner_race", "outer_race", "ball")]

    def confirmed_crash_patterns(
        self, score_tensors: bool = True
    ) -> list[DecisionRecord]:
        """
        Confirmed crash records — source material for EpisodicMemory storage.

        The caller (MemoryConsolidation) is responsible for pushing the
        associated spike vectors into the ImplicitLayer's episodic memory;
        here we only surface the records whose outcome is verified as a crash.
        """
        return [r for r in self._confirmed.values() if r.actual_label == 1]

    def action_effectiveness(self) -> dict[str, dict[str, float]]:
        """
        For each recommended action step, how often the recommendation was
        verified effective among confirmed records that used it.

        Returns
        -------
        {step: {"success_rate": float, "count": int}}
        """
        per_step: dict[str, list[bool]] = {}
        for rec in self._confirmed.values():
            success = rec.verified_success
            if success is None:
                continue
            steps = rec.actions_taken or (rec.action_plan.get("steps") or [])
            for step in steps:
                per_step.setdefault(step, []).append(bool(success))
        out: dict[str, dict[str, float]] = {}
        for step, flags in per_step.items():
            n = len(flags)
            out[step] = {
                "success_rate": round(sum(flags) / max(n, 1), 4),
                "count": n,
            }
        return out

    def effective_actions(self) -> list[str]:
        """Action steps whose verified success rate is >= 0.6 (>= 3 uses)."""
        eff: list[str] = []
        for step, info in self.action_effectiveness().items():
            if info["count"] >= 3 and info["success_rate"] >= 0.6:
                eff.append(step)
        return eff

    def stats(self) -> dict[str, Any]:
        cm = self.confusion_matrix()
        return {
            "pending": len(self._pending),
            "confirmed": len(self._confirmed),
            "tp": cm["tp"],
            "fp": cm["fp"],
            "fn": cm["fn"],
            "tn": cm["tn"],
            "confirmed_crashes": len(self.confirmed_crash_patterns()),
            "false_negatives": len(self.false_negatives()),
            "false_positives": len(self.false_positives()),
        }

    def __len__(self) -> int:
        return len(self._pending) + len(self._confirmed)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"ActionMemory(pending={s['pending']}, confirmed={s['confirmed']}, "
            f"tp={s['tp']}, fp={s['fp']}, fn={s['fn']}, tn={s['tn']})"
        )
