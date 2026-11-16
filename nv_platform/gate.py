"""
/platform/gate.py

Layer 3 — Adaptive Reasoning Gate
====================================
Decides whether to escalate from fast SNN processing to the LLM agent.

Interface
---------
    learner = GateLearner()
    gate    = ReasoningGate(learner=learner)
    history = EscalationHistory()

    escalate, confidence = gate.should_escalate(reasoning_score, history)

    # Later, when the true label is known:
    gate.log_outcome(decision_id, actual_label)

Design
------
The gate combines two signals:
  1. A rule-based floor: always escalate if any single sub-signal is above
     ``HARD_ESCALATE_THRESHOLD`` (default 0.95).  This prevents missed crashes
     in the early period before the learner has enough data.
  2. The ``GateLearner`` predicted probability.  The gate escalates when
     this probability exceeds a tunable ``threshold`` (default 0.5).

Every decision is logged to ``EscalationHistory`` with a UUID so the
outcome can be attached later via ``log_outcome()``.

False-negative protection
-------------------------
``should_escalate`` applies the ``fn_penalty`` from ``GateLearner`` implicitly
through the learned boundary.  In addition, the hard floor ensures the gate
never silently ignores an extremely high-signal event.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import torch
from torch import Tensor

from nv_platform.implicit_layer import ReasoningScore
from nv_platform.gate_learner import GateLearner
from nv_platform.automatic_action import AutomaticAction, AutomaticActionResult


# ---------------------------------------------------------------------------
# EscalationHistory
# ---------------------------------------------------------------------------


@dataclass
class EscalationHistory:
    """
    Rolling log of escalation decisions (last ``maxlen`` entries).

    Each entry is a dict::

        {
            "id":           str,           # uuid4
            "score_tensor": Tensor,        # shape (4,)
            "decision":     bool,          # True = escalated
            "confidence":   float,
            "timestamp":    str,           # ISO-8601
            "outcome":      int | None,    # 0 / 1 once known; None until then
        }
    """

    maxlen: int = 500
    log: deque = field(default_factory=lambda: deque(maxlen=500))

    def __post_init__(self) -> None:
        self.log = deque(maxlen=self.maxlen)

    def append(self, entry: dict) -> None:
        self.log.append(entry)

    def set_outcome(self, decision_id: str, outcome: int) -> bool:
        """
        Attach the true label to a previously logged decision.

        Returns ``True`` if the entry was found, ``False`` otherwise.
        """
        for entry in self.log:
            if entry["id"] == decision_id:
                entry["outcome"] = outcome
                return True
        return False

    def __len__(self) -> int:
        return len(self.log)


# ---------------------------------------------------------------------------
# Reasoning Gate
# ---------------------------------------------------------------------------


# Hard escalation floor — override learner if any signal this extreme
HARD_ESCALATE_THRESHOLD = 0.95


class ReasoningGate:
    """
    Layer 3 of the NeuroVerse pipeline.

    Parameters
    ----------
    learner   : GateLearner   The meta-learner that sets the decision boundary.
    threshold : float         Probability threshold for escalation (default 0.5).

    Examples
    --------
    >>> learner = GateLearner()
    >>> gate    = ReasoningGate(learner=learner)
    >>> history = EscalationHistory()
    >>> score   = ReasoningScore(novelty=0.9, uncertainty=0.8,
    ...                          prediction_error=0.7, memory_confidence=0.1)
    >>> escalate, conf = gate.should_escalate(score, history)
    """

    def __init__(
        self,
        learner: GateLearner,
        threshold: float = 0.5,
        automatic_action: AutomaticAction | None = None,
        training_mode: bool = False,
    ) -> None:
        self._learner = learner
        self._threshold = threshold
        self._auto = automatic_action or AutomaticAction()
        # When True, bypass the hard floor so the learner's own decisions
        # populate the history during offline pre-training (train_gate.py).
        # The hard floor must NOT override during training — if it does,
        # every entry in history has decision=True regardless of the learner,
        # and the learner trains on corrupted labels.
        self.training_mode = training_mode

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        self._threshold = value

    def should_escalate(
        self,
        score: ReasoningScore,
        history: EscalationHistory,
        extra_features: "Tensor | None" = None,
    ) -> tuple[bool, float]:
        """
        Decide whether to escalate to the LLM agent.

        Parameters
        ----------
        score           : ReasoningScore
        history         : EscalationHistory
        extra_features  : Optional Tensor of additional raw features (e.g.
                          de_rms, de_kurtosis) to append to the 5-dim
                          ImplicitLayer vector before the learner scores them.
                          Must match the extra dims the checkpoint was trained
                          with.  None = use 5-feature baseline (back-compat).

        Returns
        -------
        (escalate: bool, confidence: float)
            ``confidence`` is the learner's predicted escalation probability.
        """
        score_tensor = score.as_tensor()

        # --- Hard escalation floor (FN protection) ---
        if self.training_mode:
            hard_escalate = False
        else:
            _danger = torch.stack([
                score_tensor[0],  # novelty
                score_tensor[1],  # uncertainty
                score_tensor[2],  # prediction_error
                score_tensor[4],  # spatial_correlation
                score_tensor[5],  # risk
            ])
            hard_escalate = bool((_danger > HARD_ESCALATE_THRESHOLD).any())

        # --- Learned probability ---
        prob = self._learner.predict_proba(
            score_tensor, history, extra_features=extra_features
        )
        escalate = hard_escalate or (prob >= self._threshold)

        # --- Log the decision ---
        decision_id = str(uuid.uuid4())
        entry: dict = {
            "id": decision_id,
            "score_tensor": score_tensor,
            "decision": escalate,
            "confidence": prob,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome": None,
        }
        history.append(entry)

        # --- Trigger learner update on schedule ---
        n = len(history)
        if n > 0 and n % self._learner.update_every == 0:
            self._learner.update(history)

        return escalate, prob

    def log_outcome(
        self,
        decision_id: str,
        actual_label: int,
        history: EscalationHistory,
    ) -> bool:
        """
        Attach the true outcome of a past decision so the learner can use it.

        Parameters
        ----------
        decision_id  : str   UUID returned (implicitly stored) by the last call.
        actual_label : int   Ground-truth label (0 = normal, 1 = anomaly/crash).
        history      : EscalationHistory

        Returns
        -------
        bool — True if the entry was found and updated.
        """
        return history.set_outcome(decision_id, actual_label)

    def fast_action(
        self,
        score: ReasoningScore,
        event: dict | None = None,
        context: dict | None = None,
    ) -> AutomaticActionResult:
        """
        Produce an AutomaticActionResult for the low-need path.

        Call this when ``should_escalate`` returns False.

        Parameters
        ----------
        score   : ReasoningScore
        event   : dict  Optional raw event.
        context : dict  Optional context.

        Returns
        -------
        AutomaticActionResult
        """
        return self._auto.act(score, event, context)
