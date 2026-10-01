"""
/baselines/always_threshold_gate.py

Baseline 1 — Rule-based threshold gate
========================================
Escalates whenever ANY ``ReasoningScore`` sub-signal exceeds a fixed
threshold.  No learning, no history.

This is the simplest possible gate and provides a lower-bound reference:
it's fast and interpretable but will not adapt to domain statistics.

Usage
-----
    from baselines.always_threshold_gate import AlwaysThresholdGate
    from nv_platform.implicit_layer import ReasoningScore

    gate = AlwaysThresholdGate(threshold=0.5)
    escalate, conf = gate.should_escalate(score)
"""

from __future__ import annotations

from nv_platform.implicit_layer import ReasoningScore


class AlwaysThresholdGate:
    """
    Rule-based gate: escalate if max(sub-signals) >= threshold.

    Parameters
    ----------
    threshold : float   Any sub-signal at or above this triggers escalation.
                        Default 0.5 (half of normalised range).

    This baseline is intentionally stateless — it takes no history and
    does not update.  Compare its escalation rate and recall against
    ``ReasoningGate`` (learned) and ``AlwaysEscalateGate`` (upper bound).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def should_escalate(
        self, score: ReasoningScore
    ) -> tuple[bool, float]:
        """
        Parameters
        ----------
        score : ReasoningScore

        Returns
        -------
        (escalate: bool, confidence: float)
            ``confidence`` is the maximum sub-signal value.
        """
        max_signal = max(
            score.novelty,
            score.uncertainty,
            score.prediction_error,
            1.0 - score.memory_confidence,  # low confidence → high signal
        )
        escalate = max_signal >= self._threshold
        return escalate, float(max_signal)

    def evaluate(
        self,
        scores: list[ReasoningScore],
        labels: list[int],
    ) -> dict[str, float]:
        """
        Evaluate gate performance on a labelled sequence.

        Parameters
        ----------
        scores : list[ReasoningScore]
        labels : list[int]  0 = normal, 1 = anomaly

        Returns
        -------
        dict with keys: escalation_rate, recall, precision, f1
        """
        assert len(scores) == len(labels), "scores and labels must be same length"

        decisions = [self.should_escalate(s)[0] for s in scores]
        tp = sum(1 for d, l in zip(decisions, labels) if d and l == 1)
        fp = sum(1 for d, l in zip(decisions, labels) if d and l == 0)
        fn = sum(1 for d, l in zip(decisions, labels) if not d and l == 1)
        total = len(decisions)
        escalated = sum(decisions)

        recall = tp / (tp + fn + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        return {
            "escalation_rate": escalated / total,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
