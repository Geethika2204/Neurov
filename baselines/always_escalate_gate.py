"""
/baselines/always_escalate_gate.py

Baseline 2 — Always-escalate gate
====================================
Escalates every single event to the LLM agent regardless of the
``ReasoningScore``.

This provides the upper-bound reference: 100% recall by definition, but
100% LLM call rate.  The learned ``ReasoningGate`` must reduce LLM calls
by ≥30% vs. this baseline while maintaining crash recall >60%.

Usage
-----
    from baselines.always_escalate_gate import AlwaysEscalateGate
    from nv_platform.implicit_layer import ReasoningScore

    gate = AlwaysEscalateGate()
    escalate, conf = gate.should_escalate(score)
    # always returns (True, 1.0)
"""

from __future__ import annotations

from nv_platform.implicit_layer import ReasoningScore


class AlwaysEscalateGate:
    """
    Degenerate gate that always escalates.

    Escalation rate: 100%
    Crash recall:    100%  (assuming the LLM agent is reliable)
    LLM cost:        maximum — every event hits the LLM

    Use this as the upper-bound baseline to measure the benefit of the
    learned ``ReasoningGate``.
    """

    def should_escalate(
        self, score: ReasoningScore
    ) -> tuple[bool, float]:
        """
        Always returns (True, 1.0) regardless of score.

        Parameters
        ----------
        score : ReasoningScore  (ignored)

        Returns
        -------
        (True, 1.0)
        """
        return True, 1.0

    def evaluate(
        self,
        scores: list[ReasoningScore],
        labels: list[int],
    ) -> dict[str, float]:
        """
        Evaluate gate performance on a labelled sequence.

        Since this gate always escalates, escalation_rate = 1.0 and
        recall = 1.0 (assuming perfect downstream classifier).

        Parameters
        ----------
        scores : list[ReasoningScore]  (lengths must match labels)
        labels : list[int]  0 = normal, 1 = anomaly

        Returns
        -------
        dict with keys: escalation_rate, recall, precision, f1
        """
        assert len(scores) == len(labels)
        total = len(labels)
        positives = sum(labels)

        recall = 1.0
        precision = positives / total if total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "escalation_rate": 1.0,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
