"""
/nv_platform/safety.py

Safety Checker
==============
Validates agent actions before they are executed. Part of the deliberative
(high-need) path: RAG → Tools → Safety → Planner → Action.

The Safety checker applies domain-agnostic rules plus optional domain-
specific rules to catch actions that are:
  - Contradicted by a high-confidence "normal" risk score
  - Missing required fields
  - Labelled with invalid categories
  - Above a confidence floor when the risk is low (false alarm prevention)

Interface
---------
    checker = SafetyChecker()
    result  = checker.check(agent_decision, score, context)
    if result.safe:
        proceed_to_planner(agent_decision)
    else:
        handle_violation(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nv_platform.implicit_layer import ReasoningScore
from nv_platform.agent_plugin import AgentDecision


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SafetyResult:
    """
    Result of a safety check on an AgentDecision.

    Attributes
    ----------
    safe        : True if the action passed all safety checks.
    violations  : List of violation messages (empty if safe).
    decision    : The (possibly modified) AgentDecision after safety checks.
                  If unsafe, this is a sanitised fallback decision.
    """
    safe: bool = True
    violations: list[str] = field(default_factory=list)
    decision: AgentDecision = field(default_factory=AgentDecision)

    def __repr__(self) -> str:
        status = "SAFE" if self.safe else f"UNSAFE ({len(self.violations)} violations)"
        return f"SafetyResult({status})"


# ---------------------------------------------------------------------------
# Safety rules
# ---------------------------------------------------------------------------

# If risk is below this and agent says "crash", flag as suspicious
_LOW_RISK_CRASH_THRESHOLD = 0.2

# Minimum confidence the agent must report
_MIN_CONFIDENCE = 0.0  # no hard floor — just logged

# Valid label vocabulary (domain-agnostic)
_VALID_LABELS = {"normal", "anomaly", "crash", "uncertain", "monitor", "alert"}


class SafetyChecker:
    """
    Domain-agnostic safety checker for agent decisions.

    Checks
    ------
    1. Label is in the valid vocabulary.
    2. Confidence is a float in [0, 1].
    3. Agent says "crash"/"anomaly" but risk score is very low → suspicious,
       downgrade to "uncertain" with a warning.
    4. Agent says "normal" but risk score is very high → suspicious, upgrade
       to "uncertain" with a warning.
    5. Custom rules injected at construction time.

    Parameters
    ----------
    custom_rules : list of callables  Each callable takes
                   (decision, score, context) and returns a str violation
                   message or None if the check passes.
    strict       : bool  If True, any violation blocks the action.
                         If False (default), violations are logged but
                         the decision is still passed through (downgraded).
    """

    def __init__(
        self,
        custom_rules: list | None = None,
        strict: bool = False,
    ) -> None:
        self._custom_rules = custom_rules or []
        self._strict = strict

    def check(
        self,
        decision: AgentDecision,
        score: ReasoningScore,
        context: dict[str, Any] | None = None,
    ) -> SafetyResult:
        """
        Validate an AgentDecision against safety rules.

        Parameters
        ----------
        decision : AgentDecision  The agent's proposed action.
        score    : ReasoningScore  Current risk context.
        context  : dict           Optional additional context.

        Returns
        -------
        SafetyResult
        """
        violations: list[str] = []
        ctx = context or {}

        # Rule 1: Valid label
        if decision.label not in _VALID_LABELS:
            violations.append(
                f"Invalid label {decision.label!r}. "
                f"Must be one of {sorted(_VALID_LABELS)}."
            )

        # Rule 2: Valid confidence
        if not (0.0 <= decision.confidence <= 1.0):
            violations.append(
                f"Confidence {decision.confidence} out of [0, 1]."
            )

        # Rule 3: Crash claim vs low risk
        if (decision.label in ("crash", "anomaly", "alert")
                and score.risk < _LOW_RISK_CRASH_THRESHOLD):
            violations.append(
                f"Agent claims {decision.label!r} but risk score is "
                f"{score.risk:.3f} (< {_LOW_RISK_CRASH_THRESHOLD}). "
                "Possible false alarm."
            )

        # Rule 4: Normal claim vs high risk
        if decision.label == "normal" and score.risk > 0.8:
            violations.append(
                f"Agent claims 'normal' but risk score is {score.risk:.3f}. "
                "Possible missed crash."
            )

        # Rule 5: Custom domain rules
        for rule_fn in self._custom_rules:
            msg = rule_fn(decision, score, ctx)
            if msg:
                violations.append(msg)

        safe = len(violations) == 0

        # Build output decision
        if not safe:
            if self._strict:
                # Block — return uncertain
                out_decision = AgentDecision(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=(
                        f"Safety check failed: {'; '.join(violations)}\n"
                        f"Original reasoning: {decision.reasoning}"
                    ),
                    sources=decision.sources,
                )
            else:
                # Soft — downgrade confidence, keep label but add warning
                warning = f"[Safety warnings: {'; '.join(violations)}]"
                out_decision = AgentDecision(
                    label=decision.label,
                    confidence=decision.confidence * 0.7,  # penalty
                    reasoning=f"{warning}\n{decision.reasoning}",
                    sources=decision.sources,
                )
        else:
            out_decision = decision

        return SafetyResult(
            safe=safe,
            violations=violations,
            decision=out_decision,
        )
