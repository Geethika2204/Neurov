"""NeuroVerse platform — domain-agnostic four-layer plugin architecture."""
from nv_platform.domain_plugin import DomainPlugin
from nv_platform.snn_plugin import SNNInputPlugin
from nv_platform.implicit_layer import ImplicitPredictiveLayer, ReasoningScore
from nv_platform.gate import ReasoningGate, EscalationHistory
from nv_platform.gate_learner import GateLearner
from nv_platform.agent_plugin import LLMAgentPlugin, AgentDecision
from nv_platform.automatic_action import AutomaticAction, AutomaticActionResult
from nv_platform.safety import SafetyChecker, SafetyResult
from nv_platform.planner import Planner, ActionPlan
from nv_platform.memory_consolidation import MemoryConsolidation, Feedback, ConsolidationSummary

__all__ = [
    "DomainPlugin",
    "SNNInputPlugin",
    "ImplicitPredictiveLayer",
    "ReasoningScore",
    "ReasoningGate",
    "EscalationHistory",
    "GateLearner",
    "LLMAgentPlugin",
    "AgentDecision",
    "AutomaticAction",
    "AutomaticActionResult",
    "SafetyChecker",
    "SafetyResult",
    "Planner",
    "ActionPlan",
    "MemoryConsolidation",
    "Feedback",
    "ConsolidationSummary",
]
