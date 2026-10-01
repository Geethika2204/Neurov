"""tests/test_agent_plugin.py — Unit tests for LLMAgentPlugin and AgentDecision."""

import json
import pytest
from unittest.mock import MagicMock, patch
from nv_platform.agent_plugin import LLMAgentPlugin, AgentDecision


# ---------------------------------------------------------------------------
# AgentDecision
# ---------------------------------------------------------------------------


def test_agent_decision_defaults():
    d = AgentDecision()
    assert d.label == "uncertain"
    assert d.confidence == 0.0
    assert d.reasoning == ""
    assert d.sources == []


def test_agent_decision_to_dict():
    d = AgentDecision(label="crash", confidence=0.9, reasoning="speed drop", sources=["doc_0"])
    result = d.to_dict()
    assert result["label"] == "crash"
    assert result["confidence"] == 0.9
    assert result["sources"] == ["doc_0"]


def test_agent_decision_repr():
    d = AgentDecision(label="normal", confidence=0.8, sources=["a", "b"])
    r = repr(d)
    assert "normal" in r
    assert "2 docs" in r


# ---------------------------------------------------------------------------
# LLMAgentPlugin — unit tests with mocked LLM
# ---------------------------------------------------------------------------


class _MockResponse:
    """Mimics a LangChain AIMessage with no tool calls."""

    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []


def _make_plugin():
    return LLMAgentPlugin(model="claude-3-5-sonnet-20241022")


def test_reason_returns_agent_decision_on_mock():
    """With a mocked LLM, reason() returns a populated AgentDecision."""
    plugin = _make_plugin()

    mock_response = _MockResponse(
        content='{"label": "crash", "confidence": 0.85, "reasoning": "Speed dropped 40%"}'
    )

    with patch.object(plugin, "_run_agent") as mock_run:
        mock_run.return_value = AgentDecision(
            label="crash", confidence=0.85,
            reasoning="Speed dropped 40%", sources=[]
        )
        decision = plugin.reason({"speed": 20.0}, {}, "/tmp/fake_kb")

    assert isinstance(decision, AgentDecision)
    assert decision.label == "crash"
    assert decision.confidence == pytest.approx(0.85)


def test_reason_degrades_gracefully_on_exception():
    """Any exception inside _run_agent returns the fallback AgentDecision."""
    plugin = _make_plugin()

    with patch.object(plugin, "_run_agent", side_effect=TimeoutError("timeout")):
        decision = plugin.reason({"speed": 20.0}, {}, "/tmp/fake_kb")

    assert decision.label == "uncertain"
    assert decision.confidence == 0.0
    assert "LLM unavailable" in decision.reasoning


def test_reason_degrades_on_import_error():
    """Missing langchain-anthropic causes graceful fallback, not crash."""
    plugin = _make_plugin()

    with patch.object(plugin, "_run_agent", side_effect=ImportError("no module")):
        decision = plugin.reason({}, {}, "/tmp/kb")

    assert decision.label == "uncertain"


def test_reason_degrades_on_api_error():
    """Simulated API error triggers fallback."""
    plugin = _make_plugin()

    with patch.object(plugin, "_run_agent", side_effect=Exception("API rate limit")):
        decision = plugin.reason({"sensor": 1.0}, {"gate_score": 0.9}, "/tmp/kb")

    assert decision.label == "uncertain"
    assert "API rate limit" in decision.reasoning


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------


def test_parse_response_valid_json():
    plugin = _make_plugin()
    content = '{"label": "normal", "confidence": 0.95, "reasoning": "All clear"}'
    decision = plugin._parse_response(content, sources=[])
    assert decision.label == "normal"
    assert decision.confidence == pytest.approx(0.95)


def test_parse_response_json_in_prose():
    plugin = _make_plugin()
    content = 'Based on the data: {"label": "anomaly", "confidence": 0.7, "reasoning": "speed drop"} end.'
    decision = plugin._parse_response(content, sources=["doc_1"])
    assert decision.label == "anomaly"
    assert decision.sources == ["doc_1"]


def test_parse_response_heuristic_crash():
    plugin = _make_plugin()
    content = "There appears to be a crash event based on speed patterns."
    decision = plugin._parse_response(content, sources=[])
    assert decision.label == "anomaly"


def test_parse_response_heuristic_normal():
    plugin = _make_plugin()
    content = "Traffic is clear and normal, no incidents detected."
    decision = plugin._parse_response(content, sources=[])
    assert decision.label == "normal"


def test_parse_response_empty_content():
    plugin = _make_plugin()
    decision = plugin._parse_response("", sources=[])
    assert decision.label == "uncertain"


# ---------------------------------------------------------------------------
# Plugin properties
# ---------------------------------------------------------------------------


def test_model_name_property():
    plugin = LLMAgentPlugin(model="gpt-4o")
    assert plugin.model_name == "gpt-4o"
