"""
/nv_platform/agent_plugin.py

Layer 4 — LLM Agent Plugin
============================
When the gate escalates an event, this plugin invokes an LLM agent to produce
a structured decision with reasoning and cited sources.

Interface
---------
    agent = LLMAgentPlugin(model="claude-3-5-sonnet-20241022")
    decision = agent.reason(event, context, rag_source)

    event      : dict  — raw sensor event from the domain
    context    : dict  — any structured context (gate score, domain hints, …)
    rag_source : str   — path to the ChromaDB persist directory

Design
------
Uses a LangGraph single-agent ReAct loop with two tools:
  1. ``rag_retrieve``    — ChromaDB similarity search (top-k=5 chunks)
  2. ``lookup_context``  — pulls a key from the ``context`` dict

The agent loop is capped at ``max_tool_calls`` iterations (default 3).

Graceful degradation
--------------------
Any exception (API error, timeout, missing ChromaDB collection) returns a safe
``AgentDecision`` with ``label="uncertain"`` and ``confidence=0.0`` rather than
crashing the pipeline.

Model agnosticism
-----------------
The model name is injected at construction.  Any LangChain-compatible chat model
can be substituted — the graph code is identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentDecision:
    """
    Structured decision returned by the LLM agent.

    Attributes
    ----------
    label      : str    Predicted category (e.g. "crash", "normal", "uncertain").
    confidence : float  Agent's self-reported confidence ∈ [0, 1].
    reasoning  : str    Free-text reasoning chain from the agent.
    sources    : list   Document IDs / chunks cited from RAG retrieval.
    """

    label: str = "uncertain"
    confidence: float = 0.0
    reasoning: str = ""
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "sources": self.sources,
        }

    def __repr__(self) -> str:
        return (
            f"AgentDecision(label={self.label!r}, "
            f"confidence={self.confidence:.2f}, "
            f"sources={len(self.sources)} docs)"
        )


# ---------------------------------------------------------------------------
# Fallback decision (returned on any LLM failure)
# ---------------------------------------------------------------------------

def _make_fallback(reason: str = "LLM unavailable — pipeline degraded gracefully.") -> AgentDecision:
    """Construct a safe fallback AgentDecision with a descriptive reason."""
    return AgentDecision(
        label="uncertain",
        confidence=0.0,
        reasoning=reason,
        sources=[],
    )


# ---------------------------------------------------------------------------
# Lazy imports — avoid hard-failing at import time if deps not installed
# ---------------------------------------------------------------------------


def _import_langgraph():
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.prebuilt import ToolNode
        return StateGraph, END, ToolNode
    except ImportError as exc:
        raise ImportError(
            "langgraph is required for the LLM agent plugin. "
            "Install it with: pip install langgraph"
        ) from exc


def _import_langchain_anthropic(model_name: str):
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, timeout=30)
    except ImportError as exc:
        raise ImportError(
            "langchain-anthropic is required. "
            "Install it with: pip install langchain-anthropic"
        ) from exc


def _import_chromadb():
    try:
        import chromadb
        return chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required. Install it with: pip install chromadb"
        ) from exc


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _make_rag_tool(rag_source: str, k: int = 5):
    """Build a RAG retrieval tool bound to the given ChromaDB path."""

    def rag_retrieve(query: str) -> str:
        """Retrieve relevant knowledge-base passages for the given query."""
        try:
            chromadb = _import_chromadb()
            client = chromadb.PersistentClient(path=rag_source)
            collections = client.list_collections()
            if not collections:
                return "No knowledge base available."
            collection = client.get_collection(collections[0].name)
            results = collection.query(
                query_texts=[query],
                n_results=min(k, collection.count()),
            )
            docs = results.get("documents", [[]])[0]
            ids = results.get("ids", [[]])[0]
            if not docs:
                return "No relevant documents found."
            return "\n\n".join(
                f"[{doc_id}] {doc}" for doc_id, doc in zip(ids, docs)
            )
        except Exception as exc:  # noqa: BLE001
            return f"RAG retrieval failed: {exc}"

    return rag_retrieve


def _make_context_tool(context: dict):
    """Build a context lookup tool bound to the given context dict."""

    def lookup_context(key: str) -> str:
        """Look up a value from the structured event context by key."""
        val = context.get(key)
        if val is None:
            available = list(context.keys())
            return f"Key {key!r} not found. Available keys: {available}"
        return str(val)

    return lookup_context


# ---------------------------------------------------------------------------
# Main plugin
# ---------------------------------------------------------------------------


class LLMAgentPlugin:
    """
    Layer 4 of the NeuroVerse pipeline.

    Parameters
    ----------
    model          : str   LangChain-compatible model name.
    max_tool_calls : int   Maximum ReAct tool calls per invocation (default 3).

    Examples
    --------
    >>> agent = LLMAgentPlugin()
    >>> decision = agent.reason(event, context, "domains/traffic/rag_kb")
    """

    SYSTEM_PROMPT = (
        "You are an anomaly detection agent for a sensor monitoring system. "
        "You will receive a sensor event and may call tools to retrieve "
        "relevant context from a knowledge base. "
        "Always respond with a JSON object with keys: "
        "label (string), confidence (0-1 float), reasoning (string). "
        "Be conservative: when uncertain, prefer to flag as anomalous "
        "rather than dismiss."
    )

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        max_tool_calls: int = 3,
    ) -> None:
        self._model_name = model
        self._max_tool_calls = max_tool_calls

    @property
    def model_name(self) -> str:
        return self._model_name

    def reason(
        self,
        event: dict[str, Any],
        context: dict[str, Any],
        rag_source: str,
    ) -> AgentDecision:
        """
        Invoke the LLM agent on an escalated event.

        Parameters
        ----------
        event      : dict  Raw sensor event.
        context    : dict  Structured context (gate score, domain hints, …).
        rag_source : str   Path to ChromaDB persist directory.

        Returns
        -------
        AgentDecision — always returns (never raises).
        """
        try:
            return self._run_agent(event, context, rag_source)
        except Exception as exc:  # noqa: BLE001
            return _make_fallback(f"LLM unavailable: {exc}")

    def _run_agent(
        self,
        event: dict[str, Any],
        context: dict[str, Any],
        rag_source: str,
    ) -> AgentDecision:
        """Core agent execution — may raise; wrapped by ``reason``."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.tools import tool as lc_tool

        # Build tools
        rag_fn = _make_rag_tool(rag_source)
        ctx_fn = _make_context_tool(context)

        # Wrap as LangChain tools
        rag_tool = lc_tool(rag_fn)
        ctx_tool = lc_tool(ctx_fn)
        tools = [rag_tool, ctx_tool]

        # Build LLM
        llm = _import_langchain_anthropic(self._model_name)
        llm_with_tools = llm.bind_tools(tools)

        # Run a simple ReAct loop (no full LangGraph graph needed for ≤3 calls)
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Sensor event: {json.dumps(event)}\n"
                    f"Context: {json.dumps({k: str(v) for k, v in context.items()})}\n"
                    "Classify this event and explain your reasoning."
                )
            ),
        ]

        sources: list[str] = []
        tool_calls_made = 0

        while tool_calls_made < self._max_tool_calls:
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            # Execute tool calls
            from langchain_core.messages import ToolMessage

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                if tool_name == "rag_retrieve":
                    result = rag_fn(**tool_args)
                    # Extract cited doc IDs from result
                    for line in result.split("\n"):
                        if line.startswith("["):
                            doc_id = line.split("]")[0].lstrip("[")
                            sources.append(doc_id)
                elif tool_name == "lookup_context":
                    result = ctx_fn(**tool_args)
                else:
                    result = f"Unknown tool: {tool_name}"

                messages.append(
                    ToolMessage(content=result, tool_call_id=tool_id)
                )
                tool_calls_made += 1

        # Parse final response
        final_content = messages[-1].content if messages else ""
        return self._parse_response(final_content, sources)

    def _parse_response(
        self, content: str, sources: list[str]
    ) -> AgentDecision:
        """
        Extract structured fields from the agent's text response.

        Tries JSON first; falls back to heuristic parsing.
        """
        if not content:
            return AgentDecision(
                label="uncertain", confidence=0.0,
                reasoning="Empty response", sources=sources
            )

        # Try JSON block
        try:
            # Look for JSON object in the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(content[start:end])
                return AgentDecision(
                    label=str(parsed.get("label", "uncertain")),
                    confidence=float(parsed.get("confidence", 0.5)),
                    reasoning=str(parsed.get("reasoning", content)),
                    sources=sources,
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Heuristic: look for label keywords
        lower = content.lower()
        # Check for explicit normal/clear phrases first (higher specificity)
        normal_phrases = ("no incident", "no crash", "traffic is clear",
                          "normal", "all clear", "no anomaly")
        anomaly_phrases = ("crash", "anomaly", "alert")
        # 'incident' alone is checked only when not negated
        has_negated_incident = "no incident" in lower
        has_incident = "incident" in lower and not has_negated_incident

        if any(p in lower for p in normal_phrases):
            label = "normal"
            confidence = 0.7
        elif any(p in lower for p in anomaly_phrases) or has_incident:
            label = "anomaly"
            confidence = 0.7
        else:
            label = "uncertain"
            confidence = 0.4

        return AgentDecision(
            label=label,
            confidence=confidence,
            reasoning=content[:500],
            sources=sources,
        )
