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


def _import_langchain_ollama(model_name: str):
    """Return a ChatOllama LLM bound to the given local model name."""
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=0.1)
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for local Ollama models. "
            "Install it with: pip install langchain-ollama"
        ) from exc


def _build_llm(model_name: str):
    """
    Auto-select the right LangChain backend based on the model name prefix.

    - ``"ollama:<name>"`` or ``"<name>"`` without a slash → ChatOllama
    - Everything else → ChatAnthropic (default)

    Examples::

        _build_llm("ollama:llama3.2:3b")  → ChatOllama(model="llama3.2:3b")
        _build_llm("llama3.2:3b")         → ChatOllama(model="llama3.2:3b")
        _build_llm("claude-3-5-sonnet-…") → ChatAnthropic(model=…)
    """
    # Explicit ollama: prefix
    if model_name.startswith("ollama:"):
        return _import_langchain_ollama(model_name[len("ollama:"):])

    # Heuristic: llama / mistral / gemma / phi / qwen → local Ollama
    _LOCAL_PREFIXES = ("llama", "mistral", "gemma", "phi", "qwen",
                       "deepseek", "codellama", "vicuna", "falcon")
    if any(model_name.lower().startswith(p) for p in _LOCAL_PREFIXES):
        return _import_langchain_ollama(model_name)

    # Default: Anthropic Claude
    return _import_langchain_anthropic(model_name)


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

    # Generic, domain-agnostic classification prompt.  Domain-specific
    # guidance is injected at call time via the ``context`` dict
    # (``domain``, ``classification_guide``, ``valid_labels``), keeping this
    # module free of any one dataset's rules.  This is the fallback used when
    # the caller supplies no domain context.
    BASE_SYSTEM_PROMPT = (
        "You are a NeuroVerse anomaly classifier. "
        "You receive sensor features and reasoning signals from a spike-based "
        "predictive layer. Decide whether the event is normal or an anomaly "
        "(crash / fault / incident) and explain your reasoning. "
        "If domain-specific guidance is provided, follow it. "
        'Respond ONLY with valid JSON, nothing else: '
        '{"label": "<label>", '
        '"confidence": <0.0-1.0>, '
        '"reasoning": "<one sentence>"}'
    )

    def __init__(
        self,
        model: str = "llama3.2:3b",
        max_tool_calls: int = 1,
    ) -> None:
        self._model_name = model
        self._max_tool_calls = max_tool_calls
        # Set True by ``reason`` whenever an LLM invocation is attempted on
        # the deliberative path.  Read by the decision recorder to populate
        # ``llm_called``.  Path A never calls ``reason``, so it stays False.
        self.last_llm_invoked: bool = False

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        """
        Construct a domain-aware system prompt from the supplied context.

        Recognised context keys (all optional):
        - ``domain``             : human-readable domain name.
        - ``classification_guide``: domain-supplied rules / thresholds string.
        - ``valid_labels``        : list of acceptable label strings.

        Domains own their classification knowledge and pass it through the
        platform-agnostic ``context`` dict, so the platform never hardcodes
        one dataset's labels or thresholds.
        """
        if not context:
            return self.BASE_SYSTEM_PROMPT

        domain = context.get("domain")
        guide = context.get("classification_guide")
        valid_labels = context.get("valid_labels")

        # No domain context at all → generic prompt.
        if not (domain or guide or valid_labels):
            return self.BASE_SYSTEM_PROMPT

        parts: list[str] = []
        if domain:
            parts.append(f"You are the NeuroVerse classifier for the {domain} domain.")
        else:
            parts.append("You are the NeuroVerse anomaly classifier.")
        parts.append(
            "Given the sensor features and reasoning signals, classify the "
            "event and explain your reasoning."
        )
        if guide:
            parts.append(f"Classification guidance: {guide}")
        if valid_labels:
            label_list = ", ".join(str(l) for l in valid_labels)
            parts.append(f"Valid labels: {label_list}.")
        parts.append(
            'Respond ONLY with valid JSON, nothing else: '
            '{"label": "<label>", '
            '"confidence": <0.0-1.0>, '
            '"reasoning": "<one sentence>"}'
        )
        return " ".join(parts)

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
        # An LLM invocation is being attempted on the deliberative path.
        # This is set before the call so the recorder reflects "LLM invoked"
        # even when the call degrades to the fallback decision.
        self.last_llm_invoked = True
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
        # For local Ollama models, skip tool-binding (llama3.2 doesn't handle
        # it reliably) and do a single direct call: RAG retrieve → prompt → parse.
        llm = _build_llm(self._model_name)
        is_ollama = self._model_name.startswith("ollama:") or any(
            self._model_name.lower().startswith(p)
            for p in ("llama", "mistral", "gemma", "phi", "qwen",
                      "deepseek", "codellama", "vicuna", "falcon")
        )

        if is_ollama:
            return self._run_ollama_direct(llm, event, context, rag_source)

        return self._run_langgraph(llm, event, context, rag_source)

    def _run_ollama_direct(
        self,
        llm: Any,
        event: dict[str, Any],
        context: dict[str, Any],
        rag_source: str,
    ) -> AgentDecision:
        """
        Single-shot Ollama call with inline knowledge (no RAG round-trip).
        Domain knowledge is injected via ``context`` (``inline_kb`` /
        ``classification_guide``) so the platform never hardcodes a dataset's
        thresholds.  Falls back to the generic prompt when none is supplied.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Inline knowledge is optional and domain-supplied.  When absent we
        # rely on the model's general reasoning plus the reasoning signals.
        inline_kb = context.get("inline_kb") or context.get("classification_guide") or ""

        feature_summary = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in event.items()
            if k.startswith("de_") or k.startswith("fe_")
            or k.startswith("lane") or isinstance(v, (int, float))
        }
        # Keep the feature summary compact for small-context local models.
        if len(feature_summary) > 24:
            feature_summary = dict(list(feature_summary.items())[:24])

        human_content = (
            (f"Knowledge: {inline_kb}\n\n" if inline_kb else "")
            + f"Features: {json.dumps(feature_summary)}\n\n"
            "Reasoning signals: "
            + json.dumps({k: context[k] for k in (
                "novelty", "uncertainty", "prediction_error",
                "memory_confidence", "spatial_correlation", "risk", "gate_confidence"
            ) if k in context})
            + "\n\nReply JSON only."
        )

        messages = [
            SystemMessage(content=self._build_system_prompt(context)),
            HumanMessage(content=human_content),
        ]

        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        return self._parse_response(content, [])

    def _run_langgraph(
        self,
        llm: Any,
        event: dict[str, Any],
        context: dict[str, Any],
        rag_source: str,
    ) -> AgentDecision:
        """Full ReAct tool-calling path for Anthropic / capable models."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.tools import tool as lc_tool

        rag_fn = _make_rag_tool(rag_source)
        ctx_fn = _make_context_tool(context)
        rag_tool = lc_tool(rag_fn)
        ctx_tool = lc_tool(ctx_fn)
        tools = [rag_tool, ctx_tool]

        llm_with_tools = llm.bind_tools(tools)

        messages = [
            SystemMessage(content=self._build_system_prompt(context)),
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

            from langchain_core.messages import ToolMessage

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                if tool_name == "rag_retrieve":
                    result = rag_fn(**tool_args)
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
        # Bearing fault labels (checked first — most specific)
        bearing_labels = ("inner_race", "outer_race", "ball fault",
                          "ball_fault", "rolling element")
        if "inner_race" in lower or "inner race" in lower:
            label, confidence = "inner_race", 0.7
        elif "outer_race" in lower or "outer race" in lower:
            label, confidence = "outer_race", 0.7
        elif "ball fault" in lower or "ball_fault" in lower or "rolling element" in lower:
            label, confidence = "ball", 0.7
        # Generic anomaly / crash / normal keywords
        elif any(p in lower for p in ("no fault", "healthy", "no anomaly")):
            label, confidence = "normal", 0.7
        elif any(p in lower for p in ("no incident", "no crash", "traffic is clear",
                                       "normal", "all clear")):
            label, confidence = "normal", 0.7
        elif any(p in lower for p in ("crash", "anomaly", "fault", "alert", "incident")):
            label, confidence = "anomaly", 0.7
        else:
            label, confidence = "uncertain", 0.4

        return AgentDecision(
            label=label,
            confidence=confidence,
            reasoning=content[:500],
            sources=sources,
        )
