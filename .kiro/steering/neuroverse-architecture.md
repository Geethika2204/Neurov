# NeuroVerse — Locked Architecture & Conventions

## Core claim
NeuroVerse is a generalizable plugin architecture, not a domain-specific system.
The research contribution is a *learned* reasoning gate that decides when to
escalate from fast automatic processing to expensive LLM reasoning.

## Platform layers (in order — do not merge or reorder)
1. SNN Input Plugin — `process_events(sensor_stream) -> spike_tensor`
   - Default backend: snnTorch LIFCell. Must stay swappable with SpikingJelly/Norse.
   - Domain-agnostic — no traffic-specific logic here.
2. Implicit Predictive Layer (research core) — `compute_state(spikes) -> reasoning_score`
   - Outputs four sub-signals: novelty, uncertainty (epistemic — dropout/ensemble),
     prediction_error, memory_confidence.
   - Learned, not fixed thresholds.
3. Adaptive Reasoning Gate — `should_escalate(reasoning_score, history) -> (bool, confidence)`
   - Meta-learner: logs every escalation + its eventual outcome, updates its own
     decision boundary from that history.
4. LLM Agent Plugin — `reason(event, context, rag_source) -> decision`
   - Orchestrated with LangGraph. RAG backend: ChromaDB. Model-agnostic interface,
     default Claude.

## Domain plugin contract
Every domain must supply: input_schema, output_schema, knowledge_base_path, threshold_hints.

## Repo layout (do not deviate)
/platform/snn_plugin.py
/platform/implicit_layer.py
/platform/gate.py
/platform/gate_learner.py
/platform/agent_plugin.py
/platform/domain_plugin.py
/domains/traffic/
/domains/traffic/rag_kb/

## Tech stack (locked)
snnTorch, LangGraph, ChromaDB, FastAPI, Streamlit, FT-AED dataset (radar, ~3.7M
samples, ~400 labeled crashes).

## Non-negotiable success criteria
- SNN backend swappable without touching /domains code.
- Gate cuts LLM calls 30%+ vs. an always-escalate baseline while crash recall stays >60%.
- False negatives (missed real crashes) never traded away for fewer LLM calls.
- Docs clear enough for another team to add a new domain without touching /platform.

## What NOT to do
- Don't just substitute SNNs for a Transformer and call it done.
- Don't claim energy-efficiency wins without real hardware measurement.
- Don't put traffic-specific logic in /platform — it lives only under /domains/traffic.
