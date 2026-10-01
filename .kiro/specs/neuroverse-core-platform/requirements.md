# Requirements: NeuroVerse Core Platform

## Overview
Build the four platform-layer modules that form the spine of NeuroVerse. These modules
are domain-agnostic and live entirely under `/platform/`. The traffic domain under
`/domains/traffic/` is the validation harness — it must work end-to-end, but no
traffic-specific logic may leak into `/platform/`.

---

## Requirements

### 1 — SNN Input Plugin (`/platform/snn_plugin.py`)

**1.1** Expose a single public interface: `process_events(sensor_stream) -> spike_tensor`  
**1.2** Default backend: snnTorch `LIFCell` with rate coding (matching current `preprocess.py`
conventions: `NUM_STEPS=25`, sigmoid-then-rate encoding).  
**1.3** Backend must be swappable via a constructor argument (`backend: str = "snntorch"`) 
without any change to `/domains/` code. Supported backends: `"snntorch"`, `"spikingjelly"`, `"norse"`.  
**1.4** Input shape: `(batch, time_window, features)`. Output shape: `(num_steps, batch, features)`.  
**1.5** Unit test: swap backend to a stub and verify output tensor shape is unchanged.

---

### 2 — Implicit Predictive Layer (`/platform/implicit_layer.py`)

**2.1** Expose: `compute_state(spikes: Tensor) -> ReasoningScore`  
**2.2** `ReasoningScore` is a dataclass with four float fields:
- `novelty` — cosine distance of current spike pattern from a running prototype store
- `uncertainty` — epistemic uncertainty via MC-dropout (≥10 forward passes)
- `prediction_error` — MSE between predicted next-step spikes and actual
- `memory_confidence` — similarity to closest stored episode (episodic memory buffer, ring buffer of last N states)

**2.3** All four sub-signals normalised to [0, 1].  
**2.4** The layer is learned end-to-end — no hand-tuned thresholds anywhere in this file.  
**2.5** Provide a `fit(dataloader)` method so the layer can be pre-trained or fine-tuned.  
**2.6** Unit test: given a fixed spike tensor, `compute_state` returns a `ReasoningScore`
with all four fields in [0, 1].

---

### 3 — Adaptive Reasoning Gate (`/platform/gate.py` + `/platform/gate_learner.py`)

**3.1** `gate.py` exposes: `should_escalate(reasoning_score: ReasoningScore, history: EscalationHistory) -> tuple[bool, float]`  
**3.2** `gate_learner.py` implements the meta-learner that updates the gate's decision
boundary from logged outcomes.  
**3.3** Every call to `should_escalate` is logged: `(reasoning_score, decision, timestamp)`.
Outcomes are appended later via `log_outcome(decision_id, actual_label)`.  
**3.4** The meta-learner retrains on the log whenever `len(log) % UPDATE_EVERY == 0`
(default `UPDATE_EVERY = 50`).  
**3.5** Gate must reduce LLM escalations by ≥30% vs. always-escalate on the traffic
validation set, with crash recall >60%.  
**3.6** False negatives (missed crashes, label=1 not escalated) are penalised 5× more
than false positives in the learner's loss.  
**3.7** `EscalationHistory` is a dataclass holding the rolling log (last 500 decisions).  
**3.8** Unit tests:
- Gate never escalates on a batch of identical low-score inputs after warm-up.
- Gate always escalates when all four sub-signals are at 1.0.
- `gate_learner` log grows correctly and triggers retraining at the right interval.

---

### 4 — LLM Agent Plugin (`/platform/agent_plugin.py`)

**4.1** Expose: `reason(event: dict, context: dict, rag_source: str) -> AgentDecision`  
**4.2** `AgentDecision` dataclass: `label: str`, `confidence: float`, `reasoning: str`, `sources: list[str]`.  
**4.3** Orchestrated with LangGraph (single-agent ReAct loop, ≤3 tool calls per invocation).  
**4.4** RAG backend: ChromaDB collection at `rag_source` path. Retrieved top-k=5 chunks.  
**4.5** Model-agnostic: model name injected at construction (`model: str = "claude-3-5-sonnet-20241022"`).  
**4.6** Graceful degradation: if LLM call fails (timeout, API error), return an
`AgentDecision` with `label="uncertain"`, `confidence=0.0`, and `reasoning="LLM unavailable"`.  
**4.7** Unit test: mock LLM + ChromaDB; verify `AgentDecision` fields are populated correctly
and fallback fires on simulated timeout.

---

### 5 — Domain Plugin Base (`/platform/domain_plugin.py`)

**5.1** Define an abstract base class `DomainPlugin` with abstract properties:
`input_schema`, `output_schema`, `knowledge_base_path`, `threshold_hints`.  
**5.2** Provide a `validate_event(event: dict) -> bool` default method using `input_schema`.  
**5.3** The traffic domain (`/domains/traffic/`) must subclass `DomainPlugin` and pass
its own contract tests without modifying any `/platform/` file.  
**5.4** Unit test: instantiating `DomainPlugin` directly raises `TypeError`.

---

### 6 — End-to-end smoke test

**6.1** A script `tests/test_e2e_smoke.py` feeds one batch from `data/processed/train.pt`
through the full pipeline: `snn_plugin → implicit_layer → gate → agent_plugin` (mocked LLM).  
**6.2** The script must complete without error and print a summary:
`escalated N/M events, gate_confidence mean=X`.  
**6.3** No traffic-specific imports appear in any `/platform/` file (enforced by the test
via `ast.walk` import check).

---

## Out of scope for this spec
- FastAPI / Streamlit UI layer
- Full ChromaDB knowledge base population for traffic
- Gate hyperparameter tuning beyond the basic meta-learner
- SpikingJelly / Norse backend implementations (stubs acceptable)
