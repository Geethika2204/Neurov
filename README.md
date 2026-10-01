# NeuroVerse

A generalizable plugin architecture for anomaly detection using spiking neural
networks, implicit predictive coding, and a learned reasoning gate that decides
when to escalate to an LLM agent.

---

## Architecture

```
sensor_stream
     │
     ▼
┌──────────────────────┐
│  Layer 1             │  platform/snn_plugin.py
│  SNN Input Plugin    │  process_events(stream) → spike_tensor
└──────────┬───────────┘  default: snnTorch LIFCell (swappable)
           │
           ▼
┌──────────────────────┐
│  Layer 2             │  platform/implicit_layer.py
│  Implicit Predictive │  compute_state(spikes) → ReasoningScore
│  Layer               │  (novelty, uncertainty, prediction_error,
└──────────┬───────────┘   memory_confidence)
           │
           ▼
┌──────────────────────┐
│  Layer 3             │  platform/gate.py + gate_learner.py
│  Adaptive Reasoning  │  should_escalate(score, history) → (bool, conf)
│  Gate                │  meta-learner updates boundary from outcomes
└──────┬───────────────┘
       │ escalate?
  ┌────┴───────┐
  │ yes        │ no → fast-path decision
  ▼            ▼
┌──────────────────────┐
│  Layer 4             │  platform/agent_plugin.py
│  LLM Agent Plugin    │  reason(event, context, rag_source) → AgentDecision
└──────────────────────┘  LangGraph ReAct + ChromaDB RAG + Claude (default)
```

The four layers are domain-agnostic.  Domain-specific sensor schemas, RAG
knowledge bases, and gate hints live entirely under `/domains/<name>/`.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

---

## How to add a new domain

You can add a new domain without touching any file in `/platform/`.

### 1. Create the directory

```bash
mkdir -p domains/<name>/rag_kb
touch domains/<name>/__init__.py
touch domains/<name>/domain.py
```

### 2. Subclass `DomainPlugin`

```python
# domains/<name>/domain.py
from platform.domain_plugin import DomainPlugin

class MyDomain(DomainPlugin):

    @property
    def input_schema(self) -> dict:
        """JSON Schema for one sensor event."""
        return {
            "type": "object",
            "required": ["timestamp", "sensor_value"],
            "properties": {
                "timestamp":    {"type": "number"},
                "sensor_value": {"type": "number"},
            },
        }

    @property
    def output_schema(self) -> dict:
        """JSON Schema for the agent decision."""
        return {
            "type": "object",
            "required": ["label", "confidence"],
            "properties": {
                "label":      {"type": "string"},
                "confidence": {"type": "number"},
            },
        }

    @property
    def knowledge_base_path(self) -> str:
        """Path to the ChromaDB persist directory."""
        return "domains/<name>/rag_kb"

    @property
    def threshold_hints(self) -> dict:
        """Soft starting hints for the gate learner (adapted over time)."""
        return {
            "novelty":           0.6,
            "uncertainty":       0.65,
            "prediction_error":  0.5,
            "memory_confidence": 0.3,
        }
```

### 3. Populate the knowledge base (optional)

```python
import chromadb

client = chromadb.PersistentClient(path="domains/<name>/rag_kb")
col = client.get_or_create_collection("knowledge")
col.add(
    documents=["Domain fact 1", "Domain fact 2"],
    ids=["doc_0", "doc_1"],
)
```

### 4. Wire into the pipeline

```python
import torch
from platform.snn_plugin import SNNInputPlugin
from platform.implicit_layer import ImplicitPredictiveLayer
from platform.gate import ReasoningGate, EscalationHistory
from platform.gate_learner import GateLearner
from platform.agent_plugin import LLMAgentPlugin, AgentDecision
from domains.<name>.domain import MyDomain

domain  = MyDomain()
snn     = SNNInputPlugin()                         # swap backend here if needed
layer2  = ImplicitPredictiveLayer(input_dim=<N>)   # N = your feature count
learner = GateLearner()
gate    = ReasoningGate(learner=learner)
agent   = LLMAgentPlugin()                         # swap model here if needed
history = EscalationHistory()

for raw_event in your_sensor_stream:
    if not domain.validate_event(raw_event):
        continue

    # Layer 1 — spikes
    tensor = torch.tensor(
        [raw_event[k] for k in your_feature_cols], dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    spikes = snn.process_events(tensor)

    # Layer 2 — reasoning score
    score = layer2.compute_state(spikes)

    # Layer 3 — gate decision
    escalate, conf = gate.should_escalate(score, history)

    # Layer 4 — LLM (only when gate says so)
    if escalate:
        decision = agent.reason(raw_event, {"gate_conf": conf},
                                domain.knowledge_base_path)
    else:
        decision = AgentDecision(label="normal", confidence=1 - conf)

    # Attach ground truth when known (trains the gate meta-learner)
    if raw_event.get("label") is not None:
        last_id = list(history.log)[-1]["id"]
        gate.log_outcome(last_id, raw_event["label"], history)
```

### 5. Add tests

Place tests in `/tests/`, not inside `/domains/`.  The smoke test
`tests/test_e2e_smoke.py` checks at CI time that no `/platform/` file
ever imports from `/domains/`.

---

## Swapping the SNN backend

Pass `backend=` to `SNNInputPlugin`:

```python
snn = SNNInputPlugin(backend="snntorch")      # default
snn = SNNInputPlugin(backend="spikingjelly")  # stub — implement encode()
snn = SNNInputPlugin(backend="norse")         # stub — implement encode()
```

To implement a new backend, add a class to `platform/snn_plugin.py`:

```python
class _MyBackend(_SNNBackend):
    def encode(self, sensor_stream: Tensor, num_steps: int) -> Tensor:
        # sensor_stream: (batch, time_window, features)
        # return:        (num_steps, batch, features)
        ...
```

Then register it: `_BACKENDS["mybackend"] = _MyBackend`.  Nothing else changes.

---

## Success criteria

| Criterion | Target | How measured |
|---|---|---|
| SNN backend swappable | ✓ | `test_snn_plugin.py::test_backend_swap_preserves_shape` |
| Gate reduces LLM calls | ≥ 30% vs always-escalate | `test_baselines.py::test_always_escalate_vs_threshold_reduction` |
| Crash recall | > 60% | `AlwaysThresholdGate.evaluate()` on val set |
| No domain imports in platform | ✓ | `test_e2e_smoke.py::test_platform_does_not_import_domains` |

---

## Repo layout

```
/platform/
    domain_plugin.py      ← DomainPlugin ABC
    snn_plugin.py         ← Layer 1
    implicit_layer.py     ← Layer 2
    gate.py               ← Layer 3
    gate_learner.py       ← Layer 3 meta-learner
    agent_plugin.py       ← Layer 4

/domains/
    traffic/
        domain.py         ← TrafficDomain (FT-AED crash detection)
        rag_kb/           ← ChromaDB knowledge base

/baselines/
    always_escalate_gate.py   ← upper bound (100% LLM calls)
    always_threshold_gate.py  ← rule-based (no learning)

/tests/
    test_domain_plugin.py
    test_snn_plugin.py
    test_implicit_layer.py
    test_gate.py
    test_agent_plugin.py
    test_traffic_domain.py
    test_baselines.py
    test_e2e_smoke.py      ← architecture invariant + full pipeline

DOMAIN_PLUGIN_SPEC.md     ← full contract reference
requirements.txt
```

---

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` for the LLM agent layer (only needed for real runs;
tests mock the LLM by default).
