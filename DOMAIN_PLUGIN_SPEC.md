# Domain Plugin Specification

> **Version:** 1.0  
> **Status:** Locked — changes require updating all existing domain implementations.

---

## Overview

A NeuroVerse **domain plugin** wires a specific sensor modality (traffic radar,
medical ECG, industrial vibration, …) into the four platform layers.  The
platform layers (`/platform/`) never import concrete domain code.  Every domain
lives entirely under `/domains/<name>/`.

---

## Required files

```
/domains/<name>/
    __init__.py
    domain.py        ← must contain exactly one class that subclasses DomainPlugin
    rag_kb/          ← ChromaDB persist directory (may be empty at dev time)
```

---

## Contract: `DomainPlugin` abstract base class

Import path: `from platform.domain_plugin import DomainPlugin`

### Abstract properties

| Property | Type | Description |
|---|---|---|
| `input_schema` | `dict` (JSON Schema draft-7) | Shape of one sensor event |
| `output_schema` | `dict` (JSON Schema draft-7) | Shape of one agent decision |
| `knowledge_base_path` | `str` | Path to the ChromaDB directory |
| `threshold_hints` | `dict[str, float]` | Soft starting hints for the gate learner |

### Inherited methods (do not override unless necessary)

| Method | Signature | Description |
|---|---|---|
| `validate_event` | `(event: dict) -> bool` | Validates event against `input_schema` |
| `domain_name` | `() -> str` | Returns class name as a string |

---

## Implementing a new domain — step by step

### Step 1 — Create the directory structure

```bash
mkdir -p domains/<name>/rag_kb
touch domains/<name>/__init__.py
touch domains/<name>/domain.py
```

### Step 2 — Subclass `DomainPlugin`

```python
# domains/<name>/domain.py
from platform.domain_plugin import DomainPlugin

class MyDomain(DomainPlugin):

    @property
    def input_schema(self) -> dict:
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
        return {
            "type": "object",
            "required": ["label", "confidence"],
            "properties": {
                "label":      {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }

    @property
    def knowledge_base_path(self) -> str:
        return "domains/<name>/rag_kb"

    @property
    def threshold_hints(self) -> dict:
        # Soft hints — the gate learner will adapt these automatically.
        # Start conservative: escalate when things look novel or uncertain.
        return {
            "novelty":            0.6,
            "uncertainty":        0.65,
            "prediction_error":   0.5,
            "memory_confidence":  0.3,   # low confidence → escalate
        }
```

### Step 3 — Populate the RAG knowledge base (optional at dev time)

```python
import chromadb

client = chromadb.PersistentClient(path="domains/<name>/rag_kb")
collection = client.get_or_create_collection("knowledge")
collection.add(
    documents=["<domain fact 1>", "<domain fact 2>"],
    ids=["doc_0", "doc_1"],
)
```

### Step 4 — Wire into the platform pipeline

```python
from platform.snn_plugin import SNNInputPlugin
from platform.implicit_layer import ImplicitPredictiveLayer
from platform.gate import ReasoningGate, EscalationHistory
from platform.gate_learner import GateLearner
from platform.agent_plugin import LLMAgentPlugin
from domains.<name>.domain import MyDomain

domain  = MyDomain()
snn     = SNNInputPlugin()
implicit = ImplicitPredictiveLayer(input_dim=<feature_count>)
learner  = GateLearner()
gate     = ReasoningGate(learner=learner)
agent    = LLMAgentPlugin()
history  = EscalationHistory()

for raw_event in sensor_stream:
    if not domain.validate_event(raw_event):
        continue
    spikes        = snn.process_events(raw_event["tensor"])
    score         = implicit.compute_state(spikes)
    escalate, conf = gate.should_escalate(score, history)
    if escalate:
        decision = agent.reason(raw_event, {}, domain.knowledge_base_path)
    else:
        decision = {"label": "normal", "confidence": conf}
```

### Step 5 — Add domain-specific tests

Place tests in `/tests/` (not inside `/domains/`).  Tests may import from
`/domains/<name>/` but `/platform/` files must never import from `/domains/`.

---

## Invariants enforced by `tests/test_e2e_smoke.py`

1. No file under `/platform/` imports from `/domains/` (checked via `ast.walk`).
2. A `DomainPlugin` subclass can be instantiated and its four properties return
   non-empty values.
3. `validate_event` returns `True` for a conforming event and `False` for a
   non-conforming one.

---

## `threshold_hints` reference

| Key | Meaning | Typical starting range |
|---|---|---|
| `novelty` | Escalate if current pattern is this novel vs. memory | 0.5 – 0.8 |
| `uncertainty` | Escalate if epistemic uncertainty exceeds this | 0.5 – 0.75 |
| `prediction_error` | Escalate if prediction error exceeds this | 0.4 – 0.7 |
| `memory_confidence` | Escalate if memory similarity falls *below* this | 0.2 – 0.5 |

All values are in [0, 1].  The gate learner treats these as a prior — it will
update the decision boundary automatically as outcomes are logged.
