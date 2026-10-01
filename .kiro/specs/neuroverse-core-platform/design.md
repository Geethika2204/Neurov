# Design: NeuroVerse Core Platform

## Architecture overview

```
sensor_stream
     │
     ▼
┌─────────────────────┐
│  SNN Input Plugin   │  /platform/snn_plugin.py
│  process_events()   │
└────────┬────────────┘
         │ spike_tensor  (num_steps, batch, features)
         ▼
┌─────────────────────┐
│ Implicit Predictive │  /platform/implicit_layer.py
│  compute_state()    │
└────────┬────────────┘
         │ ReasoningScore (novelty, uncertainty, prediction_error, memory_confidence)
         ▼
┌─────────────────────┐
│  Adaptive Gate      │  /platform/gate.py + gate_learner.py
│  should_escalate()  │
└────────┬────────────┘
         │ (bool, confidence)
    ┌────┴────┐
    │ escalate│
    ▼         ▼
┌──────────┐  fast-path decision
│LLM Agent │  /platform/agent_plugin.py
│ reason() │
└──────────┘
```

---

## Module designs

### `snn_plugin.py`

```python
class SNNInputPlugin:
    def __init__(self, backend: str = "snntorch", num_steps: int = 25): ...
    def process_events(self, sensor_stream: Tensor) -> Tensor: ...

class _SNNTorchBackend:   # default
class _SpikingJellyBackend:  # stub — raises NotImplementedError
class _NorseBackend:         # stub — raises NotImplementedError

BACKENDS = {"snntorch": _SNNTorchBackend, "spikingjelly": _SpikingJellyBackend, "norse": _NorseBackend}
```

The snnTorch backend replicates the `to_spikes()` logic from `preprocess.py`:
`sigmoid → spikegen.rate(num_steps=num_steps)`.

---

### `implicit_layer.py`

```python
@dataclass
class ReasoningScore:
    novelty: float            # [0, 1]
    uncertainty: float        # [0, 1]
    prediction_error: float   # [0, 1]
    memory_confidence: float  # [0, 1]

class ImplicitPredictiveLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 memory_size: int = 256, mc_dropout_passes: int = 10): ...
    def compute_state(self, spikes: Tensor) -> ReasoningScore: ...
    def fit(self, dataloader: DataLoader, epochs: int = 10): ...
```

Internal sub-modules:
- `_PrototypeStore` — online k-means (k=16) over flattened spike vectors; novelty = 1 − max_cosine_sim.
- `_MCDropoutHead` — 2-layer MLP with Dropout(0.3); uncertainty = std of logits over N passes, normalised by max observed std.
- `_PredictorHead` — 1-step-ahead spike predictor (MSE loss during `fit`); error normalised by running max.
- `_EpisodicMemory` — `collections.deque(maxlen=memory_size)` of mean-pooled spike vectors; confidence = 1 − min_cosine_dist.

All normalisation uses running statistics updated via `torch.no_grad()` to avoid leaking gradients.

---

### `gate.py` + `gate_learner.py`

```python
# gate.py
@dataclass
class EscalationHistory:
    log: deque  # maxlen=500, each entry: {id, score, decision, outcome, timestamp}

class ReasoningGate:
    def __init__(self, learner: "GateLearner"): ...
    def should_escalate(self, score: ReasoningScore,
                        history: EscalationHistory) -> tuple[bool, float]: ...
    def log_outcome(self, decision_id: str, actual_label: int): ...

# gate_learner.py
class GateLearner:
    def __init__(self, update_every: int = 50,
                 fn_penalty: float = 5.0): ...
    def update(self, history: EscalationHistory): ...
    def predict_proba(self, score: ReasoningScore) -> float: ...
```

Decision boundary model: lightweight logistic regression over the four sub-signals +
3 history features (recent escalation rate, recent recall, recent precision).
Trained with weighted cross-entropy: FN weight = `fn_penalty`, FP weight = 1.0.
`update()` is called inside `should_escalate` whenever `len(log) % update_every == 0`.

`decision_id` is a `uuid4` string generated per call; returned as part of the gate's
internal log so callers can attach the eventual outcome.

---

### `agent_plugin.py`

```python
@dataclass
class AgentDecision:
    label: str
    confidence: float
    reasoning: str
    sources: list[str]

class LLMAgentPlugin:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022",
                 max_tool_calls: int = 3): ...
    def reason(self, event: dict, context: dict,
               rag_source: str) -> AgentDecision: ...
```

LangGraph graph: single `ReActNode` with two tools:
1. `rag_retrieve(query, collection_path, k=5)` — ChromaDB similarity search.
2. `lookup_context(key)` — pulls structured fields from `context` dict.

Timeout: 30 s (configurable). On any exception, returns the degraded `AgentDecision`.

---

### `domain_plugin.py`

```python
from abc import ABC, abstractmethod

class DomainPlugin(ABC):
    @property
    @abstractmethod
    def input_schema(self) -> dict: ...

    @property
    @abstractmethod
    def output_schema(self) -> dict: ...

    @property
    @abstractmethod
    def knowledge_base_path(self) -> str: ...

    @property
    @abstractmethod
    def threshold_hints(self) -> dict: ...

    def validate_event(self, event: dict) -> bool:
        # jsonschema.validate against self.input_schema
        ...
```

---

## File layout produced by this spec

```
/platform/
    snn_plugin.py
    implicit_layer.py
    gate.py
    gate_learner.py
    agent_plugin.py
    domain_plugin.py
    __init__.py

/domains/
    traffic/
        __init__.py
        domain.py          # TrafficDomain(DomainPlugin)
        rag_kb/            # ChromaDB persist directory (populated separately)

/tests/
    test_snn_plugin.py
    test_implicit_layer.py
    test_gate.py
    test_agent_plugin.py
    test_domain_plugin.py
    test_e2e_smoke.py
```

---

## Key invariants enforced by tests

| Invariant | Enforced in |
|---|---|
| No traffic imports in /platform | `test_e2e_smoke.py` (ast.walk) |
| Backend swap preserves output shape | `test_snn_plugin.py` |
| All ReasoningScore fields in [0,1] | `test_implicit_layer.py` |
| FN penalty ≥ 5× FP in gate loss | `test_gate.py` |
| LLM fallback fires on timeout | `test_agent_plugin.py` |
| DomainPlugin abstract (no direct instantiation) | `test_domain_plugin.py` |

---

## Dependencies (additions to venv)

All already present or minimal additions:
- `snntorch` — already installed
- `torch` — already installed
- `langgraph` — to install
- `langchain-anthropic` — to install
- `chromadb` — to install
- `jsonschema` — to install
- `pytest` — to install
