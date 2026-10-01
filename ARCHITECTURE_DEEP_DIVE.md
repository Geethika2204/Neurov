# NeuroVerse — Deep-Dive Architecture & Algorithm Reference

> Every component explained: what it does, the algorithm behind it, how data flows
> through it, and why it was designed that way.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Layer 1 — SNN Input Plugin](#2-layer-1--snn-input-plugin)
3. [Layer 2 — Implicit Predictive Layer](#3-layer-2--implicit-predictive-layer)
   - 3.1 [PrototypeStore — Novelty via Online K-Means](#31-prototypestore--novelty-via-online-k-means)
   - 3.2 [MCDropoutHead — Uncertainty via Monte Carlo Dropout](#32-mcdropouthead--uncertainty-via-monte-carlo-dropout)
   - 3.3 [PredictorHead — Prediction Error via One-Step-Ahead Forecasting](#33-predictorhead--prediction-error-via-one-step-ahead-forecasting)
   - 3.4 [EpisodicMemory — Memory Confidence via Cosine Ring Buffer](#34-episodicmemory--memory-confidence-via-cosine-ring-buffer)
   - 3.5 [SpatialCorrelationHead — Cross-Sensor Anomaly via Online Z-Score](#35-spatialcorrelationhead--cross-sensor-anomaly-via-online-z-score)
   - 3.6 [RiskHead — Composite Risk Score](#36-riskhead--composite-risk-score)
   - 3.7 [ReasoningScore — The Output Contract](#37-reasoningscore--the-output-contract)
4. [Layer 3 — Adaptive Reasoning Gate](#4-layer-3--adaptive-reasoning-gate)
   - 4.1 [GateLearner — Logistic Regression Meta-Learner](#41-gatelearner--logistic-regression-meta-learner)
   - 4.2 [ReasoningGate — Decision Logic](#42-reasoninggate--decision-logic)
   - 4.3 [EscalationHistory — Rolling Decision Log](#43-escalationhistory--rolling-decision-log)
5. [Layer 4 — LLM Agent Plugin](#5-layer-4--llm-agent-plugin)
6. [Supporting Platform Modules](#6-supporting-platform-modules)
   - 6.1 [AutomaticAction — Fast Path](#61-automaticaction--fast-path)
   - 6.2 [SafetyChecker](#62-safetychecker)
   - 6.3 [Planner](#63-planner)
   - 6.4 [MemoryConsolidation — Feedback Loop](#64-memoryconsolidation--feedback-loop)
7. [Domain Plugin System](#7-domain-plugin-system)
   - 7.1 [DomainPlugin ABC](#71-domainplugin-abc)
   - 7.2 [TrafficDomain — FT-AED Crash Detection](#72-trafficdomain--ft-aed-crash-detection)
8. [Baselines](#8-baselines)
9. [Training & Evaluation Scripts](#9-training--evaluation-scripts)
10. [Data Flow: Full End-to-End Trace](#10-data-flow-full-end-to-end-trace)
11. [Key Design Decisions & Trade-offs](#11-key-design-decisions--trade-offs)

---

## 1. System Overview

NeuroVerse is a **generalizable anomaly detection pipeline** organized into four
sequential layers. The core research contribution is Layer 3 — a **learned gate**
that decides when fast automatic processing (Layers 1–2) is sufficient and when
to invoke an expensive LLM agent (Layer 4).

```
Raw Sensor Stream
       │
       ▼
┌─────────────────────┐
│  Layer 1            │   SNNInputPlugin
│  SNN Encoding       │   sensor_stream → spike_tensor
│                     │   Algorithm: Rate coding via snnTorch LIFCell
└────────┬────────────┘
         │  spike_tensor: (num_steps=25, batch, features)
         ▼
┌─────────────────────┐
│  Layer 2            │   ImplicitPredictiveLayer
│  Score Extraction   │   spike_tensor → ReasoningScore
│                     │   6 sub-signals via 5 independent algorithms
└────────┬────────────┘
         │  ReasoningScore: {novelty, uncertainty, prediction_error,
         │                   memory_confidence, spatial_correlation, risk}
         ▼
┌─────────────────────┐
│  Layer 3            │   ReasoningGate + GateLearner
│  Escalation Gate    │   ReasoningScore → (should_escalate: bool, confidence: float)
│                     │   Algorithm: online logistic regression, FN-penalized BCE
└────────┬────────────┘
         │
    ┌────┴─────┐
    │ escalate?│
    │   YES    │   NO → AutomaticAction (microseconds)
    ▼          ▼
┌─────────────────────┐
│  Layer 4            │   LLMAgentPlugin
│  LLM Agent          │   event + context + RAG → AgentDecision
│                     │   Algorithm: LangGraph ReAct loop + ChromaDB RAG
└─────────────────────┘
         │
         ▼
  SafetyChecker → Planner → ActionPlan
         │
         ▼
  MemoryConsolidation (feedback loop back to Layer 2)
```

The four layers are **domain-agnostic**. All traffic-specific logic lives under
`/domains/traffic/`. The platform never imports from `/domains/`.

---

## 2. Layer 1 — SNN Input Plugin

**File:** `nv_platform/snn_plugin.py`

### What it does

Converts a normalized sensor stream (floating-point measurements) into a
**binary spike tensor** — the input format expected by Layer 2.

### The algorithm: Rate Coding with snnTorch

The default backend uses **rate coding**, one of the simplest and most robust
spike encoding schemes. The idea: instead of encoding a value as a number,
encode it as a *firing probability* — how often a neuron fires over a time window.

**Step-by-step:**

```
Input:  sensor_stream  shape (batch, time_window, features)
                       e.g.  (4, 20, 12) — 4 events, 20 time steps, 12 sensors

Step 1: Mean-pool the time window
        pooled = sensor_stream.mean(dim=1)    → (batch, features)
        Collapses the 20 time steps into one representative value per sensor.

Step 2: Sigmoid activation
        firing_prob = sigmoid(pooled)          → (batch, features)  ∈ (0, 1)
        Maps any real-valued input to a firing rate.
        Large positive values → firing prob near 1 (always fires).
        Large negative values → firing prob near 0 (rarely fires).
        Near zero → fires about 50% of the time.

Step 3: Rate coding via snnTorch
        spikes = snntorch.spikegen.rate(firing_prob, num_steps=25)
                                               → (25, batch, features)
        For each of the 25 time steps, independently sample a Bernoulli(p)
        for each neuron using the firing probability from Step 2.
        Output is binary: 1 = spike, 0 = no spike.
```

**Why rate coding?** It is hardware-efficient (binary operations), biologically
plausible, and naturally handles the stochasticity of real sensor data. A sensor
reading of 0.9 (after sigmoid) produces a spike ~90% of the time steps, while a
reading of 0.1 produces a spike ~10% of the time — the *rate* carries the
information.

### Backend swappability

The `_SNNBackend` ABC defines the contract: any backend must implement
`encode(sensor_stream, num_steps) → spike_tensor`. Three backends are registered:

| Backend | Status | Algorithm |
|---|---|---|
| `snntorch` | Default, implemented | LIF rate coding |
| `spikingjelly` | Stub | Implement `encode()` |
| `norse` | Stub | Implement `encode()` |

Swapping is a one-line change: `SNNInputPlugin(backend="spikingjelly")`. No other
layer or domain file changes.

### Output shape contract

```
Input:  (batch, time_window, features)   e.g. (4, 20, 12)
Output: (num_steps, batch, features)     e.g. (25, 4, 12)  — always binary {0, 1}
```

The plugin validates this shape after every `encode()` call and raises
`RuntimeError` on any mismatch.

---

## 3. Layer 2 — Implicit Predictive Layer

**File:** `nv_platform/implicit_layer.py`

### What it does

Takes the spike tensor from Layer 1 and produces a **ReasoningScore** — six
normalized signals (all ∈ [0,1]) that collectively answer: *how surprising,
uncertain, or anomalous is this input?*

Five independent sub-modules compute five of the signals. A sixth (risk) is a
learned combination of the others. There are **no hard thresholds** anywhere —
every signal is a continuous value.

### Entry point

```python
layer = ImplicitPredictiveLayer(input_dim=16)
score = layer.compute_state(spikes)   # spikes: (num_steps, batch, features)
```

Internally, `compute_state` first **mean-pools** the spike tensor:

```
spikes: (num_steps, batch, features)
         mean over dims (0, 1)
vec:    (features,)     — one representative vector for this window
```

This pooled vector `vec` is then passed independently to each sub-module.

---

### 3.1 PrototypeStore — Novelty via Online K-Means

**Algorithm:** Online K-Means with cosine similarity

**What it answers:** *Have I seen something like this before?*

The prototype store maintains `k=16` prototype vectors representing "typical"
patterns the system has observed. Novelty is how different the current input is
from all stored prototypes.

**Initialization:**
```python
prototypes: Tensor  shape (16, features)
# Initialized with random unit vectors, updated online
```

**Computing novelty:**
```
1. Normalize both vec and all prototypes to unit length (L2 norm = 1)
2. Compute cosine similarity to every prototype:
   cosine_sims = vec_norm @ prototypes_norm.T    → shape (16,)
3. Take the maximum similarity (closest prototype):
   max_sim = cosine_sims.max()
4. Novelty = 1 − max_sim
   → 0.0 means this pattern is identical to a known prototype (not novel)
   → 1.0 means this pattern is completely unlike anything seen before
```

**Why cosine similarity (not Euclidean distance)?**
Cosine similarity ignores magnitude and focuses on *direction* (the shape of the
pattern). Two windows with different absolute sensor readings but the same
proportional pattern will have high similarity — which is the right behavior for
anomaly detection on normalized data.

**Updating prototypes (online K-Means soft assignment):**
```
1. Find the nearest prototype (argmax of cosine similarities)
2. Compute adaptive learning rate:
   lr = 1 / (count[nearest] + 1)     — decreases as we see more data
3. Move the prototype toward the new observation:
   prototype[nearest] = (1 - lr) * prototype[nearest] + lr * vec
4. Increment count[nearest]
```

This is the online (streaming) equivalent of standard K-Means. Early observations
move prototypes quickly; later observations have diminishing effect — the
prototypes converge.

---

### 3.2 MCDropoutHead — Uncertainty via Monte Carlo Dropout

**Algorithm:** Monte Carlo Dropout (Gal & Ghahramani, 2016)

**What it answers:** *How confident is the model about this input?*

Standard neural networks give one output for one input. MC Dropout runs the same
input through the network **multiple times with dropout randomly active**, treating
each run as a sample from a Bayesian posterior. The variance across runs measures
**epistemic uncertainty** — uncertainty due to limited training data.

**Architecture:**
```
Input (features,)
    │
Linear(features → 128)  →  ReLU  →  Dropout(p=0.3)
    │
Linear(128 → 64)         →  ReLU  →  Dropout(p=0.3)
    │
Linear(64 → 1)           →  logit (single output)
```

**Computing uncertainty:**
```
1. Set model to train() mode (keeps dropout active during inference)
2. Run the input through the network N=10 times:
   logits = [net(vec) for _ in range(10)]    → 10 different values
3. Compute standard deviation across runs:
   std = stdev(logits)
4. Normalize by running maximum std seen so far:
   uncertainty = std / running_max_std        → ∈ [0, 1]
```

Each of the 10 runs randomly drops 30% of neurons, so each run sees a slightly
different sub-network. A confident region of the input space produces similar
logits regardless of which neurons drop — low variance, low uncertainty.
An unfamiliar or ambiguous input produces wildly varying logits — high variance,
high uncertainty.

**Why normalize by running max?** The raw std values depend on the weight
magnitudes, which change as the model trains. Normalizing by the largest std seen
keeps the output in [0, 1] without requiring re-calibration.

---

### 3.3 PredictorHead — Prediction Error via One-Step-Ahead Forecasting

**Algorithm:** Online next-step prediction with MSE error

**What it answers:** *Was this input predictable from the previous one?*

This is the core idea of **predictive coding** — a theory from neuroscience that
the brain is constantly predicting incoming signals and only propagates the
*surprise* (prediction error) upward. High prediction error = something
unexpected happened.

**Architecture:**
```
Input (features,)
    │
Linear(features → 128)  →  ReLU
    │
Linear(128 → features)  →  Sigmoid    (output values ∈ [0, 1], matching spikes)
```

**Computing prediction error:**
```
Time step t:
  1. Store the current vec as prev_vec
  2. Return 0.0 (no previous step to compare against)

Time step t+1:
  1. Use prev_vec to predict what t+1 should look like:
     predicted = net(prev_vec)
  2. Compute MSE between prediction and actual:
     mse = mean((predicted - vec)²)
  3. Normalize by running maximum MSE:
     error = mse / running_max_mse        → ∈ [0, 1]
  4. Update prev_vec = current vec
```

For normal, predictable traffic flow, the predictor learns to anticipate the next
reading well — low MSE, low prediction error. When a crash occurs, the sudden
speed drop and occupancy spike are *not predicted* from the pre-crash state —
high MSE, high prediction error.

**Key insight:** The predictor is trained online and continuously adapts to the
current traffic regime. If traffic is always slow on Friday evenings, the predictor
will eventually learn this and not flag it as anomalous.

---

### 3.4 EpisodicMemory — Memory Confidence via Cosine Ring Buffer

**Algorithm:** Cosine similarity against a fixed-size ring buffer

**What it answers:** *Does this input resemble something I've recently seen?*

This is the inverse of novelty — where the prototype store captures *long-term*
typical patterns, episodic memory captures *recent* experience. Low memory
confidence means the current input doesn't resemble anything in recent memory,
which can indicate either a novel situation or a returning rare event.

**Structure:**
```python
_buffer: deque(maxlen=256)   # stores the last 256 mean-pooled spike vectors
```

**Computing confidence:**
```
1. Stack all stored vectors:
   stored = stack(buffer)          → (N, features)
2. Normalize both query and stored vectors
3. Compute cosine similarity to every stored vector:
   sims = vec_norm @ stored_norm.T → (N,)
4. memory_confidence = max(sims).clamp(0, 1)
```

High confidence (near 1.0): the current pattern closely matches a recent
observation — situation is familiar, low escalation pressure.

Low confidence (near 0.0): the current pattern doesn't match anything recent —
could be a new anomaly type, a regime change, or a rare event returning.

**Note:** During cold start (empty buffer), this returns 0.0 — which triggers
higher escalation in early operation. This is intentional: the system is
conservative when it has no experience to draw on.

---

### 3.5 SpatialCorrelationHead — Cross-Sensor Anomaly via Online Z-Score

**Algorithm:** Welford online variance estimation + sigmoid normalization

**What it answers:** *Is the spatial pattern across sensors anomalous?*

In the FT-AED traffic dataset, the last 4 features in each window are
freeway-wide spatial features: `speed_std`, `speed_min`, `occ_max`,
`speed_range`. During crashes, the standard deviation of speed across
milemarkers jumps from ~3 mph (normal) to ~17 mph (crash) — a 5.9× separation.

This head captures that cross-sensor pattern without requiring explicit knowledge
of which sensors are adjacent.

**Algorithm:**
```
1. Extract the last 4 features (spatial context):
   spatial = vec[-4:]

2. Compute L2 norm as a scalar anomaly score:
   raw_score = ||spatial||₂

3. Maintain running mean and std via Welford's online algorithm:
   n += 1
   delta = raw_score - running_mean
   running_mean += delta / n
   running_std  = sqrt(running_std² * (n-1)/n + delta * delta₂ / n)

   (Welford's method computes exact variance in one pass, no stored history)

4. Z-score:
   z = (raw_score - running_mean) / (running_std × 3)

5. Sigmoid to [0, 1]:
   spatial_correlation = sigmoid(z)
```

If `input_dim < 16` (domain has no spatial features), this returns 0.0
gracefully — the head degrades without error for non-spatial domains.

**Why Welford's algorithm?** It computes exact variance in a single pass without
storing past values — O(1) memory and O(1) time per update. This is critical for
streaming data at 3.7M samples.

---

### 3.6 RiskHead — Composite Risk Score

**Algorithm:** Weighted linear combination + trend detection + sigmoid

**What it answers:** *Overall, how likely is this to be worth escalating?*

The risk score is the **primary signal used by the gate**. It combines all five
other sub-signals into one number, augmented by a short-term trend component
(is risk rising?).

**Default weights** (set from domain knowledge of traffic crashes):
```
novelty             → 0.15   (moderate — novel doesn't always mean dangerous)
uncertainty         → 0.15   (moderate — uncertainty is necessary but not sufficient)
prediction_error    → 0.25   (high — crashes cause sudden unpredictable changes)
1 - memory_conf     → 0.20   (medium-high — unfamiliar patterns are risky)
spatial_correlation → 0.25   (high — cross-sensor spatial anomalies are strongest signal)
```

**Computing risk:**
```
1. Build signal vector:
   signals = [novelty, uncertainty, pred_error, 1 - mem_conf, spatial]

2. Weighted sum:
   raw = dot(weights, signals).clamp(0, 1)

3. Compute trend (is risk rising over the last 5 steps?):
   if history has ≥ 5 entries:
     trend = (raw - mean(last_5_in_history)).clamp(-1, 1)
   else:
     trend = 0.0

4. Combine with trend:
   combined = raw + 0.3 × trend

5. Rescale and sigmoid (centers at 0.5, spreads the range):
   risk = sigmoid(combined × 4 - 2)
```

**Why add trend?** A single anomalous window might be noise. But if risk has been
steadily rising for 5 windows, that trend is itself a strong signal — pre-crash
deceleration patterns take 30–90 seconds to develop. The trend component captures
this temporal structure without adding recurrent architecture.

**Weights are adaptive:** The `MemoryConsolidation` module (Section 6.4) updates
these weights based on which signals were actually predictive of confirmed crashes.

---

### 3.7 ReasoningScore — The Output Contract

```python
@dataclass
class ReasoningScore:
    novelty:             float   # 0 = familiar, 1 = completely new
    uncertainty:         float   # 0 = confident, 1 = maximally uncertain
    prediction_error:    float   # 0 = perfectly predicted, 1 = total surprise
    memory_confidence:   float   # 0 = unfamiliar, 1 = recently seen
    spatial_correlation: float   # 0 = normal spatial pattern, 1 = anomalous
    risk:                float   # 0 = definitely safe, 1 = escalate immediately
```

All values ∈ [0, 1]. All values are computed **without hard thresholds** — the
gate decides what levels trigger escalation, and that decision is learned.

```python
score.as_tensor()  →  Tensor shape (6,)   # used by GateLearner as input features
```

---

## 4. Layer 3 — Adaptive Reasoning Gate

**Files:** `nv_platform/gate.py`, `nv_platform/gate_learner.py`

### What it does

Decides, for each incoming `ReasoningScore`, whether to invoke the expensive LLM
agent (Layer 4) or handle the event on the fast automatic path. The key
requirement: reduce LLM calls by ≥30% while keeping crash recall >60%.

The gate has two components that work together:

---

### 4.1 GateLearner — Logistic Regression Meta-Learner

**Algorithm:** Online logistic regression with asymmetric loss (FN-penalized BCE)

**Feature vector (9-dimensional):**
```
From ReasoningScore.as_tensor() (6 dims):
  [0] novelty
  [1] uncertainty
  [2] prediction_error
  [3] memory_confidence
  [4] spatial_correlation
  [5] risk

From EscalationHistory (last 50 decisions, 3 dims):
  [6] recent_escalation_rate   = #escalated / total
  [7] recent_recall            = TP / (TP + FN)  over labelled decisions
  [8] recent_precision         = TP / (TP + FP)  over labelled decisions
```

The history features give the learner **self-awareness** — it knows how it has
been performing recently and can adjust. If recall is dropping (more crashes being
missed), history features shift in a direction that pushes the decision boundary
toward more escalations.

**Model architecture:**
```
nn.Linear(9, 1)   →   sigmoid   →   probability of escalation
```

One linear layer with one output. Deliberately minimal — the feature engineering
in `_extract_features()` does the work; the learner just finds the right weights.

**Loss function:**
```
Binary Cross-Entropy with asymmetric sample weights:

  weight[i] = fn_penalty (default 5.0)   if label=1 and not escalated  (false negative)
  weight[i] = 1.0                        otherwise

L = -sum_i( w_i × [y_i × log(p_i) + (1-y_i) × log(1-p_i)] )
```

False negatives (missed crashes, label=1, decision=False) are penalized **5×
more** than false positives. This is the mechanism that protects against the
trade-off of fewer LLM calls at the cost of missing crashes. The gate is
structurally biased toward "escalate when in doubt."

**Update schedule:**
```
Every 50 decisions (when len(history) % 50 == 0):
  1. Collect all labelled entries from the last 500 decisions
  2. Build the (X, y, w) dataset
  3. Run 20 epochs of SGD with learning_rate=0.01
  4. Increment update_count
```

This is **online learning with batch replay** — each update trains on a growing
window of recent history (not just the last 50), which prevents forgetting while
keeping the model current.

**Initialization:**
```python
nn.init.zeros_(model.weight)
nn.init.constant_(model.bias, 0.0)
```

Starting with all-zero weights means `sigmoid(0) = 0.5`, right at the default
threshold of 0.5 — the gate starts by escalating ~50% of events before it has
learned. In practice, `train_gate.py` pre-trains on the full training set so the
gate arrives at evaluation with a meaningful boundary.

---

### 4.2 ReasoningGate — Decision Logic

```python
gate = ReasoningGate(learner=GateLearner(), threshold=0.5)
escalate, confidence = gate.should_escalate(score, history)
```

**Decision algorithm:**
```
1. Compute score tensor from ReasoningScore (6-dim)

2. Hard floor check:
   hard_escalate = any(score_tensor > 0.95)
   → Always escalate if ANY signal is extreme (FN protection)
   → This fires before the learner is trained, preventing cold-start missed crashes

3. Learned probability:
   prob = learner.predict_proba(score_tensor, history)

4. Final decision:
   escalate = hard_escalate OR (prob >= threshold)

5. Log to EscalationHistory:
   entry = {id: uuid4(), score_tensor, decision, confidence=prob, timestamp, outcome=None}
   history.append(entry)

6. Trigger learner update:
   if len(history) % learner.update_every == 0:
       learner.update(history)

7. Return (escalate, prob)
```

The hard floor at 0.95 is a **non-negotiable safety net** — if novelty,
uncertainty, or risk is at 95% of their maximum, the gate escalates regardless
of what the learner says. This prevents a poorly calibrated learner from
catastrophically missing an obvious crash during early training.

**Outcome logging:**
```python
gate.log_outcome(decision_id, actual_label=1, history)
# → history.set_outcome(id, 1)  — marks the entry so the learner can use it
```

When the ground truth label becomes known (e.g., from a human reviewer or a
downstream database), it is attached to the logged decision entry. The learner
only trains on entries that have known outcomes.

---

### 4.3 EscalationHistory — Rolling Decision Log

```python
@dataclass
class EscalationHistory:
    maxlen: int = 500
    log: deque  # rolling window of last 500 decisions
```

Each entry:
```python
{
    "id":           str,       # uuid4 — used to attach outcomes later
    "score_tensor": Tensor,    # shape (6,) — full ReasoningScore as tensor
    "decision":     bool,      # True = escalated
    "confidence":   float,     # learner's output probability
    "timestamp":    str,       # ISO-8601 UTC
    "outcome":      int|None,  # 0/1 when known, None until then
}
```

The deque's `maxlen=500` is a hyperparameter balancing two concerns:
- Too small: learner forgets historical patterns, can oscillate
- Too large: learner adapts too slowly to distribution shifts

During `train_gate.py`, `maxlen` is set to `N + 100` (the full training set) to
allow full-history learning during offline pre-training.

---

## 5. Layer 4 — LLM Agent Plugin

**File:** `nv_platform/agent_plugin.py`

### What it does

When the gate escalates an event, this plugin invokes a language model agent that
reasons about the event using a knowledge base and returns a structured decision.

### Architecture: LangGraph ReAct with RAG

```
Escalated Event + Context
         │
         ▼
   System Prompt:
   "You are an anomaly detection agent..."
         │
         ▼
   ┌─────────────┐
   │  LLM Loop   │  LangChain ChatAnthropic (Claude 3.5 Sonnet)
   │  (ReAct)    │  Max 3 tool calls per invocation
   └──────┬──────┘
          │ tool call?
    ┌─────┴─────┐
    │           │
    ▼           ▼
rag_retrieve  lookup_context
(ChromaDB)    (context dict)
    │           │
    └─────┬─────┘
          │ tool result
          ▼
   LLM processes result,
   may call another tool
          │
   (max 3 iterations)
          │
          ▼
   Final response: JSON
   {label, confidence, reasoning}
         │
         ▼
    AgentDecision
```

**ReAct (Reason + Act)** is an agent loop where the LLM alternates between:
1. **Reasoning** — thinking about what it knows and what it needs
2. **Acting** — calling a tool to get more information

This continues until the LLM produces a final answer without calling any tools.

**RAG tool (`rag_retrieve`):**
```python
def rag_retrieve(query: str) -> str:
    client = chromadb.PersistentClient(path=rag_source)
    collection = client.get_collection("knowledge")
    results = collection.query(query_texts=[query], n_results=5)
    return "\n\n".join(f"[{id}] {doc}" for id, doc in zip(ids, docs))
```

ChromaDB performs semantic similarity search over the domain knowledge base
(e.g., traffic crash patterns, freeway management procedures) and returns the
top-5 most relevant passages. The LLM can cite these in its reasoning.

**Context tool (`lookup_context`):**
```python
def lookup_context(key: str) -> str:
    return str(context.get(key))
```

Allows the agent to look up structured metadata — gate confidence score,
domain threshold hints, any other key-value pairs passed in by the caller.

### Output

```python
@dataclass
class AgentDecision:
    label:      str    # "crash" | "normal" | "anomaly" | "uncertain"
    confidence: float  # agent's self-reported confidence ∈ [0, 1]
    reasoning:  str    # free-text chain-of-thought
    sources:    list   # ChromaDB document IDs cited
```

### Graceful degradation

Every public method is wrapped in `try/except`:
```python
def reason(self, event, context, rag_source) -> AgentDecision:
    try:
        return self._run_agent(...)
    except Exception as exc:
        return AgentDecision(label="uncertain", confidence=0.0,
                             reasoning=f"LLM unavailable: {exc}")
```

API failures, network timeouts, missing ChromaDB collections — all produce a safe
`uncertain` fallback rather than crashing the pipeline.

### Response parsing

The parser first tries to extract a JSON block from the response:
```python
start = content.find("{")
end   = content.rfind("}") + 1
parsed = json.loads(content[start:end])
```

If JSON parsing fails, it falls back to keyword matching:
- "crash", "anomaly", "alert", "incident" → label = "anomaly", confidence = 0.7
- "no incident", "normal", "all clear" → label = "normal", confidence = 0.7
- No match → label = "uncertain", confidence = 0.4

### Model agnosticism

The model name is injected at construction: `LLMAgentPlugin(model="gpt-4o")`.
Any LangChain-compatible chat model works — the agent graph code is identical.
Only the `_import_langchain_anthropic()` helper needs updating for non-Anthropic
models.

---

## 6. Supporting Platform Modules

### 6.1 AutomaticAction — Fast Path

**File:** `nv_platform/automatic_action.py`

When `gate.should_escalate()` returns False, the pipeline takes the fast path
through `AutomaticAction.act()` — no LLM call, runs in microseconds.

**Decision tiers:**
```
risk < 0.25  → label="normal",  confidence = 1 - risk,  action="continue_monitoring"
risk < 0.45  → label="monitor", confidence = 0.6 - risk×0.5, action="increase_sampling_rate"
risk ≥ 0.45  → label="monitor", confidence = max(0.1, 0.5 - risk×0.3)
               (gate should have escalated this — safety net)
```

The domain can customize the action vocabulary by passing `domain_actions` at
construction:
```python
policy = AutomaticAction(domain_actions={"normal": "green_signal", "monitor": "amber_alert"})
```

---

### 6.2 SafetyChecker

**File:** `nv_platform/safety.py`

Validates the LLM agent's decision before it reaches the planner. Runs four
domain-agnostic checks:

| Rule | Check | Action on violation |
|---|---|---|
| 1 | Label is in `{normal, anomaly, crash, uncertain, monitor, alert}` | Flag violation |
| 2 | Confidence ∈ [0, 1] | Flag violation |
| 3 | Agent says "crash" but `risk < 0.20` | Downgrade confidence ×0.7 (possible false alarm) |
| 4 | Agent says "normal" but `risk > 0.80` | Downgrade confidence ×0.7 (possible missed crash) |
| 5 | Custom domain rules (injected at construction) | Domain-defined |

**Two modes:**
- `strict=False` (default): violations are logged, confidence is penalized, the
  decision passes through to the planner
- `strict=True`: any violation blocks the action, returns `label="uncertain"`

This is specifically designed to catch **hallucination-induced false alarms** —
an LLM that confidently says "crash" when the risk score is 0.05 should not
trigger an emergency response.

---

### 6.3 Planner

**File:** `nv_platform/planner.py`

Takes the safety-checked decision and converts it into a sequenced, executable
`ActionPlan` — a list of concrete steps the downstream system should take.

**Step templates:**
```python
"crash":   ["alert_traffic_management_centre",
            "activate_variable_message_signs",
            "notify_emergency_services",
            "increase_upstream_sensor_sampling_rate",
            "log_incident_to_database"]

"anomaly": ["flag_for_human_review",
            "increase_upstream_sensor_sampling_rate",
            "log_anomaly_to_database"]

"normal":  ["continue_monitoring"]
```

**Confidence adjustment formula:**
```
For crash/anomaly/alert labels:
    final_confidence = agent_confidence × 0.6 + risk × 0.4
    → Blends LLM's self-assessment with the objective risk score
    → Prevents a low-risk event from getting high crash confidence

For normal label:
    final_confidence = agent_confidence × (1 - risk × 0.3)
    → Reduces confidence if risk is elevated despite "normal" label
```

If the safety checker flagged violations, a `human_review_required` step is
prepended to every plan — ensuring human oversight on uncertain decisions.

---

### 6.4 MemoryConsolidation — Feedback Loop

**File:** `nv_platform/memory_consolidation.py`

**Algorithm:** Discriminability-based weight update with Welford-style averaging

Closes the feedback loop from confirmed outcomes back to Layer 2. After the
system acts and ground truth becomes available, this module updates the
`RiskHead` signal weights so the layer learns *which signals were actually
predictive* of crashes in this domain.

**Feedback input:**
```python
@dataclass
class Feedback:
    decision_id:      str             # UUID linking back to gate log
    actual_label:     int             # 0 or 1
    predicted_label:  str             # what system said
    score:            ReasoningScore  # signals at decision time
    spike_vec:        Tensor|None     # for episodic memory storage
    outcome_quality:  float           # [0,1] how well did the action work?
```

**Consolidation algorithm (runs every 20 feedbacks):**
```
1. Separate buffer into crashes and normals

2. Compute mean signal values for each group:
   crash_signals  = mean([novelty, uncertainty, pred_error, 1-mem_conf, spatial]
                         for all crash feedbacks)
   normal_signals = mean(...for all normal feedbacks)

3. Discriminability = (crash_signals - normal_signals).clamp(min=0)
   → How much higher is each signal during crashes vs. normal traffic?
   → High discriminability = this signal is a reliable crash predictor

4. New weights = 0.7 × discriminability + 0.3 × prior_weights
   → Blend: trust new evidence 70%, keep prior 30% (prevents overfitting)

5. Normalize weights to sum to 1.0

6. FN amplification: if there were missed crashes in the buffer:
   boost = 1 + fn_weight × (missed_crashes / total_crashes)
   new_weights = (new_weights × boost).clamp(0,1), renormalize
   → Amplifies ALL crash signals when crashes are being missed

7. Store confirmed crash spike_vec in EpisodicMemory
   → Future windows compared against this actual crash pattern
```

This creates a **continuous learning loop**: the system observes outcomes, figures
out which signals were predictive, and adjusts the risk computation accordingly —
without retraining the SNN or the gate learner.

---

## 7. Domain Plugin System

### 7.1 DomainPlugin ABC

**File:** `nv_platform/domain_plugin.py`

The abstract contract that enforces platform/domain separation. The platform
layers (`/nv_platform/`) never import concrete domain classes — they only call
the interface defined here.

```python
class DomainPlugin(ABC):

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """JSON Schema for one sensor event"""

    @property
    @abstractmethod
    def output_schema(self) -> dict:
        """JSON Schema for agent decisions"""

    @property
    @abstractmethod
    def knowledge_base_path(self) -> str:
        """Path to ChromaDB knowledge base"""

    @property
    @abstractmethod
    def threshold_hints(self) -> dict[str, float]:
        """Soft initial hints for the gate learner"""

    def validate_event(self, event: dict) -> bool:
        """jsonschema.validate(event, self.input_schema) — returns bool"""
```

**Schema validation** uses `jsonschema.validate()` (JSON Schema Draft 7). If an
event fails validation, the pipeline drops it silently and continues.

**The critical invariant** (enforced by `test_e2e_smoke.py`):
```python
# For every .py file in /nv_platform/:
for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Import):
        assert "domains" not in name
    if isinstance(node, ast.ImportFrom):
        assert not module.startswith("domains")
```

This AST-level check runs at CI time and will fail if any platform module
accidentally imports from `/domains/`.

---

### 7.2 TrafficDomain — FT-AED Crash Detection

**File:** `domains/traffic/domain.py`

The concrete domain implementation for the Nashville Freeway Anomaly (FT-AED)
dataset.

**Dataset:**
- ~3.7 million radar sensor readings (5-minute intervals)
- 4 lanes × 3 features per lane = 12 features per window
- `lane{1-4}_speed` (mph), `lane{1-4}_volume` (vehicles/5min), `lane{1-4}_occ` (occupancy %)
- ~400 labeled crash/near-crash events (severe class imbalance: ~0.01%)

**Threshold hints** (calibrated to the dataset's known statistics):
```python
{
    "novelty":           0.55,   # crash patterns are novel
    "uncertainty":       0.60,   # model is uncertain during crashes
    "prediction_error":  0.50,   # speed drops are unpredicted
    "memory_confidence": 0.25,   # crash events are rare in memory
}
```

These are *starting points* — the `GateLearner` adapts them over time.

**Helper methods** (domain-specific, not part of the platform contract):
```python
TrafficDomain.event_from_row(csv_row)   → platform event dict
TrafficDomain.event_to_tensor(event)    → Tensor shape (1, 1, 12)
```

---

## 8. Baselines

### AlwaysEscalateGate

**File:** `baselines/always_escalate_gate.py`

```python
def should_escalate(score) -> (True, 1.0)   # always
```

Upper bound reference: 100% recall, 100% LLM call rate. The learned gate must
achieve ≥30% reduction from this rate. Every metric is measured relative to this
baseline.

### AlwaysThresholdGate

**File:** `baselines/always_threshold_gate.py`

```python
max_signal = max(novelty, uncertainty, pred_error, 1 - mem_conf)
escalate = max_signal >= threshold     # default threshold=0.5
```

Rule-based, stateless, no learning. Represents what a simple production system
would do without any machine learning. Observed performance on FT-AED val set at
threshold=0.5:
- Escalation rate: 43.5% (↓56.5% reduction vs always-escalate)
- Recall: 57.0% (below the 60% floor — misses too many crashes)

This is the baseline the learned gate must beat on **both** dimensions
simultaneously: lower escalation rate than threshold, higher recall than threshold.

---

## 9. Training & Evaluation Scripts

### `preprocess.py`

Converts the raw `nashville_freeway_anomaly.csv` into windowed, spike-encoded
tensors saved as `.pt` files.

**Pipeline:**
```
CSV → normalize features (zero-mean, unit-variance per feature)
    → sliding window (window_size=20, stride=5)
    → encode as spike tensor via SNNInputPlugin
    → train/val/test split (70/15/15)
    → save to data/processed/{train,val,test}.pt
```

Adds 4 spatial features per window:
- `speed_std`: cross-lane speed standard deviation
- `speed_min`: minimum lane speed (slowest lane)
- `occ_max`: maximum occupancy (most congested lane)
- `speed_range`: max - min speed across lanes

Final shape: `data["X"]` is `(num_steps=25, N_windows, time_window=20, features=16)`.

### `train_gate.py`

Pre-trains the `GateLearner` on the full training split offline, then saves
weights to `data/processed/gate_weights.pt`.

**What it saves:**
```python
{
    "learner_state":       model.state_dict(),   # Linear(9,1) weights + bias
    "update_count":        int,
    "fn_penalty":          float,
    "update_every":        int,
    "implicit_prototypes": Tensor,               # PrototypeStore state
    "input_dim":           int,
}
```

### `evaluate.py`

Loads `gate_weights.pt`, runs all three gates on val/test split, writes results
to `analysis/results.csv`.

**Metrics reported per gate:**
```
eval_windows, escalated, escalation_rate,
tp, fp, fn, tn, recall, precision, f1, fp_rate, mean_confidence
```

**Key success thresholds:**
- `ReasoningGate.escalation_rate ≤ 0.70` (≥30% reduction from AlwaysEscalate)
- `ReasoningGate.recall ≥ 0.60` (no more than 40% of crashes missed)

---

## 10. Data Flow: Full End-to-End Trace

Tracing a single crash event (label=1) through the entire system:

```
1. RAW SENSOR EVENT
   {"timestamp": 1727000120, "lane1_speed": 15.2, "lane2_speed": 12.1, ...}
   → domain.validate_event(event)  ✓ passes jsonschema

2. LAYER 1 — SNN ENCODING
   tensor = event_to_tensor(event)       → shape (1, 1, 12)
   spikes = snn.process_events(tensor)   → shape (25, 1, 12)
   Rate-coded spike trains: lane1_speed sigmoid'd to ~0.8 firing prob
   (speed is low = post-normalization it's very negative = high sigmoid)

3. LAYER 2 — SCORE EXTRACTION
   vec = mean_pool(spikes)               → shape (12,)

   PrototypeStore:     cosine sim to 16 prototypes → closest is 0.21
                       novelty = 1 - 0.21 = 0.79   (very novel!)

   MCDropoutHead:      10 passes → logits std = 0.43, running_max = 0.45
                       uncertainty = 0.43/0.45 = 0.96

   PredictorHead:      predicted = net(prev_vec) vs actual vec
                       mse = 0.38, running_max = 0.40
                       prediction_error = 0.38/0.40 = 0.95

   EpisodicMemory:     max cosine sim to 256 stored vecs = 0.18
                       memory_confidence = 0.18   (very unfamiliar)

   SpatialCorrelation: spatial norm = 19.2, z-score = +2.1
                       spatial_correlation = sigmoid(2.1) = 0.89

   RiskHead:           raw = 0.15×0.79 + 0.15×0.96 + 0.25×0.95
                            + 0.20×(1-0.18) + 0.25×0.89
                          = 0.119 + 0.144 + 0.238 + 0.164 + 0.223 = 0.888
                       trend (risk rising for 5 steps): +0.12
                       risk = sigmoid(0.888×4 - 2 + 0.3×0.12) = 0.943

   ReasoningScore(novelty=0.79, uncertainty=0.96, prediction_error=0.95,
                  memory_confidence=0.18, spatial_correlation=0.89, risk=0.943)

4. LAYER 3 — GATE DECISION
   Hard floor: risk=0.943 > 0.95?  No. But uncertainty=0.96 > 0.95!
   hard_escalate = True  ← safety net fires

   learner.predict_proba([0.79, 0.96, 0.95, 0.18, 0.89, 0.94, 0.91, 0.82, 0.79])
                        → logit = 2.4 → sigmoid = 0.917

   escalate = True  (both hard floor and learner agree)
   confidence = 0.917

   history.append({id: "abc-123", decision: True, confidence: 0.917, outcome: None})

5. LAYER 4 — LLM AGENT
   agent.reason(event, {"gate_conf": 0.917}, "domains/traffic/rag_kb")

   System: "You are an anomaly detection agent..."
   Human:  "Sensor event: {lane1_speed: 15.2, ...}\nClassify this event."

   LLM calls rag_retrieve("sudden speed drop across multiple lanes")
   → ChromaDB returns: "[doc_3] Pre-crash signature: speed drops >40mph
                         within 2 minutes across ≥2 lanes..."

   LLM produces:
   {"label": "crash", "confidence": 0.92,
    "reasoning": "Sudden multi-lane speed collapse from 65→15mph..."}

   AgentDecision(label="crash", confidence=0.92, sources=["doc_3"])

6. SAFETY CHECK
   SafetyChecker.check(decision, score):
   - label "crash" ∈ valid_labels  ✓
   - confidence 0.92 ∈ [0,1]       ✓
   - label "crash", risk=0.943 > 0.20  ✓ (no false alarm concern)
   SafetyResult(safe=True, violations=[])

7. PLANNING
   Planner.plan(safety_result, score):
   confidence = 0.92×0.6 + 0.943×0.4 = 0.552 + 0.377 = 0.929
   steps = ["alert_traffic_management_centre",
            "activate_variable_message_signs",
            "notify_emergency_services",
            "increase_upstream_sensor_sampling_rate",
            "log_incident_to_database"]
   ActionPlan(label="crash", confidence=0.929, steps=[5 steps], safe=True)

8. FEEDBACK (when ground truth confirmed)
   consolidator.consolidate(Feedback(actual_label=1, score=score, spike_vec=vec))
   → stores spike_vec in EpisodicMemory  (future crashes will match this)
   → after 20 feedbacks: recomputes RiskHead weights from discriminability
   gate.log_outcome("abc-123", actual_label=1, history)
   → marks history entry for learner's next update
```

---

## 11. Key Design Decisions & Trade-offs

### Why spiking neural networks?

SNNs encode sensor data as spike trains — binary events in time. They are
biologically plausible models of real sensors (events happen, not continuous
values) and are inherently suited to streaming, event-driven data. The rate-coding
scheme maps naturally to sensor firing rates. The architecture is swappable for
future neuromorphic hardware execution.

### Why not a transformer for Layer 2?

Transformers require a full sequence upfront and have O(n²) attention. For
streaming anomaly detection at 3.7M samples, the five independent sub-modules of
the Implicit Predictive Layer are O(1) per sample (online updates), have no
sequence length limit, and produce interpretable sub-signals rather than a black
box latent.

### Why logistic regression for the gate learner?

Interpretability and sample efficiency. With only ~400 labeled crashes in 3.7M
samples, a complex model would overfit. Logistic regression is equivalent to
learning a linear decision boundary in the 9-dimensional feature space — enough
expressivity for this problem, and the weights directly indicate which features
drive escalation decisions.

### Why asymmetric FN penalty?

In safety-critical detection, the cost of a false negative (missed crash,
potentially fatal) vastly exceeds the cost of a false positive (unnecessary LLM
API call, ~$0.01). The 5× FN penalty ensures the learned boundary is structurally
biased toward escalation even as it learns to reduce unnecessary calls.

### Why ChromaDB for RAG?

ChromaDB is embedded (no external server), persists to disk, and has a clean
Python API. For domain knowledge bases that are updated infrequently and accessed
in batch, it's the right size of tool — production deployments might swap to
Pinecone or Weaviate without changing any platform code.

### The separation invariant

The AST import check in `test_e2e_smoke.py` is not just a code style rule — it is
the architectural guarantee that makes the system reusable. If any platform file
imports from `/domains/`, the platform layers are no longer domain-agnostic and a
new team can't add their domain without understanding traffic-specific code. The
CI test makes this guarantee mechanically enforceable.
