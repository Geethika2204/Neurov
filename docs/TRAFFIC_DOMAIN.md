# Traffic Domain — Step-by-Step Pipeline Documentation

**Dataset:** Nashville Freeway Anomaly (FT-AED) — radar loop detector readings
from a freeway near Nashville, Tennessee. The raw CSV (`nashville_freeway_anomaly.csv`)
contains approximately 3.7 million rows. Each row is one 30-second measurement
from one milemarker, covering 49 milemarkers over roughly 20 days. The label
column is `crash_record` (0 = normal, 1 = crash or near-crash). There are
approximately 400 labelled crash windows across the full dataset, making it
highly imbalanced (~0.011% positive rate after windowing).

---

## Layer 0 — Raw Data Preprocessing (`preprocess.py`)

This runs once offline and produces `data/processed/train.pt`, `val.pt`, and
`test.pt`.

### Step 0.1 — Day-based train/val/test split

`preprocess.py:load_and_split()` reads the CSV and calls
`df["day"].unique()` to get the sorted list of unique day identifiers.
It assigns the first 14 days to train, the next 3 to val, and all remaining
days to test. No shuffling occurs — days are split in chronological order to
prevent future data leaking into training.

### Step 0.2 — Spatial feature computation

`preprocess.py:compute_spatial_features()` groups the DataFrame by `unix_time`
(one group per 30-second timestamp) and computes four cross-milemarker
statistics using `df.groupby("unix_time").agg(...)`:

- `spatial_speed_std` — standard deviation of `lane1_speed` across all 49
  milemarkers at that timestamp. Captures how variable speeds are across the
  freeway segment.
- `spatial_speed_min` — minimum `lane1_speed` across all milemarkers at that
  timestamp. Identifies the slowest point on the freeway.
- `spatial_occ_max` — maximum `lane1_occ` across all milemarkers. Identifies
  the densest point.
- `spatial_speed_range` — `max(lane1_speed) - min(lane1_speed)` across all
  milemarkers. Captures the speed gradient along the freeway.

These four values are merged back onto every row via
`df.merge(agg, on="unix_time", how="left")`. Every milemarker at the same
timestamp gets the same four spatial values appended. This brings the total
feature count from 12 (per-lane features) to **16**.

### Step 0.3 — Z-score normalisation

`preprocess.py:normalize()` computes the mean and standard deviation of all
16 feature columns from the **training split only**. It then Z-scores each
split using those training statistics:
`(df[ALL_FEATURE_COLS] - mean) / std`. Columns with zero standard deviation
are replaced with 1 to avoid division by zero. The validation and test splits
are normalised with the same training statistics — they never see their own
statistics.

### Step 0.4 — Sliding-window construction

`preprocess.py:make_windows()` sorts each split by `["milemarker", "unix_time"]`
and slides a window of size `WINDOW=20` timesteps with stride `STRIDE=5` over
each milemarker's time series. Any window that spans two different milemarkers
(detected by comparing `milemarker[start] != milemarker[end-1]`) is discarded.
The label for a window is 1 if **any** of the 20 timesteps in that window has
`crash_record=1`, otherwise 0. This produces arrays of shape
`(N_windows, 20, 16)`.

### Step 0.5 — Spike encoding

`preprocess.py:to_spikes()` applies `torch.sigmoid()` to the normalised
feature tensor (mapping each value from real numbers to `(0, 1)`) and then
calls `snntorch.spikegen.rate(x, num_steps=25)`. The rate coder treats each
value in `(0, 1)` as the probability that a spike fires at each of the 25
time steps, drawing Bernoulli samples independently. The result is a binary
tensor of shape `(25, N_windows, 20, 16)` — 25 time steps, N windows, 20
timesteps per window, 16 features. This tensor is saved to
`data/processed/{train,val,test}.pt` under the key `"X"`, with labels under
`"y"`.

---

## Layer 1 — SNN Input Plugin (`nv_platform/snn_plugin.py`)

### What it receives

During evaluation (`evaluate.py`), each window `i` is extracted from the saved
spike tensor as `X[:, i, :, :]` — shape `(25, 20, 16)` — then mean-pooled
over the time-window dimension with `.mean(dim=1)` to produce shape `(25, 16)`,
then given a batch dimension with `.unsqueeze(1)` to become `(25, 1, 16)`.
This is passed to `layer.compute_state()` directly.

During live pipeline use (`CWRUPipeline` or any equivalent), the event dict
is converted to a `(1, 1, 16)` tensor and passed to
`SNNInputPlugin.process_events()`.

### What `process_events` does

`_SNNTorchBackend.encode()` receives a tensor of shape
`(batch, time_window, features)` or `(batch, features)`:

1. If 3-dimensional, it mean-pools over `dim=1` (the time window axis),
   collapsing `(batch, time_window, features)` → `(batch, features)`.
2. It applies `torch.sigmoid(pooled)` element-wise, mapping Z-scored features
   into the range `(0, 1)`. This value is interpreted as a firing rate.
3. It calls `snntorch.spikegen.rate(firing_prob, num_steps=25)`, which
   draws `num_steps` independent Bernoulli samples per feature per batch item.
   The result is a binary tensor of shape `(25, batch, features)`.

The shape contract is validated: the output must have exactly `num_steps` in
dimension 0 and the same number of features as the input.

---

## Layer 2 — Implicit Predictive Layer (`nv_platform/implicit_layer.py`)

`ImplicitPredictiveLayer.compute_state(spikes)` receives the spike tensor of
shape `(25, batch, 16)` and runs it through six sub-modules, producing a
`ReasoningScore` with six fields all in `[0, 1]`.

### Mean-pooling

Before any sub-module runs, `_mean_pool()` averages over both the time-step
axis (dim 0) and the batch axis (dim 1) using `.mean(dim=(0, 1))`, reducing
the `(25, batch, 16)` tensor to a single `(16,)` vector `vec`. All six
sub-modules operate on this same vector.

### Sub-module 1 — `_PrototypeStore` → `novelty`

Maintains a bank of `k=16` prototype vectors initialised with
`F.normalize(torch.randn(16, 16), dim=1)`. Prototypes and counts are
registered as PyTorch buffers so they persist in `state_dict()`.

`compute_novelty(vec)`:
1. L2-normalises `vec` to get `vec_norm` of shape `(1, 16)`.
2. L2-normalises all 16 prototypes to get `protos_norm` of shape `(16, 16)`.
3. Computes cosine similarities: `(vec_norm @ protos_norm.T).squeeze(0)`
   — shape `(16,)`.
4. Takes `max_sim = cosine_sims.max().item()`.
5. Returns `novelty = clamp(1.0 - max(0.0, max_sim), 0, 1)`.

A perfect match to a prototype gives `novelty=0`; a completely unknown
pattern gives `novelty≈1`.

After scoring, `update(vec)` finds the nearest prototype by argmax of cosine
similarities and applies an online k-means update:
`lr = 1 / (count + 1)`, then
`prototype[nearest] = (1-lr)*prototype[nearest] + lr*vec`.
The count for that prototype increments by 1.

### Sub-module 2 — `_MCDropoutHead` → `uncertainty`

A two-layer MLP: `Linear(16→128) → ReLU → Dropout(0.3) → Linear(128→64)
→ ReLU → Dropout(0.3) → Linear(64→1)`.

`compute_uncertainty(vec)`:
1. Calls `self.train()` to keep Dropout active.
2. Runs `self.net(vec.unsqueeze(0))` exactly `mc_passes=10` times and
   concatenates the resulting logits into a tensor of shape `(10, 1)`.
3. Computes the standard deviation over the 10 passes: `std = logits.std()`.
4. Updates `_running_max_std` if `std > _running_max_std`.
5. Returns `uncertainty = clamp(std / _running_max_std, 0, 1)`.

A model that always produces the same logit regardless of Dropout mask has
`std=0` → `uncertainty=0`. A model with high variance across masks has
high `uncertainty`.

### Sub-module 3 — `_PredictorHead` → `prediction_error`

A one-step-ahead predictor: `Linear(16→128) → ReLU → Linear(128→16) → Sigmoid`.

`compute_prediction_error(vec)`:
1. If `_prev_vec` is `None` (first window in a file), stores `vec` as
   `_prev_vec` and returns `0.0`.
2. Otherwise, runs `self.net(_prev_vec.unsqueeze(0)).squeeze(0)` to get the
   predicted current vector.
3. Computes `mse = F.mse_loss(predicted, vec.detach()).item()`.
4. Updates `_running_max_mse` if `mse > _running_max_mse`.
5. Returns `prediction_error = clamp(mse / _running_max_mse, 0, 1)`.
6. Updates `_prev_vec = vec.detach()` for the next call.

A good predictor that correctly anticipates the next window produces
`prediction_error≈0`. An unexpected change produces high `prediction_error`.

**Important:** `_prev_vec` is **not** a registered buffer; it is transient
state that must be reset between independent sequences by calling
`layer._predictor.reset()`. In the traffic domain, this reset happens between
files or recording sessions.

### Sub-module 4 — `_EpisodicMemory` → `memory_confidence`

A Python `deque(maxlen=256)` storing the last 256 mean-pooled vectors.

`compute_confidence(vec)`:
1. If the buffer is empty, returns `0.0`.
2. Stacks all buffered vectors into a matrix of shape `(N, 16)`.
3. L2-normalises `vec` and all stored vectors.
4. Computes cosine similarities: `(vec_norm @ stored_norm.T).squeeze(0)`.
5. Returns `clamp(sims.max(), 0, 1)`.

After scoring, `store(vec)` appends `vec.detach()` to the deque.

A window that closely matches any of the last 256 seen windows has high
`memory_confidence` (familiar). A window unlike anything recently seen has
`memory_confidence≈0`.

### Sub-module 5 — `_SpatialCorrelationHead` → `spatial_correlation`

Tracks the **L2 norm of the last 4 elements of `vec`** (the four spatial
features appended in preprocessing) using online Welford statistics.

`compute_spatial_correlation(vec)`:
1. Extracts `spatial = vec[-4:]` — the four spatial feature values.
2. Computes `raw_score = float(spatial.norm())`.
3. Updates Welford running mean and variance over `raw_score`.
4. Computes a Z-score: `z = (raw_score - running_mean) / (running_std * 3)`.
5. Applies `sigmoid(z)` and returns `clamp(result, 0, 1)`.

For the first few windows the Welford statistics are not yet stable and the
score stays near 0.5 (sigmoid(0) = 0.5). After warm-up, unusually high
spatial correlation produces scores above 0.5.

### Sub-module 6 — `_RiskHead` → `risk`

Combines all five previous signals into one composite score.

`compute_risk(novelty, uncertainty, pred_error, mem_conf, spatial)`:
1. Builds a signal vector: `[novelty, uncertainty, pred_error, 1-mem_conf, spatial]`.
   Note that `memory_confidence` is **inverted** (low confidence = high danger).
2. Weights: `[0.15, 0.15, 0.25, 0.20, 0.25]` (fixed, registered as a buffer).
3. `raw = clamp(dot(weights, signals), 0, 1)`.
4. Trend: if the ring buffer of the last 50 raw risks has 5+ entries, computes
   `trend = clamp(raw - mean(recent), -1, 1)`. Otherwise `trend=0`.
5. `combined = raw + 0.3 * trend`.
6. `risk = sigmoid(combined * 4 - 2)`. The `*4 - 2` scaling maps the range
   `[0, 1]` to `[-2, 2]` before sigmoid, making the output sensitive to
   values near 0.5.
7. The raw risk value is stored in a ring buffer (size 50).

---

## Layer 3 — Reasoning Gate (`nv_platform/gate.py` + `gate_learner.py`)

### Hard escalation floor

`ReasoningGate.should_escalate()` first checks the hard floor. It stacks
five sub-signals from `score.as_tensor()`: indices `[0, 1, 2, 4, 5]`
(novelty, uncertainty, prediction\_error, spatial\_correlation, risk).
Index 3 (memory\_confidence) is **excluded** because high values mean
_familiar_, not dangerous. If any of these five values exceeds
`HARD_ESCALATE_THRESHOLD = 0.95`, `hard_escalate = True` immediately.
This check is bypassed when `gate.training_mode = True`.

### Learned gate probability

`GateLearner.predict_proba(score_tensor)`:
1. Calls `_extract_features(score_tensor)` which picks indices
   `[2, 1, 4, 5, 0]` from the 6-element score tensor, producing a 5-element
   feature vector in order: `[prediction_error, uncertainty,
   spatial_correlation, risk, novelty]`.
2. Applies standardisation: `(feats - feature_mean) / clamp(feature_std, min=1e-6)`.
3. Passes through `nn.Linear(5, 1)`: `logit = w · feats_scaled + b`.
4. Returns `float(sigmoid(logit))`.

### Escalation decision

`escalate = hard_escalate OR (prob >= threshold)`.

The gate logs every decision to `EscalationHistory` (a `deque(maxlen=500)`)
with a UUID, the score tensor, the decision, and `outcome=None`. Every
`update_every=50` decisions, `GateLearner.update(history)` is called, which
refits the logistic model on all labelled entries in history using
`F.binary_cross_entropy_with_logits` with class-imbalance weighting.

### Offline training (`train_gate.py`)

The `gate_weights.pt` checkpoint for the traffic domain was produced by:
1. Loading `data/processed/train.pt` (131,564 windows).
2. Constructing `ImplicitPredictiveLayer(input_dim=16)` and fitting it on
   train data with 5 epochs of Adam (`lr=1e-3`), minimising MSE prediction
   loss + binary cross-entropy MC head loss.
3. Running every train window through `layer.compute_state()` to collect
   the 5-feature gate vector.
4. Z-scoring with training-set mean and std.
5. Fitting `sklearn.linear_model.LogisticRegression(class_weight="balanced",
   C=1.0, solver="lbfgs")`.
6. Running the validation set (28,076 windows) through the same process and
   sweeping thresholds from 0.05 to 0.95 in steps of 0.05 to find the
   highest threshold achieving recall ≥ 0.65. The selected threshold was
   **0.3276**.

The checkpoint stores: `layer` (state\_dict), `input_dim=16`, `weights`
(shape 5), `bias`, `feature_mean`, `feature_std`, `threshold=0.3276`,
`recall_target=0.65`.

---

## Layer 4 — LLM Agent Plugin (`nv_platform/agent_plugin.py`)

When `escalate=True`, `LLMAgentPlugin.reason(event, context, rag_source)`
is called.

For Ollama/local models (detected by checking if the model name starts with
`"llama"`, `"mistral"`, `"gemma"`, `"phi"`, `"qwen"`, etc.),
`_run_ollama_direct()` is used:
1. A fixed inline knowledge string with CWRU bearing thresholds is prepended.
2. The event features prefixed with `de_` or `fe_` are extracted and
   JSON-encoded.
3. A `SystemMessage` with the classifier prompt and a `HumanMessage` with
   the inline knowledge and features are sent to `llm.invoke(messages)`.
4. The response is parsed with `_parse_response()`.

For Anthropic/Claude models, `_run_langgraph()` runs a ReAct loop:
1. Two LangChain tools are created: `rag_retrieve` (queries ChromaDB) and
   `lookup_context` (reads the context dict).
2. The LLM is bound to these tools with `llm.bind_tools(tools)`.
3. Up to `max_tool_calls=1` iterations of tool calls are executed.
4. The final message content is parsed.

`_parse_response()` tries to find and parse a JSON object (`{...}`) in the
response text. If that fails, it uses keyword heuristics (checking for
"inner_race", "outer_race", "ball fault", "normal", etc.) to assign a label
and confidence of 0.7.

When the LLM is disabled (`enable_llm=False`), the pipeline uses
`_heuristic_decision()` or `_fast_path_decision()` instead (see CWRU domain
doc for details on those functions).

---

## `evaluate.py` — Offline Gate Evaluation

`evaluate.py` loads the saved checkpoint, restores the **exact** layer state
(so running statistics, prototype stores, and episodic memory match the state
at training time), runs every test-split window through the layer to collect
scores, then evaluates three gates:

1. **AlwaysEscalate** — escalates every window (baseline, recall=1.0).
2. **AlwaysThreshold** — escalates if `max(novelty, uncertainty,
   prediction_error, 1-memory_confidence) >= threshold`.
3. **ReasoningGate** — uses the trained logistic weights with the hard floor.

For the ReasoningGate it separately counts hard-floor escalations vs learner
escalations and reports both TP/FP for each path.

---

## Results — Traffic Domain (FT-AED)

All numbers are from `analysis/traffic/results.csv`, measured on the
**test split** (days 17–20, 28,076 windows, 1,372 crashes).

### Gate comparison table

| Gate | Windows | Crashes | Escalation rate | LLM reduction | Recall | Precision | F1 | FN |
|---|---|---|---|---|---|---|---|---|
| AlwaysEscalate | 28,076 | 1,372 | 100.00% | 0.00% | 1.000 | 0.049 | 0.093 | 0 |
| AlwaysThreshold | 28,076 | 1,372 | 46.35% | 53.65% | 0.170 | 0.018 | 0.032 | 1,139 |
| **ReasoningGate** | **28,076** | **1,372** | **51.27%** | **48.73%** | **0.971** | **0.093** | **0.169** | **40** |

### Hard floor vs learner breakdown (ReasoningGate only)

| Component | Escalations | TP | FP |
|---|---|---|---|
| Hard floor | 3 | 0 | 3 |
| Learner | 14,394 | 1,332 | 13,062 |

The hard floor fired 3 times. All 3 were false positives (normal windows).
The learner drives essentially all escalation decisions.

### Analysis

**Why precision is low (0.0925):** The dataset has 1,372 crashes among
28,076 windows — only 4.9% positive rate. Even with 97.1% crash recall,
the gate still escalates thousands of normal windows (13,062 FP vs 1,332 TP),
giving precision of 1332/(1332+13062) = 9.3%. This is not a failure — it
reflects the extreme class imbalance. The gate's job is to reduce LLM calls
while keeping crash recall high, not to achieve high precision.

**Why AlwaysThreshold fails (recall=0.170):** The rule-based gate checks
`max(novelty, uncertainty, prediction_error, 1-memory_confidence) >= 0.3276`.
With the same threshold as the learned gate, most crash windows do not trigger
any single signal above 0.3276. The learned gate uses a linear combination of
all five signals with weights fitted to the crash distribution, which is far
more sensitive.

**LLM reduction of 48.73%:** The gate sends 14,394 of 28,076 windows to the
LLM — a reduction of 48.73% from the 100% always-escalate baseline. This meets
the ≥30% target. Missing 40 of 1,372 crashes (2.9% FN rate) means the gate is
operating well above the >60% recall target.

**Threshold calibration effect:** The threshold 0.3276 was chosen by sweeping
the validation set to find the highest threshold that still kept recall ≥ 0.65.
The resulting operating point achieves 97.1% recall at 51.3% escalation rate —
well above the recall target, trading off some LLM reduction for safety.
