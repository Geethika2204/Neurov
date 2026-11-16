# CWRU Domain — Step-by-Step Pipeline Documentation

**Dataset:** Case Western Reserve University (CWRU) Bearing Fault Dataset.
Vibration signals recorded from a motor-driven mechanical test rig at the
Drive-End (DE) and Fan-End (FE) accelerometers at 12,000 Hz sampling rate.
Four fault classes:

| Class | Integer label | Description |
|---|---|---|
| normal | 0 | Healthy bearing, no fault |
| inner_race | 1 | Defect on the inner raceway |
| ball | 2 | Defect on a rolling element (ball) |
| outer_race | 3 | Defect on the outer raceway |

Each class has four `.mat` or `.mat.zip` files recorded at different motor
loads (RPM). The dataset is split strictly by file — 3 files per class for
training, 1 file per class (a different RPM) for the held-out test.

---

## Layer 0 — Data Loading and Preprocessing

### Step 0.1 — Loading `.mat` files (`domains/cwru/loader.py`)

`CWRULoader.load_all()` iterates over `normal/`, `inner_race/`, `ball/`, and
`outer_race/` subdirectories. For each file ending in `.mat` or `.mat.zip`:

- **Plain `.mat`:** reads the file bytes directly with `Path.read_bytes()`,
  then passes them to `scipy.io.loadmat(io.BytesIO(data))`.
- **`.mat.zip`:** opens the ZIP archive, finds the first entry ending in
  `.mat` by scanning `zf.namelist()`, reads those bytes with `zf.read()`,
  then calls `scipy.io.loadmat()`.

`_extract_channels()` scans every key in the loaded mat dict. Keys matching
`_DE_TIME` (case-insensitive) become `de_time`, keys matching `_FE_TIME`
become `fe_time`, keys matching `_BA_TIME` become `ba_time`, and keys ending
in `RPM` become the `rpm` scalar. All time channels are flattened to 1-D float64
arrays. The file ID is extracted by stripping non-numeric characters from the
filename stem with `re.sub(r"[^0-9]", "", path.stem)`.

Each loaded file becomes a `CWRURecord` dataclass with fields:
`fault_class`, `label` (0–3), `file_id`, `source_path`, `de_time`,
`fe_time`, `ba_time`, `rpm`, `sample_rate=12000`.

### Step 0.2 — Windowing and feature extraction (`domains/cwru/preprocess.py`)

`CWRUPreprocessor(window_size=1024, overlap=0.5, use_fe=True)` sets
`step = max(1, int(1024 * (1 - 0.5))) = 512` samples. For each record it
slides a window from index 0 to `len(de_time) - 1024 + 1` in steps of 512.

For each window `de_seg = de_time[start:start+1024]`, `_extract_features()`
computes 9 statistics from the 1-D float64 segment:

1. `rms` = `sqrt(mean(x²))`
2. `std` = `std(x)`
3. `peak` = `max(|x|)`
4. `crest_factor` = `peak / rms`  (or `peak / 1e-9` if `rms=0`)
5. `kurtosis` = `(mean((x - mean(x))⁴) / std⁴) - 3`  (Fisher normalised,
   equals 0 for a Gaussian; fault signals produce elevated values)
6. `skewness` = `mean((x - mean(x))³) / std³`
7. `peak_to_peak` = `max(x) - min(x)`
8. `shape_factor` = `rms / mean(|x|)`
9. `impulse_factor` = `peak / mean(|x|)`

All values are clipped to `[-1e6, 1e6]` before returning.

If `use_fe=True` and the record has a non-empty `fe_time`, the same 9 features
are computed for the FE channel and concatenated: DE features first, then FE
features, giving a **18-dimensional** vector. If FE is absent, only 9 DE
features are returned.

The resulting `CWRUWindow` has: `fault_class`, `label`, `file_id`,
`window_idx`, `features` (shape 18), `feature_cols` (18 names), `rpm`,
`timestamp = window_idx * (512/12000)` seconds.

Feature column names follow the pattern `de_<stat>` and `fe_<stat>`:
`de_rms`, `de_std`, `de_peak`, `de_crest_factor`, `de_kurtosis`,
`de_skewness`, `de_peak_to_peak`, `de_shape_factor`, `de_impulse_factor`,
then the same 9 for `fe_`.

### Step 0.3 — Z-score normalisation (`CWRUPreprocessor.normalize_events`)

`normalize_events(events, fit=True)` builds a `(N, num_features)` matrix by
extracting all feature keys (everything except `timestamp`, `fault_class`,
`label`, `file_id`, `window_idx`, `rpm`). When `fit=True` it computes
`mean = mat.mean(axis=0)` and `std = mat.std(axis=0) + 1e-8` from these
events, stores them in `self._feature_mean` and `self._feature_std`, and
returns a new list of dicts with normalised values. When `fit=False` it
uses the previously stored statistics. The normalised event dicts contain
only metadata keys and Z-scored feature values.

**Data leakage prevention:** In `evaluate_held_out.py`, `fit=True` is called
on train events only. The test events are normalised with `fit=False`, reusing
the train statistics. The test files are never touched during statistics fitting.

### Step 0.4 — Heuristic threshold calibration (`evaluate_held_out.py:calibrate_thresholds`)

Calibrated thresholds are derived from train events only. For each class pair,
`calibrate_thresholds()` computes percentiles of `de_kurtosis` and `de_rms`:

- **Z-scored outer_race threshold:** midpoint between `inner_race de_kurtosis p95`
  and `outer_race de_kurtosis p5`.
- **Z-scored inner_race threshold:** midpoint between `normal de_kurtosis p95`
  and `inner_race de_kurtosis p5`.
- **Z-scored ball threshold:** midpoint between `normal de_rms p95` and
  `ball de_rms p5`.

The same three thresholds are computed on raw (un-normalised) event values
for the fast-path classifier. These six threshold values are stored in a dict
and never re-fitted on test data.

---

## Layer 1 — SNN Input Plugin (`nv_platform/snn_plugin.py`)

`CWRUDomain.event_to_tensor(event)` extracts the 18 feature values in
`feature_cols` order and returns a tensor of shape `(1, 1, 18)` —
`(batch=1, time_window=1, features=18)`.

`SNNInputPlugin.process_events(tensor)` calls `_SNNTorchBackend.encode()`:
1. `tensor.dim() == 3`, so it mean-pools over `dim=1`: `(1,1,18) → (1,18)`.
2. `torch.sigmoid(pooled)` maps each Z-scored feature to `(0,1)`.
3. `snntorch.spikegen.rate(firing_prob, num_steps=25)` draws 25 Bernoulli
   samples per feature, producing a binary tensor of shape `(25, 1, 18)`.

The `(25, 1, 18)` spike tensor is what flows into the ImplicitPredictiveLayer.

---

## Layer 2 — Implicit Predictive Layer (`nv_platform/implicit_layer.py`)

`ImplicitPredictiveLayer(input_dim=18)` constructs the same six sub-modules
as the traffic domain but with all linear layers and prototype stores sized
to 18 features instead of 16.

`compute_state(spikes)` with `spikes` of shape `(25, 1, 18)`:
1. `_mean_pool()` averages over `dim=(0,1)` → `(18,)` vector `vec`.
2. All six sub-modules receive this same `vec`.

### `_PrototypeStore(input_dim=18, k=16)` → `novelty`

16 prototypes of size 18. Cosine similarity computation and online k-means
update are identical to the traffic domain, but operating on 18-dimensional
vectors.

### `_MCDropoutHead(input_dim=18)` → `uncertainty`

`Linear(18→128) → ReLU → Dropout(0.3) → Linear(128→64) → ReLU →
Dropout(0.3) → Linear(64→1)`. MC uncertainty estimated over 10 stochastic
forward passes. Identical logic, different input size.

### `_PredictorHead(input_dim=18)` → `prediction_error`

`Linear(18→128) → ReLU → Linear(128→18) → Sigmoid`. Same one-step-ahead
predictor, 18 dimensions.

**Predictor reset between files:** In `evaluate_held_out.py`, the evaluation
loop checks `if file_id != prev_file_id: layer2._predictor.reset()`. Each
`.mat` file is an independent recording; carrying `_prev_vec` from the last
window of one file into the first window of the next would produce a
meaningless prediction error.

### `_EpisodicMemory(memory_size=256)` → `memory_confidence`

Same `deque(maxlen=256)` logic, 18-dimensional vectors.

### `_SpatialCorrelationHead(input_dim=18)` → `spatial_correlation`

Checks `input_dim >= MIN_SPATIAL_DIM=16` — 18 ≥ 16 so `_has_spatial=True`.
Takes `vec[-4:]` — the last 4 elements of the 18-dim vector, which are
`fe_shape_factor`, `fe_impulse_factor` and two others from the FE channel.
Computes their L2 norm and applies Welford normalisation + sigmoid.

### `_RiskHead` → `risk`

Identical formula: `raw = dot([0.15, 0.15, 0.25, 0.20, 0.25],
[novelty, uncertainty, pred_error, 1-mem_conf, spatial])`, then
`risk = sigmoid(raw*4 - 2)` with a trend correction from the last 50 steps.

### Layer state saved in checkpoint

`train_cwru_gate.py` warms the layer on all 4,496 training windows before
collecting features for the logistic regression. The resulting layer
`state_dict()` — containing all prototype vectors, running maxima, Welford
statistics, and episodic memory — is saved in the checkpoint under the key
`"layer"`. During evaluation, `layer2.load_state_dict(ckpt["layer"])` is
called before the test loop so the layer starts from the same state it had
after seeing the training data.

---

## Layer 3 — Reasoning Gate (`nv_platform/gate.py` + `gate_learner.py` + `domains/cwru/train_cwru_gate.py`)

### Feature vector — 7 dimensions

The CWRU gate uses a **7-dimensional** feature vector, not the standard 5.
Two raw vibration features are appended after the 5 ImplicitLayer signals:

| Index | Feature | Source |
|---|---|---|
| 0 | prediction_error | ImplicitLayer index 2 |
| 1 | uncertainty | ImplicitLayer index 1 |
| 2 | spatial_correlation | ImplicitLayer index 4 |
| 3 | risk | ImplicitLayer index 5 |
| 4 | novelty | ImplicitLayer index 0 |
| 5 | de_rms | Raw physical value from `_raw` dict |
| 6 | de_kurtosis | Raw physical value from `_raw` dict |

**Why these two were added:** The 5 ImplicitLayer signals cannot distinguish
ball faults from normal windows. Ball-fault windows produce similar novelty,
uncertainty, spatial correlation, risk, and prediction error to normal windows
(measured AUC ≤ 0.55 for every signal on the validation set). Raw `de_rms`
achieves AUC = 1.000 for ball vs normal. Raw `de_kurtosis` achieves AUC =
0.819 for all faults vs normal. These physics-informed features are already
computed during preprocessing (they are part of the 18-feature vector) so
adding them to the gate does not require any new computation.

### Training procedure (`train_cwru_gate.py`)

1. **Load records:** Identifies 12 training files (3 per class) and 4 val files
   (1 per class) by `file_id`. Test files are never loaded.

2. **Preprocess train:** `CWRUPreprocessor` windows and extracts 18 features.
   `normalize_events(fit=True)` computes Z-score statistics from train only.
   `_raw` physical values are attached to every train and val event under the
   `"_raw"` key.

3. **Preprocess val:** `normalize_events(fit=False)` applies train statistics
   to val events.

4. **Warm the ImplicitPredictiveLayer:** Iterates over all 4,496 train events
   in file order (resetting `_predictor` at each file boundary), passing each
   through the full `compute_state()` pipeline. This builds up the prototype
   store, Welford statistics, episodic memory, and running maxima. The scores
   are discarded; only the layer's internal state matters.

5. **Collect train features:** Iterates over all 4,496 train events again
   (resetting `_predictor` at file boundaries). For each event, calls
   `_extract_features(score.as_tensor())` to get the 5 ImplicitLayer signals,
   then appends `float(raw["de_rms"])` and `float(raw["de_kurtosis"])` to form
   the 7-element feature vector. Labels are 0 for normal, 1 for any fault class.

6. **Compute normalisation statistics:** `feat_mean = train_feats.mean(axis=0)`
   and `feat_std = max(train_feats.std(axis=0), 1e-6)` over shape `(4496, 7)`.

7. **Fit logistic regression:** Z-scores the train features with the train stats,
   then fits `sklearn.linear_model.LogisticRegression(class_weight="balanced",
   C=1.0, solver="lbfgs", max_iter=1000, random_state=0)`. The `class_weight=
   "balanced"` setting automatically weights the loss inversely proportional
   to class frequency.

8. **Collect val features:** Same process over the 1,659 val events.

9. **Threshold sweep:** Evaluates thresholds from 0.05 to 0.95 in steps of 0.05
   on the val set. For each threshold, computes TP, FP, FN, recall, precision,
   F1, escalation rate, and normal false escalation rate. Selects the **highest**
   threshold where `recall >= recall_target` (default 0.73). This ensures the
   gate is as conservative as possible while still meeting the recall floor.

10. **Save checkpoint:** Writes `data/processed/cwru_gate_weights.pt` with keys:
    `domain="cwru"`, `layer` (state\_dict), `input_dim=18`, `weights` (shape 7),
    `bias`, `feature_mean` (shape 7), `feature_std` (shape 7),
    `feature_names` (list of 7 names), `n_features=7`, `threshold`,
    `recall_target`, `train_file_ids`, `val_file_ids`, plus provenance metadata.

### Domain tag validation

When `pipeline.py` or `evaluate_held_out.py` loads a checkpoint, it reads
`ckpt.get("domain", None)`. If the tag is not `None` and not `"cwru"`, the
checkpoint is **rejected** with a `UserWarning` and the loop continues to the
next candidate checkpoint. The old `gate_weights.pt` has no domain tag
(`domain=None`), so it is not rejected — it falls through as a fallback with
a warning that domain is unknown. The new `cwru_gate_weights.pt` has
`domain="cwru"` and is always preferred.

### How `predict_proba` handles 7 features

`GateLearner.predict_proba(score_tensor, extra_features=tensor([de_rms, de_kurt]))`:
1. Calls `_extract_features(score_tensor)` → shape `(5,)`.
2. Concatenates `extra_features`: `torch.cat([feats, extra_features.float()])` → shape `(7,)`.
3. Applies `_standardise`: `(feats - feature_mean) / clamp(feature_std, 1e-6)`.
4. If `self._model.in_features != 7`, the linear layer is resized in-place to
   `nn.Linear(7, 1)`, copying the first 5 weights from the old model.
5. `logit = self._model(feats_scaled.unsqueeze(0))`.
6. Returns `float(sigmoid(logit))`.

### Hard floor

Operates on the same 5 ImplicitLayer signals (indices 0,1,2,4,5 of the
`ReasoningScore` tensor). Fires if any exceeds 0.95. On the CWRU test set,
the hard floor fired 0 times (all escalations were learner-driven).

---

## Layer 4 — Classification (heuristic, no LLM in held-out evaluation)

The held-out evaluation does not use an LLM. After the gate decision, one of
two functions classifies the window:

**`classify_heuristic(event, thresh)`** — used for escalated windows.
Operates on Z-scored feature values:
1. If `de_kurtosis > thresh["z_outer"]` (+1.3581): → `outer_race`
2. Elif `de_kurtosis > thresh["z_inner"]` (+0.0149): → `inner_race`
3. Elif `de_rms > thresh["z_ball"]` (-0.5020): → `ball`
4. Else: → `normal`

**`classify_fast_path(raw_event, thresh)`** — used for non-escalated windows.
Operates on raw (physical-unit) feature values:
1. If `de_kurtosis > thresh["raw_outer"]` (3.5634): → `outer_race`
2. Elif `de_kurtosis > thresh["raw_inner"]` (1.0813): → `inner_race`
3. Elif `de_rms > thresh["raw_ball"]` (0.1028): → `ball`
4. Else: → `normal`

Both functions use the same decision logic (kurtosis first, then RMS) but
with thresholds calibrated to the respective feature space.

---

## Results

### Optimised CWRU gate (current, `cwru_gate_weights.pt`)

Measured on 1,659 held-out test windows (1 unseen file per class).

| Class | Test file | Windows | Escalated | Esc% | Correct | Accuracy |
|---|---|---|---|---|---|---|
| normal | 99 | 946 | 0 | 0.0% | 946 | 100.0% |
| inner_race | 108 | 239 | 239 | 100.0% | 239 | 100.0% |
| ball | 121 | 236 | 219 | 92.8% | 236 | 100.0% |
| outer_race | 133 | 238 | 238 | 100.0% | 238 | 100.0% |
| **OVERALL** | — | **1,659** | **696** | **42.0%** | **1,659** | **100.0%** |

| Gate metric | Value |
|---|---|
| Escalation rate | 42.0% |
| LLM reduction vs always-escalate | **58.0%** |
| Fault escalation recall | **97.6%** |
| Normal false escalation rate | **0.0%** |
| Gate precision | **100.0%** |
| Hard-floor fires | 0 |
| Checkpoint threshold | 0.95 |
| Gate confidence mean | 0.434 |
| Gate confidence std | 0.485 |

### Confusion matrix

|  | normal | inner_race | ball | outer_race |
|---|---|---|---|---|
| **normal** | 946 | 0 | 0 | 0 |
| **inner_race** | 0 | 239 | 0 | 0 |
| **ball** | 0 | 0 | 236 | 0 |
| **outer_race** | 0 | 0 | 0 | 238 |

Zero misclassifications across all 1,659 windows.

### Frozen baseline (BASELINE.md, before any gate optimisation)

| Metric | Frozen baseline | Optimised gate | Change |
|---|---|---|---|
| Escalation rate | 70.9% | 42.0% | −28.9 pp |
| LLM reduction | 29.1% | 58.0% | +28.9 pp |
| Fault escalation recall | 76.0% | 97.6% | +21.6 pp |
| Normal false escalation | ~82% | 0.0% | −82 pp |
| Classification accuracy | 100.0% | 100.0% | 0 |

### Analysis

**Why accuracy is 100% regardless of gate routing:** Both `classify_heuristic`
and `classify_fast_path` use kurtosis and RMS thresholds calibrated from the
training distribution. Outer_race windows have de_kurtosis well above 3.56 raw
(mean ~12 for outer_race vs ~3 for normal). Inner_race windows have kurtosis
above 1.08 raw. Ball windows have de_rms above 0.103 raw. Normal windows have
de_rms below 0.08 raw. The separation is clean enough that both classifiers
achieve 100% accuracy whether or not the gate escalates.

**Why the gate still escalates 42% of windows:** Outer_race (100%) and
inner_race (100%) must be escalated because their probability scores are very
high (outer_race mean gate_conf ≈ 0.98, inner_race mean ≈ 0.61). Ball at
92.8% also has high scores from de_rms. The only windows that could be
suppressed are ball windows where de_rms is near the decision boundary, plus
any borderline inner_race windows. The 17 non-escalated ball windows have
de_rms below the learned decision boundary.

**Why normal escalation rate is 0%:** The 7th feature `de_rms` for normal
windows is always low (raw mean ≈ 0.07 vs ball mean ≈ 0.15). The logistic
weight on de_rms is +16.3, strongly penalising high values. The logistic
regression learned to associate low de_rms with normal and high de_rms with
ball. After normalisation with train mean=0.20, std=0.195, a normal window
with de_rms=0.07 gets z = (0.07-0.20)/0.195 = -0.67, contributing −10.9 to
the logit via the +16.3 weight. This pushes the probability well below the
0.95 threshold for every normal window.

**Why the threshold is 0.95:** The val sweep found that precision is 100% at
threshold ≥ 0.90 (no normal windows reach that confidence) while still
achieving recall ≥ 0.73. The algorithm selects the highest threshold meeting
the recall target, which was 0.95 (recall=0.978 on val). This is not arbitrary
— it reflects the fact that the gate produces bimodal confidence values:
normal windows cluster near 0 and fault windows cluster near 1 after the
de_rms feature is included.

**Why ball fault recall is 92.8% (not 100%):** 17 of 236 ball windows are
not escalated. These are ball windows where `de_rms` is relatively low (near
the decision boundary). They are still classified correctly by the fast-path
classifier because de_rms > 0.103 raw for all of them — only the gate score
falls below 0.95. This is an acceptable trade-off: classification accuracy
is preserved even when the gate does not escalate.

**Data leakage verification:** The test file IDs (99, 108, 121, 133) never
appear in the training or val records. Z-score statistics are computed only
from train events. The ImplicitLayer is warmed only on train events. The
threshold is selected only from val events. The test set is touched exactly
once during the evaluation loop.
