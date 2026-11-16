# CWRU Domain — Theoretical Pipeline Explanation

**Domain:** Case Western Reserve University (CWRU) bearing fault detection.  
**Goal:** Given 1024-sample vibration recordings from a rotating motor bearing, classify whether the bearing is healthy or has a fault — and if faulty, identify which of three fault types it is — while minimising unnecessary calls to an LLM.

The pipeline has two completely separate phases. The **training phase** builds and calibrates every component from labelled recordings at known motor loads. The **testing phase** runs recordings at a different motor load through the already-built components to produce classification decisions.

---

## Why This Architecture Exists

A bearing monitoring system samples vibration continuously. Most windows during normal operation are healthy — escalating every window to an LLM for analysis would waste compute on the overwhelming majority of normal readings. The same four-layer architecture as the traffic domain is used, but adapted to the physics of bearing vibration.

The key insight driving the CWRU design is that bearing faults produce **specific, well-understood vibration signatures**:
- Outer race faults create strong periodic impulses at a characteristic defect frequency, producing very high kurtosis (a statistical measure of impulsiveness).
- Inner race faults also produce impulses but at a different frequency and modulated by shaft rotation, giving moderately elevated kurtosis.
- Ball faults cause subtle amplitude modulation in the vibration signal, mostly detectable through elevated RMS amplitude rather than extreme kurtosis.
- Healthy bearings produce essentially Gaussian vibration noise with low kurtosis.

The pipeline uses these physics to inform its feature engineering.

---

## Training Phase

### Step 1 — File-level train/test split

The dataset contains four `.mat` or `.mat.zip` files per class (normal, inner_race, ball, outer_race). Each file was recorded at a different motor load in RPM. The first three files per class (files 100/972/98 for normal, 105/106/107 for inner_race, 118/119/120 for ball, 130/131/132 for outer_race) are assigned to the training split. The fourth file per class (99, 108, 121, 133) is held out for testing.

**Why file-level splitting rather than random window splitting?** Each file is a continuous recording at one motor load. If you randomly split windows across files, windows from the same recording would appear in both training and test sets. The model could memorise recording-specific noise characteristics — small variations in the test rig's vibration signature that are unrelated to fault type — and appear to generalise when it is actually memorising. File-level splitting ensures the model never sees any window from the test recording during training.

**Why different motor loads matter:** The bearing's vibration amplitude scales with rotation speed. A threshold that separates normal from faulty at 1750 RPM may not work at 1500 RPM. By training on three loads and testing on the fourth, the system is forced to learn features that generalise across operating conditions.

### Step 2 — Loading the vibration signals

Each `.mat` file is loaded using MATLAB's native format reader. The file contains time-series vibration recordings from two accelerometers: the Drive-End (DE) sensor close to the fault location, and the Fan-End (FE) sensor at the opposite end of the shaft. The file also contains the shaft speed in RPM. All channels are extracted and stored as 1-D arrays of float64 values sampled at 12,000 Hz.

### Step 3 — Windowing the signal

The continuous vibration recording is sliced into overlapping windows of 1024 samples each. A new window starts every 512 samples (50% overlap). At 12,000 Hz, one window covers about 85 milliseconds of vibration, which is long enough to capture several complete shaft revolutions and therefore several fault impulse cycles.

**Why 50% overlap?** A fault impulse that falls near the edge of a window might be partially captured. Overlapping windows ensure that every impulse is fully contained within at least one window, giving the feature extraction a clean view of it.

### Step 4 — Feature extraction from each window

Nine statistical and signal-processing features are computed separately for the DE channel and (if available) the FE channel, giving 18 features per window. The choice of these nine features is grounded in decades of bearing diagnostics literature.

**RMS (root mean square):** The square root of the average squared amplitude. This measures the overall energy level of the vibration signal. Ball faults increase RMS noticeably; outer and inner race faults also elevate RMS but the increase is dominated by their effect on kurtosis.

**Standard deviation:** Similar to RMS but mean-corrected. Measures the spread of the vibration signal around its mean value.

**Peak:** The maximum absolute value in the window. Fault impulses create sharp peaks, so this captures impulsive events.

**Crest factor:** Peak divided by RMS. This normalises the peak value by the signal's overall energy level. A high crest factor means the signal has sharp spikes relative to its background level, which is characteristic of impulse-type bearing faults.

**Kurtosis (Fisher normalised):** The fourth statistical moment, normalised by the standard deviation to the fourth power, minus 3. For a purely Gaussian signal, kurtosis equals zero. Each impulse from a bearing defect adds a sharp spike to the signal, pushing kurtosis well above zero. Outer race faults typically produce kurtosis above 3–5; inner race faults around 1–2; healthy bearings stay near 0. Kurtosis is the single most discriminative feature for bearing fault detection.

**Skewness:** The third statistical moment normalised by standard deviation cubed. Measures asymmetry in the vibration distribution. Faults can introduce asymmetric impulses.

**Peak-to-peak:** Maximum value minus minimum value. Captures the full dynamic range of the vibration in the window.

**Shape factor:** RMS divided by mean absolute value. This is a dimensionless measure of the signal's waveform shape, related to how "peaky" or "flat" the signal is.

**Impulse factor:** Peak divided by mean absolute value. Similar to crest factor but uses mean absolute value as the reference instead of RMS.

Together, these nine features capture the energy, impulsiveness, and shape characteristics of the bearing vibration at multiple statistical orders, making them sensitive to the specific fault signatures described above.

### Step 5 — Z-score normalisation

The 18 features from all training windows are assembled into a matrix and Z-scored: each feature is shifted by its training-set mean and divided by its training-set standard deviation. This ensures that features measured in different physical units (e.g., peak amplitude in g's vs. dimensionless kurtosis) have the same scale going into the downstream neural network layers.

The mean and standard deviation are computed from the training files only. When the test files are later normalised, the training statistics are reused — the test data never sees its own statistics.

### Step 6 — Heuristic threshold calibration

Using the Z-scored training data, percentile-based decision thresholds are calibrated for the classifier:

- The threshold between outer_race and inner_race is set at the midpoint between the 95th percentile of inner_race kurtosis values and the 5th percentile of outer_race kurtosis values in the Z-scored training set.
- The threshold between inner_race and normal is set at the midpoint between the 95th percentile of normal kurtosis values and the 5th percentile of inner_race kurtosis values.
- The threshold between ball and normal is set using de_rms (since kurtosis does not separate ball from normal), at the midpoint between the 95th percentile of normal de_rms and the 5th percentile of ball de_rms.

The same procedure is repeated on the raw (un-normalised) feature values to produce a second set of thresholds for the fast-path classifier (used when the gate does not escalate).

**Why percentile midpoints?** This is a data-driven version of the standard signal processing approach of choosing thresholds where class distributions separate. By using percentiles instead of means, the thresholds are robust to outliers within each class.

### Step 7 — Warming the ImplicitPredictiveLayer

The ImplicitPredictiveLayer is built with `input_dim=18` to match the 18-feature vector. It is then run through all 4,496 training windows in file order (the predictor is reset at each file boundary because consecutive files are independent recordings, not a continuous sequence).

The purpose of this warm-up pass is to build up the layer's internal state from the training data distribution:

- The 16 prototype vectors learn to represent the range of normal and faulty vibration patterns seen in training.
- The predictor network learns to anticipate the next window's features given the current one, within each file.
- The episodic memory buffer fills with the last 256 seen vectors.
- The Welford running statistics for spatial correlation stabilise around the training distribution.
- The running maximum values for prediction error and uncertainty normalisation adjust to the training range.

The scores produced during this warm-up pass are discarded — the purpose is purely to put the layer's internal state in the same condition it will be in when real test windows arrive.

**Why warm-up matters:** The novelty signal measures how different a window is from the learned prototypes. If the prototypes have only been exposed to a few random windows (cold start), every window looks novel. After warm-up, the prototypes represent the actual range of patterns in the data, and the novelty signal becomes meaningful.

### Step 8 — Collecting gate training features

After the warm-up pass, every training window is run through the warmed ImplicitPredictiveLayer a second time (again in file order, resetting the predictor at file boundaries) to collect the 7-dimensional gate feature vector for each window:

**From ImplicitPredictiveLayer (5 signals):**
1. Prediction error — how badly did the predictor anticipate this window?
2. Uncertainty — how much did the MC-dropout outputs vary across 10 passes?
3. Spatial correlation — how anomalous is the cross-channel pattern of vibration features?
4. Risk — the weighted composite of all signals.
5. Novelty — how different is this window from the learned prototypes?

**Raw vibration features added directly (2 signals):**
6. de_rms — the raw (non-Z-scored) Drive-End RMS amplitude.
7. de_kurtosis — the raw Drive-End kurtosis.

**Why add raw vibration features to the gate?** This is the most important design decision in the CWRU gate. When only the five ImplicitLayer signals are used, the gate cannot distinguish ball faults from normal windows. The measured AUC (area under the ROC curve) for separating ball from normal is below 0.56 for every one of the five signals — barely better than random guessing. The reason is that the ImplicitLayer learns patterns of temporal dynamics and novelty, but ball faults do not create strikingly novel or temporally unpredictable patterns — they just increase overall vibration energy.

Raw de_rms has an AUC of 1.000 for ball vs normal on the validation set — it perfectly separates them. Raw de_kurtosis has an AUC of 0.82 for all faults vs normal. These two physics-informed features give the gate exactly the information it needs to distinguish ball faults from normal windows. They are already computed during preprocessing, so adding them costs nothing extra.

### Step 9 — Computing gate normalisation statistics

The 7-dimensional feature vectors from all 4,496 training windows are assembled into a matrix. The mean and standard deviation of each of the 7 features are computed from this matrix. The standard deviation is clamped to a minimum of 1e-6 to prevent division-by-zero if any feature happens to be constant across training windows.

These 7 mean values and 7 standard deviation values are essential for consistent inference at test time.

### Step 10 — Fitting the logistic regression gate

The 7-dimensional feature vectors are Z-scored using the training statistics. A logistic regression with balanced class weighting is fitted to predict whether a window is faulty (label 1) or normal (label 0). Balanced class weighting adjusts the loss function so that misclassifying a fault costs proportionally more than misclassifying a normal window, compensating for the fact that normal windows are slightly more numerous in the training set.

The model learns 7 weights (one per feature) and one bias term. The de_rms weight turns out to be strongly positive (+16.3), reflecting its high discriminating power for ball faults. The prediction_error weight turns out to be negative — this is correct for CWRU, where the predictor within a file actually becomes better at anticipating normal patterns than fault patterns as it warms up, so lower prediction error is associated with more familiar (normal) conditions within a file.

**Why logistic regression rather than a deep neural network?** The 7-feature space is small and well-behaved after the ImplicitLayer and raw feature extraction. A deep network would likely overfit on 4,496 training windows. Logistic regression is regularised (C=1.0 in sklearn controls regularisation strength), interpretable, and fast to fit.

### Step 11 — Threshold calibration on the validation set

The 1,659 validation windows (the held-out files) are processed through the same pipeline — Z-normalised with training statistics, run through the warmed ImplicitPredictiveLayer, 7 features collected, Z-scored with training gate statistics, logistic regression applied to get probabilities. A sweep over thresholds from 0.05 to 0.95 is performed on these validation probabilities. The highest threshold where fault recall is at least 73% is selected.

The selected threshold is 0.95. This seems very high, but it reflects the bimodal nature of the gate output after the de_rms feature is added: normal windows get probabilities near 0 (because their de_rms is low, which the +16.3 weight maps to a strongly negative logit), and fault windows get probabilities near 1 (because their de_rms is elevated or their kurtosis is high). The distribution is not spread across the full range; it is concentrated at both ends, so a threshold of 0.95 correctly separates them while being the most conservative possible setting.

### Step 12 — Saving the checkpoint

The checkpoint saves: the domain tag "cwru" (to prevent this checkpoint from being accidentally loaded for traffic data), the full ImplicitPredictiveLayer state (all prototype vectors, Welford statistics, episodic memory, running maxima), the 7 logistic regression weights and bias, the 7 gate feature means and standard deviations, the feature names (to verify ordering at load time), the calibrated threshold, and provenance information (which files were used for training and validation, the random seed, the recall target).

---

## Testing Phase

### Step 1 — Loading the checkpoint

The checkpoint is loaded. The ImplicitPredictiveLayer's state is restored: all 16 prototype vectors reflect patterns seen in the training files, the predictor's running maximum MSE reflects the training range, the episodic memory contains the last 256 training windows, and the Welford statistics reflect the training distribution of spatial correlation values. The logistic regression weights and the gate feature normalisation statistics are also restored.

The layer's predictor state (`_prev_vec`) is not restored from the checkpoint — it is intentionally set to None. The test files are independent recordings; there is no meaningful "previous window" at the start of each file.

### Step 2 — Preprocessing test files

The test files (99, 108, 121, 133 — one per class, all recorded at a motor load not seen during training) are loaded the same way as training files. Each signal is windowed into 1024-sample windows with 512-sample overlap. The same 9 features are extracted per channel (18 total). These raw feature values are stored separately for use by the fast-path classifier.

The 18 features are then Z-scored using the training-set mean and standard deviation. The test data is never used to compute its own normalisation statistics.

Both the Z-scored and raw versions of each event's features are attached to the event dict. The Z-scored values feed the SNN and ImplicitPredictiveLayer. The raw values feed the gate's de_rms and de_kurtosis features, and also the fast-path classifier.

### Step 3 — SNN encoding (Layer 1)

The 18 Z-scored feature values are extracted from the event dict as a tensor of shape (1, 1, 18) — one batch, one time step, 18 features. The SNN applies sigmoid to map Z-scores into (0, 1), then performs rate coding with 25 time steps, producing a binary spike tensor of shape (25, 1, 18).

**Why keep the SNN for CWRU?** The SNN is part of the fixed platform architecture. While its contribution is primarily to standardise the input range before the ImplicitPredictiveLayer, keeping it maintains architectural consistency across domains. The sigmoid transformation also compresses extreme Z-score values, which is useful because bearing faults can produce very large kurtosis values.

### Step 4 — Computing the ReasoningScore (Layer 2)

The (25, 1, 18) spike tensor is mean-pooled over both the 25 time steps and the batch dimension, producing a single 18-dimensional vector. This vector captures the average firing rate of each feature across the simulated time steps.

Each sub-module runs against the already-warmed internal state:

**Novelty:** The current 18-dimensional vector is compared against the 16 learned prototype vectors using cosine similarity. If the vibration pattern matches a prototype that was formed from, say, outer_race fault windows, novelty will be low. If it looks unlike anything in training, novelty will be high. After the warm-up pass on training data, the prototypes have settled on representative patterns, so this signal is meaningful.

**Uncertainty:** The MC-dropout network produces 10 stochastic outputs for the current vector. The variance across those 10 outputs is the uncertainty. The network was trained on training windows, so it tends to be confident (low variance) about patterns it was trained on and uncertain (high variance) about patterns it was not.

**Prediction error:** The predictor takes the previous window's vector (from the same file, since the predictor is reset at file boundaries) and predicts what the current window should look like. Within a recording of outer_race fault data, consecutive windows are highly similar (the fault repeats every shaft revolution), so the predictor learns to anticipate them. When an unexpected pattern appears — such as the beginning of a ball fault — the prediction error spikes.

**Memory confidence:** The current vector is compared against the last 256 vectors in the episodic memory, which after the warm-up pass contains a mixture of normal and faulty patterns from training. High confidence means this pattern was recently seen.

**Spatial correlation:** The last 4 elements of the 18-dimensional vector (which are the last 4 FE channel features: de_crest_factor is not at the end — the actual last 4 are `fe_shape_factor`, `fe_impulse_factor`, plus the two preceding FE features) are extracted. Their L2 norm is computed and compared against the Welford running statistics to produce a Z-score, then sigmoid-mapped to (0, 1).

**Risk:** The weighted combination:
```
raw = 0.15 × novelty + 0.15 × uncertainty + 0.25 × prediction_error
    + 0.20 × (1 − memory_confidence) + 0.25 × spatial_correlation
```
Plus a trend correction from the last 50 raw risk values. The result passes through `sigmoid(value × 4 − 2)`.

After computing all six signals, the prototype store and episodic memory are updated. This means the layer's state keeps evolving during the test pass — it adapts to what it is seeing in the test file. This is intentional: online adaptation is part of the design.

### Step 5 — Gate decision (Layer 3)

**Hard floor:** Five signals are checked against 0.95. On the CWRU test set, the hard floor fires zero times — the fault signals in CWRU do not reach 0.95 at this stage. All escalation decisions are made by the learned gate.

**Learned gate:** The 7-dimensional feature vector is assembled:
- 5 ImplicitLayer signals extracted in the order [prediction_error, uncertainty, spatial_correlation, risk, novelty].
- de_rms: the raw (non-Z-scored) RMS value from the event's `_raw` dict.
- de_kurtosis: the raw kurtosis value from the `_raw` dict.

This vector is Z-scored using the gate's training statistics (from the checkpoint) and passed through the logistic regression. The output is a probability that this window deserves LLM attention.

If the probability is at or above 0.95 (the calibrated threshold), the window is escalated. If not, it goes to the fast-path classifier.

**What drives the gate decision for each class:**
- **Normal windows:** de_rms ≈ 0.07 raw, z-scored to about −0.67. The de_rms weight of +16.3 pushes the logit strongly negative (about −10.9 from this feature alone). The probability stays near zero and the threshold of 0.95 is never reached. Zero normal windows escalate.
- **Ball windows:** de_rms ≈ 0.15 raw, z-scored to about −0.26. The logit contribution from de_rms is about −4.2. Combined with positive contributions from other features, the total pushes most ball windows above 0.95. 93% of ball windows escalate.
- **Inner_race windows:** de_kurtosis is elevated (1–3 raw). The de_kurtosis weight is +0.078, which contributes modestly. But uncertainty and risk are also higher for inner_race, pushing the total above 0.95. 100% of inner_race windows escalate.
- **Outer_race windows:** de_kurtosis is very high (4–15 raw). The de_rms is also high (~0.6 raw). Both raw features push the logit strongly positive. 100% of outer_race windows escalate.

### Step 6 — Classification (Layer 4)

**Not escalated (fast-path):** The window is classified using the raw feature values and the raw thresholds calibrated from training data:
- If de_kurtosis > 3.5634 raw → outer_race
- Elif de_kurtosis > 1.0813 raw → inner_race
- Elif de_rms > 0.1028 raw → ball
- Else → normal

**Escalated (heuristic path):** The window is classified using the Z-scored feature values and the Z-score thresholds:
- If de_kurtosis > +1.3581 Z → outer_race
- Elif de_kurtosis > +0.0149 Z → inner_race
- Elif de_rms > −0.5020 Z → ball
- Else → normal

Both paths use the same decision logic (kurtosis first, then RMS) because these are the most physically meaningful features for bearing fault type identification. The separation between classes is clean enough in both feature spaces that both classifiers achieve 100% accuracy.

**Why does classification accuracy stay 100% regardless of which path is taken?** The Z-scored and raw thresholds are both calibrated from the same underlying distributions. A window whose Z-scored de_kurtosis is +2.0 (outer_race territory in Z-space) will also have raw de_kurtosis above 3.56 (outer_race territory in raw space). The two classifiers are equivalent — they describe the same physical boundary using different units. So whether a window goes through the gate or bypasses it makes no difference to the final label.

---

## What Each Signal Actually Measures in This Domain

| Signal | What it captures for bearing vibration | High value means |
|---|---|---|
| Novelty | Does this vibration pattern match any prototype from training? | New type of vibration |
| Uncertainty | Does the MC-dropout network agree with itself about this window? | Ambiguous pattern |
| Prediction error | Did vibration change unexpectedly from the previous window? | Sudden change within recording |
| Memory confidence | Has this exact type of vibration been seen recently? | Familiar pattern |
| Spatial correlation | Is the cross-channel (DE vs FE sensor) pattern unusual? | Unusual sensor relationship |
| Risk | Weighted combination of above five signals | Overall danger level |
| de_rms (raw) | Total vibration energy level | Elevated for ball and other faults |
| de_kurtosis (raw) | Impulsiveness of the vibration signal | Elevated for outer/inner race faults |

---

## Why Ball Faults Are Harder Than Other Faults

Ball faults (rolling element defects) create a distinctive challenge. When a ball with a surface defect rolls over the inner and outer races, it produces impacts at a characteristic frequency called the ball spin frequency. However, because the ball contacts both races alternately, the impacts are modulated — they appear and disappear as the ball spins. This modulation makes ball fault vibration look less impulsive and less periodic than outer or inner race faults.

As a result, kurtosis barely rises above normal for ball faults. The main signature is a slight increase in overall vibration energy (RMS). The ImplicitLayer signals miss this because they are sensitive to pattern novelty and temporal unpredictability, not to raw amplitude. Only by including the raw de_rms value directly in the gate feature vector can the gate reliably identify ball faults.

---

## The Domain Mismatch Problem

The checkpoint contains a domain tag "cwru". When the pipeline loads a checkpoint, it checks this tag. If a checkpoint was trained on traffic data (domain tag "traffic" or absent), it is rejected. This matters because the gate weights and normalisation statistics from a traffic checkpoint are completely wrong for CWRU:

- The traffic gate was trained with traffic data where the risk feature has mean 0.230 and standard deviation 0.023. CWRU risk values span 0.19–0.63 with standard deviation 0.070 — three times larger. Z-scoring CWRU values with traffic statistics produces Z-scores of ±15 instead of ±2, which saturates the sigmoid and makes the gate output meaningless.
- The logistic weights learned from traffic data assign negative weights to uncertainty and novelty because in traffic data, high uncertainty and novelty are associated with unusual but non-crash events. In CWRU, those signals carry different information.

The domain tag check prevents this silent failure.

---

## Summary of Training vs Testing

| Aspect | Training | Testing |
|---|---|---|
| Data | Files 100/972/98, 105/106/107, 118/119/120, 130/131/132 | Files 99, 108, 121, 133 |
| Motor loads | Three loads per class | One unseen load per class |
| Normalisation stats | Computed from training windows | Loaded from checkpoint |
| Heuristic thresholds | Calibrated from training distribution | Applied to test features |
| ImplicitLayer warm-up | Full pass over all 4,496 training windows | Layer starts from checkpoint state |
| Gate weights | Fitted by logistic regression | Loaded from checkpoint |
| Threshold | Calibrated on validation files | Loaded from checkpoint (0.95) |
| Predictor state | Reset at file boundaries during warm-up | Reset at each test file boundary |
| Outcome | Checkpoint saved with full layer state | 100% accuracy, 58% LLM reduction |
