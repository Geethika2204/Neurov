# Traffic Domain — Theoretical Pipeline Explanation

**Domain:** Nashville Freeway Anomaly (FT-AED) crash detection.  
**Goal:** Given 30-second radar loop detector readings from a freeway, decide whether a crash or near-crash is happening — and do it without calling an expensive LLM for every single window.

The pipeline has two completely separate phases. The **training phase** builds and calibrates every component from labelled historical data. The **testing phase** runs new, unseen data through the already-built components to make decisions in near real-time.

---

## Why This Architecture Exists

A freeway radar system produces readings continuously. If you sent every reading to a large language model for analysis, you would waste enormous compute on the 99.98% of windows that are completely normal. The system is therefore designed around a question: *can we tell cheaply whether a window is interesting enough to need expensive analysis?*

The four-layer architecture answers that question:

1. Convert raw sensor numbers into a biologically-inspired spike representation.
2. Measure how surprising, uncertain, and unfamiliar the current window is.
3. Decide — using a learned rule — whether the window is unusual enough to escalate.
4. Only for escalated windows, invoke the LLM to produce a final reasoned decision.

---

## Training Phase

### Step 1 — Splitting data by time

The dataset covers roughly 20 days of freeway measurements. The first 14 days become the training set, the next 3 days become the validation set, and the remaining days become the test set. The split is strictly chronological — no shuffling. This prevents the model from "seeing the future" during training, which would give artificially optimistic performance estimates.

**Why time-based splitting matters:** If you randomly shuffled all windows and split them, a window from 2am on day 15 might end up in the training set while a window from 2am on day 14 ends up in the test set. The model could learn patterns from the future and appear to generalise when it is actually memorising. Keeping time order intact forces the model to truly predict forward.

### Step 2 — Adding spatial features

Each row in the raw dataset is one sensor at one milemarker at one timestamp. At any given timestamp, all 49 milemarkers are measured simultaneously. Four summary statistics are computed across all 49 milemarkers for each timestamp:

- How spread out are the speeds across the whole freeway segment right now?
- What is the minimum speed anywhere on the segment right now (the slowest point)?
- What is the maximum lane occupancy anywhere (the most congested point)?
- How large is the speed difference between the fastest and slowest point?

These four values are the same for every milemarker at the same timestamp — they describe the freeway as a whole rather than just one sensor. They are appended to the 12 per-sensor features, giving 16 features per window.

**Why spatial features help:** A crash at one milemarker typically causes a ripple — speeds drop upstream, occupancy rises nearby. Looking at only one sensor misses this pattern. The spatial summary captures whether the whole freeway is behaving unusually, not just the sensor directly at the incident.

### Step 3 — Z-score normalisation

Each of the 16 features has a different natural scale. Speed is measured in miles per hour (tens to hundreds), occupancy as a fraction (0 to 1). To stop large-magnitude features from dominating, each feature is rescaled so that it has mean zero and standard deviation one on the training set. This is called Z-score normalisation:

```
z = (raw_value - training_mean) / training_std
```

**Critical rule:** Only the training set's mean and standard deviation are used. The validation and test sets are scaled using those same training numbers. If the test set computed its own mean and standard deviation, it would implicitly use future information.

### Step 4 — Sliding window construction

Each milemarker's time series is broken into overlapping windows of 20 consecutive timesteps. A new window starts every 5 timesteps (stride of 5), so consecutive windows overlap by 15 steps. Each window is labelled 1 (crash) if any of its 20 timesteps contains a labelled crash event, and 0 otherwise. Any window that would span two different milemarkers is discarded.

**Why overlapping windows?** A crash does not happen at a single instant — it unfolds over time. Overlapping windows ensure the model sees the same event from multiple starting points, giving it more examples of the buildup and aftermath of a crash.

### Step 5 — Spike encoding

The Z-scored window values are converted into spike trains — sequences of binary (0 or 1) signals that mimic how biological neurons communicate. First, each Z-scored value is passed through a sigmoid function, which squashes it into the range (0, 1). This value is then interpreted as a firing probability: how likely is this feature to fire a spike at any given moment?

For each of 25 simulated time steps, an independent coin flip is performed for each feature at each sample in the window. The result is a binary tensor: for each feature, at each time step, did it fire (1) or not (0)?

**Why use spikes at all?** Spiking neural networks (SNNs) model information as the timing and rate of discrete events rather than as continuous numbers. They are biologically plausible, potentially more energy-efficient on specialised hardware, and produce a natural measure of activity level (firing rate) that the downstream layer can use.

### Step 6 — Fitting the ImplicitPredictiveLayer

The ImplicitPredictiveLayer contains several internal components that learn from the training data before any gate training happens. Each component is trained by passing all training windows through it in temporal order (never shuffled):

**Predictor head:** A small neural network that is trained to predict the next window's features from the current window's features. It is trained by minimising the mean squared error between its prediction and the actual next window. After training, good predictions correspond to expected, normal traffic states; bad predictions correspond to sudden, unusual changes.

**MC-Dropout head:** A small neural network with Dropout layers, trained to classify whether a window contains a crash. The Dropout layers randomly deactivate neurons during each forward pass. The same window run through the network 10 times in a row will produce 10 slightly different outputs — the variance across those 10 outputs measures how uncertain the model is about its prediction.

**Prototype store:** Maintains 16 representative vectors (prototypes) that are updated online using a simplified version of the k-means algorithm. Each new window is compared to all 16 prototypes; the most similar prototype is updated to move slightly toward the new window. After seeing many training windows, the prototypes settle on representative patterns of normal traffic.

**Episodic memory:** A ring buffer that stores the last 256 windows seen. Each new window is compared against everything in this buffer to measure how recently a similar pattern was seen.

**Spatial correlation tracker:** Tracks the statistical distribution of spatial activity (the last 4 elements of the feature vector, which are the cross-milemarker spatial features) using Welford's online algorithm, which maintains a running mean and variance without storing all past values.

**Risk head:** A fixed formula that combines the other five signals into one composite danger score, with a short-term trend component that checks whether risk is rising or falling compared to the recent average.

### Step 7 — Collecting gate training features

With the ImplicitPredictiveLayer fully warmed on training data, every training window is run through it again to collect the five-dimensional feature vector used to train the gate:

```
[prediction_error, uncertainty, spatial_correlation, risk, novelty]
```

The sixth signal, memory_confidence, is excluded because after the layer warms up on normal traffic patterns it saturates near 1.0 for virtually everything — it has near-zero variance and therefore no discriminating power for the logistic regression.

### Step 8 — Training the logistic regression gate

The five-dimensional feature vectors are Z-scored (again, using training-set statistics only) and fed into a logistic regression with balanced class weighting. Balanced class weighting means the model is penalised more for missing a rare crash than for misclassifying a common normal window — this compensates for the extreme imbalance (~0.01% crash rate).

Logistic regression learns a set of five weights (one per feature) and a bias term. The output is a probability between 0 and 1: how likely is this window to be a crash that deserves LLM attention?

**Why logistic regression instead of a neural network?** Interpretability and stability. With only five features and a highly imbalanced dataset, a complex neural network risks overfitting. Logistic regression is a linear classifier that generalises well and whose weights can be directly interpreted — a positive weight means that feature pushes toward escalation.

### Step 9 — Threshold calibration on the validation set

The validation set (days 15–17, never seen during training) is run through the same pipeline to collect five-dimensional feature vectors and gate probabilities. A sweep is performed over thresholds from 0.05 to 0.95 in steps of 0.05. For each threshold, the fraction of windows escalated and the fraction of crashes caught (recall) are measured. The highest threshold that still achieves at least 65% crash recall is selected.

**Why not just use 0.5?** Because the classes are so imbalanced, the logistic regression's output probabilities are compressed toward zero for most windows. Using 0.5 would miss many crashes. The validation sweep finds the threshold that balances the safety requirement (catch enough crashes) against the efficiency requirement (don't escalate too many normal windows).

### Step 10 — Saving the checkpoint

Everything learned is saved together: the ImplicitPredictiveLayer's internal state (prototype vectors, running statistics, episodic memory), the logistic regression weights and bias, the training-set feature means and standard deviations, and the calibrated threshold. This is a single file that fully defines the trained gate.

---

## Testing Phase

### Step 1 — Loading the checkpoint

The saved checkpoint is loaded. The ImplicitPredictiveLayer's state is restored exactly as it was at the end of training. This is not just the neural network weights — it includes the learned prototype vectors, the running maximum values used for normalisation, and the episodic memory buffer. Without restoring this state, the layer would start from scratch and produce feature values with a completely different distribution than training.

### Step 2 — Receiving a new window

A new 20-timestep window of Z-scored traffic features arrives. It has already been normalised using the training-set mean and standard deviation saved in the checkpoint.

### Step 3 — Spike encoding (Layer 1)

The Z-scored window values go through the same sigmoid-to-spike-rate encoding used during training, producing a binary spike tensor of shape (25 timesteps × batch × 16 features).

### Step 4 — Computing the ReasoningScore (Layer 2)

The spike tensor is mean-pooled over the 25 time steps and the batch dimension, collapsing it to a single 16-dimensional vector that represents the average activity level of each feature during this window.

Each sub-module then computes its signal using the already-warmed internal state:

**Novelty:** The 16-dimensional vector is compared against all 16 prototype vectors using cosine similarity (which measures the angle between vectors, not their magnitude). The maximum similarity across all 16 prototypes is found. Novelty = 1 − that maximum similarity. If the window closely matches a known normal traffic pattern, novelty is near zero. If it matches nothing the system has seen before, novelty is near one.

**Uncertainty:** The same 16-dimensional vector is passed through the MC-Dropout network 10 times with dropout active. Each pass produces a slightly different output because random neurons are dropped each time. The standard deviation across those 10 outputs measures how much the model disagrees with itself about this window — this is epistemic uncertainty, meaning uncertainty due to limited knowledge rather than random noise.

**Prediction error:** The predictor network takes the previous window's vector and predicts what the current window should look like. The mean squared error between that prediction and the actual current window is computed. A large error means the current window was unexpected given the previous one — a potential sign of a sudden change.

**Memory confidence:** The current vector is compared against all vectors in the 256-entry ring buffer using cosine similarity. The maximum similarity found is memory confidence — how recently was a similar pattern seen? Low confidence means this is an unfamiliar pattern.

**Spatial correlation:** The four spatial features at the end of the vector are extracted and their L2 norm is computed. This norm is then compared against the running statistics maintained by Welford's algorithm to produce a Z-score, which is passed through a sigmoid to give a value in (0, 1). An unusually high or low spatial correlation score indicates abnormal cross-sensor patterns.

**Risk:** The weighted combination of all five signals, plus a trend component:
```
raw = 0.15 × novelty + 0.15 × uncertainty + 0.25 × prediction_error
    + 0.20 × (1 − memory_confidence) + 0.25 × spatial_correlation
```
The trend checks whether this raw value is higher than the average of the last 50 raw values. If risk is rising, the final risk score is nudged upward; if it is falling, it is nudged downward. The result is passed through `sigmoid(value × 4 − 2)` to produce a value in (0, 1).

After computing all six signals, the prototype store and episodic memory are updated with the current window, so the layer's state evolves continuously as new data arrives.

### Step 5 — Gate decision (Layer 3)

**Hard floor check:** Five of the six signals (all except memory_confidence) are checked against the hard floor threshold of 0.95. Memory confidence is excluded because a high value means the pattern is *familiar*, not dangerous. If any of the five danger signals exceeds 0.95, the window is escalated immediately, regardless of what the logistic regression says. This catches extreme anomalies that the trained model might otherwise miss.

**Learned gate:** The five-dimensional feature vector is extracted from the ReasoningScore in the order [prediction_error, uncertainty, spatial_correlation, risk, novelty], Z-scored using the training-set statistics saved in the checkpoint, and multiplied by the trained logistic regression weights plus the bias. The sigmoid of that sum is the escalation probability. If this probability is at or above the calibrated threshold (0.3276), the window is escalated.

The final decision is: escalate if either the hard floor fired or the learned probability exceeds the threshold.

### Step 6 — Routing the decision (Layer 4)

**Not escalated:** The gate suppresses this window. No LLM call is made. The window is handled by the automatic action module, which produces a simple "continue monitoring" or "increase sampling rate" response based on the raw risk score.

**Escalated:** The window is passed to the LLM agent with the sensor readings (in physical units, not Z-scored), the gate confidence score, and contextual information. For local Ollama models, the agent makes a single direct call with inline knowledge. For Claude/Anthropic models, it runs a ReAct loop with access to two tools: a ChromaDB knowledge base retrieval tool and a context lookup tool. The agent responds with a structured JSON label, confidence, and reasoning chain.

---

## What Each Signal Actually Measures in This Domain

| Signal | What it captures for traffic | High value means |
|---|---|---|
| Novelty | Has this combination of lane speeds, volumes and occupancies been seen before? | Unusual traffic pattern |
| Uncertainty | Does the model disagree with itself about how to classify this window? | Model is unsure — treat carefully |
| Prediction error | Did traffic suddenly change relative to the previous window? | Abrupt transition — possible incident |
| Memory confidence | Was a similar pattern seen recently? | Familiar pattern |
| Spatial correlation | Is the cross-freeway pattern of speeds and densities abnormal? | Anomalous spatial spread |
| Risk | Weighted composite of the above five signals plus trend | Overall danger level |

---

## Why the Gate Cannot Achieve Perfect Separation

Crashes are extremely rare (1,372 in 28,076 test windows — 4.9%). Even a perfect classifier would have low precision because for every 1 crash there are roughly 20 normal windows. The gate is not trying to perfectly classify — it is trying to cheaply filter out windows that are obviously normal, so the expensive LLM only sees windows that have some chance of being crashes.

At the operating point selected (threshold 0.3276, recall 97.1%), the gate catches 1,332 of 1,372 crashes while also flagging 13,062 normal windows. This gives 97.1% recall at the cost of 46.4% false positive rate among normals. That trade-off is intentional: missing a crash (false negative) is far more costly than making an unnecessary LLM call (false positive).

---

## Summary of Training vs Testing

| Aspect | Training | Testing |
|---|---|---|
| Data | Days 1–14 (train) + days 15–17 (val) | Days 18–20 (test) |
| Normalisation stats | Computed from training data | Loaded from checkpoint |
| ImplicitLayer state | Built up by processing training windows | Loaded from checkpoint |
| Gate weights | Fitted by logistic regression on training features | Loaded from checkpoint |
| Threshold | Calibrated on validation set | Loaded from checkpoint |
| Layer updates | Prototypes and memory update during warm-up | Prototypes and memory continue updating |
| Outcome | Checkpoint saved | Decisions made in real time |
