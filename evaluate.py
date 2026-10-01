"""
evaluate.py
===========
Evaluates the trained NeuroVerse gate on the validation and/or test split.

What this does
--------------
1.  Loads the gate checkpoint produced by ``train_gate.py``.
2.  Restores the SAME ``ImplicitPredictiveLayer`` state (identical weights,
    buffers, running statistics) — NOT a freshly initialised layer.
3.  Runs every window through the layer to obtain five danger signals.
4.  Applies the SAME feature mean / std from training.
5.  Applies the SAME logistic weights / bias and validation-selected threshold.
6.  Reports hard-floor escalations and learner escalations separately.
7.  Compares against AlwaysEscalate and AlwaysThreshold baselines.
8.  Writes results.csv and score_distributions.csv (with all five signals).

Usage
-----
    python evaluate.py                    # default: test split
    python evaluate.py --split val        # val split
    python evaluate.py --max 10000        # subsample for speed
    python evaluate.py --no-checkpoint    # fresh random layer (sanity check only)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from nv_platform.implicit_layer import ImplicitPredictiveLayer, ReasoningScore
from nv_platform.gate import ReasoningGate, EscalationHistory, HARD_ESCALATE_THRESHOLD
from nv_platform.gate_learner import GateLearner, _extract_features
from baselines.always_escalate_gate import AlwaysEscalateGate
from baselines.always_threshold_gate import AlwaysThresholdGate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM        = 16
OUTPUT_DIR       = REPO_ROOT / "analysis"
CHECKPOINT_PATH  = REPO_ROOT / "data" / "processed" / "gate_weights.pt"
THRESHOLD_DEFAULT = 0.5   # fallback if no checkpoint


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, y) from data/processed/<split>.pt.

    X shape: (num_steps, N, time_window, features)
    y shape: (N,)
    """
    path = REPO_ROOT / "data" / "processed" / f"{split}.pt"
    data = torch.load(path, weights_only=True)
    return data["X"], data["y"]


# ---------------------------------------------------------------------------
# Score generation
# ---------------------------------------------------------------------------

def generate_scores(
    X: torch.Tensor,
    layer: ImplicitPredictiveLayer,
    max_windows: int | None,
    desc: str = "",
) -> list[ReasoningScore]:
    N = X.shape[1]
    if max_windows is not None:
        N = min(N, max_windows)

    scores: list[ReasoningScore] = []
    t0 = time.time()
    for i in range(N):
        window = X[:, i, :, :].mean(dim=1).unsqueeze(1)  # (num_steps, 1, F)
        scores.append(layer.compute_state(window))
        if i % 5000 == 0 and i > 0:
            pct = 100 * i / N
            print(f"  {desc} {i:>7,}/{N:,} ({pct:.0f}%)  [{time.time()-t0:.0f}s]",
                  end="\r", flush=True)
    print(f"  {desc} {N:,}/{N:,} (100%)  [{time.time()-t0:.1f}s]      ")
    return scores


# ---------------------------------------------------------------------------
# Learned gate evaluation (with hard-floor/learner breakdown)
# ---------------------------------------------------------------------------

def evaluate_reasoning_gate(
    scores: list[ReasoningScore],
    labels: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    threshold: float,
) -> dict:
    """
    Evaluate the offline logistic gate with explicit hard-floor / learner breakdown.

    Hard floor: escalate if ANY of [novelty, uncertainty, pred_error,
                spatial_correlation, risk] > HARD_ESCALATE_THRESHOLD (0.95).
    Learner   : escalate if sigmoid(w·x + b) >= threshold.

    Reports both paths separately.
    """
    y = labels.tolist()
    N = len(scores)

    hard_floor_count = 0
    learner_count    = 0
    total_esc        = 0

    tp_total = tp_hard = tp_learner = 0
    fp_total = fp_hard = fp_learner = 0
    fn_total = 0
    tn_total = 0

    for i, sc in enumerate(scores):
        st      = sc.as_tensor()
        label   = int(y[i])

        # --- Hard floor (same indices as gate.py) ---
        danger = torch.stack([st[0], st[1], st[2], st[4], st[5]])
        hard   = bool((danger > HARD_ESCALATE_THRESHOLD).any())

        # --- Learner probability ---
        feats       = _extract_features(st)
        feats_scaled = (feats - feat_mean) / feat_std
        with torch.no_grad():
            logit = (feats_scaled @ weights) + bias
            prob  = float(torch.sigmoid(logit))
        learned = (prob >= threshold)

        escalate = hard or learned

        if escalate:
            total_esc += 1
            if hard:
                hard_floor_count += 1
            if learned:
                learner_count += 1

            if label == 1:
                tp_total += 1
                if hard:    tp_hard    += 1
                if learned: tp_learner += 1
            else:
                fp_total += 1
                if hard:    fp_hard    += 1
                if learned: fp_learner += 1
        else:
            if label == 1:
                fn_total += 1
            else:
                tn_total += 1

    recall    = tp_total / max(tp_total + fn_total, 1)
    precision = tp_total / max(tp_total + fp_total, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    esc_rate  = total_esc / max(N, 1)
    fp_rate   = fp_total  / max(fp_total + tn_total, 1)
    reduction = 1.0 - esc_rate

    n_pos = int(labels.sum())

    return {
        "gate":                     "ReasoningGate",
        "split_windows":            N,
        "positive_windows":         n_pos,
        "threshold":                round(threshold, 4),
        # ── overall ──────────────────────────────────────────────────────
        "total_escalations":        total_esc,
        "escalation_rate":          round(esc_rate, 4),
        "llm_call_reduction":       round(reduction, 4),
        "tp":                       tp_total,
        "fp":                       fp_total,
        "fn":                       fn_total,
        "tn":                       tn_total,
        "recall":                   round(recall, 4),
        "precision":                round(precision, 4),
        "f1":                       round(f1, 4),
        "fp_rate":                  round(fp_rate, 4),
        # ── hard-floor breakdown ─────────────────────────────────────────
        "hard_floor_escalations":   hard_floor_count,
        "hard_floor_esc_rate":      round(hard_floor_count / max(N, 1), 4),
        "hard_floor_tp":            tp_hard,
        "hard_floor_fp":            fp_hard,
        # ── learner breakdown ────────────────────────────────────────────
        "learner_escalations":      learner_count,
        "learner_esc_rate":         round(learner_count / max(N, 1), 4),
        "learner_tp":               tp_learner,
        "learner_fp":               fp_learner,
    }


# ---------------------------------------------------------------------------
# Baseline gate evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(
    name: str,
    gate,
    scores: list[ReasoningScore],
    labels: torch.Tensor,
) -> dict:
    """Evaluate an AlwaysEscalate or AlwaysThreshold gate."""
    y  = labels.tolist()
    N  = len(scores)

    tp = fp = fn = tn = 0
    escalated = 0

    for sc, label in zip(scores, y):
        esc, _ = gate.should_escalate(sc)
        if esc:
            escalated += 1
            if label == 1: tp += 1
            else:          fp += 1
        else:
            if label == 1: fn += 1
            else:          tn += 1

    recall    = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    esc_rate  = escalated / max(N, 1)
    fp_rate   = fp / max(fp + tn, 1)
    reduction = 1.0 - esc_rate
    n_pos     = int(labels.sum())

    return {
        "gate":                     name,
        "split_windows":            N,
        "positive_windows":         n_pos,
        "threshold":                getattr(gate, "_threshold", getattr(gate, "threshold", "N/A")),
        "total_escalations":        escalated,
        "escalation_rate":          round(esc_rate, 4),
        "llm_call_reduction":       round(reduction, 4),
        "tp":                       tp,
        "fp":                       fp,
        "fn":                       fn,
        "tn":                       tn,
        "recall":                   round(recall, 4),
        "precision":                round(precision, 4),
        "f1":                       round(f1, 4),
        "fp_rate":                  round(fp_rate, 4),
        "hard_floor_escalations":   "N/A",
        "hard_floor_esc_rate":      "N/A",
        "hard_floor_tp":            "N/A",
        "hard_floor_fp":            "N/A",
        "learner_escalations":      "N/A",
        "learner_esc_rate":         "N/A",
        "learner_tp":               "N/A",
        "learner_fp":               "N/A",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeuroVerse gate pipeline.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--max", type=int, default=None,
                        help="Cap evaluation windows (default: all)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Skip checkpoint load — use random layer (sanity check only)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load checkpoint ─────────────────────────────────────────────────
    weights   = None
    bias      = None
    feat_mean = None
    feat_std  = None
    threshold = THRESHOLD_DEFAULT
    input_dim = INPUT_DIM

    if not args.no_checkpoint and CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint: {CHECKPOINT_PATH.relative_to(REPO_ROOT)}")
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False)

        input_dim = ckpt.get("input_dim", INPUT_DIM)
        weights   = ckpt["weights"]
        bias      = ckpt["bias"]
        feat_mean = ckpt["feature_mean"]
        feat_std  = ckpt["feature_std"]
        threshold = float(ckpt["threshold"])

        print(f"  input_dim  = {input_dim}")
        print(f"  threshold  = {threshold:.4f}  "
              f"(target recall={ckpt.get('recall_target','?')})")
        print(f"  weights    = {weights.tolist()}")
        print(f"  bias       = {bias.item():.4f}")
        print(f"  feat_mean  = {feat_mean.tolist()}")
        print(f"  feat_std   = {feat_std.tolist()}")

        # Restore exact layer state
        print("  Restoring ImplicitPredictiveLayer from checkpoint …")
        layer = ImplicitPredictiveLayer(input_dim=input_dim)
        layer.load_state_dict(ckpt["layer"])
        print("  Layer state restored successfully.")
    else:
        if args.no_checkpoint:
            print("WARNING: --no-checkpoint flag set. Using randomly initialised layer.")
        else:
            print(f"WARNING: checkpoint not found at {CHECKPOINT_PATH}.")
            print("         Run train_gate.py first for a properly trained gate.")
            print("         Falling back to random layer + threshold=0.5")
        layer = ImplicitPredictiveLayer(input_dim=input_dim)
        weights   = torch.zeros(5)
        bias      = torch.tensor(1.0)
        feat_mean = torch.zeros(5)
        feat_std  = torch.ones(5)
        threshold = THRESHOLD_DEFAULT

    # Reset predictor prev_vec for the evaluation pass (fresh sequence)
    layer._predictor.reset()

    # ── 2. Load evaluation split ───────────────────────────────────────────
    print(f"\nLoading {args.split} split …")
    X, y = load_split(args.split)
    N_eval = X.shape[1] if args.max is None else min(X.shape[1], args.max)
    X = X[:, :N_eval]
    y = y[:N_eval]
    n_pos = int((y == 1).sum())
    print(f"  Windows: {N_eval:,}  |  Positives: {n_pos:,} ({100*n_pos/N_eval:.2f}%)")

    # ── 3. Generate scores ─────────────────────────────────────────────────
    print(f"\nGenerating scores …")
    scores = generate_scores(X, layer, max_windows=None, desc=args.split)
    y_eval = y[:len(scores)]

    # ── 4. Evaluate all gates ──────────────────────────────────────────────
    print("\nEvaluating gates …")
    results = []

    # AlwaysEscalate baseline
    print("  AlwaysEscalate …")
    ae_gate = AlwaysEscalateGate()
    results.append(evaluate_baseline("AlwaysEscalate", ae_gate, scores, y_eval))

    # AlwaysThreshold baseline (use same threshold as the learned gate for fair comparison)
    print(f"  AlwaysThreshold (threshold={threshold:.4f}) …")
    at_gate = AlwaysThresholdGate(threshold=threshold)
    results.append(evaluate_baseline("AlwaysThreshold", at_gate, scores, y_eval))

    # Reasoning gate (offline logistic)
    print(f"  ReasoningGate (threshold={threshold:.4f}) …")
    rg_result = evaluate_reasoning_gate(
        scores, y_eval, weights, bias, feat_mean, feat_std, threshold
    )
    results.append(rg_result)

    # ── 5. Print summary ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Gate':<22} {'Recall':>7} {'Prec':>7} {'EscRate':>9} {'Reduction':>10} {'FN':>6}")
    print(f"{'─'*70}")
    for r in results:
        print(
            f"{r['gate']:<22} "
            f"{r['recall']:>7.3f} "
            f"{r['precision']:>7.3f} "
            f"{r['escalation_rate']:>9.3f} "
            f"{r['llm_call_reduction']:>10.3f} "
            f"{r['fn']:>6}"
        )
    print(f"{'─'*70}")

    rg = results[2]
    print(f"\nReasoningGate breakdown:")
    print(f"  total_escalations    : {rg['total_escalations']:,}")
    print(f"    hard_floor_esc     : {rg['hard_floor_escalations']:,}  "
          f"({rg['hard_floor_esc_rate']:.3f} of windows)")
    print(f"    learner_esc        : {rg['learner_escalations']:,}  "
          f"({rg['learner_esc_rate']:.3f} of windows)")
    print(f"  recall               : {rg['recall']:.3f}")
    print(f"  fn (missed crashes)  : {rg['fn']:,}")

    # ── 6. Save results.csv ────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "results.csv"
    fieldnames   = list(results[0].keys())
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {results_path.relative_to(REPO_ROOT)}")

    # ── 7. Save score_distributions.csv (all five danger signals) ─────────
    dist_path = OUTPUT_DIR / "score_distributions.csv"
    with open(dist_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_idx", "label",
            "novelty", "uncertainty", "prediction_error",
            "spatial_correlation", "risk",
            "memory_confidence",
        ])
        for idx, (sc, lbl) in enumerate(zip(scores, y_eval.tolist())):
            writer.writerow([
                idx, int(lbl),
                round(sc.novelty, 6),
                round(sc.uncertainty, 6),
                round(sc.prediction_error, 6),
                round(sc.spatial_correlation, 6),
                round(sc.risk, 6),
                round(sc.memory_confidence, 6),
            ])
    print(f"Score distributions written to {dist_path.relative_to(REPO_ROOT)}")

    # ── 8. Spatial correlation note ────────────────────────────────────────
    sc_vals   = np.array([sc.spatial_correlation for sc in scores])
    sc_labels = y_eval.numpy()
    if sc_vals.std() > 0.01:
        pos_mean = sc_vals[sc_labels == 1].mean() if (sc_labels == 1).any() else float("nan")
        neg_mean = sc_vals[sc_labels == 0].mean() if (sc_labels == 0).any() else float("nan")
        print(f"\nspatial_correlation note:")
        print(f"  positive class mean : {pos_mean:.4f}")
        print(f"  negative class mean : {neg_mean:.4f}")
        print(f"  overall std         : {sc_vals.std():.4f}")
        print("  (bimodal distribution expected — see objective §10 for context)")

    print("\nDone.")


if __name__ == "__main__":
    main()
