"""
train_gate.py
=============
Offline gate training pipeline.

Steps
-----
1.  Fixed seed for full reproducibility.
2.  Construct and fit ``ImplicitPredictiveLayer`` on the training split
    (shuffle=False — temporal order must be preserved for the predictor).
3.  Generate the five danger signals for every training window.
4.  Standardise features using training-set mean / std.
5.  Fit a weighted logistic regression (sklearn + class_weight='balanced',
    or a full-batch LBFGS fallback) to predict escalation from the five signals.
6.  Generate validation scores and pick the escalation threshold that achieves
    recall >= RECALL_TARGET while minimising unnecessary escalation.
7.  Save the complete checkpoint:
        layer state_dict
        logistic weights + bias
        feature_mean, feature_std
        threshold

Usage
-----
    python train_gate.py              # full training set
    python train_gate.py --max 20000  # subsample for speed
    python train_gate.py --epochs 3   # fewer fit epochs

After training run:
    python evaluate.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from nv_platform.implicit_layer import ImplicitPredictiveLayer
from nv_platform.gate_learner import GateLearner, _extract_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_DIM    = 16       # 12 per-sensor + 4 spatial features
FIT_EPOCHS   = 5        # predictive-layer pre-training epochs
RECALL_TARGET = 0.65    # minimum recall we want on the val set
SAVE_PATH    = REPO_ROOT / "data" / "processed" / "gate_weights.pt"
RANDOM_SEED  = 0


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
) -> list:
    """Run every window through the layer and return a list of ReasoningScore."""
    N = X.shape[1]
    if max_windows is not None:
        N = min(N, max_windows)

    scores = []
    t0 = time.time()
    for i in range(N):
        # X[:, i, :, :] → (num_steps, time_window, features)
        # mean over time_window → (num_steps, features)
        # unsqueeze batch dim → (num_steps, 1, features)
        window = X[:, i, :, :].mean(dim=1).unsqueeze(1)
        scores.append(layer.compute_state(window))
        if i % 5000 == 0 and i > 0:
            pct = 100 * i / N
            print(f"  {desc} {i:>7,}/{N:,} ({pct:.0f}%)  [{time.time()-t0:.0f}s]",
                  end="\r", flush=True)
    print(f"  {desc} {N:,}/{N:,} (100%)  [{time.time()-t0:.0f}s]      ")
    return scores


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(scores: list) -> torch.Tensor:
    """Stack five danger signals into a (N, 5) feature matrix."""
    rows = []
    for sc in scores:
        rows.append(_extract_features(sc.as_tensor()))
    return torch.stack(rows)   # (N, 5)


# ---------------------------------------------------------------------------
# Offline logistic regression (sklearn preferred for stability)
# ---------------------------------------------------------------------------

def fit_logistic_sklearn(
    X_scaled: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit balanced logistic regression with sklearn.

    Returns (weights, bias) as 1-D / scalar Tensors.
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
        C=1.0,
    )
    clf.fit(X_scaled.numpy(), y.numpy().astype(int))
    weights = torch.tensor(clf.coef_[0], dtype=torch.float32)
    bias    = torch.tensor(clf.intercept_[0], dtype=torch.float32)
    return weights, bias


def fit_logistic_lbfgs(
    X_scaled: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full-batch weighted logistic regression via PyTorch LBFGS.

    Fallback if sklearn is unavailable.
    """
    npos      = y.sum().clamp(min=1)
    nneg      = len(y) - npos
    pos_weight = torch.clamp(nneg / npos, max=20.0)
    print(f"  pos_weight = {pos_weight.item():.2f}  "
          f"(npos={int(npos.item())}, nneg={int(nneg.item())})")

    model = torch.nn.Linear(X_scaled.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.constant_(model.bias, 1.0)

    opt = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=200, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        logits = model(X_scaled).squeeze(-1)
        loss   = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight
        )
        loss.backward()
        return loss

    opt.step(closure)

    weights = model.weight.data.squeeze(0)
    bias    = model.bias.data.squeeze()
    return weights, bias


# ---------------------------------------------------------------------------
# Threshold selection on validation set
# ---------------------------------------------------------------------------

def select_threshold(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    recall_target: float = RECALL_TARGET,
) -> float:
    """
    Pick the largest threshold that still achieves recall >= recall_target.

    Higher threshold → fewer escalations → lower recall.
    We scan from high precision to find the most restrictive threshold that
    still meets the recall floor.

    Returns the selected threshold (float).
    """
    from sklearn.metrics import precision_recall_curve  # type: ignore

    if val_labels.sum() == 0:
        print("  WARNING: no positives in val set — using threshold=0.0")
        return 0.0

    precision_arr, recall_arr, thresholds = precision_recall_curve(
        val_labels, val_probs
    )
    # precision_recall_curve returns arrays of length n+1 for the last
    # precision/recall values; thresholds has length n.
    # We iterate over thresholds (low→high) and find the last one where
    # recall >= recall_target.

    best_threshold = 0.0  # if nothing qualifies, escalate everything
    for i, t in enumerate(thresholds):
        # recall_arr[i] corresponds to threshold t
        if recall_arr[i] >= recall_target:
            best_threshold = float(t)
        # recall decreases as threshold increases; once recall < target stop
        # searching further BUT don't break — lower thresholds give higher
        # recall and we want the *largest* threshold that still meets the target.

    # Double-check: at best_threshold, compute actual recall
    pred_pos = (val_probs >= best_threshold).astype(int)
    tp  = int(((pred_pos == 1) & (val_labels == 1)).sum())
    fn  = int(((pred_pos == 0) & (val_labels == 1)).sum())
    esc = int(pred_pos.sum())
    actual_recall = tp / max(tp + fn, 1)
    print(f"  Threshold={best_threshold:.4f}  "
          f"val_recall={actual_recall:.3f}  "
          f"val_esc_rate={esc/len(val_labels):.3f}")
    return best_threshold


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(max_windows: int | None = None, fit_epochs: int = FIT_EPOCHS) -> None:
    print("\nNeuroVerse — Offline Gate Training")
    print("=" * 50)

    # ── 1. Fixed seed ──────────────────────────────────────────────────────
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    # ── 2. Load training data ─────────────────────────────────────────────
    print("\n[1/6] Loading train.pt …")
    X_train, y_train = load_split("train")
    N_train = min(X_train.shape[1], max_windows) if max_windows else X_train.shape[1]
    X_train = X_train[:, :N_train]
    y_train = y_train[:N_train]
    actual_features = X_train.shape[-1]
    n_crashes = int((y_train == 1).sum())
    print(f"  Training windows : {N_train:,}  |  Crashes: {n_crashes:,} ({100*n_crashes/N_train:.2f}%)")

    # ── 3. Construct and fit ImplicitPredictiveLayer ───────────────────────
    print(f"\n[2/6] Constructing ImplicitPredictiveLayer(input_dim={actual_features}) …")
    layer = ImplicitPredictiveLayer(input_dim=actual_features)

    print(f"  Fitting predictive layer ({fit_epochs} epochs, shuffle=False) …")
    # Build a DataLoader that preserves temporal order
    # X_train shape: (num_steps, N, time_window, features)
    # We need (N, num_steps * time_window, features) or similar flat batch;
    # mean-pool time and steps → (N, features) for simplicity in the fit loop
    X_flat = X_train.mean(dim=(0, 2))   # (N, features)
    ds  = TensorDataset(X_flat, y_train.float())
    dl  = DataLoader(ds, batch_size=256, shuffle=False)
    fit_history = layer.fit(dl, epochs=fit_epochs)
    print(f"  Final predictor loss: {fit_history['predictor_loss'][-1]:.4f}  "
          f"mc_loss: {fit_history['mc_loss'][-1]:.4f}")

    # ── 4. Generate training scores ────────────────────────────────────────
    print(f"\n[3/6] Generating training scores ({N_train:,} windows) …")
    # Reset predictor prev_vec so scoring starts fresh (consistent with eval)
    layer._predictor.reset()
    train_scores = generate_scores(X_train, layer, max_windows=None, desc="train")

    # ── 5. Build & standardise feature matrix ─────────────────────────────
    print("\n[4/6] Building feature matrix and standardising …")
    X_feat  = build_feature_matrix(train_scores)            # (N, 5)
    y_feat  = y_train[:len(train_scores)].float()

    feat_mean = X_feat.mean(dim=0)                          # (5,)
    feat_std  = X_feat.std(dim=0).clamp_min(1e-8)           # (5,)
    X_scaled  = (X_feat - feat_mean) / feat_std

    print(f"  Feature means : {feat_mean.tolist()}")
    print(f"  Feature stds  : {feat_std.tolist()}")

    # ── 6. Fit logistic regression ─────────────────────────────────────────
    print("\n[5/6] Fitting offline logistic regression …")
    npos = int((y_feat == 1).sum())
    nneg = len(y_feat) - npos
    print(f"  npos={npos:,}  nneg={nneg:,}  imbalance ratio={nneg/max(npos,1):.1f}:1")

    try:
        import sklearn  # noqa: F401
        print("  Using sklearn LogisticRegression (balanced class_weight) …")
        weights, bias = fit_logistic_sklearn(X_scaled, y_feat)
    except ImportError:
        print("  sklearn not found — using PyTorch LBFGS …")
        weights, bias = fit_logistic_lbfgs(X_scaled, y_feat)

    print(f"  Weights: {weights.tolist()}")
    print(f"  Bias   : {bias.item():.4f}")

    # Quick train-set metrics
    with torch.no_grad():
        train_logits = (X_scaled @ weights) + bias
        train_probs  = torch.sigmoid(train_logits).numpy()
    train_preds  = (train_probs >= 0.5).astype(int)
    y_np = y_feat.numpy().astype(int)
    tp = int(((train_preds == 1) & (y_np == 1)).sum())
    fn = int(((train_preds == 0) & (y_np == 1)).sum())
    fp = int(((train_preds == 1) & (y_np == 0)).sum())
    train_recall = tp / max(tp + fn, 1)
    train_esc    = (tp + fp) / max(len(train_preds), 1)
    print(f"  Train recall @ 0.5 : {train_recall:.3f}  "
          f"esc_rate={train_esc:.3f}  FN={fn:,}")

    # ── 6b. Generate validation scores ────────────────────────────────────
    print("\n[5b/6] Loading val.pt and generating validation scores …")
    X_val, y_val = load_split("val")
    N_val = X_val.shape[1] if max_windows is None else min(X_val.shape[1], max_windows)
    X_val = X_val[:, :N_val]
    y_val = y_val[:N_val]

    # Reset predictor prev_vec for val pass (fresh sequence)
    layer._predictor.reset()
    val_scores = generate_scores(X_val, layer, max_windows=None, desc="val")
    X_val_feat   = build_feature_matrix(val_scores)           # (N_val, 5)
    y_val_trimmed = y_val[:len(val_scores)]
    X_val_scaled  = (X_val_feat - feat_mean) / feat_std

    with torch.no_grad():
        val_logits = (X_val_scaled @ weights) + bias
        val_probs  = torch.sigmoid(val_logits).numpy()

    val_labels = y_val_trimmed.numpy().astype(int)
    n_val_crashes = int(val_labels.sum())
    print(f"  Val windows: {len(val_labels):,}  |  Crashes: {n_val_crashes:,}")

    # ── 7. Select validation threshold ────────────────────────────────────
    print(f"\n[6/6] Selecting threshold for recall >= {RECALL_TARGET} …")
    threshold = select_threshold(val_probs, val_labels, recall_target=RECALL_TARGET)

    # ── 8. Save complete checkpoint ───────────────────────────────────────
    print(f"\nSaving checkpoint → {SAVE_PATH.relative_to(REPO_ROOT)}")
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "layer":         layer.state_dict(),
        "input_dim":     actual_features,
        "weights":       weights,
        "bias":          bias,
        "feature_mean":  feat_mean,
        "feature_std":   feat_std,
        "threshold":     threshold,
        "recall_target": RECALL_TARGET,
        "random_seed":   RANDOM_SEED,
        "fit_epochs":    fit_epochs,
        "n_train":       N_train,
        "n_val":         len(val_labels),
    }, SAVE_PATH)

    print("\n" + "=" * 50)
    print("Training complete.")
    print(f"  Threshold        : {threshold:.4f}")
    print(f"  Checkpoint saved : {SAVE_PATH.relative_to(REPO_ROOT)}")
    print("Run  python evaluate.py  to evaluate on the test set.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline train the NeuroVerse gate learner."
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Cap number of training windows (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=FIT_EPOCHS,
        help=f"ImplicitLayer fit epochs (default: {FIT_EPOCHS})",
    )
    args = parser.parse_args()
    train(max_windows=args.max, fit_epochs=args.epochs)


if __name__ == "__main__":
    main()
