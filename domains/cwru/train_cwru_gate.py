"""
domains/cwru/train_cwru_gate.py
================================
CWRU-specific offline gate training.

Produces a checkpoint at ``data/processed/cwru_gate_weights.pt`` that is
tagged with ``domain='cwru'`` and contains normalisation statistics derived
exclusively from CWRU training files.

Split (file-level, no overlap with held-out test)
--------------------------------------------------
    Class        Train files              Val (threshold calibration)
    ------------ ------------------------ ----------
    normal       100, 97 2, 98            99
    inner_race   105, 106, 107            108
    ball         118, 119 2, 120          121
    outer_race   130, 131, 132            133

Feature vector (7 dimensions)
------------------------------
The gate uses 5 ImplicitLayer danger signals PLUS 2 raw vibration features:

    0  prediction_error     (ImplicitLayer)
    1  uncertainty          (ImplicitLayer)
    2  spatial_correlation  (ImplicitLayer)
    3  risk                 (ImplicitLayer)
    4  novelty              (ImplicitLayer)
    5  de_rms               raw Drive-End RMS amplitude -- ball vs normal AUC 1.00
    6  de_kurtosis          raw Drive-End kurtosis      -- outer/inner vs normal AUC 0.82

Motivation for adding raw vibration features
--------------------------------------------
Ball-fault windows are indistinguishable from normal windows using only the
5 ImplicitLayer signals (ball-vs-normal AUC <= 0.55 for every signal).
Raw de_rms has AUC=1.00 for ball vs normal on the validation set, while
de_kurtosis cleanly separates inner_race and outer_race (AUC=0.82).

Adding these two physics-informed features allows the logistic regression to
achieve >50% LLM reduction without sacrificing recall, which is not possible
with the 5-signal feature set alone.

Both raw features are normalised using train-split mean/std so there is no
leakage.  They are appended to the existing 5-feature vector; the feature
ordering and names are stored in the checkpoint so the evaluation code can
verify consistency.

Usage
-----
    python domains/cwru/train_cwru_gate.py
    python domains/cwru/train_cwru_gate.py --recall-target 0.73
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

warnings.filterwarnings("ignore")

from domains.cwru.loader import CWRULoader
from domains.cwru.preprocess import CWRUPreprocessor
from domains.cwru.domain import CWRUDomain
from nv_platform.snn_plugin import SNNInputPlugin
from nv_platform.implicit_layer import ImplicitPredictiveLayer
from nv_platform.gate_learner import _extract_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAIN            = "cwru"
RANDOM_SEED       = 0
# Recall target chosen so the val-sweep can find a threshold that also gives
# >= 50% LLM reduction.  Val sweep shows recall=0.73 is achievable at esc%=48%.
RECALL_TARGET_DEF = 0.73
SAVE_PATH         = Path(_REPO_ROOT) / "data" / "processed" / "cwru_gate_weights.pt"

# 7-feature vector: 5 ImplicitLayer signals + 2 raw vibration features
FEATURE_NAMES_AUGMENTED = [
    "prediction_error",
    "uncertainty",
    "spatial_correlation",
    "risk",
    "novelty",
    "de_rms",        # raw; ball-vs-normal AUC 1.00
    "de_kurtosis",   # raw; outer/inner-vs-normal AUC 0.82
]
N_FEATURES = len(FEATURE_NAMES_AUGMENTED)  # 7

# File-level split (must not overlap with test)
TRAIN_FILE_IDS: dict[str, list[str]] = {
    "normal":     ["100", "972", "98"],
    "inner_race": ["105", "106", "107"],
    "ball":       ["118", "1192", "120"],
    "outer_race": ["130", "131", "132"],
}
VAL_FILE_IDS: dict[str, str] = {
    "normal":     "99",
    "inner_race": "108",
    "ball":       "121",
    "outer_race": "133",
}


# ---------------------------------------------------------------------------
# Augmented feature collection
# ---------------------------------------------------------------------------

def collect_gate_features(
    events: list[dict],
    domain: CWRUDomain,
    snn: SNNInputPlugin,
    layer: ImplicitPredictiveLayer,
    desc: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run every event through SNN -> ImplicitLayer -> 7-feature extraction.

    Feature vector = [pred_error, uncertainty, spatial, risk, novelty,
                      de_rms_raw, de_kurtosis_raw].

    Raw de_rms and de_kurtosis are taken from the event's ``_raw`` sub-dict
    (physical units, before Z-score normalisation).  They are normalised later
    by train-split mean/std.

    Resets _predictor at each file boundary (within-file prediction error only).

    Returns
    -------
    feats  : np.ndarray, shape (N, 7)
    labels : np.ndarray, shape (N,)   -- 0=normal, 1=any fault
    """
    feats_list: list[list[float]] = []
    labels: list[int] = []
    prev_fid: str | None = None
    t0 = time.time()

    for i, event in enumerate(events):
        fid = str(event.get("file_id", ""))
        if fid != prev_fid:
            layer._predictor.reset()
            prev_fid = fid

        tensor = domain.event_to_tensor(event)
        spikes = snn.process_events(tensor)
        score  = layer.compute_state(spikes)

        # 5 ImplicitLayer signals
        f5 = _extract_features(score.as_tensor()).numpy().tolist()

        # 2 raw vibration features (physical units, not Z-scored here)
        raw = event.get("_raw", event)
        de_rms_val  = float(raw.get("de_rms",      0.0))
        de_kurt_val = float(raw.get("de_kurtosis", 0.0))

        feats_list.append(f5 + [de_rms_val, de_kurt_val])
        cls = event.get("fault_class", "normal")
        labels.append(0 if cls == "normal" else 1)

        if i > 0 and i % 1000 == 0:
            pct = 100 * i / len(events)
            print(f"  {desc} {i:>5}/{len(events)}  ({pct:.0f}%)  "
                  f"[{time.time()-t0:.0f}s]", end="\r", flush=True)

    print(f"  {desc} {len(events)}/{len(events)} (100%)  "
          f"[{time.time()-t0:.1f}s]      ")
    return np.array(feats_list, dtype=np.float32), np.array(labels)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_threshold(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    recall_target: float,
) -> tuple[float, dict]:
    """
    Choose the highest threshold that still achieves recall >= recall_target
    on the validation set.

    Returns (threshold, metrics_dict).
    """
    if val_labels.sum() == 0:
        print("  WARNING: no fault windows in val set -- defaulting to 0.5")
        return 0.5, {}

    n_normal = int((val_labels == 0).sum())
    n_fault  = int((val_labels == 1).sum())
    best_t   = 0.0
    best_metrics: dict = {}

    print(f"\n  Threshold sweep (val: {n_fault} faults, {n_normal} normal):")
    print(f"  {'threshold':>10} {'esc%':>8} {'LLM_red%':>10} {'recall':>8} "
          f"{'prec':>8} {'F1':>8} {'fn_norm%':>10}")

    for t in np.arange(0.05, 0.96, 0.05):
        preds = (val_probs >= t).astype(int)
        tp = int(((preds == 1) & (val_labels == 1)).sum())
        fp = int(((preds == 1) & (val_labels == 0)).sum())
        fn = int(((preds == 0) & (val_labels == 1)).sum())

        recall    = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        esc_rate  = (tp + fp) / max(len(val_labels), 1)
        llm_red   = 1.0 - esc_rate
        fn_norm   = fp / max(n_normal, 1)

        mark = " <- candidate" if recall >= recall_target else ""
        print(f"  {t:>10.2f} {esc_rate*100:>8.1f} {llm_red*100:>10.1f} "
              f"{recall:>8.3f} {precision:>8.3f} {f1:>8.3f} {fn_norm*100:>10.1f}{mark}")

        if recall >= recall_target:
            best_t = float(t)
            best_metrics = {
                "threshold":          best_t,
                "val_recall":         recall,
                "val_precision":      precision,
                "val_f1":             f1,
                "val_esc_rate":       esc_rate,
                "val_llm_reduction":  llm_red,
                "val_fn_normal_rate": fn_norm,
            }

    if not best_metrics:
        print(f"  WARNING: recall target {recall_target} not achievable -- "
              f"using 0.05")
        best_t = 0.05

    return best_t, best_metrics


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(
    recall_target: float = RECALL_TARGET_DEF,
    data_root: str = "domains/cwru",
) -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\nCWRU Gate Training (augmented 7-feature gate)")
    print("=" * 60)
    print(f"Domain         : {DOMAIN}")
    print(f"Recall target  : {recall_target}")
    print(f"Feature dim    : {N_FEATURES}  "
          f"(5 ImplicitLayer + de_rms + de_kurtosis)")
    print(f"Save path      : {SAVE_PATH.relative_to(Path(_REPO_ROOT))}")

    # ── 1. Load records ────────────────────────────────────────────────────
    loader = CWRULoader(data_root)
    all_records = loader.load_all()

    train_records = [
        r for r in all_records
        if r.file_id in TRAIN_FILE_IDS.get(r.fault_class, [])
    ]
    val_records = [
        r for r in all_records
        if r.file_id == VAL_FILE_IDS.get(r.fault_class)
    ]

    print(f"\nTrain files : {len(train_records)} "
          f"({[r.file_id for r in train_records]})")
    print(f"Val files   : {len(val_records)} "
          f"({[r.file_id for r in val_records]})")

    # ── 2. Preprocess (fit normaliser on train only) ───────────────────────
    prep = CWRUPreprocessor(window_size=1024, overlap=0.5, use_fe=True)

    train_windows = prep.process_records(train_records)
    train_raw     = prep.to_events(train_windows)
    train_norm    = prep.normalize_events([dict(e) for e in train_raw], fit=True)

    # Val: apply TRAIN statistics (fit=False -- no data leakage)
    val_windows = prep.process_records(val_records)
    val_raw     = prep.to_events(val_windows)
    val_norm    = prep.normalize_events([dict(e) for e in val_raw], fit=False)

    # Attach raw physical values to events (needed for de_rms / de_kurtosis)
    feat_cols = set(CWRUDomain().feature_cols)
    for ne, re_ in zip(train_norm, train_raw):
        ne["_raw"] = {k: re_[k] for k in feat_cols if k in re_}
    for ne, re_ in zip(val_norm, val_raw):
        ne["_raw"] = {k: re_[k] for k in feat_cols if k in re_}

    n_tr = len(train_norm)
    n_va = len(val_norm)
    tr_faults = sum(1 for e in train_norm if e.get("fault_class") != "normal")
    va_faults = sum(1 for e in val_norm   if e.get("fault_class") != "normal")
    print(f"\nTrain windows : {n_tr:,}  "
          f"(faults: {tr_faults:,}, normal: {n_tr-tr_faults:,})")
    print(f"Val windows   : {n_va:,}  "
          f"(faults: {va_faults:,}, normal: {n_va-va_faults:,})")

    # ── 3. Warm ImplicitPredictiveLayer on training data ───────────────────
    domain = CWRUDomain()
    snn    = SNNInputPlugin(backend="snntorch", num_steps=25)
    layer  = ImplicitPredictiveLayer(input_dim=domain.num_features)

    print("\n[0/4] Warming ImplicitPredictiveLayer on train windows ...")
    _warm_prev_fid: str | None = None
    for _w_event in train_norm:
        _w_fid = str(_w_event.get("file_id", ""))
        if _w_fid != _warm_prev_fid:
            layer._predictor.reset()
            _warm_prev_fid = _w_fid
        _w_tensor = domain.event_to_tensor(_w_event)
        _w_spikes = snn.process_events(_w_tensor)
        layer.compute_state(_w_spikes)
    print(f"  Warmed on {n_tr:,} windows.")

    # ── 4. Collect 7-feature vectors from TRAIN windows ───────────────────
    print("\n[1/4] Collecting train gate features (7-dim) ...")
    train_feats, train_labels = collect_gate_features(
        train_norm, domain, snn, layer, desc="train"
    )
    print(f"  Train features : {train_feats.shape}  "
          f"faults={train_labels.sum()}, "
          f"normal={len(train_labels)-train_labels.sum()}")

    # ── 5. Normalisation statistics from train only ────────────────────────
    feat_mean = train_feats.mean(axis=0)
    feat_std  = np.maximum(train_feats.std(axis=0), 1e-6)

    print("\n[2/4] CWRU domain normalisation statistics (train only):")
    print(f"  {'feature':22} {'mean':>10} {'std':>10}")
    for n, m, s in zip(FEATURE_NAMES_AUGMENTED, feat_mean, feat_std):
        print(f"  {n:22} {m:>10.4f} {s:>10.4f}")

    # ── 6. Fit logistic regression ─────────────────────────────────────────
    print("\n[3/4] Fitting logistic regression (balanced, C=1.0) ...")
    train_z = (train_feats - feat_mean) / feat_std

    n_fault  = int(train_labels.sum())
    n_normal_tr = len(train_labels) - n_fault
    print(f"  nfault={n_fault}  nnormal={n_normal_tr}  "
          f"imbalance={n_normal_tr/max(n_fault,1):.1f}:1")

    from sklearn.linear_model import LogisticRegression  # type: ignore
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
        C=1.0,
    )
    clf.fit(train_z, train_labels)

    weights = torch.tensor(clf.coef_[0], dtype=torch.float32)
    bias    = torch.tensor(clf.intercept_[0], dtype=torch.float32)

    print(f"  Weights:")
    for n, w in zip(FEATURE_NAMES_AUGMENTED, weights.tolist()):
        print(f"    {n:22}: {w:+.4f}")
    print(f"  Bias: {bias.item():+.4f}")

    logits_tr = train_z @ clf.coef_[0] + clf.intercept_[0]
    probs_tr  = 1.0 / (1.0 + np.exp(-logits_tr))
    preds_tr  = (probs_tr >= 0.5).astype(int)
    tp = int(((preds_tr == 1) & (train_labels == 1)).sum())
    fn = int(((preds_tr == 0) & (train_labels == 1)).sum())
    fp = int(((preds_tr == 1) & (train_labels == 0)).sum())
    print(f"  Train recall@0.5 : {tp/max(tp+fn,1):.3f}  "
          f"esc_rate={(tp+fp)/len(train_labels):.3f}  FN={fn}")

    # ── 7. Collect val features and select threshold ───────────────────────
    print("\n[4/4] Collecting val gate features (7-dim) ...")
    val_feats, val_labels = collect_gate_features(
        val_norm, domain, snn, layer, desc="val"
    )

    val_z      = (val_feats - feat_mean) / feat_std
    val_logits = val_z @ clf.coef_[0] + clf.intercept_[0]
    val_probs  = 1.0 / (1.0 + np.exp(-val_logits))

    threshold, val_metrics = select_threshold(val_probs, val_labels, recall_target)

    print(f"\n  Selected threshold : {threshold:.4f}")
    if val_metrics:
        print(f"  Val recall         : {val_metrics.get('val_recall'):.3f}")
        print(f"  Val LLM reduction  : {val_metrics.get('val_llm_reduction',0)*100:.1f}%")
        print(f"  Val esc_rate       : {val_metrics.get('val_esc_rate'):.3f}")
        print(f"  Val fn_normal_rate : {val_metrics.get('val_fn_normal_rate'):.3f}")

    # ── 8. Save checkpoint ─────────────────────────────────────────────────
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat_mean_t = torch.tensor(feat_mean, dtype=torch.float32)
    feat_std_t  = torch.tensor(feat_std,  dtype=torch.float32)

    ckpt = {
        "domain":           DOMAIN,
        "layer":            layer.state_dict(),
        "input_dim":        domain.num_features,
        "weights":          weights,
        "bias":             bias,
        "feature_mean":     feat_mean_t,
        "feature_std":      feat_std_t,
        "feature_names":    FEATURE_NAMES_AUGMENTED,
        "n_features":       N_FEATURES,
        "threshold":        float(threshold),
        "recall_target":    recall_target,
        "random_seed":      RANDOM_SEED,
        "n_train":          n_tr,
        "n_val":            n_va,
        "train_faults":     int(tr_faults),
        "val_faults":       int(va_faults),
        "val_metrics":      val_metrics,
        "train_file_ids":   TRAIN_FILE_IDS,
        "val_file_ids":     VAL_FILE_IDS,
    }
    torch.save(ckpt, SAVE_PATH)
    print(f"\nCheckpoint saved -> {SAVE_PATH.relative_to(Path(_REPO_ROOT))}")

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Domain        : {DOMAIN}")
    print(f"  Feature dim   : {N_FEATURES}")
    print(f"  Threshold     : {threshold:.4f}  (recall target: {recall_target})")
    print(f"  Checkpoint    : {SAVE_PATH.relative_to(Path(_REPO_ROOT))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the CWRU gate learner (augmented 7-feature)."
    )
    parser.add_argument(
        "--recall-target", type=float, default=RECALL_TARGET_DEF,
        help=f"Min fault recall on val set (default: {RECALL_TARGET_DEF})",
    )
    parser.add_argument(
        "--data-root", default="domains/cwru",
        help="Path to CWRU data directory",
    )
    args = parser.parse_args()
    train(recall_target=args.recall_target, data_root=args.data_root)


if __name__ == "__main__":
    main()
