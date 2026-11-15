"""
domains/cwru/evaluate_held_out.py
==================================
Proper file-level held-out evaluation for the CWRU pipeline.

Split (fixed, by file ID):
    Class        Train files          Test file
    ──────────── ──────────────────── ──────────
    normal       100, 972, 98         99
    inner_race   105, 106, 107        108
    ball         118, 119 2, 120      121
    outer_race   130, 131, 132        133

Procedure
---------
1. Load and preprocess TRAIN files only.
2. Fit Z-score normalisation statistics on train windows.
3. Derive heuristic thresholds (kurtosis / RMS) from train distributions.
4. Load and preprocess TEST files, applying train normalisation stats.
5. Run the full four-layer pipeline on test windows — gate + classifier.
6. Report per-class and overall accuracy, gate metrics, and confusion matrix.

The test file for each class is never touched until step 4, so thresholds
and normalisation statistics are genuinely unseen-data estimates.

Usage
-----
    python domains/cwru/evaluate_held_out.py
    python domains/cwru/evaluate_held_out.py --max-per-class 200
"""

from __future__ import annotations

import argparse
import json
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

from domains.cwru.loader import CWRULoader, CWRURecord
from domains.cwru.preprocess import CWRUPreprocessor
from domains.cwru.domain import CWRUDomain
from domains.cwru.pipeline import CWRUPipeline, PipelineResult, _stratified_sample
from nv_platform.snn_plugin import SNNInputPlugin
from nv_platform.implicit_layer import ImplicitPredictiveLayer
from nv_platform.gate import ReasoningGate, EscalationHistory
from nv_platform.gate_learner import GateLearner
from nv_platform.agent_plugin import AgentDecision

# ---------------------------------------------------------------------------
# File-level split definition
# ---------------------------------------------------------------------------

TRAIN_FILE_IDS: dict[str, list[str]] = {
    "normal":     ["100", "972", "98"],
    "inner_race": ["105", "106", "107"],
    "ball":       ["118", "1192", "120"],   # "119 2.mat" → file_id "1192"
    "outer_race": ["130", "131", "132"],
}

TEST_FILE_IDS: dict[str, str] = {
    "normal":     "99",
    "inner_race": "108",
    "ball":       "121",
    "outer_race": "133",
}

CLASSES = ["normal", "inner_race", "ball", "outer_race"]


# ---------------------------------------------------------------------------
# Threshold calibration from train distributions
# ---------------------------------------------------------------------------

def calibrate_thresholds(
    train_norm_events: list[dict],
    train_raw_events: list[dict],
) -> dict:
    """
    Derive kurtosis / RMS decision boundaries from train-set distributions.

    Strategy: find the midpoint of the gap between adjacent class clusters
    for Z-scored features (heuristic path) and raw features (fast-path).
    Returns a dict of thresholds used by both decision functions.
    """
    by_class_norm = {c: [] for c in CLASSES}
    by_class_raw  = {c: [] for c in CLASSES}
    for e in train_norm_events:
        by_class_norm[e["fault_class"]].append(e)
    for e in train_raw_events:
        by_class_raw[e["fault_class"]].append(e)

    def p(events, feat, pct):
        vals = [e[feat] for e in events]
        return float(np.percentile(vals, pct)) if vals else 0.0

    # ── Z-score thresholds (for _heuristic_decision) ─────────────────────
    # outer_race / inner_race gap: midpoint between inner_race p95 and outer_race p5
    ir_kurt_p95_z  = p(by_class_norm["inner_race"], "de_kurtosis", 95)
    or_kurt_p5_z   = p(by_class_norm["outer_race"], "de_kurtosis",  5)
    z_outer_thresh = (ir_kurt_p95_z + or_kurt_p5_z) / 2

    # inner_race / normal+ball gap: midpoint between normal p95 and inner_race p5
    norm_kurt_p95_z = p(by_class_norm["normal"],     "de_kurtosis", 95)
    ir_kurt_p5_z    = p(by_class_norm["inner_race"], "de_kurtosis",  5)
    z_inner_thresh  = (norm_kurt_p95_z + ir_kurt_p5_z) / 2

    # ball / normal gap on de_rms (kurtosis alone can't separate them)
    norm_rms_p95_z  = p(by_class_norm["normal"], "de_rms", 95)
    ball_rms_p5_z   = p(by_class_norm["ball"],   "de_rms",  5)
    z_ball_thresh   = (norm_rms_p95_z + ball_rms_p5_z) / 2

    # ── Raw thresholds (for _fast_path_decision) ──────────────────────────
    ir_kurt_p95_r   = p(by_class_raw["inner_race"], "de_kurtosis", 95)
    or_kurt_p5_r    = p(by_class_raw["outer_race"], "de_kurtosis",  5)
    raw_outer_thresh = (ir_kurt_p95_r + or_kurt_p5_r) / 2

    norm_kurt_p95_r  = p(by_class_raw["normal"],     "de_kurtosis", 95)
    ir_kurt_p5_r     = p(by_class_raw["inner_race"], "de_kurtosis",  5)
    raw_inner_thresh = (norm_kurt_p95_r + ir_kurt_p5_r) / 2

    norm_rms_p95_r   = p(by_class_raw["normal"], "de_rms", 95)
    ball_rms_p5_r    = p(by_class_raw["ball"],   "de_rms",  5)
    raw_ball_thresh  = (norm_rms_p95_r + ball_rms_p5_r) / 2

    return {
        # Z-score thresholds
        "z_outer": z_outer_thresh,
        "z_inner": z_inner_thresh,
        "z_ball":  z_ball_thresh,
        # Raw thresholds
        "raw_outer": raw_outer_thresh,
        "raw_inner": raw_inner_thresh,
        "raw_ball":  raw_ball_thresh,
    }


# ---------------------------------------------------------------------------
# Classifier functions using calibrated thresholds
# ---------------------------------------------------------------------------

def classify_heuristic(event: dict, thresh: dict) -> str:
    """Classify a Z-scored event using calibrated thresholds."""
    k = float(event.get("de_kurtosis", 0.0))
    r = float(event.get("de_rms", 0.0))
    if k > thresh["z_outer"]:
        return "outer_race"
    elif k > thresh["z_inner"]:
        return "inner_race"
    elif r > thresh["z_ball"]:
        return "ball"
    else:
        return "normal"


def classify_fast_path(raw_event: dict, thresh: dict) -> str:
    """Classify a raw-valued event using calibrated thresholds."""
    k = float(raw_event.get("de_kurtosis", 0.0))
    r = float(raw_event.get("de_rms", 0.0))
    if k > thresh["raw_outer"]:
        return "outer_race"
    elif k > thresh["raw_inner"]:
        return "inner_race"
    elif r > thresh["raw_ball"]:
        return "ball"
    else:
        return "normal"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run(max_per_class: int | None = None, data_root: str = "domains/cwru") -> None:
    loader = CWRULoader(data_root)
    all_records = loader.load_all()

    # ── Split records by file ID ──────────────────────────────────────────
    train_records: list[CWRURecord] = []
    test_records:  list[CWRURecord] = []

    for rec in all_records:
        fid = rec.file_id
        cls = rec.fault_class
        if fid == TEST_FILE_IDS.get(cls):
            test_records.append(rec)
        elif fid in TRAIN_FILE_IDS.get(cls, []):
            train_records.append(rec)
        # else: unrecognised file — skip

    print(f"Train records : {len(train_records)} files")
    print(f"Test  records : {len(test_records)} files")
    for cls in CLASSES:
        tr = [r for r in train_records if r.fault_class == cls]
        te = [r for r in test_records  if r.fault_class == cls]
        print(f"  {cls:<14}  train={[r.file_id for r in tr]}  test={[r.file_id for r in te]}")

    # ── Preprocess train ──────────────────────────────────────────────────
    print("\nPreprocessing train files …")
    prep = CWRUPreprocessor(window_size=1024, overlap=0.5, use_fe=True)
    train_windows  = prep.process_records(train_records)
    train_raw      = prep.to_events(train_windows)
    train_norm     = prep.normalize_events([dict(e) for e in train_raw], fit=True)

    print(f"  Train windows : {len(train_norm):,}")

    # ── Calibrate thresholds on train ─────────────────────────────────────
    thresh = calibrate_thresholds(train_norm, train_raw)
    print("\nCalibrated thresholds (from train files only):")
    print(f"  Z-score  outer_race  de_kurtosis > {thresh['z_outer']:+.4f}")
    print(f"  Z-score  inner_race  de_kurtosis > {thresh['z_inner']:+.4f}")
    print(f"  Z-score  ball        de_rms       > {thresh['z_ball']:+.4f}")
    print(f"  Raw      outer_race  de_kurtosis > {thresh['raw_outer']:.4f}")
    print(f"  Raw      inner_race  de_kurtosis > {thresh['raw_inner']:.4f}")
    print(f"  Raw      ball        de_rms       > {thresh['raw_ball']:.4f}")

    # ── Preprocess test using TRAIN normalisation stats (fit=False) ───────
    print("\nPreprocessing test files (applying train normalisation stats) …")
    test_windows = prep.process_records(test_records)
    test_raw     = prep.to_events(test_windows)
    test_norm    = prep.normalize_events([dict(e) for e in test_raw], fit=False)

    print(f"  Test windows  : {len(test_norm):,}")

    # Attach raw values for the fast-path classifier
    feat_cols = set(CWRUDomain().feature_cols)
    for ne, re_ in zip(test_norm, test_raw):
        ne["_raw"] = {k: re_[k] for k in feat_cols if k in re_}

    # ── Optionally cap per class ──────────────────────────────────────────
    if max_per_class is not None:
        by_cls: dict[str, list] = {c: [] for c in CLASSES}
        for e in test_norm:
            by_cls[e["fault_class"]].append(e)
        test_norm = []
        for c in CLASSES:
            evs = by_cls[c]
            step = max(1, len(evs) // max_per_class)
            test_norm.extend(evs[::step][:max_per_class])
        print(f"  Capped to ≤{max_per_class} per class → {len(test_norm)} total")

    # ── Set up platform layers ────────────────────────────────────────────
    domain = CWRUDomain()
    snn    = SNNInputPlugin(backend="snntorch", num_steps=25)
    layer2 = ImplicitPredictiveLayer(input_dim=domain.num_features)

    learner = GateLearner()
    weights_path = Path(_REPO_ROOT) / "data" / "processed" / "gate_weights.pt"
    if weights_path.exists():
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        with torch.no_grad():
            learner._model.weight.copy_(ckpt["weights"].unsqueeze(0))
            learner._model.bias.copy_(ckpt["bias"].unsqueeze(0))
        learner.feature_mean = ckpt["feature_mean"]
        learner.feature_std  = ckpt["feature_std"]
        gate_threshold = float(ckpt["threshold"])
        print(f"\nLoaded GateLearner weights (threshold={gate_threshold:.4f})")
    else:
        gate_threshold = 0.5
        print("\nNo checkpoint found — using default gate threshold 0.5")

    gate    = ReasoningGate(learner=learner, threshold=gate_threshold)
    history = EscalationHistory()

    # ── Run pipeline on test windows ──────────────────────────────────────
    print("\nRunning pipeline on TEST windows …")
    correct = 0
    total   = 0
    escalated = 0
    per_class_correct: dict[str, int] = {c: 0 for c in CLASSES}
    per_class_total:   dict[str, int] = {c: 0 for c in CLASSES}
    conf_mat: dict[str, dict[str, int]] = {
        c: {c2: 0 for c2 in CLASSES} for c in CLASSES
    }

    prev_file_id: str | None = None

    for event in test_norm:
        file_id = str(event.get("file_id", ""))
        if file_id != prev_file_id:
            layer2._predictor.reset()
            prev_file_id = file_id

        true_cls = event["fault_class"]

        # Layer 1
        tensor = domain.event_to_tensor(event)
        spikes = snn.process_events(tensor)

        # Layer 2
        score = layer2.compute_state(spikes)

        # Layer 3
        escalate, gate_conf = gate.should_escalate(score, history)

        # Layer 4 — calibrated heuristic (no LLM)
        raw = event.get("_raw", event)
        if escalate:
            pred = classify_heuristic(event, thresh)
        else:
            pred = classify_fast_path(raw, thresh)

        # Tally
        per_class_total[true_cls] += 1
        total += 1
        if escalate:
            escalated += 1
        if pred == true_cls:
            correct += 1
            per_class_correct[true_cls] += 1
        conf_mat[true_cls][pred] += 1

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HELD-OUT EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test file per class  : one unseen file each")
    print(f"  Total test windows   : {total:,}")
    print(f"  Escalation rate      : {escalated/total:.1%}  ({escalated}/{total})")
    print(f"  LLM call reduction   : {1 - escalated/total:.1%}  vs always-escalate")
    print()
    print(f"  {'Class':<14}  {'Correct':>8}  {'Total':>7}  {'Accuracy':>9}")
    print(f"  {'─'*44}")
    for cls in CLASSES:
        acc = per_class_correct[cls] / max(per_class_total[cls], 1)
        print(f"  {cls:<14}  {per_class_correct[cls]:>8}  "
              f"{per_class_total[cls]:>7}  {acc:>9.1%}")
    print(f"  {'─'*44}")
    overall = correct / max(total, 1)
    print(f"  {'OVERALL':<14}  {correct:>8}  {total:>7}  {overall:>9.1%}")

    print("\n  Confusion matrix (true \\ predicted):")
    hdr = f"  {'':14}" + "".join(f"{c[:8]:>10}" for c in CLASSES)
    print(hdr)
    print(f"  {'─' * (14 + 10 * len(CLASSES))}")
    for tc in CLASSES:
        row = f"  {tc:<14}" + "".join(f"{conf_mat[tc].get(pc,0):>10}" for pc in CLASSES)
        print(row)

    print("\n  Calibrated thresholds used:")
    print(f"    Z  outer_race  kurt > {thresh['z_outer']:+.3f}")
    print(f"    Z  inner_race  kurt > {thresh['z_inner']:+.3f}")
    print(f"    Z  ball        rms  > {thresh['z_ball']:+.3f}")
    print(f"    Raw outer_race kurt > {thresh['raw_outer']:.3f}")
    print(f"    Raw inner_race kurt > {thresh['raw_inner']:.3f}")
    print(f"    Raw ball       rms  > {thresh['raw_ball']:.3f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="File-level held-out evaluation for CWRU pipeline."
    )
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap test windows per class (default: all)")
    parser.add_argument("--data-root", default="domains/cwru",
                        help="Path to CWRU data directory")
    args = parser.parse_args()
    run(max_per_class=args.max_per_class, data_root=args.data_root)
