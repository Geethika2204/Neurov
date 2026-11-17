"""
domains/cwru/evaluate_held_out.py
==================================
Proper file-level held-out evaluation for the CWRU pipeline.

Split (fixed, by file ID):
    Class        Train files          Test file
    ──────────── ──────────────────────── ──────────
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
7. Optionally compare BEFORE (traffic checkpoint) vs AFTER (CWRU checkpoint).

The test file for each class is never touched until step 4, so thresholds
and normalisation statistics are genuinely unseen-data estimates.

Usage
-----
    python domains/cwru/evaluate_held_out.py
    python domains/cwru/evaluate_held_out.py --max-per-class 200
    python domains/cwru/evaluate_held_out.py --compare
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

    ir_kurt_p95_z  = p(by_class_norm["inner_race"], "de_kurtosis", 95)
    or_kurt_p5_z   = p(by_class_norm["outer_race"], "de_kurtosis",  5)
    z_outer_thresh = (ir_kurt_p95_z + or_kurt_p5_z) / 2

    norm_kurt_p95_z = p(by_class_norm["normal"],     "de_kurtosis", 95)
    ir_kurt_p5_z    = p(by_class_norm["inner_race"], "de_kurtosis",  5)
    z_inner_thresh  = (norm_kurt_p95_z + ir_kurt_p5_z) / 2

    norm_rms_p95_z  = p(by_class_norm["normal"], "de_rms", 95)
    ball_rms_p5_z   = p(by_class_norm["ball"],   "de_rms",  5)
    z_ball_thresh   = (norm_rms_p95_z + ball_rms_p5_z) / 2

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
        "z_outer": z_outer_thresh, "z_inner": z_inner_thresh, "z_ball": z_ball_thresh,
        "raw_outer": raw_outer_thresh, "raw_inner": raw_inner_thresh, "raw_ball": raw_ball_thresh,
    }


def classify_heuristic(event: dict, thresh: dict) -> str:
    k = float(event.get("de_kurtosis", 0.0))
    r = float(event.get("de_rms", 0.0))
    if k > thresh["z_outer"]:   return "outer_race"
    elif k > thresh["z_inner"]: return "inner_race"
    elif r > thresh["z_ball"]:  return "ball"
    else:                        return "normal"


def classify_fast_path(raw_event: dict, thresh: dict) -> str:
    k = float(raw_event.get("de_kurtosis", 0.0))
    r = float(raw_event.get("de_rms", 0.0))
    if k > thresh["raw_outer"]:   return "outer_race"
    elif k > thresh["raw_inner"]: return "inner_race"
    elif r > thresh["raw_ball"]:  return "ball"
    else:                          return "normal"


# ---------------------------------------------------------------------------
# Gate loader helper
# ---------------------------------------------------------------------------

def load_gate(ckpt_path: Path, expected_domain: str = "cwru") -> tuple[GateLearner, float, str]:
    """
    Load a gate checkpoint and return (learner, threshold, domain_tag).

    Handles both 5-feature (legacy) and 7-feature (augmented) checkpoints.
    Raises a warning (not an error) if the domain tag is missing or mismatched.
    """
    import torch.nn as _nn
    learner = GateLearner()
    if not ckpt_path.exists():
        return learner, 0.5, "none"

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    domain_tag = ckpt.get("domain", "unknown")

    if domain_tag != expected_domain and expected_domain != "__any__":
        warnings.warn(
            f"Checkpoint {ckpt_path.name}: domain='{domain_tag}', "
            f"expected='{expected_domain}'. "
            "Results may be inaccurate.",
            UserWarning,
        )

    # Determine feature dimension from saved weights
    saved_weights = ckpt["weights"]          # shape (n_features,)
    n_features    = saved_weights.shape[0]

    # Resize the learner's linear layer to match checkpoint dimensions
    if n_features != 5:
        learner._model = _nn.Linear(n_features, 1)
        _nn.init.zeros_(learner._model.weight)
        _nn.init.constant_(learner._model.bias, 1.0)

    with torch.no_grad():
        learner._model.weight.copy_(saved_weights.unsqueeze(0))
        learner._model.bias.copy_(ckpt["bias"].unsqueeze(0))
    learner.feature_mean = ckpt["feature_mean"]
    learner.feature_std  = ckpt["feature_std"]
    threshold = float(ckpt["threshold"])
    return learner, threshold, domain_tag


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    test_norm: list[dict],
    thresh: dict,
    gate: ReasoningGate,
    domain: CWRUDomain,
    snn: SNNInputPlugin,
    layer2: ImplicitPredictiveLayer,
    label: str = "gate",
    n_extra_features: int = 0,
) -> dict:
    """
    Run one pass of the pipeline and return a metrics dict.

    Parameters
    ----------
    n_extra_features : int
        Number of extra raw features the gate was trained with (0 or 2).
        When 2, passes [de_rms, de_kurtosis] as extra_features to
        gate.should_escalate so the learner scores the full 7-dim vector.
    """
    history = EscalationHistory()

    correct = 0
    total   = 0
    escalated = 0
    hard_floor_fires = 0
    per_class_correct: dict[str, int]  = {c: 0 for c in CLASSES}
    per_class_total:   dict[str, int]  = {c: 0 for c in CLASSES}
    per_class_esc:     dict[str, int]  = {c: 0 for c in CLASSES}
    conf_mat: dict[str, dict[str, int]] = {
        c: {c2: 0 for c2 in CLASSES} for c in CLASSES
    }

    prev_file_id: str | None = None
    all_confs: list[float] = []

    for event in test_norm:
        file_id = str(event.get("file_id", ""))
        if file_id != prev_file_id:
            layer2._predictor.reset()
            prev_file_id = file_id

        true_cls = event["fault_class"]

        tensor = domain.event_to_tensor(event)
        spikes = snn.process_events(tensor)
        score  = layer2.compute_state(spikes)

        # Build extra features tensor if the checkpoint expects them
        extra = None
        if n_extra_features > 0:
            raw = event.get("_raw", event)
            de_rms_v  = float(raw.get("de_rms",      0.0))
            de_kurt_v = float(raw.get("de_kurtosis", 0.0))
            extra = torch.tensor([de_rms_v, de_kurt_v], dtype=torch.float32)

        escalate, gate_conf = gate.should_escalate(
            score, history, extra_features=extra
        )
        all_confs.append(float(gate_conf))

        # Detect hard-floor fires: escalated but gate_conf < threshold
        if escalate and gate_conf < gate.threshold:
            hard_floor_fires += 1

        raw = event.get("_raw", event)
        if escalate:
            pred = classify_heuristic(event, thresh)
        else:
            pred = classify_fast_path(raw, thresh)

        per_class_total[true_cls] += 1
        total += 1
        if escalate:
            escalated += 1
            per_class_esc[true_cls] += 1
        if pred == true_cls:
            correct += 1
            per_class_correct[true_cls] += 1
        conf_mat[true_cls][pred] += 1

    # Gate metrics: faults = escalation should happen; normal = should not
    n_fault_windows = sum(per_class_total[c] for c in CLASSES if c != "normal")
    n_normal_windows = per_class_total["normal"]
    # Fault escalation recall = how many fault windows were escalated
    fault_esc  = sum(per_class_esc[c]  for c in CLASSES if c != "normal")
    normal_esc = per_class_esc["normal"]
    fault_recall = fault_esc  / max(n_fault_windows, 1)
    normal_fp    = normal_esc / max(n_normal_windows, 1)
    precision_gate = fault_esc / max(escalated, 1)

    return {
        "label":          label,
        "total":          total,
        "escalated":      escalated,
        "not_escalated":  total - escalated,
        "esc_rate":       escalated / max(total, 1),
        "correct":        correct,
        "accuracy":       correct / max(total, 1),
        "fault_recall":   fault_recall,
        "normal_fp_rate": normal_fp,
        "gate_precision": precision_gate,
        "llm_calls_saved": total - escalated,
        "hard_floor_fires": hard_floor_fires,
        "gate_conf_mean": float(np.mean(all_confs)) if all_confs else 0.0,
        "gate_conf_std":  float(np.std(all_confs))  if all_confs else 0.0,
        "per_class_correct": per_class_correct,
        "per_class_total":   per_class_total,
        "per_class_esc":     per_class_esc,
        "conf_mat":          conf_mat,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    max_per_class: int | None = None,
    data_root: str = "domains/cwru",
    compare: bool = False,
    output_path: str | None = None,
) -> None:
    loader = CWRULoader(data_root)
    all_records = loader.load_all()

    train_records: list[CWRURecord] = []
    test_records:  list[CWRURecord] = []

    for rec in all_records:
        fid = rec.file_id
        cls = rec.fault_class
        if fid == TEST_FILE_IDS.get(cls):
            test_records.append(rec)
        elif fid in TRAIN_FILE_IDS.get(cls, []):
            train_records.append(rec)

    print(f"Train records : {len(train_records)} files")
    print(f"Test  records : {len(test_records)} files")
    for cls in CLASSES:
        tr = [r for r in train_records if r.fault_class == cls]
        te = [r for r in test_records  if r.fault_class == cls]
        print(f"  {cls:<14}  train={[r.file_id for r in tr]}  "
              f"test={[r.file_id for r in te]}")

    prep = CWRUPreprocessor(window_size=1024, overlap=0.5, use_fe=True)
    print("\nPreprocessing train files …")
    train_windows  = prep.process_records(train_records)
    train_raw      = prep.to_events(train_windows)
    train_norm     = prep.normalize_events([dict(e) for e in train_raw], fit=True)
    print(f"  Train windows : {len(train_norm):,}")

    thresh = calibrate_thresholds(train_norm, train_raw)
    print("\nCalibrated thresholds (from train files only):")
    print(f"  Z-score  outer_race  de_kurtosis > {thresh['z_outer']:+.4f}")
    print(f"  Z-score  inner_race  de_kurtosis > {thresh['z_inner']:+.4f}")
    print(f"  Z-score  ball        de_rms       > {thresh['z_ball']:+.4f}")
    print(f"  Raw      outer_race  de_kurtosis > {thresh['raw_outer']:.4f}")
    print(f"  Raw      inner_race  de_kurtosis > {thresh['raw_inner']:.4f}")
    print(f"  Raw      ball        de_rms       > {thresh['raw_ball']:.4f}")

    print("\nPreprocessing test files (applying train normalisation stats) …")
    test_windows = prep.process_records(test_records)
    test_raw     = prep.to_events(test_windows)
    test_norm    = prep.normalize_events([dict(e) for e in test_raw], fit=False)
    print(f"  Test windows  : {len(test_norm):,}")

    feat_cols = set(CWRUDomain().feature_cols)
    for ne, re_ in zip(test_norm, test_raw):
        ne["_raw"] = {k: re_[k] for k in feat_cols if k in re_}

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

    domain = CWRUDomain()
    snn    = SNNInputPlugin(backend="snntorch", num_steps=25)
    layer2 = ImplicitPredictiveLayer(input_dim=domain.num_features)

    _repo = Path(_REPO_ROOT)
    cwru_ckpt_path    = _repo / "data" / "processed" / "cwru_gate_weights.pt"
    generic_ckpt_path = _repo / "data" / "processed" / "gate_weights.pt"

    # ── Evaluate with CWRU checkpoint (primary) ───────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION: CWRU-domain gate checkpoint")
    print("=" * 60)
    if cwru_ckpt_path.exists():
        learner_cwru, threshold_cwru, domain_tag = load_gate(cwru_ckpt_path, "cwru")
        print(f"Loaded {cwru_ckpt_path.name}  domain='{domain_tag}'  threshold={threshold_cwru:.4f}")
        _ckpt_for_layer = torch.load(str(cwru_ckpt_path), map_location="cpu", weights_only=False)
        # Determine how many extra (raw) features the gate uses
        n_saved_weights  = int(_ckpt_for_layer["weights"].shape[0])
        n_extra_features = max(0, n_saved_weights - 5)
        feat_names_saved = _ckpt_for_layer.get("feature_names", [])
        print(f"  Gate feature dim : {n_saved_weights}  "
              f"(5 ImplicitLayer + {n_extra_features} raw)")
        if feat_names_saved:
            print(f"  Feature names    : {feat_names_saved}")
        if "layer" in _ckpt_for_layer:
            try:
                layer2.load_state_dict(_ckpt_for_layer["layer"])
                print("  ImplicitPredictiveLayer state restored from checkpoint.")
            except Exception as _layer_exc:
                print(f"  WARNING: could not restore layer state — {_layer_exc}")
    else:
        print(f"CWRU checkpoint not found at {cwru_ckpt_path.name}")
        print("Run:  python domains/cwru/train_cwru_gate.py")
        learner_cwru, threshold_cwru, domain_tag = GateLearner(), 0.5, "none"
        n_extra_features = 0
    learner_cwru.update_every = 10**9  # disable online update during eval
    gate_cwru = ReasoningGate(learner=learner_cwru, threshold=threshold_cwru)
    results_cwru = run_evaluation(test_norm, thresh, gate_cwru, domain, snn, layer2,
                                   label="CWRU gate",
                                   n_extra_features=n_extra_features)
    results_cwru["threshold"] = threshold_cwru
    _print_results(results_cwru, thresh)

    # Save results to JSON if output path provided
    if output_path:
        _save_results(results_cwru, thresh, output_path)

    if not compare:
        return

    # ── Evaluate with original traffic checkpoint (for comparison) ────────
    print("\n" + "=" * 60)
    print("COMPARISON: Original traffic-domain gate checkpoint")
    print("=" * 60)
    if generic_ckpt_path.exists():
        learner_old, threshold_old, domain_tag_old = load_gate(
            generic_ckpt_path, expected_domain="__any__"
        )
        print(f"Loaded {generic_ckpt_path.name}  domain='{domain_tag_old}'  "
              f"threshold={threshold_old:.4f}")
    else:
        print(f"Generic checkpoint not found — skipping comparison")
        return

    learner_old.update_every = 10**9  # disable online update during eval
    gate_old = ReasoningGate(learner=learner_old, threshold=threshold_old)
    results_old = run_evaluation(test_norm, thresh, gate_old, domain, snn, layer2,
                                  label="Traffic gate (old)",
                                  n_extra_features=0)
    _print_results(results_old, thresh)

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BEFORE / AFTER COMPARISON")
    print("=" * 60)
    _print_comparison(results_old, results_cwru)


def _save_results(r: dict, thresh: dict, output_path: str) -> None:
    """Save evaluation results to a JSON file."""
    import json
    from pathlib import Path

    # Convert non-serializable values
    out = {
        "label": r["label"],
        "threshold": r.get("threshold", "N/A"),
        "total_test_windows": r["total"],
        "escalated": r["escalated"],
        "not_escalated": r["not_escalated"],
        "escalation_rate": round(r["esc_rate"], 4),
        "llm_calls_saved": r["llm_calls_saved"],
        "hard_floor_fires": r["hard_floor_fires"],
        "gate_conf_mean": round(r["gate_conf_mean"], 4),
        "gate_conf_std": round(r["gate_conf_std"], 4),
        "classification_accuracy": round(r["accuracy"], 4),
        "fault_escalation_recall": round(r["fault_recall"], 4),
        "normal_false_esc_rate": round(r["normal_fp_rate"], 4),
        "gate_precision": round(r["gate_precision"], 4),
        "per_class": {},
        "confusion_matrix": r["conf_mat"],
        "thresholds": {k: round(v, 4) for k, v in thresh.items()},
    }
    for cls in CLASSES:
        out["per_class"][cls] = {
            "correct": r["per_class_correct"][cls],
            "total": r["per_class_total"][cls],
            "accuracy": round(r["per_class_correct"][cls] / max(r["per_class_total"][cls], 1), 4),
            "escalated": r["per_class_esc"][cls],
            "esc_pct": round(r["per_class_esc"][cls] / max(r["per_class_total"][cls], 1), 4),
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {output_path}")


def _print_results(r: dict, thresh: dict) -> None:
    total = r["total"]
    print(f"\n  Total test windows   : {total:,}")
    print(f"  Escalated            : {r['escalated']:,}  ({r['esc_rate']:.1%})")
    print(f"  Not escalated        : {r['not_escalated']:,}  ({1-r['esc_rate']:.1%})")
    print(f"  LLM calls saved      : {r['llm_calls_saved']:,} vs always-escalate")
    print(f"  Hard-floor fires     : {r['hard_floor_fires']:,}")
    print()
    print(f"  Gate confidence      : mean={r['gate_conf_mean']:.3f}  "
          f"std={r['gate_conf_std']:.3f}")
    print()
    print(f"  Classification accuracy : {r['accuracy']:.1%}  ({r['correct']}/{total})")
    print(f"  Fault escalation recall : {r['fault_recall']:.1%}")
    print(f"  Normal false esc rate   : {r['normal_fp_rate']:.1%}")
    print(f"  Gate precision          : {r['gate_precision']:.1%}")
    print()
    print(f"  {'Class':<14}  {'Correct':>8}  {'Total':>7}  {'Accuracy':>9}  "
          f"{'Escalated':>10}  {'Esc%':>6}")
    print(f"  {'─' * 56}")
    for cls in ["normal", "inner_race", "ball", "outer_race"]:
        acc = r["per_class_correct"][cls] / max(r["per_class_total"][cls], 1)
        ep  = r["per_class_esc"][cls] / max(r["per_class_total"][cls], 1)
        print(f"  {cls:<14}  {r['per_class_correct'][cls]:>8}  "
              f"{r['per_class_total'][cls]:>7}  {acc:>9.1%}  "
              f"{r['per_class_esc'][cls]:>10}  {ep:>6.1%}")
    print(f"  {'─' * 56}")
    tot_c = sum(r["per_class_correct"].values())
    tot_t = sum(r["per_class_total"].values())
    print(f"  {'OVERALL':<14}  {tot_c:>8}  {tot_t:>7}  "
          f"{tot_c/max(tot_t,1):>9.1%}")
    print()
    print("  Confusion matrix (true \\ predicted):")
    hdr = f"  {'':14}" + "".join(f"{c[:8]:>10}" for c in CLASSES)
    print(hdr)
    print(f"  {'─' * (14 + 10 * len(CLASSES))}")
    conf_mat = r["conf_mat"]
    for tc in CLASSES:
        row = f"  {tc:<14}" + "".join(f"{conf_mat[tc].get(pc,0):>10}" for pc in CLASSES)
        print(row)


def _print_comparison(old: dict, new: dict) -> None:
    metrics = [
        ("Metric",                    "Original gate",   "Improved gate"),
        ("─"*30,                      "─"*15,            "─"*15),
        ("Escalation rate",           f"{old['esc_rate']:.1%}",    f"{new['esc_rate']:.1%}"),
        ("LLM calls saved",           f"{old['llm_calls_saved']}",  f"{new['llm_calls_saved']}"),
        ("Classification accuracy",   f"{old['accuracy']:.1%}",    f"{new['accuracy']:.1%}"),
        ("Fault escalation recall",   f"{old['fault_recall']:.1%}", f"{new['fault_recall']:.1%}"),
        ("Normal false esc rate",     f"{old['normal_fp_rate']:.1%}",f"{new['normal_fp_rate']:.1%}"),
        ("Gate precision",            f"{old['gate_precision']:.1%}",f"{new['gate_precision']:.1%}"),
        ("Hard-floor fires",          str(old['hard_floor_fires']), str(new['hard_floor_fires'])),
        ("Gate conf mean",            f"{old['gate_conf_mean']:.3f}", f"{new['gate_conf_mean']:.3f}"),
        ("Gate threshold used",       "0.3276",                     f"{new.get('threshold', 'N/A')}"),
    ]
    col_w = [32, 16, 16]
    for row in metrics:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, col_w)))
    print()
    # Delta
    delta_esc = old["esc_rate"] - new["esc_rate"]
    delta_acc = new["accuracy"] - old["accuracy"]
    delta_rec = new["fault_recall"] - old["fault_recall"]
    delta_fp  = old["normal_fp_rate"] - new["normal_fp_rate"]
    print(f"  Escalation rate change  : {delta_esc:+.1%}  ({'reduced' if delta_esc > 0 else 'increased'})")
    print(f"  Accuracy change         : {delta_acc:+.1%}  ({'improved' if delta_acc >= 0 else 'degraded'})")
    print(f"  Fault recall change     : {delta_rec:+.1%}  ({'improved' if delta_rec >= 0 else 'degraded'})")
    print(f"  Normal false esc change : {delta_fp:+.1%}  ({'reduced' if delta_fp > 0 else 'increased'})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="File-level held-out evaluation for CWRU pipeline."
    )
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--data-root", default="domains/cwru")
    parser.add_argument("--compare", action="store_true",
                        help="Also run with the original traffic checkpoint for comparison")
    parser.add_argument("--output", default=None,
                        help="Optional JSON file path to save results.")
    args = parser.parse_args()
    run(max_per_class=args.max_per_class, data_root=args.data_root,
        compare=args.compare, output_path=args.output)
