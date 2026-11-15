"""
analysis/cwru/report.py
=======================
Reads cwru_results.json from the repo root and prints a formatted analysis
report to stdout.  Optionally saves plots to analysis/cwru/.

The JSON has two top-level keys:
  "summary"  — aggregate stats produced by domains/cwru/pipeline.py
  "results"  — one record per evaluation window, fields:
                 event_id, fault_class, true_label, escalated,
                 gate_confidence, decision_label, decision_conf,
                 reasoning, sources, latency_ms, correct

Fault classes: normal (0), inner_race (1), ball (2), outer_race (3)

Usage
-----
    python analysis/cwru/report.py            # text report only
    python analysis/cwru/report.py --plots    # text report + save PNG plots
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent.parent
RESULTS_JSON = REPO_ROOT / "cwru_results.json"
OUT_DIR      = REPO_ROOT / "analysis" / "cwru"

CLASSES      = ["normal", "inner_race", "ball", "outer_race"]
FAULT_CLASSES = ["inner_race", "ball", "outer_race"]   # non-normal

# label used by pipeline for the "no fault" class
NORMAL_CLASS = "normal"


# ── helpers ────────────────────────────────────────────────────────────────

def load_results() -> tuple[dict, list[dict]]:
    with open(RESULTS_JSON) as f:
        data = json.load(f)
    return data["summary"], data["results"]


def bar(value: float, width: int = 30, char: str = "█") -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return char * filled + "░" * (width - filled)


def pct(v: float) -> str:
    return f"{v * 100:6.2f}%"


def hdr(title: str, width: int = 62) -> str:
    return f"\n{'─' * width}\n  {title}\n{'─' * width}"


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


# ── binary confusion helpers ───────────────────────────────────────────────
# For the gate-comparison section we treat the problem as binary:
# "fault" (any non-normal class) vs "normal".
# "escalated" means the gate sent the window to the agent.

def binary_confusion(results: list[dict]) -> dict[str, int]:
    """TP/FP/FN/TN treating any non-normal class as the positive."""
    tp = sum(1 for r in results if r["fault_class"] != NORMAL_CLASS and     r["escalated"])
    fn = sum(1 for r in results if r["fault_class"] != NORMAL_CLASS and not r["escalated"])
    fp = sum(1 for r in results if r["fault_class"] == NORMAL_CLASS and     r["escalated"])
    tn = sum(1 for r in results if r["fault_class"] == NORMAL_CLASS and not r["escalated"])
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


# ── report sections ────────────────────────────────────────────────────────

def section_gate_comparison(results: list[dict], summary: dict) -> None:
    """
    Compare the ReasoningGate (from cwru_results.json) against an
    AlwaysEscalate baseline reconstructed analytically.
    """
    print(hdr("GATE COMPARISON"))

    total = len(results)
    cm    = binary_confusion(results)
    tp, fn, fp, tn = cm["tp"], cm["fn"], cm["fp"], cm["tn"]
    n_faults  = tp + fn
    n_normal  = fp + tn
    esc_total = tp + fp

    # Gate metrics
    esc_rate  = safe_div(esc_total, total)
    recall    = safe_div(tp, n_faults)
    precision = safe_div(tp, esc_total)
    f1        = safe_div(2 * precision * recall, precision + recall)

    # AlwaysEscalate baseline (escalate every window → 100 % call rate)
    ae_recall    = 1.0
    ae_precision = safe_div(n_faults, total)
    ae_f1        = safe_div(2 * ae_precision * ae_recall, ae_precision + ae_recall)

    col_w = [22, 9, 8, 10, 8]
    header = (
        f"  {'Gate':<{col_w[0]}} {'Esc%':>{col_w[1]}} "
        f"{'Recall':>{col_w[2]}} {'Prec':>{col_w[3]}} {'F1':>{col_w[4]}}"
    )
    print(f"\n  Binary view: fault (inner_race | ball | outer_race) vs normal")
    print(f"  Total windows: {total:,}   fault: {n_faults:,}   normal: {n_normal:,}\n")
    print(header)
    print(f"  {'─' * sum(col_w)}")

    for label, er, rec, prec, f1v in [
        ("AlwaysEscalate",  1.0,       ae_recall,    ae_precision, ae_f1),
        ("ReasoningGate",   esc_rate,  recall,       precision,    f1),
    ]:
        print(
            f"  {label:<{col_w[0]}} {pct(er):>{col_w[1]}} "
            f"{rec:>{col_w[2]}.3f} {prec:>{col_w[3]}.3f} {f1v:>{col_w[4]}.3f}"
        )

    # Per-class accuracy from the pipeline summary
    print(f"\n  Per-class accuracy (classifier):")
    for cls, acc in summary.get("per_class_accuracy", {}).items():
        print(f"    {cls:<14} {pct(acc)}  {bar(acc, 20)}")

    print(f"\n  Overall classifier accuracy : {pct(summary.get('accuracy', 0))}")
    print(f"  Average latency             : {summary.get('avg_latency_ms', 0):.1f} ms")


def section_llm_reduction(results: list[dict]) -> None:
    """How much does the ReasoningGate cut LLM calls vs always-escalate?"""
    print(hdr("LLM CALL REDUCTION vs ALWAYS-ESCALATE"))

    total     = len(results)
    cm        = binary_confusion(results)
    esc_total = cm["tp"] + cm["fp"]
    esc_rate  = safe_div(esc_total, total)
    reduction = 1.0 - esc_rate                 # vs AlwaysEscalate baseline of 1.0
    recall    = safe_div(cm["tp"], cm["tp"] + cm["fn"])

    meets_reduction = reduction >= 0.30
    meets_recall    = recall > 0.60

    print(f"\n  ReasoningGate")
    print(f"    Escalation rate  : {pct(esc_rate)}    reduction={pct(reduction)}  "
          f"{'✓' if meets_reduction else '✗'} (target ≥30%)")
    print(f"    Fault recall     : {recall:.3f}            "
          f"{'✓' if meets_recall else '✗'} (target >60%)")
    print(f"    LLM call bar     : {bar(esc_rate)}")

    # Escalation breakdown by class
    print(f"\n  Escalation breakdown by fault class:")
    for cls in CLASSES:
        sub = [r for r in results if r["fault_class"] == cls]
        esc = sum(1 for r in sub if r["escalated"])
        print(f"    {cls:<14} {esc:>3}/{len(sub)}  ({pct(safe_div(esc, len(sub)))})")


def section_confusion(results: list[dict]) -> None:
    """Full confusion matrix: predicted × true for all four classes."""
    print(hdr("CONFUSION MATRIX (4-class)"))

    # Build confusion matrix
    # rows = true class, cols = predicted class
    conf_mat: dict[str, dict[str, int]] = {
        c: {c2: 0 for c2 in CLASSES + ["uncertain"]} for c in CLASSES
    }
    for r in results:
        true_cls  = r["fault_class"]
        pred_cls  = r["decision_label"]
        conf_mat[true_cls][pred_cls] += 1

    all_pred_labels = CLASSES + ["uncertain"]
    col_w = 12
    header = f"  {'True \\ Pred':<14}" + "".join(f"{c:>{col_w}}" for c in all_pred_labels)
    print(f"\n{header}")
    print(f"  {'─' * (14 + col_w * len(all_pred_labels))}")
    for true_cls in CLASSES:
        row_str = f"  {true_cls:<14}"
        for pred_cls in all_pred_labels:
            row_str += f"{conf_mat[true_cls].get(pred_cls, 0):>{col_w}}"
        print(row_str)

    # Binary confusion summary (fault detection)
    cm = binary_confusion(results)
    total = len(results)
    print(f"\n  Binary confusion (fault = any non-normal):")
    print(f"    True  positives (faults escalated)   : {cm['tp']:>6,}")
    print(f"    False positives (normal escalated)   : {cm['fp']:>6,}")
    print(f"    False negatives (faults MISSED)      : {cm['fn']:>6,}  ← key metric")
    print(f"    True  negatives (normal suppressed)  : {cm['tn']:>6,}")
    print(f"    Total evaluation windows             : {total:>6,}")


def section_score_distributions(results: list[dict]) -> None:
    """Gate confidence distribution per fault class."""
    print(hdr("GATE CONFIDENCE DISTRIBUTIONS (mean per class)"))

    print(f"\n  {'Class':<14} {'Mean conf':>10} {'Min':>8} {'Max':>8} {'Escalated%':>12}")
    print(f"  {'─' * 54}")
    for cls in CLASSES:
        sub   = [r for r in results if r["fault_class"] == cls]
        confs = [r["gate_confidence"] for r in sub]
        esc   = sum(1 for r in sub if r["escalated"])
        mean_c = _mean(confs)
        print(
            f"  {cls:<14} {mean_c:>10.4f} {min(confs):>8.4f} {max(confs):>8.4f} "
            f"{pct(safe_div(esc, len(sub))):>12}"
        )

    # Decision confidence
    print(f"\n  Decision confidence (classifier output):")
    print(f"  {'Class':<14} {'Mean dec_conf':>14} {'Correct%':>10}")
    print(f"  {'─' * 40}")
    for cls in CLASSES:
        sub = [r for r in results if r["fault_class"] == cls]
        dc  = _mean([r["decision_conf"] for r in sub])
        cor = sum(1 for r in sub if r["correct"])
        print(f"  {cls:<14} {dc:>14.4f} {pct(safe_div(cor, len(sub))):>10}")

    # Latency
    lats = sorted(r["latency_ms"] for r in results)
    n    = len(lats)
    print(f"\n  Latency (ms): mean={_mean(lats):.1f}  "
          f"median={lats[n // 2]:.1f}  "
          f"min={lats[0]:.1f}  max={lats[-1]:.1f}")


def section_success_criteria(results: list[dict], summary: dict) -> None:
    """Check each success criterion from the NeuroVerse spec."""
    print(hdr("SUCCESS CRITERIA CHECK"))

    total  = len(results)
    cm     = binary_confusion(results)
    esc_rate  = safe_div(cm["tp"] + cm["fp"], total)
    reduction = 1.0 - esc_rate
    recall    = safe_div(cm["tp"], cm["tp"] + cm["fn"])
    accuracy  = summary.get("accuracy", 0.0)

    criteria: list[tuple[str, bool | None, str]] = [
        (
            "Gate reduces LLM calls ≥30% vs always-escalate",
            reduction >= 0.30,
            f"{pct(reduction)} reduction  (esc rate = {pct(esc_rate)})"
        ),
        (
            "Fault recall >60% (binary escalation-based)",
            recall > 0.60,
            f"recall = {recall:.4f}"
        ),
        (
            "Overall classifier accuracy",
            None,                              # informational
            f"accuracy = {pct(accuracy)}"
        ),
        (
            "Normal class accuracy (false-alarm rate)",
            None,
            f"normal acc = {pct(summary.get('per_class_accuracy', {}).get('normal', 0))}"
        ),
        (
            "Inner-race fault accuracy",
            summary.get("per_class_accuracy", {}).get("inner_race", 0) > 0.60,
            f"inner_race acc = {pct(summary.get('per_class_accuracy', {}).get('inner_race', 0))}"
        ),
        (
            "Ball fault accuracy",
            summary.get("per_class_accuracy", {}).get("ball", 0) > 0.60,
            f"ball acc = {pct(summary.get('per_class_accuracy', {}).get('ball', 0))}"
        ),
        (
            "Outer-race fault accuracy",
            summary.get("per_class_accuracy", {}).get("outer_race", 0) > 0.60,
            f"outer_race acc = {pct(summary.get('per_class_accuracy', {}).get('outer_race', 0))}"
        ),
        (
            "Missed faults (FN) — informational",
            None,
            f"FN = {cm['fn']:,} of {cm['tp'] + cm['fn']:,} fault windows"
        ),
    ]

    print()
    for desc, meets, detail in criteria:
        icon = "ℹ" if meets is None else ("✓" if meets else "✗")
        print(f"  {icon}  {desc:<52}  {detail}")


# ── plots (optional) ───────────────────────────────────────────────────────

def save_plots(results: list[dict], summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n  matplotlib not available — skipping plots.")
        return

    out = OUT_DIR
    colors = {
        "normal":     "#3498db",
        "inner_race": "#e74c3c",
        "ball":       "#2ecc71",
        "outer_race": "#f39c12",
    }

    # ── Plot 1: Per-class accuracy bar chart ─────────────────────────────
    per_class = summary.get("per_class_accuracy", {})
    cls_names = list(per_class.keys())
    cls_accs  = [per_class[c] for c in cls_names]
    cls_cols  = [colors.get(c, "#95a5a6") for c in cls_names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cls_names, [a * 100 for a in cls_accs], color=cls_cols)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.axhline(60, color="orange", linestyle="--", linewidth=0.9,
               label="60% target")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CWRU — Per-Class Classification Accuracy")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "plot_per_class_accuracy.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p.relative_to(REPO_ROOT)}")

    # ── Plot 2: Gate confidence distribution per class ───────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    bins = np.linspace(0, 1, 30)
    for i, cls in enumerate(CLASSES):
        ax  = axes[i]
        sub = [r for r in results if r["fault_class"] == cls]
        esc_confs   = [r["gate_confidence"] for r in sub if     r["escalated"]]
        noesc_confs = [r["gate_confidence"] for r in sub if not r["escalated"]]
        if esc_confs:
            ax.hist(esc_confs,   bins=bins, alpha=0.7,
                    color="#e74c3c", label="Escalated",     density=True)
        if noesc_confs:
            ax.hist(noesc_confs, bins=bins, alpha=0.7,
                    color="#3498db", label="Not escalated", density=True)
        ax.set_title(cls.replace("_", " ").title())
        ax.set_xlabel("Gate confidence")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("CWRU — Gate Confidence by Class\n(escalated vs not escalated)",
                 fontsize=11)
    fig.tight_layout()
    p = out / "plot_gate_confidence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p.relative_to(REPO_ROOT)}")

    # ── Plot 3: False negatives (missed faults) per class ────────────────
    fn_counts = []
    for cls in FAULT_CLASSES:
        sub = [r for r in results if r["fault_class"] == cls]
        fn_counts.append(sum(1 for r in sub if not r["escalated"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_colors = [colors.get(c, "#95a5a6") for c in FAULT_CLASSES]
    rects = ax.barh(FAULT_CLASSES, fn_counts, color=bar_colors)
    ax.bar_label(rects, padding=3, fontsize=9)
    ax.set_xlabel("Missed Faults (False Negatives)")
    ax.set_title("CWRU — Missed Fault Windows per Class\n(lower is better)")
    fig.tight_layout()
    p = out / "plot_false_negatives.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p.relative_to(REPO_ROOT)}")

    # ── Plot 4: Decision label distribution ──────────────────────────────
    dl_counts = Counter(r["decision_label"] for r in results)
    labels = sorted(dl_counts.keys())
    counts = [dl_counts[l] for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts,
           color=[colors.get(l, "#95a5a6") for l in labels])
    ax.set_ylabel("Count")
    ax.set_title("CWRU — Decision Label Distribution")
    for idx, cnt in enumerate(counts):
        ax.text(idx, cnt + 1, str(cnt), ha="center", fontsize=9)
    fig.tight_layout()
    p = out / "plot_decision_labels.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p.relative_to(REPO_ROOT)}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuroVerse CWRU domain analysis report"
    )
    parser.add_argument("--plots", action="store_true",
                        help="Save matplotlib PNG plots to analysis/cwru/")
    args = parser.parse_args()

    if not RESULTS_JSON.exists():
        print(f"ERROR: {RESULTS_JSON} not found. Run domains/cwru/pipeline.py first.")
        sys.exit(1)

    summary, results = load_results()

    total   = len(results)
    n_fault = sum(1 for r in results if r["fault_class"] != NORMAL_CLASS)
    n_norm  = total - n_fault

    print("\n" + "=" * 62)
    print("  NEUROVERSE — CWRU DOMAIN ANALYSIS REPORT")
    print("=" * 62)
    print(f"  Results file  : {RESULTS_JSON.relative_to(REPO_ROOT)}")
    print(f"  Total windows : {total:,}")
    print(f"  Fault windows : {n_fault:,}  ({100 * n_fault / total:.1f}%)")
    print(f"  Normal windows: {n_norm:,}  ({100 * n_norm / total:.1f}%)")
    print(f"  Fault classes : {', '.join(FAULT_CLASSES)}")

    section_gate_comparison(results, summary)
    section_llm_reduction(results)
    section_confusion(results)
    section_score_distributions(results)
    section_success_criteria(results, summary)

    if args.plots:
        print(hdr("PLOTS"))
        save_plots(results, summary)

    print("\n" + "=" * 62)
    print("  END OF REPORT")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
