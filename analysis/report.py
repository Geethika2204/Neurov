"""
analysis/report.py
==================
Reads analysis/results.csv and analysis/score_distributions.csv and prints
a formatted analysis report to stdout.  Optionally saves plots to analysis/.

Usage
-----
    python analysis/report.py            # text report only
    python analysis/report.py --plots    # text report + save PNG plots
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_CSV = REPO_ROOT / "analysis" / "results.csv"
DISTS_CSV   = REPO_ROOT / "analysis" / "score_distributions.csv"


# ── CSV helpers ────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str) -> float:
    return float(row[key])


# ── formatting helpers ─────────────────────────────────────────────────────

def bar(value: float, width: int = 30, char: str = "█") -> str:
    filled = round(value * width)
    return char * filled + "░" * (width - filled)


def pct(v: float) -> str:
    return f"{v*100:6.2f}%"


def hdr(title: str, width: int = 62) -> str:
    return f"\n{'─'*width}\n  {title}\n{'─'*width}"


# ── report sections ────────────────────────────────────────────────────────

def section_gate_comparison(rows: list[dict]) -> None:
    """Table: escalation rate, recall, precision, F1 per gate per split."""
    print(hdr("GATE COMPARISON"))

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    col_w = [34, 9, 8, 10, 8]
    header = (
        f"  {'Gate':<{col_w[0]}} {'Esc%':>{col_w[1]}} "
        f"{'Recall':>{col_w[2]}} {'Prec':>{col_w[3]}} {'F1':>{col_w[4]}}"
    )

    for split, split_rows in sorted(by_split.items()):
        print(f"\n  Split: {split.upper()}")
        print(header)
        print(f"  {'─'*sum(col_w)}")
        for r in split_rows:
            esc  = as_float(r, "escalation_rate")
            rec  = as_float(r, "recall")
            prec = as_float(r, "precision")
            f1   = as_float(r, "f1")
            print(
                f"  {r['gate']:<{col_w[0]}} {pct(esc):>{col_w[1]}} "
                f"{rec:>{col_w[2]}.3f} {prec:>{col_w[3]}.3f} {f1:>{col_w[4]}.3f}"
            )


def section_llm_reduction(rows: list[dict]) -> None:
    """How much do non-always-escalate gates cut LLM calls?"""
    print(hdr("LLM CALL REDUCTION vs ALWAYS-ESCALATE"))

    by_split: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_split[r["split"]][r["gate"]] = as_float(r, "escalation_rate")

    for split, gate_rates in sorted(by_split.items()):
        baseline = gate_rates.get("AlwaysEscalate", 1.0)
        print(f"\n  Split: {split.upper()}  (baseline LLM call rate = 100%)")
        for gate, rate in gate_rates.items():
            if gate == "AlwaysEscalate":
                continue
            reduction = (baseline - rate) / baseline
            recall_row = next(
                r for r in rows if r["split"] == split and r["gate"] == gate
            )
            recall = as_float(recall_row, "recall")
            meets_target = "✓" if reduction >= 0.30 else "✗"
            recall_ok    = "✓" if recall >= 0.60 else "✗"
            print(
                f"\n  {gate}")
            print(
                f"    Escalation rate : {pct(rate)}  "
                f"  reduction={pct(reduction)}  {meets_target} (target ≥30%)")
            print(
                f"    Crash recall    : {recall:.3f}              "
                f"  {recall_ok} (target >60%)")
            print(
                f"    LLM call bar    : {bar(rate)}"
            )


def section_confusion(rows: list[dict]) -> None:
    """Confusion matrix numbers per gate."""
    print(hdr("CONFUSION MATRIX DETAIL"))

    for r in rows:
        tp = int(r["tp"]); fp = int(r["fp"])
        fn = int(r["fn"]); tn = int(r["tn"])
        total = tp + fp + fn + tn
        print(
            f"\n  [{r['split'].upper()}] {r['gate']}\n"
            f"    True  positives (crashes caught)    : {tp:>6,}\n"
            f"    False positives (normal escalated)  : {fp:>6,}\n"
            f"    False negatives (crashes MISSED)    : {fn:>6,}  ← key metric\n"
            f"    True  negatives (normal suppressed) : {tn:>6,}\n"
            f"    Total eval windows                  : {total:>6,}"
        )


def section_score_distributions(dist_rows: list[dict]) -> None:
    """Mean ReasoningScore sub-signals for normal vs crash windows."""
    print(hdr("REASONING SCORE DISTRIBUTIONS (mean per class)"))

    fields = ["novelty", "uncertainty", "prediction_error", "memory_confidence", "spatial_correlation", "risk"]

    by_split_label: dict[tuple, list[dict]] = defaultdict(list)
    for r in dist_rows:
        key = (r["split"], int(r["label"]))
        by_split_label[key].append(r)

    for split in sorted({r["split"] for r in dist_rows}):
        print(f"\n  Split: {split.upper()}")
        header = f"  {'Sub-signal':<22} {'Normal (0)':>12} {'Crash (1)':>12} {'Δ':>8}"
        print(header)
        print(f"  {'─'*56}")
        normal_rows = by_split_label[(split, 0)]
        crash_rows  = by_split_label[(split, 1)]
        for field in fields:
            if not normal_rows or not crash_rows:
                continue
            n_mean = sum(as_float(r, field) for r in normal_rows) / len(normal_rows)
            c_mean = sum(as_float(r, field) for r in crash_rows)  / len(crash_rows)
            delta  = c_mean - n_mean
            arrow  = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "~")
            print(
                f"  {field:<22} {n_mean:>12.4f} {c_mean:>12.4f} "
                f"{delta:>+7.4f} {arrow}"
            )


def section_success_criteria(rows: list[dict]) -> None:
    """Check each success criterion from the spec."""
    print(hdr("SUCCESS CRITERIA CHECK"))

    # Gather test-split metrics
    test_rows = {r["gate"]: r for r in rows if r["split"] == "test"}
    val_rows  = {r["gate"]: r for r in rows if r["split"] == "val"}

    criteria = []

    # 1. Gate reduces LLM calls ≥30% vs always-escalate
    for gate_key, gate_rows in [("test", test_rows), ("val", val_rows)]:
        baseline = as_float(gate_rows.get("AlwaysEscalate", {"escalation_rate": "1.0"}),
                            "escalation_rate")
        for gate, row in gate_rows.items():
            if gate == "AlwaysEscalate":
                continue
            esc   = as_float(row, "escalation_rate")
            red   = (baseline - esc) / max(baseline, 1e-9)
            meets = red >= 0.30
            criteria.append((
                f"[{gate_key.upper()}] {gate}: LLM call reduction ≥30%",
                meets, f"{pct(red)} reduction"
            ))

    # 2. Crash recall >60%
    for gate_key, gate_rows in [("test", test_rows), ("val", val_rows)]:
        for gate, row in gate_rows.items():
            recall = as_float(row, "recall")
            meets  = recall > 0.60
            criteria.append((
                f"[{gate_key.upper()}] {gate}: crash recall >60%",
                meets, f"recall={recall:.3f}"
            ))

    # 3. False negatives reported (lower is better — no hard target, just report)
    for gate_key, gate_rows in [("test", test_rows)]:
        for gate, row in gate_rows.items():
            fn = int(row["fn"])
            criteria.append((
                f"[TEST] {gate}: missed crashes (FN)",
                None, f"FN={fn:,}"  # None = informational only
            ))

    print()
    for desc, meets, detail in criteria:
        if meets is None:
            icon = "ℹ"
        elif meets:
            icon = "✓"
        else:
            icon = "✗"
        print(f"  {icon}  {desc:<55}  {detail}")


# ── plots (optional) ───────────────────────────────────────────────────────

def save_plots(rows: list[dict], dist_rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("\n  matplotlib not available — skipping plots.")
        return

    out = REPO_ROOT / "analysis"
    colors = {"AlwaysEscalate": "#e74c3c",
              "ReasoningGate": "#2ecc71"}

    # ── Plot 1: Escalation rate vs Recall (test split) ────────────────────
    test_rows = [r for r in rows if r["split"] == "test"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in test_rows:
        esc  = as_float(r, "escalation_rate")
        rec  = as_float(r, "recall")
        gate = r["gate"]
        color = colors.get(gate, "#3498db")
        ax.scatter(esc, rec, s=120, color=color, zorder=5)
        ax.annotate(gate, (esc, rec), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    ax.axvline(0.70, color="gray", linestyle="--", linewidth=0.8,
               label="30% reduction line (esc=0.70)")
    ax.axhline(0.60, color="orange", linestyle="--", linewidth=0.8,
               label="recall target (0.60)")
    ax.set_xlabel("Escalation Rate (LLM call fraction)")
    ax.set_ylabel("Crash Recall")
    ax.set_title("Gate Comparison — Test Split\nEscalation Rate vs Recall")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    p = out / "plot_esc_vs_recall.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p.relative_to(REPO_ROOT)}")

    # ── Plot 2: ReasoningScore distributions (val split) ─────────────────
    val_dists = [r for r in dist_rows if r["split"] == "val"]
    fields = ["novelty", "uncertainty", "prediction_error", "memory_confidence", "spatial_correlation", "risk"]
    normal = [r for r in val_dists if int(r["label"]) == 0]
    crash  = [r for r in val_dists if int(r["label"]) == 1]

    # Sample to keep plotting fast
    import random
    random.seed(42)
    normal_s = random.sample(normal, min(2000, len(normal)))
    crash_s  = random.sample(crash,  min(2000, len(crash)))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for i, field in enumerate(fields):
        ax = axes[i]
        n_vals = [float(r[field]) for r in normal_s]
        c_vals = [float(r[field]) for r in crash_s]
        bins = np.linspace(0, 1, 40)
        ax.hist(n_vals, bins=bins, alpha=0.6, color="#3498db",
                label="Normal", density=True)
        ax.hist(c_vals, bins=bins, alpha=0.6, color="#e74c3c",
                label="Crash",  density=True)
        ax.set_title(field.replace("_", " ").title())
        ax.set_xlabel("Score [0-1]")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("ReasoningScore Distributions — Val Split\n(Normal vs Crash windows)",
                 fontsize=11)
    fig.tight_layout()
    p = out / "plot_score_distributions.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p.relative_to(REPO_ROOT)}")

    # ── Plot 3: Bar chart — FN count per gate (test split) ───────────────
    test_rows_sorted = sorted(test_rows, key=lambda r: int(r["fn"]))
    gates  = [r["gate"] for r in test_rows_sorted]
    fn_vals = [int(r["fn"]) for r in test_rows_sorted]
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = [colors.get(g, "#3498db") for g in gates]
    bars = ax.barh(gates, fn_vals, color=bar_colors)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlabel("Missed Crashes (False Negatives)")
    ax.set_title("False Negatives per Gate — Test Split\n(lower is better)")
    fig.tight_layout()
    p = out / "plot_false_negatives.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved → {p.relative_to(REPO_ROOT)}")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroVerse analysis report")
    parser.add_argument("--plots", action="store_true",
                        help="Save matplotlib PNG plots to analysis/")
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run evaluate.py first.")
        sys.exit(1)
    if not DISTS_CSV.exists():
        print(f"ERROR: {DISTS_CSV} not found. Run evaluate.py first.")
        sys.exit(1)

    rows      = load_csv(RESULTS_CSV)
    dist_rows = load_csv(DISTS_CSV)

    print("\n" + "="*62)
    print("  NEUROVERSE — GATE COMPARISON ANALYSIS REPORT")
    print("="*62)
    print(f"  Results file : {RESULTS_CSV.relative_to(REPO_ROOT)}")
    print(f"  Windows      : {len(dist_rows):,} total")
    crashes = sum(1 for r in dist_rows if int(r["label"]) == 1)
    print(f"  Crashes      : {crashes:,} ({100*crashes/len(dist_rows):.2f}%)")

    section_gate_comparison(rows)
    section_llm_reduction(rows)
    section_confusion(rows)
    section_score_distributions(dist_rows)
    section_success_criteria(rows)

    if args.plots:
        print(hdr("PLOTS"))
        save_plots(rows, dist_rows)

    print("\n" + "="*62)
    print("  END OF REPORT")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
