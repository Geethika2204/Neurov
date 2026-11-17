"""
domains/traffic/pipeline.py
============================
End-to-end FT-AED Traffic Crash Detection Pipeline.

Runs the complete NeuroVerse four-layer architecture on the FT-AED radar
dataset:

  data/processed/{split}.pt  (spike tensors + labels)
       │
       ▼
  [TrafficDomain]        — schema validation
       │
       ▼
  Layer 1: (done in preprocess.py — spike tensors already encoded)
       │
       ▼
  Layer 2: ImplicitPredictiveLayer — novelty / uncertainty / prediction_error
       │
       ▼
  Layer 3: ReasoningGate — escalate or fast-path
       │ escalate?
  ┌────┴──────┐
  │ yes       │ no → fast-path heuristic
  ▼           ▼
  Layer 4: LLMAgentPlugin — LLM → structured decision
       │
       ▼
  [TrafficPipelineResult] — per-event decision objects

Usage
-----
    python domains/traffic/pipeline.py --max-events 500
    python domains/traffic/pipeline.py --max-events 500 --no-llm
    python domains/traffic/pipeline.py --max-events 500 --split val
    python domains/traffic/pipeline.py --max-events 500 --llm-model claude-3-5-sonnet-20241022
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Allow running standalone from repo root
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nv_platform.implicit_layer import ImplicitPredictiveLayer, ReasoningScore
from nv_platform.gate import ReasoningGate, EscalationHistory
from nv_platform.gate_learner import GateLearner
from nv_platform.agent_plugin import LLMAgentPlugin, AgentDecision
from domains.traffic.domain import TrafficDomain

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path(_REPO_ROOT) / "data" / "processed" / "gate_weights.pt"
INPUT_DIM = 16


# ---------------------------------------------------------------------------
# Result / Summary dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrafficPipelineResult:
    """Per-event result from the traffic pipeline."""
    window_idx: int
    true_label: int
    escalated: bool
    gate_confidence: float
    decision_label: str
    decision_confidence: float
    reasoning: str = ""
    latency_ms: float = 0.0

    def is_correct(self) -> bool:
        if self.true_label == 1:
            return self.decision_label in ("crash", "anomaly")
        return self.decision_label == "normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "true_label": self.true_label,
            "escalated": self.escalated,
            "gate_confidence": round(self.gate_confidence, 4),
            "decision_label": self.decision_label,
            "decision_confidence": round(self.decision_confidence, 4),
            "reasoning": self.reasoning[:200] if self.reasoning else "",
            "latency_ms": round(self.latency_ms, 2),
            "correct": self.is_correct(),
        }


@dataclass
class TrafficPipelineSummary:
    """Aggregate metrics over all pipeline results."""
    total_events: int = 0
    escalated: int = 0
    correct: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    total_latency_ms: float = 0.0

    @property
    def escalation_rate(self) -> float:
        return self.escalated / self.total_events if self.total_events else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total_events if self.total_events else 0.0

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_events if self.total_events else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "escalated": self.escalated,
            "escalation_rate": round(self.escalation_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TrafficPipelineSummary("
            f"events={self.total_events}, "
            f"escalation={self.escalation_rate:.1%}, "
            f"accuracy={self.accuracy:.1%}, "
            f"recall={self.recall:.1%}, "
            f"avg_latency={self.avg_latency_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TrafficPipeline:
    """
    End-to-end FT-AED Traffic Crash Detection Pipeline.

    Loads preprocessed spike tensors from data/processed/{split}.pt, runs
    them through Layers 2–4 of the NeuroVerse architecture, and reports
    classification results.

    Parameters
    ----------
    split       : str         Data split to evaluate ("train", "val", "test").
    max_events  : int         Maximum events to process (stratified sample).
    enable_llm  : bool        If False, use heuristic for escalated events.
    llm_model   : str         LLM model name (default: llama3.2:3b via Ollama).
    """

    def __init__(
        self,
        split: str = "test",
        max_events: int = 500,
        enable_llm: bool = True,
        llm_model: str = "llama3.2:3b",
    ) -> None:
        self.split = split
        self.max_events = max_events
        self.enable_llm = enable_llm
        self.llm_model = llm_model

        self.domain = TrafficDomain()

        # Load checkpoint and restore layer + gate
        self._load_checkpoint()

        # Layer 4: LLM agent
        self.agent = LLMAgentPlugin(model=llm_model) if enable_llm else None
        self.history = EscalationHistory()

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> None:
        """Load gate weights and restore ImplicitPredictiveLayer state."""
        if not CHECKPOINT_PATH.exists():
            print(f"WARNING: No checkpoint at {CHECKPOINT_PATH}")
            print("         Using random layer + threshold=0.5")
            self.implicit_layer = ImplicitPredictiveLayer(input_dim=INPUT_DIM)
            self.gate_learner = GateLearner()
            self.gate = ReasoningGate(learner=self.gate_learner, threshold=0.5)
            return

        ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)

        input_dim = ckpt.get("input_dim", INPUT_DIM)
        weights = ckpt["weights"]
        bias = ckpt["bias"]
        feat_mean = ckpt["feature_mean"]
        feat_std = ckpt["feature_std"]
        threshold = float(ckpt["threshold"])

        # Restore ImplicitPredictiveLayer
        self.implicit_layer = ImplicitPredictiveLayer(input_dim=input_dim)
        if "layer" in ckpt:
            self.implicit_layer.load_state_dict(ckpt["layer"])
            print("ImplicitPredictiveLayer state restored from checkpoint.")

        # Setup GateLearner with saved weights
        n_feats = int(weights.shape[0])
        self.gate_learner = GateLearner()
        self.gate_learner.update_every = 10**9  # disable online update during eval
        if self.gate_learner._model.in_features != n_feats:
            import torch.nn as nn
            self.gate_learner._model = nn.Linear(n_feats, 1)
            nn.init.zeros_(self.gate_learner._model.weight)
            nn.init.constant_(self.gate_learner._model.bias, 1.0)
        with torch.no_grad():
            self.gate_learner._model.weight.copy_(weights.unsqueeze(0))
            self.gate_learner._model.bias.copy_(bias.unsqueeze(0))
        self.gate_learner.feature_mean = feat_mean
        self.gate_learner.feature_std = feat_std

        self.gate = ReasoningGate(learner=self.gate_learner, threshold=threshold)

        print(f"Checkpoint loaded: threshold={threshold:.4f}, input_dim={input_dim}, "
              f"features={n_feats}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load spike tensors and labels from data/processed/{split}.pt."""
        path = Path(_REPO_ROOT) / "data" / "processed" / f"{self.split}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                f"Run preprocess.py first to generate spike tensors:\n"
                f"  python preprocess.py"
            )
        data = torch.load(path, weights_only=True)
        return data["X"], data["y"]

    def _stratified_sample(
        self, y: torch.Tensor, n: int
    ) -> list[int]:
        """Sample n window indices with balanced crash/normal representation."""
        pos_indices = (y == 1).nonzero(as_tuple=True)[0].tolist()
        neg_indices = (y == 0).nonzero(as_tuple=True)[0].tolist()

        # Aim for 50/50, cap at available
        n_pos = min(n // 2, len(pos_indices))
        n_neg = min(n - n_pos, len(neg_indices))

        rng = random.Random(42)
        sampled = rng.sample(pos_indices, n_pos) + rng.sample(neg_indices, n_neg)
        rng.shuffle(sampled)
        return sampled

    # ------------------------------------------------------------------
    # Fast-path heuristic (no LLM)
    # ------------------------------------------------------------------

    def _fast_path_decision(
        self, score: ReasoningScore, gate_conf: float
    ) -> AgentDecision:
        """Lightweight classification for non-escalated windows."""
        if score.risk > 0.6:
            return AgentDecision(
                label="anomaly", confidence=0.7,
                reasoning=f"Fast-path. risk={score.risk:.3f}, gate_conf={gate_conf:.3f}.",
            )
        return AgentDecision(
            label="normal", confidence=0.85,
            reasoning=f"Fast-path. risk={score.risk:.3f}, gate_conf={gate_conf:.3f}.",
        )

    def _heuristic_decision(self, score: ReasoningScore) -> AgentDecision:
        """Classification for escalated windows when LLM is disabled."""
        if score.risk > 0.6 or score.prediction_error > 0.5:
            return AgentDecision(
                label="crash", confidence=0.7,
                reasoning=(
                    f"Heuristic. risk={score.risk:.3f}, "
                    f"prediction_error={score.prediction_error:.3f}, "
                    f"novelty={score.novelty:.3f}."
                ),
            )
        return AgentDecision(
            label="normal", confidence=0.7,
            reasoning=(
                f"Heuristic. risk={score.risk:.3f}, "
                f"prediction_error={score.prediction_error:.3f}."
            ),
        )

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> list[TrafficPipelineResult]:
        """Run the full pipeline on a stratified sample of events."""
        # Load data
        X, y = self._load_data()
        N_total = X.shape[1]
        n_pos = int((y == 1).sum())
        print(f"Loaded {self.split}: {N_total:,} windows, {n_pos:,} crashes")

        # Sample
        sampled = self._stratified_sample(y, self.max_events)
        N = len(sampled)
        sampled_labels = y[sampled]
        n_sampled_pos = int((sampled_labels == 1).sum())
        print(f"Sampled {N} events ({n_sampled_pos} crashes, "
              f"{N - n_sampled_pos} normal)")

        # Reset layer state for fresh evaluation
        self.implicit_layer._predictor.reset()

        # Run pipeline
        from tqdm import tqdm

        results: list[TrafficPipelineResult] = []
        pbar = tqdm(range(N), desc="Traffic pipeline", unit="event",
                    dynamic_ncols=True)

        for i in pbar:
            idx = sampled[i]
            t0 = time.perf_counter()

            # Layer 2: Implicit predictive layer (Layer 1 already done in preprocess)
            window = X[:, idx, :, :].mean(dim=1).unsqueeze(1)  # (num_steps, 1, F)
            score = self.implicit_layer.compute_state(window)

            # Layer 3: Gate
            escalate, gate_conf = self.gate.should_escalate(score, self.history)

            # Layer 4: LLM or fast-path
            if escalate and self.agent is not None:
                agent_ctx = {
                    "gate_confidence": gate_conf,
                    "domain": "traffic_crash_detection",
                    "classification_guide": (
                        "Traffic crash detection from freeway radar sensors. "
                        "High spatial_correlation and high risk indicate anomalous "
                        "cross-milemarker patterns. High prediction_error means the "
                        "SNN failed to predict the next timestep. High novelty means "
                        "the pattern is unfamiliar. crash_record=1 means a crash "
                        "was recorded in this time window. "
                        "Classify as 'crash' if multiple signals are elevated, "
                        "'normal' if all signals are low."
                    ),
                    "valid_labels": ["normal", "crash", "uncertain"],
                    "novelty": round(float(score.novelty), 3),
                    "uncertainty": round(float(score.uncertainty), 3),
                    "prediction_error": round(float(score.prediction_error), 3),
                    "memory_confidence": round(float(score.memory_confidence), 3),
                    "spatial_correlation": round(float(score.spatial_correlation), 3),
                    "risk": round(float(score.risk), 3),
                }
                llm_event = {
                    "window_idx": idx,
                    "novelty": round(float(score.novelty), 3),
                    "uncertainty": round(float(score.uncertainty), 3),
                    "prediction_error": round(float(score.prediction_error), 3),
                    "spatial_correlation": round(float(score.spatial_correlation), 3),
                    "risk": round(float(score.risk), 3),
                }
                decision = self.agent.reason(
                    llm_event, agent_ctx, self.domain.knowledge_base_path
                )
            elif escalate:
                decision = self._heuristic_decision(score)
            else:
                decision = self._fast_path_decision(score, gate_conf)

            # Log outcome for gate learner
            true_label = int(sampled_labels[i].item())
            if self.history.log:
                last_id = list(self.history.log)[-1]["id"]
                self.gate.log_outcome(last_id, true_label, self.history)

            latency_ms = (time.perf_counter() - t0) * 1000

            result = TrafficPipelineResult(
                window_idx=idx,
                true_label=true_label,
                escalated=escalate,
                gate_confidence=float(gate_conf),
                decision_label=decision.label,
                decision_confidence=decision.confidence,
                reasoning=decision.reasoning,
                latency_ms=latency_ms,
            )
            results.append(result)

            esc_count = sum(1 for r in results if r.escalated)
            pbar.set_postfix({
                "lbl": "crash" if true_label else "norm",
                "pred": result.decision_label[:4],
                "esc": f"{esc_count}/{len(results)}",
                "ms": f"{result.latency_ms:.0f}",
            })

        return results

    def summarize(self, results: list[TrafficPipelineResult]) -> TrafficPipelineSummary:
        """Compute aggregate metrics from pipeline results."""
        s = TrafficPipelineSummary()
        for r in results:
            s.total_events += 1
            s.total_latency_ms += r.latency_ms
            if r.escalated:
                s.escalated += 1
            if r.is_correct():
                s.correct += 1
            # Confusion matrix (crash/anomaly = positive, normal = negative)
            pred_pos = r.decision_label in ("crash", "anomaly")
            if r.true_label == 1 and pred_pos:
                s.tp += 1
            elif r.true_label == 0 and pred_pos:
                s.fp += 1
            elif r.true_label == 1 and not pred_pos:
                s.fn += 1
            else:
                s.tn += 1
        return s


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Run the FT-AED Traffic Crash Detection pipeline."
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Data split to evaluate (default: test)."
    )
    parser.add_argument(
        "--max-events", type=int, default=500,
        help="Maximum number of events to process (default 500)."
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM calls (use heuristic for escalated events)."
    )
    parser.add_argument(
        "--llm-model", default="llama3.2:3b",
        help="Ollama or Anthropic model name (default: llama3.2:3b)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON file path to write results."
    )
    args = parser.parse_args()

    print(f"[Traffic Pipeline] split={args.split!r}, "
          f"max_events={args.max_events}, "
          f"llm={'off' if args.no_llm else args.llm_model}")

    pipe = TrafficPipeline(
        split=args.split,
        max_events=args.max_events,
        enable_llm=not args.no_llm,
        llm_model=args.llm_model,
    )

    results = pipe.run()
    summary = pipe.summarize(results)

    print(f"\n[Traffic Pipeline] Summary:")
    summary_dict = summary.to_dict()
    for k, v in summary_dict.items():
        print(f"  {k}: {v}")

    if args.output:
        payload = {
            "summary": summary_dict,
            "results": [r.to_dict() for r in results],
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Traffic Pipeline] Results written to {args.output!r}")


if __name__ == "__main__":
    main()
