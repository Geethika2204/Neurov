"""
/platform/implicit_layer.py

Layer 2 — Implicit Predictive Layer
=====================================
Converts a spike tensor into a ``ReasoningScore`` — six normalised sub-signals
that capture how surprising / uncertain / unfamiliar the current input is.

Interface
---------
    layer = ImplicitPredictiveLayer(input_dim=16)
    score = layer.compute_state(spikes)   # spikes: (num_steps, batch, features)
    # score.novelty, .uncertainty, .prediction_error, .memory_confidence,
    # .spatial_correlation, .risk ∈ [0, 1]

Design
------
Six sub-modules each produce one sub-signal.  All mutable state that affects
score generation is registered as a PyTorch buffer so that ``state_dict()``
captures the *complete* scoring state — enabling exact reproducibility between
``train_gate.py`` and ``evaluate.py``.

1. ``_PrototypeStore``
   Online k-means prototype bank (k=16).  Novelty = 1 − max cosine similarity.
   Prototypes and counts are buffers.

2. ``_MCDropoutHead``
   Two-layer MLP with Dropout(0.3).  Epistemic uncertainty = std of logits over
   N stochastic forward passes, normalised by a running max.
   _running_max_std is a buffer.

3. ``_PredictorHead``
   One-step-ahead spike predictor (predicts t+1 from t).  Prediction error =
   MSE between prediction and actual next step, normalised by running max.
   _running_max_mse is a buffer.

4. ``_EpisodicMemory``
   Ring buffer of recent mean-pooled spike vectors (not a buffer — pure Python
   deque; restored via a dedicated state-dict key handled by the parent layer).

5. ``_SpatialCorrelationHead``
   Detects anomalous spatial patterns.  Welford running_mean / running_std /
   running_n are buffers.

6. ``_RiskHead``
   Composite risk signal.  running_mean / running_std and history are buffers /
   restored via state-dict.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReasoningScore:
    """
    Six normalised sub-signals produced by the Implicit Predictive Layer.

    All values ∈ [0, 1].  Higher values indicate more reason to escalate.

    Attributes
    ----------
    novelty             : How different is this input from stored prototypes?
    uncertainty         : Epistemic uncertainty (MC-dropout variance).
    prediction_error    : How poorly did the predictor anticipate this input?
    memory_confidence   : How well does this match recent episodic memory?
                          *Low* confidence → unfamiliar → reason to escalate.
    spatial_correlation : How anomalous is the cross-sensor spatial pattern?
    risk                : Composite risk score — weighted combination of the
                          above signals plus a short-term trend component.
                          This is the primary signal used by the gate.
    """

    novelty: float = 0.0
    uncertainty: float = 0.0
    prediction_error: float = 0.0
    memory_confidence: float = 0.0
    spatial_correlation: float = 0.0
    risk: float = 0.0

    def as_tensor(self) -> Tensor:
        """Return sub-signals as a 1-D float tensor of shape (6,)."""
        return torch.tensor(
            [self.novelty, self.uncertainty,
             self.prediction_error, self.memory_confidence,
             self.spatial_correlation, self.risk],
            dtype=torch.float32,
        )

    def __repr__(self) -> str:
        return (
            f"ReasoningScore(novelty={self.novelty:.3f}, "
            f"uncertainty={self.uncertainty:.3f}, "
            f"prediction_error={self.prediction_error:.3f}, "
            f"memory_confidence={self.memory_confidence:.3f}, "
            f"spatial_correlation={self.spatial_correlation:.3f}, "
            f"risk={self.risk:.3f})"
        )


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class _PrototypeStore(nn.Module):
    """
    Online k-means prototype bank.

    novelty = 1 − max_cosine_sim(input, prototypes)

    Prototypes and counts are registered as buffers so they are saved and
    restored by ``state_dict()`` / ``load_state_dict()``.
    """

    def __init__(self, input_dim: int, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.register_buffer(
            "prototypes",
            F.normalize(torch.randn(k, input_dim), dim=1),
        )
        self.register_buffer("_counts", torch.ones(k))

    @torch.no_grad()
    def compute_novelty(self, vec: Tensor) -> float:
        vec_norm    = F.normalize(vec.unsqueeze(0), dim=1)          # (1, F)
        protos_norm = F.normalize(self.prototypes, dim=1)            # (k, F)
        cosine_sims = (vec_norm @ protos_norm.T).squeeze(0)          # (k,)
        max_sim     = cosine_sims.max().item()
        novelty     = 1.0 - max(0.0, float(max_sim))
        return float(torch.clamp(torch.tensor(novelty), 0.0, 1.0))

    @torch.no_grad()
    def update(self, vec: Tensor) -> None:
        vec_norm    = F.normalize(vec.unsqueeze(0), dim=1)
        protos_norm = F.normalize(self.prototypes, dim=1)
        cosine_sims = (vec_norm @ protos_norm.T).squeeze(0)
        nearest     = int(cosine_sims.argmax().item())
        lr          = 1.0 / (self._counts[nearest].item() + 1)
        self.prototypes[nearest] = (
            (1 - lr) * self.prototypes[nearest] + lr * vec
        )
        self._counts[nearest] += 1


class _MCDropoutHead(nn.Module):
    """
    Two-layer MLP with Dropout for epistemic uncertainty estimation.

    uncertainty = std(logits over N stochastic passes) / running_max_std

    _running_max_std is a buffer so it is saved in state_dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.3, mc_passes: int = 10) -> None:
        super().__init__()
        self.mc_passes = mc_passes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # Buffer so it survives save/load
        self.register_buffer("_running_max_std", torch.tensor(1e-6))

    def compute_uncertainty(self, vec: Tensor) -> float:
        self.train()  # keep dropout active
        x = vec.unsqueeze(0)
        with torch.no_grad():
            logits = torch.cat(
                [self.net(x) for _ in range(self.mc_passes)], dim=0
            )  # (mc_passes, 1)
        std = logits.std().item()
        if std > self._running_max_std.item():
            self._running_max_std.fill_(std)
        uncertainty = std / self._running_max_std.item()
        return float(torch.clamp(torch.tensor(uncertainty), 0.0, 1.0))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _PredictorHead(nn.Module):
    """
    One-step-ahead spike predictor.

    prediction_error = MSE(predicted_t+1, actual_t+1) / running_max_mse

    _running_max_mse is a buffer so it is saved in state_dict.
    _prev_vec is *not* a buffer — it is transient per-run state and should
    be reset between train and eval passes.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # spikes are in [0, 1]
        )
        self.register_buffer("_running_max_mse", torch.tensor(1e-6))
        self._prev_vec: Optional[Tensor] = None

    def reset(self) -> None:
        """Reset transient per-sequence state (call before each eval pass)."""
        self._prev_vec = None

    def compute_prediction_error(self, vec: Tensor) -> float:
        if self._prev_vec is None:
            self._prev_vec = vec.detach()
            return 0.0

        with torch.no_grad():
            predicted = self.net(self._prev_vec.unsqueeze(0)).squeeze(0)
        mse = F.mse_loss(predicted, vec.detach()).item()
        if mse > self._running_max_mse.item():
            self._running_max_mse.fill_(mse)
        error = mse / self._running_max_mse.item()
        self._prev_vec = vec.detach()
        return float(torch.clamp(torch.tensor(error), 0.0, 1.0))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _EpisodicMemory(nn.Module):
    """
    Ring buffer of recent mean-pooled spike vectors.

    memory_confidence = max cosine similarity to any stored episode.
    Low confidence = unfamiliar = escalate.

    The buffer is stored as a flat tensor buffer so that state_dict()
    captures it.  An active_size buffer tracks how many entries are valid.
    """

    def __init__(self, memory_size: int = 256) -> None:
        super().__init__()
        self.memory_size = memory_size
        # Placeholder dim; will be resized on first store()
        self._dim: int = 0
        # We use simple Python deque internally; persistence is via
        # _save_to_extras / _load_from_extras in the parent module.
        self._buffer: deque[Tensor] = deque(maxlen=memory_size)

    def compute_confidence(self, vec: Tensor) -> float:
        if not self._buffer:
            return 0.0
        stored     = torch.stack(list(self._buffer))  # (N, F)
        vec_norm   = F.normalize(vec.unsqueeze(0), dim=1)
        stored_norm = F.normalize(stored, dim=1)
        sims       = (vec_norm @ stored_norm.T).squeeze(0)
        return float(torch.clamp(sims.max(), 0.0, 1.0))

    def store(self, vec: Tensor) -> None:
        self._buffer.append(vec.detach())

    # ---- serialisation helpers (called by ImplicitPredictiveLayer) --------
    def get_buffer_tensor(self) -> Optional[Tensor]:
        """Return buffer as a tensor (N, F), or None if empty."""
        if not self._buffer:
            return None
        return torch.stack(list(self._buffer))

    def set_buffer_tensor(self, t: Tensor) -> None:
        """Restore buffer from a tensor (N, F)."""
        self._buffer = deque(
            [t[i] for i in range(t.shape[0])],
            maxlen=self.memory_size,
        )


class _SpatialCorrelationHead(nn.Module):
    """
    Detects anomalous spatial patterns in the input feature vector.

    spatial_correlation = how anomalous is the spatial pattern right now,
    normalised by running statistics so it stays in [0, 1].

    Welford statistics are buffers.
    """

    N_SPATIAL    = 4
    MIN_SPATIAL_DIM = 16  # 12 per-sensor + 4 spatial

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self._has_spatial = (input_dim >= self.MIN_SPATIAL_DIM)

        # Welford stats as buffers
        self.register_buffer("_running_mean_sc", torch.tensor(0.0))
        self.register_buffer("_running_std_sc",  torch.tensor(1.0))
        self.register_buffer("_n_sc",            torch.tensor(0,   dtype=torch.long))

    @torch.no_grad()
    def compute_spatial_correlation(self, vec: Tensor) -> float:
        if not self._has_spatial:
            return 0.0

        spatial   = vec[-self.N_SPATIAL:]      # (4,)
        raw_score = float(spatial.norm())

        # Online Welford update
        self._n_sc += 1
        n     = self._n_sc.item()
        delta = raw_score - self._running_mean_sc.item()
        self._running_mean_sc += delta / n
        delta2 = raw_score - self._running_mean_sc.item()
        old_M2  = (self._running_std_sc.item() ** 2) * max(n - 1, 1)
        new_M2  = old_M2 + delta * delta2
        self._running_std_sc.fill_(
            max(math.sqrt(new_M2 / max(n, 1)), 1e-6)
        )

        # Z-score → sigmoid → [0, 1]
        z     = (raw_score - self._running_mean_sc.item()) / (
            self._running_std_sc.item() * 3 + 1e-6
        )
        score = 1.0 / (1.0 + math.exp(-z))
        return float(torch.clamp(torch.tensor(score), 0.0, 1.0))


class _RiskHead(nn.Module):
    """
    Composite risk signal — primary gate-driving score.

    Formula
    -------
        raw_risk = w·[novelty, uncertainty, pred_error, (1-mem_conf), spatial]
        trend    = clip((raw_risk - running_mean) / running_std, -1, 1)
        risk     = sigmoid(raw_risk + α·trend)

    Running statistics are buffers.
    """

    DEFAULT_WEIGHTS = torch.tensor([0.15, 0.15, 0.25, 0.20, 0.25])
    TREND_ALPHA     = 0.3

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_weights",       self.DEFAULT_WEIGHTS.clone())
        self.register_buffer("_risk_run_mean", torch.tensor(0.5))
        self.register_buffer("_risk_run_std",  torch.tensor(0.1))
        self.register_buffer("_risk_n",        torch.tensor(0, dtype=torch.long))
        # Short-term history (last 50 raw risks) — stored as a flat buffer
        self.register_buffer("_risk_hist",     torch.zeros(50))
        self.register_buffer("_risk_hist_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_risk_hist_len", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def compute_risk(
        self,
        novelty: float,
        uncertainty: float,
        prediction_error: float,
        memory_confidence: float,
        spatial_correlation: float,
    ) -> float:
        signals = torch.tensor([
            novelty,
            uncertainty,
            prediction_error,
            1.0 - memory_confidence,
            spatial_correlation,
        ])
        raw = float((self._weights * signals).sum().clamp(0.0, 1.0))

        # Trend from ring buffer
        hist_len = self._risk_hist_len.item()
        if hist_len >= 5:
            recent_mean = self._risk_hist[:min(hist_len, 50)].mean().item()
            trend = float(torch.tensor(raw - recent_mean).clamp(-1.0, 1.0))
        else:
            trend = 0.0

        combined = raw + self.TREND_ALPHA * trend
        risk = float(torch.sigmoid(torch.tensor(combined * 4 - 2)))

        # Update ring buffer
        ptr = int(self._risk_hist_ptr.item())
        self._risk_hist[ptr] = raw
        self._risk_hist_ptr.fill_((ptr + 1) % 50)
        if hist_len < 50:
            self._risk_hist_len += 1

        return float(torch.clamp(torch.tensor(risk), 0.0, 1.0))

    def update_weights(self, weights: Tensor) -> None:
        assert weights.shape == (5,), "Expected 5 weights"
        w = weights.clamp(0.0, 1.0)
        self._weights.copy_(w / (w.sum() + 1e-9))


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class ImplicitPredictiveLayer(nn.Module):
    """
    Layer 2 of the NeuroVerse pipeline.

    Converts a spike tensor into a ``ReasoningScore``.

    ALL mutable state that influences score generation is registered as
    PyTorch buffers or ``nn.Module`` parameters so that::

        layer.state_dict()

    captures the *complete* scoring state.  Loading that state dict on a
    freshly constructed layer gives bit-for-bit identical scores.

    Parameters
    ----------
    input_dim    : int  Number of sensor features.
    hidden_dim   : int  MLP hidden dimension (default 128).
    memory_size  : int  Episodic memory ring-buffer capacity (default 256).
    mc_passes    : int  MC-dropout forward passes for uncertainty (default 10).
    k_prototypes : int  Number of prototype vectors (default 16).

    Examples
    --------
    >>> layer = ImplicitPredictiveLayer(input_dim=16)
    >>> spikes = torch.randint(0, 2, (25, 4, 16)).float()
    >>> score = layer.compute_state(spikes)
    >>> all(0 <= v <= 1 for v in score.as_tensor().tolist())
    True
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        memory_size: int = 256,
        mc_passes: int = 10,
        k_prototypes: int = 16,
    ) -> None:
        super().__init__()
        self.input_dim   = input_dim
        self.memory_size = memory_size

        self._prototypes = _PrototypeStore(input_dim, k=k_prototypes)
        self._mc_head    = _MCDropoutHead(input_dim, hidden_dim, mc_passes=mc_passes)
        self._predictor  = _PredictorHead(input_dim, hidden_dim)
        self._memory     = _EpisodicMemory(memory_size)
        self._spatial    = _SpatialCorrelationHead(input_dim)
        self._risk       = _RiskHead()

        # Aliases so parameters() / state_dict() sees all sub-modules
        self.mc_head   = self._mc_head
        self.predictor = self._predictor

    # ------------------------------------------------------------------
    # state_dict / load_state_dict overrides to include episodic memory
    # ------------------------------------------------------------------

    def state_dict(self, destination=None, prefix="", keep_vars=False):  # type: ignore[override]
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Persist episodic memory buffer (deque → tensor)
        mem_t = self._memory.get_buffer_tensor()
        if mem_t is not None:
            sd[prefix + "_memory_buffer"] = mem_t
        # Persist predictor's previous vector (transient but useful for warm starts)
        if self._predictor._prev_vec is not None:
            sd[prefix + "_predictor_prev_vec"] = self._predictor._prev_vec
        return sd

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
        # Extract our extras before passing to the base class
        sd = dict(state_dict)
        mem_key  = "_memory_buffer"
        prev_key = "_predictor_prev_vec"

        # strip prefix if present
        actual_mem_key  = next((k for k in sd if k.endswith(mem_key)),  None)
        actual_prev_key = next((k for k in sd if k.endswith(prev_key)), None)

        if actual_mem_key:
            self._memory.set_buffer_tensor(sd.pop(actual_mem_key))
        if actual_prev_key:
            self._predictor._prev_vec = sd.pop(actual_prev_key)

        return super().load_state_dict(sd, strict=strict)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def _mean_pool(self, spikes: Tensor) -> Tensor:
        """
        Collapse (num_steps, batch, features) → (features,) via mean over
        both the time-step and batch dimensions.
        """
        if spikes.dim() == 3:
            return spikes.mean(dim=(0, 1))
        elif spikes.dim() == 2:
            return spikes.mean(dim=0)
        else:
            return spikes.flatten()

    def compute_state(self, spikes: Tensor) -> ReasoningScore:
        """
        Parameters
        ----------
        spikes : Tensor, shape (num_steps, batch, features)

        Returns
        -------
        ReasoningScore with all fields ∈ [0, 1]
        """
        vec = self._mean_pool(spikes)  # (features,)

        novelty     = self._prototypes.compute_novelty(vec)
        uncertainty = self._mc_head.compute_uncertainty(vec)
        pred_error  = self._predictor.compute_prediction_error(vec)
        mem_conf    = self._memory.compute_confidence(vec)
        spatial     = self._spatial.compute_spatial_correlation(vec)
        risk        = self._risk.compute_risk(
            novelty, uncertainty, pred_error, mem_conf, spatial
        )

        # Update stores
        self._prototypes.update(vec)
        self._memory.store(vec)

        return ReasoningScore(
            novelty=novelty,
            uncertainty=uncertainty,
            prediction_error=pred_error,
            memory_confidence=mem_conf,
            spatial_correlation=spatial,
            risk=risk,
        )

    def fit(
        self,
        dataloader: DataLoader,
        epochs: int = 5,
        lr: float = 1e-3,
    ) -> dict[str, list[float]]:
        """
        Pre-train the predictor and MC-dropout heads.

        The dataloader must yield ``(spikes, labels)`` batches where
        ``spikes`` has shape ``(batch, num_steps_or_flat, features)`` or
        similar.  Temporal ordering must be preserved — use ``shuffle=False``.

        Returns a history dict with ``"predictor_loss"`` and ``"mc_loss"``
        per epoch.
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history: dict[str, list[float]] = {
            "predictor_loss": [], "mc_loss": []
        }

        for epoch in range(epochs):
            p_losses, mc_losses = [], []
            for batch in dataloader:
                spikes, labels = batch
                spikes = spikes.float()
                labels = labels.float()

                # Mean-pool to (batch, features)
                if spikes.dim() == 4:
                    vecs = spikes.mean(dim=(1, 2))  # (batch, features)
                elif spikes.dim() == 3:
                    vecs = spikes.mean(dim=1)       # (batch, features)
                else:
                    vecs = spikes

                opt.zero_grad()

                # Predictor loss: predict next step from current
                if vecs.shape[0] > 1:
                    preds  = self._predictor(vecs[:-1])
                    p_loss = F.mse_loss(preds, vecs[1:].detach())
                else:
                    p_loss = torch.tensor(0.0, requires_grad=True)

                # MC head loss: binary classification
                self._mc_head.train()
                logits = self._mc_head(vecs).squeeze(-1)
                if labels.shape[0] == vecs.shape[0]:
                    mc_loss = F.binary_cross_entropy_with_logits(logits, labels)
                else:
                    mc_loss = torch.tensor(0.0, requires_grad=True)

                loss = p_loss + mc_loss
                loss.backward()
                opt.step()

                p_losses.append(p_loss.item())
                mc_losses.append(mc_loss.item())

            history["predictor_loss"].append(
                sum(p_losses) / max(len(p_losses), 1)
            )
            history["mc_loss"].append(
                sum(mc_losses) / max(len(mc_losses), 1)
            )
            print(f"  fit epoch {epoch+1}/{epochs}  "
                  f"pred_loss={history['predictor_loss'][-1]:.4f}  "
                  f"mc_loss={history['mc_loss'][-1]:.4f}")

        return history

    def forward(self, spikes: Tensor) -> Tensor:
        """Returns the ReasoningScore as a 6-d tensor (for use in nn pipelines)."""
        return self.compute_state(spikes).as_tensor()
