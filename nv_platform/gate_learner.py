"""
/platform/gate_learner.py

Gate Meta-Learner
==================
A lightweight logistic regression model that predicts whether to escalate
based solely on the five danger signals emitted by the ImplicitPredictiveLayer.

Feature vector (5 dimensions)
------------------------------
0: prediction_error     — how badly was this window predicted?
1: uncertainty          — epistemic uncertainty (MC-dropout std)
2: spatial_correlation  — how anomalous is the cross-sensor pattern?
3: risk                 — composite weighted score (primary signal)
4: novelty              — how new is this pattern vs. stored prototypes?

Deliberately EXCLUDED from features
-------------------------------------
- memory_confidence : always ≈ 0.998 after warm-up; near-zero variance.
- fn_rate           : history-dependent; training/eval mismatch when
                      computed from a final/current history object.
- esc_rate          : depends on the gate's own past decisions → feedback loop.

Training is done offline in ``train_gate.py`` using a standardised full-batch
logistic regression.  ``predict_proba()`` applies the saved weights directly.
The ``update()`` online path is retained for compatibility but should not be
used in the main offline pipeline.

Checkpoint domain tagging
--------------------------
Checkpoints saved by ``train_gate.py`` include a ``"domain"`` field.  When a
checkpoint is loaded the caller should verify that the domain matches the
intended application domain.  Attempting to load a checkpoint for a different
domain will raise ``DomainMismatchError`` unless ``allow_domain_mismatch=True``
is passed to the loader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from nv_platform.gate import EscalationHistory


# ---------------------------------------------------------------------------
# Domain mismatch error
# ---------------------------------------------------------------------------

class DomainMismatchError(ValueError):
    """
    Raised when a checkpoint trained for domain A is loaded for domain B
    without an explicit opt-in.

    Example
    -------
    >>> raise DomainMismatchError("traffic", "cwru")
    """
    def __init__(self, checkpoint_domain: str, expected_domain: str) -> None:
        self.checkpoint_domain = checkpoint_domain
        self.expected_domain   = expected_domain
        super().__init__(
            f"Checkpoint domain '{checkpoint_domain}' does not match "
            f"the expected domain '{expected_domain}'. "
            f"Retrain the gate on '{expected_domain}' data or pass "
            f"allow_domain_mismatch=True to suppress this error."
        )


# ---------------------------------------------------------------------------
# Feature extraction — five danger signals only
# ---------------------------------------------------------------------------

_FEATURE_DIM = 5  # prediction_error, uncertainty, spatial_correlation, risk, novelty

# Index mapping into ReasoningScore.as_tensor() → (6,):
#   [0] novelty
#   [1] uncertainty
#   [2] prediction_error
#   [3] memory_confidence   ← excluded
#   [4] spatial_correlation
#   [5] risk
_SCORE_INDICES = [2, 1, 4, 5, 0]  # pred_error, uncertainty, spatial, risk, novelty

# Human-readable names (same order as _SCORE_INDICES)
FEATURE_NAMES = [
    "prediction_error",
    "uncertainty",
    "spatial_correlation",
    "risk",
    "novelty",
]


def _extract_features(score_tensor: Tensor) -> Tensor:
    """
    Extract the five danger signals from a ReasoningScore tensor.

    Parameters
    ----------
    score_tensor : Tensor, shape (6,)
        Output of ``ReasoningScore.as_tensor()``.

    Returns
    -------
    Tensor, shape (5,) — [prediction_error, uncertainty,
                           spatial_correlation, risk, novelty]
    """
    return torch.stack([score_tensor[i] for i in _SCORE_INDICES])


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


class GateLearner:
    """
    Logistic regression meta-learner for the reasoning gate.

    In the offline pipeline (train_gate.py) the model weights, bias,
    feature_mean, and feature_std are set externally after fitting.
    ``predict_proba()`` applies the standardisation internally.

    Parameters
    ----------
    feature_mean : Tensor | None  Training-set feature mean for standardisation.
    feature_std  : Tensor | None  Training-set feature std for standardisation.
    update_every : int            Online update schedule (kept for back-compat).
    fn_penalty   : float          FN penalty for online updates (back-compat).
    lr           : float          Learning rate for online SGD (back-compat).
    epochs_per_update : int       Epochs per online update (back-compat).
    """

    def __init__(
        self,
        feature_mean: Tensor | None = None,
        feature_std:  Tensor | None = None,
        update_every: int   = 50,
        fn_penalty:   float = 5.0,
        lr:           float = 0.05,
        epochs_per_update: int = 50,
    ) -> None:
        self.update_every       = update_every
        self.fn_penalty         = fn_penalty
        self.lr                 = lr
        self.epochs_per_update  = epochs_per_update

        # Single linear layer: 5 features → 1 logit
        self._model = nn.Linear(_FEATURE_DIM, 1)
        nn.init.zeros_(self._model.weight)
        # Positive bias: sigmoid(+1) ≈ 0.73 — lean toward escalation before
        # any training data, preventing cold-start false negatives.
        nn.init.constant_(self._model.bias, 1.0)

        # Standardisation statistics (set after offline training).
        # Default to zero-mean / unit-std (identity transform) so that an
        # untrained learner still produces valid probabilities instead of
        # exploding from mis-scaled inputs.
        self.feature_mean: Tensor = (
            feature_mean if feature_mean is not None
            else torch.zeros(_FEATURE_DIM)
        )
        self.feature_std: Tensor = (
            feature_std if feature_std is not None
            else torch.ones(_FEATURE_DIM)
        )

        self._opt          = torch.optim.SGD(self._model.parameters(), lr=lr)
        self._update_count = 0

    def _standardise(self, feats: Tensor) -> Tensor:
        """
        Apply training-set standardisation to a feature vector or matrix.

        Guards against near-zero standard deviations (clamps to ≥ 1e-6) so
        a degenerate feature cannot produce infinite z-scores.
        """
        safe_std = self.feature_std.clamp(min=1e-6)
        return (feats - self.feature_mean) / safe_std

    def predict_proba(
        self,
        score_tensor: Tensor,
        history: "EscalationHistory | None" = None,  # ignored; kept for back-compat
        extra_features: Tensor | None = None,
    ) -> float:
        """
        Probability of escalation given the current score.

        Parameters
        ----------
        score_tensor    : Tensor, shape (6,)
        history         : ignored (kept for API compatibility)
        extra_features  : Optional Tensor of additional raw features to append
                          to the 5-dim ImplicitLayer vector before scoring.
                          Shape (k,) where k matches the extra dims the model
                          was trained with.  If None, only the 5 standard
                          features are used (backward-compatible).

        Returns
        -------
        float in [0, 1]
        """
        feats = _extract_features(score_tensor)
        if extra_features is not None:
            feats = torch.cat([feats, extra_features.float()])
        feats_scaled = self._standardise(feats)
        # Dynamically resize the linear layer if the checkpoint dimension
        # differs from the default (handles 5->7 augmentation without breaking
        # the existing online-update path).
        in_dim = feats_scaled.shape[0]
        if self._model.in_features != in_dim:
            import torch.nn as _nn
            old_model = self._model
            self._model = _nn.Linear(in_dim, 1)
            _nn.init.zeros_(self._model.weight)
            _nn.init.constant_(self._model.bias, 1.0)
            # Copy shared weights for the first 5 dims if sizes align
            with torch.no_grad():
                shared = min(old_model.in_features, in_dim)
                self._model.weight.data[0, :shared] = \
                    old_model.weight.data[0, :shared]
                self._model.bias.data.copy_(old_model.bias.data)
        with torch.no_grad():
            logit = self._model(feats_scaled.unsqueeze(0))
        return float(torch.sigmoid(logit).squeeze())

    def update(self, history: "EscalationHistory") -> None:
        """
        Online retraining on labelled decisions in history.

        Uses the five danger signals only (no history-dependent features).
        This path is kept for back-compat; prefer offline training.
        """
        labelled = [
            e for e in history.log if e.get("outcome") is not None
        ]
        if len(labelled) < 5:
            return

        X_list, y_list = [], []
        for entry in labelled:
            feats = _extract_features(entry["score_tensor"])
            X_list.append(feats)
            y_list.append(float(entry["outcome"]))

        X = torch.stack(X_list)
        y = torch.tensor(y_list).float()

        # Class-imbalance weighting
        npos = y.sum().clamp(min=1)
        nneg = len(y) - npos
        pos_weight = torch.clamp(nneg / npos, max=20.0)

        X_scaled = self._standardise(X)

        for _ in range(self.epochs_per_update):
            self._opt.zero_grad()
            logits = self._model(X_scaled).squeeze(-1)
            loss   = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight
            )
            loss.backward()
            self._opt.step()

        self._update_count += 1
