"""
domains/cwru/preprocess.py
==========================
Signal preprocessing for CWRU bearing fault vibration data.

Pipeline
--------
1. Segment raw vibration signal into fixed-length windows (default 1024 samples).
2. Extract 9 statistical / signal-processing features per window per channel.
3. Concatenate DE + FE features → 18-dim feature vector (or 9-dim if FE absent).
4. Z-score normalise across the batch (per-feature mean / std).

Feature set (per channel)
--------------------------
  0  rms              Root-mean-square amplitude
  1  std              Standard deviation
  2  peak             Peak absolute value
  3  crest_factor     Peak / RMS
  4  kurtosis         4th-order moment (impulsivity indicator)
  5  skewness         3rd-order moment (asymmetry)
  6  peak_to_peak     Max − Min
  7  shape_factor     RMS / mean(|x|)
  8  impulse_factor   Peak / mean(|x|)

These are the most discriminative features for bearing fault classification
(documented in Smith & Randall 2015, Chen et al. 2020).

Output
------
Each preprocessed window becomes a "sensor event" dict ready for the CWRU
domain's input_schema validator.

Public API
----------
    from domains.cwru.preprocess import CWRUPreprocessor

    prep     = CWRUPreprocessor(window_size=1024, overlap=0.5)
    windows  = prep.process_records(records)      # list of CWRUWindow
    events   = prep.to_events(windows)            # list of event dicts
    normed   = prep.normalize_events(events)      # z-score normalised copy
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .loader import CWRURecord, FAULT_CLASSES, LABEL_MAP

# ---------------------------------------------------------------------------
# Feature names (must stay in sync with domain.py FEATURE_COLS)
# ---------------------------------------------------------------------------

_CHANNEL_FEATURES = [
    "rms",
    "std",
    "peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "peak_to_peak",
    "shape_factor",
    "impulse_factor",
]

# Feature columns emitted for a window that has both DE and FE channels
FEATURE_COLS_DE_FE = [f"de_{f}" for f in _CHANNEL_FEATURES] + [
    f"fe_{f}" for f in _CHANNEL_FEATURES
]

# Feature columns when only DE channel is available
FEATURE_COLS_DE_ONLY = [f"de_{f}" for f in _CHANNEL_FEATURES]

NUM_FEATURES_DE_FE = len(FEATURE_COLS_DE_FE)      # 18
NUM_FEATURES_DE_ONLY = len(FEATURE_COLS_DE_ONLY)   # 9


# ---------------------------------------------------------------------------
# Window dataclass
# ---------------------------------------------------------------------------


@dataclass
class CWRUWindow:
    """
    A single preprocessed window extracted from a CWRURecord.

    Attributes
    ----------
    fault_class  : str              Original fault class label.
    label        : int              Integer label.
    file_id      : str              Source file ID.
    window_idx   : int              Index of this window within the file.
    features     : np.ndarray       Feature vector, shape (NUM_FEATURES,).
    feature_cols : list[str]        Feature column names matching ``features``.
    rpm          : float            Shaft speed in RPM.
    timestamp    : float            Synthetic timestamp (window_idx * dt).
    """

    fault_class: str
    label: int
    file_id: str
    window_idx: int
    features: np.ndarray
    feature_cols: list[str]
    rpm: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dict (suitable for domain.validate_event)."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "fault_class": self.fault_class,
            "label": self.label,
            "file_id": self.file_id,
            "window_idx": self.window_idx,
            "rpm": self.rpm,
        }
        for col, val in zip(self.feature_cols, self.features.tolist()):
            d[col] = float(val)
        return d


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _safe_mean_abs(x: np.ndarray) -> float:
    """Mean absolute value, protected against division-by-zero."""
    m = np.mean(np.abs(x))
    return float(m) if m != 0.0 else 1e-9


def _extract_features(segment: np.ndarray) -> np.ndarray:
    """
    Compute 9 statistical features from a 1-D vibration segment.

    Parameters
    ----------
    segment : np.ndarray  Shape (window_size,), float64.

    Returns
    -------
    np.ndarray  Shape (9,).
    """
    if segment.size == 0:
        return np.zeros(len(_CHANNEL_FEATURES), dtype=np.float32)

    x = segment.astype(np.float64)
    n = x.size

    # Basic stats
    mean_abs = _safe_mean_abs(x)
    rms_val = float(np.sqrt(np.mean(x ** 2)))
    std_val = float(np.std(x))
    peak_val = float(np.max(np.abs(x)))
    p2p_val = float(np.max(x) - np.min(x))

    # Crest factor: peak / RMS
    crest = peak_val / (rms_val if rms_val != 0.0 else 1e-9)

    # Kurtosis (Fisher normalised — 0 for Gaussian)
    mu4 = float(np.mean((x - np.mean(x)) ** 4))
    sigma4 = float(std_val ** 4)
    kurtosis_val = (mu4 / sigma4) - 3.0 if sigma4 != 0.0 else 0.0

    # Skewness
    mu3 = float(np.mean((x - np.mean(x)) ** 3))
    sigma3 = float(std_val ** 3)
    skewness_val = mu3 / sigma3 if sigma3 != 0.0 else 0.0

    # Shape factor: RMS / mean(|x|)
    shape_val = rms_val / mean_abs

    # Impulse factor: peak / mean(|x|)
    impulse_val = peak_val / mean_abs

    feats = np.array(
        [rms_val, std_val, peak_val, crest, kurtosis_val,
         skewness_val, p2p_val, shape_val, impulse_val],
        dtype=np.float32,
    )
    # Clip extreme values to avoid NaN propagation
    return np.clip(feats, -1e6, 1e6)


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------


class CWRUPreprocessor:
    """
    Segments CWRURecord vibration signals into windows and extracts features.

    Parameters
    ----------
    window_size : int    Samples per window (default 1024 ≈ 85 ms @ 12 kHz).
    overlap     : float  Fractional overlap between consecutive windows [0, 1).
                         Default 0.5 (50 % overlap).
    use_fe      : bool   Whether to include Fan-End channel features.
                         If the FE channel is absent for a record, falls back
                         to DE-only automatically.

    Examples
    --------
    >>> from domains.cwru.loader import CWRULoader
    >>> loader = CWRULoader("domains/cwru")
    >>> records = loader.load_all()
    >>> prep = CWRUPreprocessor(window_size=1024, overlap=0.5)
    >>> windows = prep.process_records(records)
    >>> events  = prep.to_events(windows)
    >>> normed  = prep.normalize_events(events)
    """

    def __init__(
        self,
        window_size: int = 1024,
        overlap: float = 0.5,
        use_fe: bool = True,
    ) -> None:
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        self.window_size = window_size
        self.overlap = overlap
        self.use_fe = use_fe
        self._step = max(1, int(window_size * (1.0 - overlap)))

        # Fit statistics for Z-score normalization (populated by normalize_events)
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_records(self, records: list[CWRURecord]) -> list[CWRUWindow]:
        """
        Segment all records into feature windows.

        Parameters
        ----------
        records : list[CWRURecord]  Loaded CWRU records.

        Returns
        -------
        list[CWRUWindow]  One entry per segmented window.
        """
        windows: list[CWRUWindow] = []
        for record in records:
            windows.extend(self._process_record(record))
        return windows

    def to_events(self, windows: list[CWRUWindow]) -> list[dict[str, Any]]:
        """Convert CWRUWindow list to event dicts."""
        return [w.to_dict() for w in windows]

    def normalize_events(
        self,
        events: list[dict[str, Any]],
        fit: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Z-score normalise all numeric feature values in-place (copy).

        Parameters
        ----------
        events : list[dict]  Output of ``to_events()``.
        fit    : bool        If True, recompute mean/std from these events.
                             If False, use previously fitted statistics.

        Returns
        -------
        list[dict]  New list with normalised values.
        """
        if not events:
            return events

        # Determine feature keys (exclude metadata fields)
        _meta = {"timestamp", "fault_class", "label", "file_id", "window_idx", "rpm"}
        feat_keys = [k for k in events[0] if k not in _meta]

        if not feat_keys:
            return events

        # Build feature matrix
        mat = np.array(
            [[e[k] for k in feat_keys] for e in events],
            dtype=np.float32,
        )

        if fit:
            self._feature_mean = mat.mean(axis=0)
            self._feature_std = mat.std(axis=0) + 1e-8

        if self._feature_mean is None:
            return events

        normed = (mat - self._feature_mean) / self._feature_std

        # Rebuild event dicts
        result: list[dict[str, Any]] = []
        for i, ev in enumerate(events):
            new_ev = {k: ev[k] for k in _meta if k in ev}
            for j, fk in enumerate(feat_keys):
                new_ev[fk] = float(normed[i, j])
            result.append(new_ev)

        return result

    @property
    def feature_cols(self) -> list[str]:
        """
        Feature column names.  Determined dynamically based on use_fe flag;
        always returns FEATURE_COLS_DE_FE when use_fe=True (falls back to
        DE-only per record if FE channel is absent).
        """
        return FEATURE_COLS_DE_FE if self.use_fe else FEATURE_COLS_DE_ONLY

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_record(self, record: CWRURecord) -> list[CWRUWindow]:
        """Segment one record into windows and extract features."""
        de = record.de_time
        fe = record.fe_time if (self.use_fe and record.fe_time.size > 0) else None

        # Time step between consecutive window starts (seconds)
        dt = self._step / record.sample_rate

        windows: list[CWRUWindow] = []
        starts = range(0, len(de) - self.window_size + 1, self._step)

        for idx, start in enumerate(starts):
            end = start + self.window_size
            de_seg = de[start:end]

            # Extract DE features (always present)
            de_feats = _extract_features(de_seg)

            if fe is not None and len(fe) >= end:
                fe_seg = fe[start:end]
                fe_feats = _extract_features(fe_seg)
                features = np.concatenate([de_feats, fe_feats])
                feat_cols = FEATURE_COLS_DE_FE
            else:
                features = de_feats
                feat_cols = FEATURE_COLS_DE_ONLY

            windows.append(
                CWRUWindow(
                    fault_class=record.fault_class,
                    label=record.label,
                    file_id=record.file_id,
                    window_idx=idx,
                    features=features,
                    feature_cols=feat_cols,
                    rpm=record.rpm,
                    timestamp=float(idx) * dt,
                )
            )

        return windows
