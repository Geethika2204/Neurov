"""
domains/cwru/domain.py
======================
CWRU Bearing Fault Detection Domain Plugin.

Plugs the Case Western Reserve University (CWRU) bearing fault dataset into
the NeuroVerse platform.  Covers four fault classes:
  - normal     (label 0) — healthy baseline bearing
  - inner_race (label 1) — inner raceway defect
  - ball       (label 2) — rolling element defect
  - outer_race (label 3) — outer raceway defect

Sensor events are dicts containing 18 vibration features (9 from Drive-End
accelerometer + 9 from Fan-End accelerometer) derived from 1024-sample windows
at 12 kHz.

This file contains ALL CWRU-specific logic.  /nv_platform/ files never
import from here.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from nv_platform.domain_plugin import DomainPlugin
from .preprocess import FEATURE_COLS_DE_FE, FEATURE_COLS_DE_ONLY, NUM_FEATURES_DE_FE

# ---------------------------------------------------------------------------
# Feature definitions (must match preprocess.py)
# ---------------------------------------------------------------------------

# Default: use all 18 features (DE + FE). The domain can be instantiated
# with de_only=True for files that lack a FE channel.
FEATURE_COLS = FEATURE_COLS_DE_FE      # 18 columns
NUM_FEATURES = NUM_FEATURES_DE_FE      # 18

# Fault class label mapping
FAULT_CLASSES = {
    "normal":     0,
    "inner_race": 1,
    "ball":       2,
    "outer_race": 3,
}

# Path to ChromaDB knowledge base (relative to repo root)
_KB_PATH = os.path.join(os.path.dirname(__file__), "rag_kb")


# ---------------------------------------------------------------------------
# JSON Schemas
# ---------------------------------------------------------------------------

def _build_input_schema(feature_cols: list[str]) -> dict[str, Any]:
    """Build input schema for the given feature column set."""
    return {
        "type": "object",
        "required": ["timestamp"] + feature_cols,
        "properties": {
            "timestamp": {
                "type": "number",
                "description": "Synthetic window timestamp (seconds)",
            },
            **{col: {"type": "number"} for col in feature_cols},
            # Optional metadata
            "fault_class":  {"type": "string"},
            "label":        {"type": "integer", "minimum": 0, "maximum": 3},
            "file_id":      {"type": "string"},
            "window_idx":   {"type": "integer"},
            "rpm":          {"type": "number"},
        },
        "additionalProperties": True,
    }


_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["label", "confidence"],
    "properties": {
        "label": {
            "type": "string",
            "enum": ["normal", "inner_race", "ball", "outer_race", "uncertain"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "reasoning": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

# Gate threshold hints for bearing fault detection.
# CWRU fault signatures are highly distinctive in vibration features, so we
# can afford aggressive (lower) escalation thresholds compared to traffic data.
# Faults manifest as elevated kurtosis, crest factor, and periodic impulsive
# energy — all captured by our 9-feature set.
_THRESHOLD_HINTS: dict[str, float] = {
    "novelty":           0.50,  # escalate if pattern is unfamiliar
    "uncertainty":       0.55,  # escalate if model is uncertain about class
    "prediction_error":  0.45,  # escalate if features shift unexpectedly
    "memory_confidence": 0.30,  # escalate if below prototype similarity
}


# ---------------------------------------------------------------------------
# Domain class
# ---------------------------------------------------------------------------


class CWRUDomain(DomainPlugin):
    """
    CWRU Bearing Fault Detection domain.

    Plugs the CWRU vibration dataset into the NeuroVerse pipeline.
    Sensor events are dicts with 18 vibration features (DE + FE channels,
    9 features each) plus a timestamp and optional metadata.

    Parameters
    ----------
    de_only : bool  Use only Drive-End features (9 dims) instead of 18.
                    Set to True when processing files without a FE channel.

    Examples
    --------
    >>> domain = CWRUDomain()
    >>> event = {
    ...     "timestamp": 0.0,
    ...     **{col: 0.0 for col in FEATURE_COLS},
    ... }
    >>> domain.validate_event(event)
    True
    """

    def __init__(self, de_only: bool = False) -> None:
        self._de_only = de_only
        self._feature_cols = FEATURE_COLS_DE_ONLY if de_only else FEATURE_COLS_DE_FE
        self._input_schema = _build_input_schema(self._feature_cols)

    # ------------------------------------------------------------------
    # DomainPlugin contract
    # ------------------------------------------------------------------

    @property
    def input_schema(self) -> dict[str, Any]:
        return self._input_schema

    @property
    def output_schema(self) -> dict[str, Any]:
        return _OUTPUT_SCHEMA

    @property
    def knowledge_base_path(self) -> str:
        return _KB_PATH

    @property
    def threshold_hints(self) -> dict[str, float]:
        return _THRESHOLD_HINTS

    # ------------------------------------------------------------------
    # Domain-specific helpers
    # ------------------------------------------------------------------

    @property
    def feature_cols(self) -> list[str]:
        """Active feature column names (18 or 9 depending on de_only)."""
        return self._feature_cols

    @property
    def num_features(self) -> int:
        """Number of features per event."""
        return len(self._feature_cols)

    def event_to_tensor(self, event: dict[str, Any]) -> "torch.Tensor":
        """
        Extract feature values from an event dict as a float tensor.

        Returns
        -------
        torch.Tensor  shape (1, 1, num_features) — (batch, time_window, feats)
        Ready to pass directly to ``SNNInputPlugin.process_events``.
        """
        values = [float(event.get(col, 0.0)) for col in self._feature_cols]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def label_to_class(label: int) -> str:
        """Convert integer label to fault class name."""
        inv = {v: k for k, v in FAULT_CLASSES.items()}
        return inv.get(label, "unknown")

    @staticmethod
    def class_to_label(fault_class: str) -> int:
        """Convert fault class name to integer label."""
        return FAULT_CLASSES.get(fault_class, -1)

    def event_from_window(self, window_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Normalise a CWRUWindow.to_dict() output into a valid platform event.

        Keeps only the keys that belong to the input schema (timestamp +
        feature columns + optional metadata).
        """
        event: dict[str, Any] = {
            "timestamp": float(window_dict.get("timestamp", 0.0)),
        }
        for col in self._feature_cols:
            event[col] = float(window_dict.get(col, 0.0))
        # Carry optional metadata if present
        for opt in ("fault_class", "label", "file_id", "window_idx", "rpm"):
            if opt in window_dict:
                event[opt] = window_dict[opt]
        return event
