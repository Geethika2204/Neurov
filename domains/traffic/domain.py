"""
/domains/traffic/domain.py

Traffic Domain Plugin — FT-AED Crash Detection
================================================
Wires the Nashville Freeway Anomaly (FT-AED) radar dataset into the
NeuroVerse platform.

Dataset features (12 columns, matching preprocess.py FEATURE_COLS):
  lane{1-4}_speed, lane{1-4}_volume, lane{1-4}_occ

Label: crash_record (0 = normal, 1 = crash / near-crash)

This file contains ALL traffic-specific logic.  /platform/ files never
import from here.
"""

from __future__ import annotations

import os
from typing import Any

from nv_platform.domain_plugin import DomainPlugin


# ---------------------------------------------------------------------------
# Feature definitions (must match preprocess.py)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "lane1_speed", "lane1_volume", "lane1_occ",
    "lane2_speed", "lane2_volume", "lane2_occ",
    "lane3_speed", "lane3_volume", "lane3_occ",
    "lane4_speed", "lane4_volume", "lane4_occ",
]

_NUM_FEATURES = len(FEATURE_COLS)  # 12

# Path to ChromaDB knowledge base (relative to repo root)
_KB_PATH = os.path.join(
    os.path.dirname(__file__), "rag_kb"
)


# ---------------------------------------------------------------------------
# JSON Schemas
# ---------------------------------------------------------------------------

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["timestamp"] + FEATURE_COLS,
    "properties": {
        "timestamp": {"type": "number", "description": "Unix timestamp (seconds)"},
        **{
            col: {"type": "number"}
            for col in FEATURE_COLS
        },
        # Optional label for supervised evaluation
        "crash_record": {"type": "integer", "enum": [0, 1]},
        "milemarker": {"type": "number"},
        "day": {"type": "integer"},
    },
    "additionalProperties": True,
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["label", "confidence"],
    "properties": {
        "label": {
            "type": "string",
            "enum": ["normal", "anomaly", "crash", "uncertain"],
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

# Soft gate hints derived from domain knowledge of the FT-AED dataset.
# ~400 labelled crashes in ~3.7M samples → very imbalanced; be aggressive
# about escalating when things look unusual.
_THRESHOLD_HINTS: dict[str, float] = {
    "novelty":           0.55,  # escalate if pattern looks new
    "uncertainty":       0.60,  # escalate if model is unsure
    "prediction_error":  0.50,  # escalate if prediction was badly wrong
    "memory_confidence": 0.25,  # escalate if below this similarity to memory
}


# ---------------------------------------------------------------------------
# Domain class
# ---------------------------------------------------------------------------


class TrafficDomain(DomainPlugin):
    """
    FT-AED traffic crash detection domain.

    Plugs the Nashville Freeway Anomaly dataset into the NeuroVerse pipeline.
    Sensor events are dicts with 12 lane-level features (speed, volume,
    occupancy) plus a timestamp.

    Examples
    --------
    >>> domain = TrafficDomain()
    >>> event = {
    ...     "timestamp": 1727000000.0,
    ...     "lane1_speed": 65.0, "lane1_volume": 120.0, "lane1_occ": 0.15,
    ...     "lane2_speed": 60.0, "lane2_volume": 110.0, "lane2_occ": 0.14,
    ...     "lane3_speed": 55.0, "lane3_volume": 100.0, "lane3_occ": 0.13,
    ...     "lane4_speed": 50.0, "lane4_volume": 90.0,  "lane4_occ": 0.12,
    ... }
    >>> domain.validate_event(event)
    True
    """

    @property
    def input_schema(self) -> dict[str, Any]:
        return _INPUT_SCHEMA

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
    # Domain-specific helpers (not part of the platform contract)
    # ------------------------------------------------------------------

    @staticmethod
    def event_from_row(row: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a raw DataFrame row (from the FT-AED CSV) into a platform event.

        Parameters
        ----------
        row : dict  A row from ``nashville_freeway_anomaly.csv`` as a dict.

        Returns
        -------
        dict  A valid event dict passable to ``validate_event``.
        """
        event: dict[str, Any] = {
            "timestamp": float(row.get("unix_time", 0)),
        }
        for col in FEATURE_COLS:
            event[col] = float(row.get(col, 0.0))
        if "crash_record" in row:
            event["crash_record"] = int(row["crash_record"])
        if "milemarker" in row:
            event["milemarker"] = float(row["milemarker"])
        if "day" in row:
            event["day"] = int(row["day"])
        return event

    @staticmethod
    def event_to_tensor(event: dict[str, Any]):
        """
        Extract the 12 feature values from an event dict as a float tensor.

        Returns
        -------
        torch.Tensor, shape (1, 1, 12)  — (batch=1, time_window=1, features=12)
        Ready to pass directly to ``SNNInputPlugin.process_events``.
        """
        import torch

        values = [float(event.get(col, 0.0)) for col in FEATURE_COLS]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
