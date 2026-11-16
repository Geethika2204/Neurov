"""
/nv_platform/decision_recorder.py

Unified Decision Recording (Path A + Path B)
============================================
A single, persistent record of every decision the system makes, regardless of
which path produced it.

Why a separate module
---------------------
The existing ``action_memory.DecisionRecord`` is the *in-memory* record for the
feedback loop (outcome verification, confirmed success/failure).  That is a
different concern.  This module is a focused, *persistent audit log* with the
exact schema the decision-recording task requires and no feedback-loop
semantics — so it deliberately does not duplicate the gate, action module,
planner, or the feedback store.

Persistence
-----------
The codebase has no general decision-log storage: the gate checkpoint uses
``torch.save`` and CWRU results use ad-hoc JSON files.  Neither is suitable for
one-record-per-event append-only logging that must survive restarts, so this
module uses SQLite (Python stdlib — no new dependency).  Records remain
available after the application restarts because they are committed to disk.

Common record schema (both paths)
---------------------------------
Each record contains exactly:

    decision_id      : Unique identifier (the gate's per-event UUID).
    timestamp         : Event processing time (ISO-8601 UTC).
    predicted_label   : Final predicted label (NORMAL, CRASH, or domain class).
    decision_path     : "Path A" (fast / automatic) or "Path B" (deliberative / LLM).
    selected_action   : The action selected by the system.
    action_plan       : The complete generated action plan (when available).
    llm_called        : Whether an LLM was actually invoked.
    gate_confidence   : Gate escalation probability (extra, for auditing).

Domain independence
-------------------
No imports from ``/domains/``.  Labels are stored as opaque strings, so any
domain's label vocabulary works without changes here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Storage location
# ---------------------------------------------------------------------------

# Default on-disk location: <repo_root>/data/decision_records.sqlite
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(REPO_ROOT / "data" / "decision_records.sqlite")


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """
    One persisted decision.  Used identically for Path A and Path B.

    Attributes
    ----------
    decision_id     : str            Unique identifier.
    timestamp       : str            Event processing time (ISO-8601 UTC).
    predicted_label : str            Final predicted label.
    decision_path   : str            "Path A" or "Path B".
    selected_action : str            Action selected by the system.
    action_plan     : dict | None    Complete generated action plan, or None.
    llm_called      : bool           Whether an LLM was actually invoked.
    gate_confidence : float          Gate escalation probability (extra).
    """

    decision_id: str = ""
    timestamp: str = ""
    predicted_label: str = "uncertain"
    decision_path: str = "Path A"
    selected_action: str = ""
    action_plan: Optional[dict[str, Any]] = None
    llm_called: bool = False
    gate_confidence: float = 0.0

    # Valid path labels (single source of truth for the schema).
    PATH_A = "Path A"
    PATH_B = "Path B"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "predicted_label": self.predicted_label,
            "decision_path": self.decision_path,
            "selected_action": self.selected_action,
            "action_plan": self.action_plan,
            "llm_called": self.llm_called,
            "gate_confidence": self.gate_confidence,
        }

    def __repr__(self) -> str:
        return (
            f"DecisionRecord(id={self.decision_id[:8]}…, "
            f"path={self.decision_path}, label={self.predicted_label!r}, "
            f"action={self.selected_action!r}, llm_called={self.llm_called})"
        )


# ---------------------------------------------------------------------------
# Recorder (SQLite-backed, thread-safe)
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_records (
    decision_id     TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    decision_path   TEXT NOT NULL,
    selected_action TEXT,
    action_plan     TEXT,
    llm_called      INTEGER NOT NULL,
    gate_confidence REAL NOT NULL
);
"""


class DecisionRecorder:
    """
    Persistent, append-only store of ``DecisionRecord`` entries.

    Parameters
    ----------
    db_path : str   SQLite database path.  Defaults to
                    ``<repo_root>/data/decision_records.sqlite`` so records
                    survive application restarts.  The parent directory is
                    created if missing.

    Thread safety
    -------------
    A single connection is opened with ``check_same_thread=False`` and all
    reads/writes are serialised by a lock, so the recorder is safe to share
    across the pipeline components.

    Examples
    --------
    >>> recorder = DecisionRecorder()
    >>> recorder.record(DecisionRecord(decision_id="d1", ...))
    >>> recorder.count()
    1
    >>> # After a process restart, a new recorder at the same path sees d1:
    >>> DecisionRecorder().get("d1") is not None
    True
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, rec: DecisionRecord) -> None:
        """Persist one decision (INSERT OR REPLACE on decision_id)."""
        plan_json = (
            json.dumps(rec.action_plan, default=str)
            if rec.action_plan is not None
            else None
        )
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO decision_records
                   (decision_id, timestamp, predicted_label, decision_path,
                    selected_action, action_plan, llm_called, gate_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec.decision_id,
                    rec.timestamp,
                    rec.predicted_label,
                    rec.decision_path,
                    rec.selected_action,
                    plan_json,
                    int(bool(rec.llm_called)),
                    float(rec.gate_confidence),
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, decision_id: str) -> Optional[DecisionRecord]:
        """Return the record with this decision_id, or None."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM decision_records WHERE decision_id = ?",
                (decision_id,),
            )
            row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def all(self) -> list[DecisionRecord]:
        """Return every record (oldest first)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM decision_records ORDER BY rowid ASC"
            )
            rows = cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self) -> int:
        """Total number of stored records."""
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM decision_records")
            return int(cur.fetchone()[0])

    def count_by_path(self) -> dict[str, int]:
        """Counts split by decision_path (Path A / Path B)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT decision_path, COUNT(*) FROM decision_records "
                "GROUP BY decision_path"
            )
            return {row[0]: int(row[1]) for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "DecisionRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: tuple) -> DecisionRecord:
        (
            decision_id, timestamp, predicted_label, decision_path,
            selected_action, plan_json, llm_called, gate_confidence,
        ) = row
        action_plan = json.loads(plan_json) if plan_json is not None else None
        return DecisionRecord(
            decision_id=decision_id,
            timestamp=timestamp,
            predicted_label=predicted_label,
            decision_path=decision_path,
            selected_action=selected_action or "",
            action_plan=action_plan,
            llm_called=bool(llm_called),
            gate_confidence=float(gate_confidence),
        )


# ---------------------------------------------------------------------------
# Default singleton accessor (optional convenience for production wiring)
# ---------------------------------------------------------------------------

_default_recorder: Optional[DecisionRecorder] = None
_default_lock = threading.Lock()


def get_default_recorder() -> DecisionRecorder:
    """
    Return a process-wide ``DecisionRecorder`` at the default path.

    Useful for production wiring where every processed event should be
    logged without each caller constructing its own recorder.  Tests should
    pass their own ``DecisionRecorder(tmp_path)`` to stay hermetic.
    """
    global _default_recorder
    with _default_lock:
        if _default_recorder is None:
            _default_recorder = DecisionRecorder(DEFAULT_DB_PATH)
        return _default_recorder
