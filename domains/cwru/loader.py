"""
domains/cwru/loader.py
======================
CWRU bearing fault .mat file loader.

Supports two formats:
  - Plain ``*.mat``    (MATLAB v5 / v7.3 — scipy.io.loadmat)
  - Zipped ``*.mat.zip`` (single .mat inside a zip archive)

CWRU channel conventions
-------------------------
Each file contains up to three vibration channels plus an RPM scalar:
  X<id>_DE_time  — Drive-End (DE) accelerometer  [primary — always present]
  X<id>_FE_time  — Fan-End (FE) accelerometer     [present in most files]
  X<id>_BA_time  — Base (BA) accelerometer         [fault files only]
  X<id>RPM       — Shaft speed (scalar)

The loader normalises channel names so downstream code always uses:
  de_time, fe_time, ba_time, rpm

Public API
----------
    from domains.cwru.loader import CWRULoader

    loader  = CWRULoader(data_root="domains/cwru")
    records = loader.load_all()
    # records: list of CWRURecord dataclasses

    # Or load a single file:
    record = CWRULoader.load_file("domains/cwru/inner_race/105.mat.zip",
                                  fault_class="inner_race")
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAULT_CLASSES = ("normal", "inner_race", "ball", "outer_race")

# Map fault class → integer label (used as target in supervised evaluation)
LABEL_MAP: dict[str, int] = {
    "normal":     0,
    "inner_race": 1,
    "ball":       2,
    "outer_race": 3,
}

# Nominal sampling rate (Hz) — CWRU DE channel is sampled at 12 kHz (or 48 kHz
# for some files; we default to 12 kHz which covers all files in this dataset).
DEFAULT_SAMPLE_RATE_HZ: int = 12_000


# ---------------------------------------------------------------------------
# Record dataclass
# ---------------------------------------------------------------------------


@dataclass
class CWRURecord:
    """
    One loaded CWRU .mat file.

    Attributes
    ----------
    fault_class  : str              One of FAULT_CLASSES.
    label        : int              Integer label (0 = normal … 3 = outer_race).
    file_id      : str              Numeric file ID extracted from filename.
    source_path  : str              Absolute path to the source file.
    de_time      : np.ndarray       Drive-End vibration, shape (N,).
    fe_time      : np.ndarray       Fan-End vibration, shape (N,) or empty.
    ba_time      : np.ndarray       Base vibration, shape (N,) or empty.
    rpm          : float            Shaft speed in RPM (scalar).
    sample_rate  : int              Sampling frequency in Hz (default 12 000).
    """

    fault_class: str
    label: int
    file_id: str
    source_path: str
    de_time: np.ndarray
    fe_time: np.ndarray = field(default_factory=lambda: np.array([]))
    ba_time: np.ndarray = field(default_factory=lambda: np.array([]))
    rpm: float = 0.0
    sample_rate: int = DEFAULT_SAMPLE_RATE_HZ

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CWRURecord(fault_class={self.fault_class!r}, "
            f"file_id={self.file_id!r}, "
            f"de_time.shape={self.de_time.shape}, "
            f"rpm={self.rpm:.0f})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_mat_bytes(data: bytes) -> dict:
    """Load MATLAB .mat bytes via scipy.io.loadmat."""
    return scipy.io.loadmat(io.BytesIO(data))


def _extract_channels(mat: dict, file_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Extract (de_time, fe_time, ba_time, rpm) from a raw mat dict.

    Channel keys follow the pattern ``X<id>_DE_time``, ``X<id>_FE_time``, etc.
    We normalise by ignoring the numeric prefix so the same code handles all IDs.
    """
    de_time = np.array([])
    fe_time = np.array([])
    ba_time = np.array([])
    rpm_val: float = 0.0

    for key, val in mat.items():
        if key.startswith("_"):
            continue
        key_upper = key.upper()
        if "_DE_TIME" in key_upper:
            de_time = val.flatten().astype(np.float64)
        elif "_FE_TIME" in key_upper:
            fe_time = val.flatten().astype(np.float64)
        elif "_BA_TIME" in key_upper:
            ba_time = val.flatten().astype(np.float64)
        elif key_upper.endswith("RPM"):
            rpm_val = float(np.array(val).flatten()[0])

    return de_time, fe_time, ba_time, rpm_val


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


class CWRULoader:
    """
    Discovers and loads all CWRU .mat / .mat.zip files under ``data_root``.

    Directory structure expected::

        data_root/
            normal/      *.mat or *.mat.zip
            inner_race/  *.mat.zip
            ball/        *.mat.zip
            outer_race/  *.mat.zip

    Parameters
    ----------
    data_root   : str | Path  Root directory containing the four class folders.
    sample_rate : int         Sampling frequency to store in each CWRURecord.

    Examples
    --------
    >>> loader = CWRULoader("domains/cwru")
    >>> records = loader.load_all()
    >>> len(records)  # one per file
    15
    """

    def __init__(
        self,
        data_root: str | os.PathLike = "domains/cwru",
        sample_rate: int = DEFAULT_SAMPLE_RATE_HZ,
    ) -> None:
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def load_all(self) -> list[CWRURecord]:
        """Load every readable .mat / .mat.zip file under data_root."""
        records: list[CWRURecord] = []
        for cls in FAULT_CLASSES:
            class_dir = self.data_root / cls
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if _is_mat_file(path):
                    try:
                        record = self.load_file(path, fault_class=cls)
                        records.append(record)
                    except Exception as exc:  # noqa: BLE001
                        # Skip corrupted / unreadable files but warn
                        import warnings
                        warnings.warn(
                            f"CWRULoader: skipping {path} — {exc}",
                            UserWarning,
                            stacklevel=2,
                        )
        return records

    @staticmethod
    def load_file(
        path: str | os.PathLike,
        fault_class: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE_HZ,
    ) -> CWRURecord:
        """
        Load a single .mat or .mat.zip file.

        Parameters
        ----------
        path        : str | Path  Path to the file.
        fault_class : str         One of FAULT_CLASSES.
        sample_rate : int         Sampling frequency.

        Returns
        -------
        CWRURecord

        Raises
        ------
        ValueError  If the file cannot be parsed or fault_class is unknown.
        FileNotFoundError  If path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CWRULoader: file not found: {path}")
        if fault_class not in FAULT_CLASSES:
            raise ValueError(
                f"Unknown fault_class {fault_class!r}. "
                f"Choose from {FAULT_CLASSES}."
            )

        # --- Read bytes ---
        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                mat_name = next(
                    (n for n in names if n.endswith(".mat")), names[0]
                )
                data = zf.read(mat_name)
        else:
            # Plain .mat — read binary
            data = path.read_bytes()

        # --- Parse ---
        mat = _load_mat_bytes(data)

        # Extract file ID from filename (numeric stem)
        file_id = re.sub(r"[^0-9]", "", path.stem)
        if not file_id:
            file_id = path.stem

        de_time, fe_time, ba_time, rpm = _extract_channels(mat, file_id)

        if de_time.size == 0:
            raise ValueError(
                f"CWRULoader: no DE_time channel found in {path}. "
                f"Available keys: {[k for k in mat if not k.startswith('_')]}"
            )

        return CWRURecord(
            fault_class=fault_class,
            label=LABEL_MAP[fault_class],
            file_id=file_id,
            source_path=str(path.resolve()),
            de_time=de_time,
            fe_time=fe_time,
            ba_time=ba_time,
            rpm=rpm,
            sample_rate=sample_rate,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _is_mat_file(path: Path) -> bool:
    """Return True for *.mat and *.mat.zip files (skips hidden / non-data)."""
    name = path.name
    if name.startswith("."):
        return False
    return name.endswith(".mat") or name.endswith(".mat.zip")
