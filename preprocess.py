"""
preprocess.py
=============
Builds sliding-window spike tensors from the FT-AED radar CSV and saves them
to data/processed/{train,val,test}.pt.

Spatial features (new)
-----------------------
Since all 49 milemarkers share the same timestamp grid, we compute freeway-wide
statistics at each timestep and append them to every window's feature vector.

Spatial features added (4 per timestep):
  - lane1_speed_std   : std of lane1_speed across all milemarkers at this t
  - lane1_speed_min   : min of lane1_speed across all milemarkers (congestion front)
  - lane1_occ_max     : max of lane1_occ   across all milemarkers (density peak)
  - lane1_speed_range : max - min speed (gradient magnitude)

These are normalised with training-set statistics and appended to the 12 per-
milemarker features, giving a final feature vector of 12 + 4 = 16 dimensions.
"""

import pandas as pd
import numpy as np
import torch
from snntorch import spikegen
from pathlib import Path

# ── Feature definitions ────────────────────────────────────────────────────

FEATURE_COLS = [
    "lane1_speed", "lane1_volume", "lane1_occ",
    "lane2_speed", "lane2_volume", "lane2_occ",
    "lane3_speed", "lane3_volume", "lane3_occ",
    "lane4_speed", "lane4_volume", "lane4_occ",
]

# Spatial feature names — computed freeway-wide per unix_time
SPATIAL_COLS = [
    "spatial_speed_std",    # cross-milemarker std of lane1_speed
    "spatial_speed_min",    # cross-milemarker min of lane1_speed
    "spatial_occ_max",      # cross-milemarker max of lane1_occ
    "spatial_speed_range",  # cross-milemarker (max - min) of lane1_speed
]

ALL_FEATURE_COLS = FEATURE_COLS + SPATIAL_COLS  # 16 total
NUM_FEATURES     = len(ALL_FEATURE_COLS)         # 16

LABEL_COL = "crash_record"
WINDOW    = 20
STRIDE    = 5
NUM_STEPS = 25


# ── Spatial feature computation ────────────────────────────────────────────

def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute freeway-wide spatial statistics per unix_time and join them back
    onto every row.

    Parameters
    ----------
    df : DataFrame with columns: unix_time, milemarker, lane1_speed, lane1_occ

    Returns
    -------
    df with SPATIAL_COLS appended
    """
    # Group by timestamp, aggregate over all milemarkers
    agg = df.groupby("unix_time").agg(
        spatial_speed_std   = ("lane1_speed", "std"),
        spatial_speed_min   = ("lane1_speed", "min"),
        spatial_occ_max     = ("lane1_occ",   "max"),
        spatial_speed_range = ("lane1_speed",
                               lambda x: x.max() - x.min()),
    ).reset_index()

    # Fill NaN std (single milemarker edge case) with 0
    agg["spatial_speed_std"] = agg["spatial_speed_std"].fillna(0.0)

    df = df.merge(agg, on="unix_time", how="left")
    return df


# ── Split, normalise, window ───────────────────────────────────────────────

def load_and_split(path: str, train_days: int = 14, val_days: int = 3):
    df = pd.read_csv(path)
    df = compute_spatial_features(df)
    days = sorted(df["day"].unique())
    train_d = days[:train_days]
    val_d   = days[train_days:train_days + val_days]
    test_d  = days[train_days + val_days:]
    return (df[df["day"].isin(train_d)].copy(),
            df[df["day"].isin(val_d)].copy(),
            df[df["day"].isin(test_d)].copy())


def normalize(train_df, val_df, test_df):
    """Normalise ALL_FEATURE_COLS using training-set statistics."""
    mean = train_df[ALL_FEATURE_COLS].mean()
    std  = train_df[ALL_FEATURE_COLS].std().replace(0, 1)
    return (
        (train_df[ALL_FEATURE_COLS] - mean) / std,
        (val_df[ALL_FEATURE_COLS]   - mean) / std,
        (test_df[ALL_FEATURE_COLS]  - mean) / std,
    )


def make_windows(df: pd.DataFrame, feats: pd.DataFrame):
    """
    Build sliding windows of shape (WINDOW, NUM_FEATURES).

    Windows that span milemarker boundaries are skipped.
    Label = 1 if any timestep in the window has crash_record = 1.
    """
    order      = df.sort_values(["milemarker", "unix_time"]).index.values
    values     = feats.loc[order].values                    # (N, 16)
    labels     = df[LABEL_COL].loc[order].values
    milemarker = df["milemarker"].loc[order].values

    X, y = [], []
    for start in range(0, len(values) - WINDOW, STRIDE):
        end = start + WINDOW
        # Skip windows that cross milemarker boundaries
        if milemarker[start] != milemarker[end - 1]:
            continue
        X.append(values[start:end])
        y.append(int(labels[start:end].max() == 1))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def to_spikes(X: np.ndarray) -> torch.Tensor:
    """Rate-encode normalised features as binary spike trains."""
    x = torch.sigmoid(torch.tensor(X))
    return spikegen.rate(x, num_steps=NUM_STEPS)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading CSV and computing spatial features …")
    train_df, val_df, test_df = load_and_split(
        "data/nashville_freeway_anomaly.csv"
    )

    print(f"Spatial features added: {SPATIAL_COLS}")
    print(f"Total features per timestep: {NUM_FEATURES}")
    print(f"  train: {len(train_df):,} rows")
    print(f"  val:   {len(val_df):,} rows")
    print(f"  test:  {len(test_df):,} rows")

    print("\nNormalising …")
    train_feats, val_feats, test_feats = normalize(train_df, val_df, test_df)

    print("Building windows …")
    X_train, y_train = make_windows(train_df, train_feats)
    X_val,   y_val   = make_windows(val_df,   val_feats)
    X_test,  y_test  = make_windows(test_df,  test_feats)

    print(f"  train: X={X_train.shape}, crashes={y_train.sum()}/{len(y_train)}")
    print(f"  val:   X={X_val.shape},   crashes={y_val.sum()}/{len(y_val)}")
    print(f"  test:  X={X_test.shape},  crashes={y_test.sum()}/{len(y_test)}")

    print("\nEncoding spikes …")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    torch.save({"X": to_spikes(X_train), "y": torch.tensor(y_train)},
               "data/processed/train.pt")
    torch.save({"X": to_spikes(X_val),   "y": torch.tensor(y_val)},
               "data/processed/val.pt")
    torch.save({"X": to_spikes(X_test),  "y": torch.tensor(y_test)},
               "data/processed/test.pt")

    print("Done → data/processed/{train,val,test}.pt")
    print(f"New X shape: (num_steps={NUM_STEPS}, N_windows, time_window={WINDOW}, features={NUM_FEATURES})")
