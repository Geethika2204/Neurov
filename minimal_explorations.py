
# minimal_exploration.py
# Minimal first-pass exploration of the FT-AED dataset

import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD DATA
# ============================================================

FILE_PATH = "nashville_freeway_anomaly.txt"

df = pd.read_csv(FILE_PATH)

print("\n" + "=" * 70)
print("FT-AED DATASET — MINIMAL EXPLORATION")
print("=" * 70)

# ============================================================
# 2. BASIC STRUCTURE
# ============================================================

print("\n[1] BASIC STRUCTURE")
print("-" * 50)

print(f"Shape: {df.shape}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for col in df.columns:
    print(f"  - {col}")

print("\nData types:")
print(df.dtypes)

# ============================================================
# 3. FIRST / LAST ROWS
# ============================================================

print("\n[2] SAMPLE DATA")
print("-" * 50)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n[3] MISSING VALUES")
print("-" * 50)

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "missing": missing,
    "missing_%": (missing / len(df) * 100).round(2)
})

print(missing_table)

# ============================================================
# 5. DUPLICATES
# ============================================================

print("\n[4] DUPLICATES")
print("-" * 50)

print(f"Duplicate rows: {df.duplicated().sum():,}")

# ============================================================
# 6. NUMERICAL SUMMARY
# ============================================================

print("\n[5] NUMERICAL FEATURES")
print("-" * 50)

numeric_cols = df.select_dtypes(include=np.number).columns

print(df[numeric_cols].describe().T)

# ============================================================
# 7. UNIQUE VALUES
# ============================================================

print("\n[6] UNIQUE VALUES")
print("-" * 50)

for col in df.columns:
    n_unique = df[col].nunique(dropna=True)

    print(f"\n{col}")
    print(f"  Unique values: {n_unique}")

    if n_unique <= 20:
        print(f"  Values: {df[col].dropna().unique()}")

# ============================================================
# 8. POTENTIAL LABEL COLUMNS
# ============================================================

print("\n[7] POSSIBLE LABEL / ANOMALY COLUMNS")
print("-" * 50)

keywords = [
    "label",
    "anomaly",
    "crash",
    "event",
    "incident",
    "human"
]

label_candidates = [
    col for col in df.columns
    if any(keyword in col.lower() for keyword in keywords)
]

if label_candidates:
    for col in label_candidates:
        print(f"\n{col}")
        print(df[col].value_counts(dropna=False))
else:
    print("No obvious label columns found.")

# ============================================================
# 9. POTENTIAL TRAFFIC FEATURES
# ============================================================

print("\n[8] POTENTIAL TRAFFIC FEATURES")
print("-" * 50)

traffic_keywords = [
    "speed",
    "volume",
    "occupancy"
]

traffic_cols = [
    col for col in df.columns
    if any(keyword in col.lower() for keyword in traffic_keywords)
]

if traffic_cols:
    print("Detected traffic features:")
    for col in traffic_cols:
        print(f"  - {col}")

        if pd.api.types.is_numeric_dtype(df[col]):
            print(
                f"      min={df[col].min():.3f}, "
                f"max={df[col].max():.3f}, "
                f"mean={df[col].mean():.3f}"
            )
else:
    print("No obvious traffic feature columns found.")

# ============================================================
# 10. SPATIAL INFORMATION
# ============================================================

print("\n[9] SPATIAL STRUCTURE")
print("-" * 50)

spatial_keywords = [
    "mile",
    "lane",
    "location",
    "node",
    "sensor"
]

spatial_cols = [
    col for col in df.columns
    if any(keyword in col.lower() for keyword in spatial_keywords)
]

for col in spatial_cols:
    print(f"\n{col}")
    print(f"  Unique: {df[col].nunique(dropna=True)}")

    if df[col].nunique(dropna=True) <= 20:
        print(f"  Values: {df[col].dropna().unique()}")

# ============================================================
# 11. TIME INFORMATION
# ============================================================

print("\n[10] TEMPORAL STRUCTURE")
print("-" * 50)

time_candidates = [
    col for col in df.columns
    if any(
        keyword in col.lower()
        for keyword in ["time", "date", "day"]
    )
]

for col in time_candidates:

    print(f"\n{col}")

    # Try parsing as datetime
    parsed = pd.to_datetime(df[col], errors="coerce")

    valid = parsed.notna().sum()

    print(f"  Datetime-parsable: {valid:,}/{len(df):,}")

    if valid > 0:
        print(f"  Start: {parsed.min()}")
        print(f"  End:   {parsed.max()}")

# ============================================================
# 12. BASIC ANOMALY RATE
# ============================================================

print("\n[11] ANOMALY / CRASH RATE")
print("-" * 50)

for col in label_candidates:

    values = df[col].value_counts(dropna=False)

    print(f"\n{col}:")
    print(values)

    if len(values) <= 10:
        print("\nPercentage:")
        print(
            (df[col].value_counts(normalize=True, dropna=False) * 100)
            .round(3)
        )

# ============================================================
# 13. CORRELATION BETWEEN NUMERICAL FEATURES
# ============================================================

print("\n[12] NUMERICAL CORRELATIONS")
print("-" * 50)

if len(numeric_cols) > 1:
    print(
        df[numeric_cols]
        .corr()
        .round(3)
    )

# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPLORATION COMPLETE")
print("=" * 70)

print(f"""
Dataset:
    Rows       : {len(df):,}
    Columns    : {len(df.columns)}
    Duplicates : {df.duplicated().sum():,}

Numeric columns:
    {list(numeric_cols)}

Potential labels:
    {label_candidates}

Potential traffic features:
    {traffic_cols}

Potential spatial columns:
    {spatial_cols}

Potential time columns:
    {time_candidates}
""")


