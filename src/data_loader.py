"""src/data_loader.py — generic telemetry loading with validation.

Owned by: Role A. This does NOT know about mission-specific column names —
that's src/mission_adapter.py's job. This module only guarantees: chronological
order, no silent row drops, duplicate detection, and a metadata dict describing
what was found.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path


class LoaderError(ValueError):
    pass


def load_csv(path: str | Path, timestamp_col: str = "timestamp") -> tuple[pd.DataFrame, dict]:
    """Load a CSV, validate it, sort chronologically. Returns (df, metadata).

    Never silently drops rows. Raises LoaderError on missing required columns.
    """
    path = Path(path)
    if not path.exists():
        raise LoaderError(f"File not found: {path}")

    df = pd.read_csv(path)

    if timestamp_col not in df.columns:
        raise LoaderError(
            f"Required column '{timestamp_col}' not found. Columns present: {list(df.columns)}"
        )

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    n_before = len(df)
    duplicate_count = df[timestamp_col].duplicated().sum()

    df = df.sort_values(timestamp_col).reset_index(drop=True)

    metadata = {
        "source_path": str(path),
        "n_rows": len(df),
        "n_rows_original": n_before,
        "columns": list(df.columns),
        "duplicate_timestamps": int(duplicate_count),
        "time_range": (str(df[timestamp_col].min()), str(df[timestamp_col].max())) if len(df) else (None, None),
    }
    return df, metadata


def load_dataframe(df: pd.DataFrame, timestamp_col: str = "timestamp") -> tuple[pd.DataFrame, dict]:
    """Same validation path as load_csv, for data already in memory
    (e.g. straight from src/simulator.py) — avoids a pointless CSV round-trip."""
    if timestamp_col not in df.columns:
        raise LoaderError(f"Required column '{timestamp_col}' not found.")
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    duplicate_count = df[timestamp_col].duplicated().sum()
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    metadata = {
        "source_path": "<in-memory>",
        "n_rows": len(df),
        "columns": list(df.columns),
        "duplicate_timestamps": int(duplicate_count),
        "time_range": (str(df[timestamp_col].min()), str(df[timestamp_col].max())) if len(df) else (None, None),
    }
    return df, metadata
