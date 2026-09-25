"""src/data_quality.py — missingness, duplicates, sampling checks.

Owned by: Role A. Operates on common-schema (long-format) data, since by
this point adapters have already run.
"""
from __future__ import annotations
import pandas as pd


def quality_report(df: pd.DataFrame, timestamp_col: str = "timestamp",
                    channel_col: str = "channel_name", value_col: str = "value") -> dict:
    """One report per call, covering the whole common-schema DataFrame.
    For per-channel detail, call per_channel_report() below."""
    n = len(df)
    n_missing_values = int(df[value_col].isna().sum())
    n_duplicate_ts = int(df.duplicated(subset=[timestamp_col, channel_col]).sum())
    channels = sorted(df[channel_col].dropna().unique().tolist())

    return {
        "n_rows": n,
        "n_channels": len(channels),
        "channels": channels,
        "n_missing_values": n_missing_values,
        "pct_missing": round(100 * n_missing_values / n, 3) if n else 0.0,
        "n_duplicate_timestamp_channel_pairs": n_duplicate_ts,
        "time_range": (str(df[timestamp_col].min()), str(df[timestamp_col].max())) if n else (None, None),
    }


def per_channel_report(df: pd.DataFrame, timestamp_col: str = "timestamp",
                        channel_col: str = "channel_name", value_col: str = "value") -> pd.DataFrame:
    """Per-channel missingness + sampling interval stats — this is what
    should actually get read before trusting a channel in the model."""
    rows = []
    for ch, sub in df.groupby(channel_col):
        sub = sub.sort_values(timestamp_col)
        intervals = sub[timestamp_col].diff().dt.total_seconds().dropna()
        rows.append({
            "channel": ch,
            "n_points": len(sub),
            "pct_missing": round(100 * sub[value_col].isna().mean(), 3),
            "median_interval_s": float(intervals.median()) if len(intervals) else None,
            "irregular_sampling": bool(intervals.std() > 0.5 * intervals.median()) if len(intervals) else None,
        })
    return pd.DataFrame(rows).sort_values("channel").reset_index(drop=True)
