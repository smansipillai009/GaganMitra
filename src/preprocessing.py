"""src/preprocessing.py — scaling and missingness handling.

Owned by: Role A. Critical rule (called out repeatedly in the plan): fit any
scaler on TRAINING data only, never on validation/test, or you get silent
leakage that inflates your metrics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


def wide_pivot(common_df: pd.DataFrame, timestamp_col: str = "timestamp",
               channel_col: str = "channel_name", value_col: str = "value") -> pd.DataFrame:
    """Common-schema long format -> wide format (one column per channel),
    which is what windowing.py and the models actually need."""
    wide = common_df.pivot_table(index=timestamp_col, columns=channel_col,
                                  values=value_col, aggfunc="first")
    return wide.sort_index()


def add_missingness_indicators(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Adds one <channel>_missing boolean column per channel BEFORE filling,
    so the model can (optionally) learn from the missingness pattern itself,
    not just have it silently interpolated away."""
    out = wide_df.copy()
    for col in wide_df.columns:
        out[f"{col}_missing"] = wide_df[col].isna()
    return out


def forward_fill_short_gaps(wide_df: pd.DataFrame, max_gap_rows: int = 5) -> pd.DataFrame:
    """Forward-fill gaps up to max_gap_rows; longer gaps are left as NaN
    rather than silently interpolated across a large missing-data burst."""
    out = wide_df.copy()
    for col in [c for c in wide_df.columns if not c.endswith("_missing")]:
        out[col] = wide_df[col].ffill(limit=max_gap_rows)
    return out


class TrainOnlyScaler:
    """Thin wrapper enforcing the 'fit on train only' rule at the type level —
    the fit_transform / transform split makes it hard to accidentally fit on
    validation or test data by mistake."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.columns_ = None

    def fit(self, train_wide_df: pd.DataFrame) -> "TrainOnlyScaler":
        numeric_cols = [c for c in train_wide_df.columns if not c.endswith("_missing")]
        self.columns_ = numeric_cols
        self.mean_ = train_wide_df[numeric_cols].mean()
        self.std_ = train_wide_df[numeric_cols].std().replace(0, 1.0)
        return self

    def transform(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fit yet — call .fit() on TRAINING data first")
        out = wide_df.copy()
        out[self.columns_] = (wide_df[self.columns_] - self.mean_) / self.std_
        return out

    def save(self, path: str | Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "TrainOnlyScaler":
        return joblib.load(path)
