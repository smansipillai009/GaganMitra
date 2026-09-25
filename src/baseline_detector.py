"""src/baseline_detector.py — robust z-score/EWMA baseline detector.

Owned by: Role B. This is the fallback detector — it must stay reliable and
simple, since the 30-hour plan's checkpoints fall back to this if the LSTM
autoencoder (src/lstm_autoencoder.py) isn't stable in time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def robust_zscore_scores(wide_df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Per-channel robust z-score using rolling median/MAD (more outlier-
    resistant than mean/std). Returns a DataFrame of per-channel scores,
    same shape as the numeric part of wide_df."""
    numeric_cols = [c for c in wide_df.columns if not c.endswith("_missing")]
    scores = pd.DataFrame(index=wide_df.index)
    for col in numeric_cols:
        series = wide_df[col]
        median = series.rolling(window, min_periods=window // 2).median()
        mad = (series - median).abs().rolling(window, min_periods=window // 2).median()
        mad = mad.replace(0, np.nan)
        z = 0.6745 * (series - median) / mad
        scores[col] = z.abs()
    return scores


def ewma_scores(wide_df: pd.DataFrame, span: int = 30) -> pd.DataFrame:
    """Alternative/complementary baseline: deviation from an EWMA of each
    channel. Useful alongside robust z-score, not a replacement for it."""
    numeric_cols = [c for c in wide_df.columns if not c.endswith("_missing")]
    scores = pd.DataFrame(index=wide_df.index)
    for col in numeric_cols:
        ewma = wide_df[col].ewm(span=span, adjust=False).mean()
        std = wide_df[col].rolling(span * 2, min_periods=span).std().replace(0, np.nan)
        scores[col] = ((wide_df[col] - ewma) / std).abs()
    return scores


def combined_score(per_channel_scores: pd.DataFrame, method: str = "max") -> pd.Series:
    """Collapse per-channel scores into one combined score per timestamp.
    'max' (any channel badly off = flag) is a reasonable, explainable
    default for a hackathon demo — mean is too easily diluted by many
    near-normal channels."""
    if method == "max":
        return per_channel_scores.max(axis=1)
    elif method == "mean":
        return per_channel_scores.mean(axis=1)
    raise ValueError(f"Unknown method: {method}")
