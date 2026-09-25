"""src/windowing.py — chronological sliding windows for the models.

Owned by: Role A/B jointly. NEVER shuffles — time series order matters.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_windows(wide_df: pd.DataFrame, sequence_length: int = 50, stride: int = 1,
                  mode_columns: list[str] | None = None) -> dict:
    """Chronological sliding windows.

    Returns a dict with:
      X: np.ndarray, shape [num_windows, sequence_length, num_channels]
      X_mode: np.ndarray or None, shape [num_windows, sequence_length, num_mode_channels]
              (the telecommand/state channels, kept SEPARATE from telemetry —
              this split is what src/lstm_autoencoder.py's conditioning needs)
      timestamps: list of (start, end) timestamp pairs, one per window
      channel_names: list of str, matches X's last dimension order

    Never shuffles. Drops any window containing NaN (a stricter early
    boilerplate choice — relax this once real gap-handling is tuned).
    """
    feature_cols = [c for c in wide_df.columns
                     if (mode_columns is None or c not in mode_columns) and not c.endswith("_missing")]
    mode_cols = mode_columns or []

    values = wide_df[feature_cols].to_numpy(dtype=float)
    mode_values = wide_df[mode_cols].to_numpy(dtype=float) if mode_cols else None
    timestamps = wide_df.index.to_numpy()

    n = len(wide_df)
    if n < sequence_length:
        raise ValueError(f"Not enough rows ({n}) for sequence_length={sequence_length}")

    X, X_mode, ts_pairs = [], [], []
    for start in range(0, n - sequence_length + 1, stride):
        end = start + sequence_length
        window = values[start:end]
        if np.isnan(window).any():
            continue
        X.append(window)
        ts_pairs.append((timestamps[start], timestamps[end - 1]))
        if mode_values is not None:
            X_mode.append(mode_values[start:end])

    return {
        "X": np.array(X),
        "X_mode": np.array(X_mode) if mode_values is not None else None,
        "timestamps": ts_pairs,
        "channel_names": feature_cols,
    }
