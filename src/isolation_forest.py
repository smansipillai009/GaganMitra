"""src/isolation_forest.py — thin wrapper around scikit-learn's Isolation
Forest, kept as its own file per the plan's module list.

Owned by: Role B.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(train_wide_df: pd.DataFrame, contamination: float = 0.02,
                          random_state: int = 42) -> IsolationForest:
    numeric_cols = [c for c in train_wide_df.columns if not c.endswith("_missing")]
    clean = train_wide_df[numeric_cols].dropna()
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(clean)
    return model


def score(model: IsolationForest, wide_df: pd.DataFrame) -> pd.Series:
    """Returns an anomaly score per timestamp (higher = more anomalous —
    note sklearn's raw decision_function is inverted, so we flip the sign
    here to keep the convention consistent with baseline_detector.py)."""
    numeric_cols = [c for c in wide_df.columns if not c.endswith("_missing")]
    valid = wide_df[numeric_cols].dropna()
    raw_scores = -model.decision_function(valid)
    return pd.Series(raw_scores, index=valid.index).reindex(wide_df.index)
