"""src/explainability.py — contributing-channel ranking.

Owned by: Role B. Two methods here:
  1. per_channel_deviation_ranking() — WORKING NOW, uses the same robust
     z-score scores baseline_detector.py already computes. This is the
     honest fallback if Gradient SHAP doesn't make it in time.
  2. gradient_shap_ranking() — STUB. This is the deck's actual promised
     method (Gradient SHAP / Integrated Gradients, triggered only on
     flagged windows). Fill this in once src/lstm_autoencoder.py exists
     and is stable — it needs a trained model to attribute against.

IMPORTANT: never call gradient_shap_ranking() on the full data stream, only
on windows that already breached the dynamic threshold — this is both a
performance necessity and the deck's stated design choice.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def per_channel_deviation_ranking(per_channel_scores: pd.DataFrame, start_time, end_time,
                                   top_k: int = 5) -> list[dict]:
    """Rank channels by their mean deviation score during the event window.
    Working today — no model dependency."""
    window = per_channel_scores.loc[start_time:end_time]
    ranking = window.mean().sort_values(ascending=False).head(top_k)
    return [
        {"channel": ch, "contribution_score": float(val)}
        for ch, val in ranking.items() if not np.isnan(val)
    ]


def gradient_shap_ranking(model, window_tensor, channel_names: list[str], top_k: int = 5) -> list[dict]:
    """STUB — implement once the LSTM autoencoder (src/lstm_autoencoder.py)
    is trained and stable. Intended approach:
      1. Use captum's GradientShap or IntegratedGradients (or a hand-rolled
         single-pass gradient attribution) on model's reconstruction error
         w.r.t. the input window.
      2. Aggregate per-timestep attributions to one score per channel.
      3. Return the same shape as per_channel_deviation_ranking() so callers
         (alert_bundle.py) don't need to know which method produced it.

    Raises NotImplementedError until Role B fills this in — this is
    deliberate: better to fail loudly than silently return fake attributions.
    """
    raise NotImplementedError(
        "gradient_shap_ranking is a stub. Use per_channel_deviation_ranking() "
        "as the working fallback until this is implemented and tested against "
        "a real trained model (see src/lstm_autoencoder.py)."
    )


def rank_contributing_channels(per_channel_scores: pd.DataFrame, start_time, end_time,
                                model=None, window_tensor=None, channel_names=None,
                                top_k: int = 5, prefer_gradient_shap: bool = False) -> tuple[list[dict], str]:
    """Convenience wrapper: tries Gradient SHAP if requested and possible,
    falls back to the deviation-based ranking otherwise. Returns
    (ranking, method_used) — ALWAYS report method_used in the alert bundle
    so nobody accidentally claims SHAP ran when it didn't."""
    if prefer_gradient_shap and model is not None and window_tensor is not None:
        try:
            return gradient_shap_ranking(model, window_tensor, channel_names, top_k), "gradient_shap"
        except NotImplementedError:
            pass
    return per_channel_deviation_ranking(per_channel_scores, start_time, end_time, top_k), "reconstruction_deviation"
