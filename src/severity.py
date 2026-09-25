"""src/severity.py — severity scoring using the formula decided in
configs/config.yaml, NOT invented live.

Owned by: Role B. The formula: a weighted combination of how far the score
exceeds threshold, how long the event persisted, and how many channels were
involved — each weight configurable, so the whole team agrees on the rule
in writing (config file) rather than each person having a different mental
model of it.
"""
from __future__ import annotations
import numpy as np


def compute_severity(
    peak_score: float,
    threshold_at_peak: float,
    duration_rows: int,
    max_expected_duration_rows: int,
    num_channels_involved: int,
    max_expected_channels: int,
    weights: dict,
) -> tuple[str, float]:
    """Returns (severity_label, severity_score in [0,1]).

    Each component is normalized to roughly [0,1] before weighting, so the
    weights in config.yaml are genuinely comparable to each other.
    """
    score_component = min(1.0, max(0.0, (peak_score - threshold_at_peak) / max(threshold_at_peak, 1e-6)))
    persistence_component = min(1.0, duration_rows / max(max_expected_duration_rows, 1))
    channel_component = min(1.0, num_channels_involved / max(max_expected_channels, 1))

    severity_score = (
        weights["score_weight"] * score_component
        + weights["persistence_weight"] * persistence_component
        + weights["channel_count_weight"] * channel_component
    )
    severity_score = float(np.clip(severity_score, 0.0, 1.0))

    thresholds = weights["thresholds"]
    if severity_score >= thresholds["high"]:
        label = "HIGH"
    elif severity_score >= thresholds["medium"]:
        label = "MEDIUM"
    elif severity_score >= thresholds["low"]:
        label = "LOW"
    else:
        label = "MINIMAL"

    return label, severity_score


def explain_severity(label: str, severity_score: float, duration_rows: int,
                      num_channels_involved: int) -> str:
    """One-sentence, judge-ready explanation — this is what gets said out
    loud, not the raw formula. Keep it this simple on purpose."""
    return (
        f"{label} severity (score {severity_score:.2f}): the anomaly persisted for "
        f"{duration_rows} readings and involved {num_channels_involved} channel(s) "
        f"deviating together above the moving threshold."
    )
