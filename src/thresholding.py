"""src/thresholding.py — Non-Parametric Dynamic Thresholding (NDT) style
moving threshold, per the pitch deck's "Dynamic Thresholding" box.

Owned by: Role B. This is a SIMPLIFIED implementation of the idea behind
NASA's Telemanom NDT method, built for the hackathon timeframe — read the
real Telemanom paper before hackathon day (per the boilerplate plan's Day 1
reading task) and refine this if time allows. The core idea kept here: the
threshold is a moving quantity (mean + k*std of a rolling window of scores),
not a single fixed number, so it adapts as normal conditions naturally shift.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def moving_ewma_bound(scores: pd.Series, span: int = 100, k: float = 3.0,
                       min_periods: int = 30) -> pd.Series:
    """Returns a moving upper bound: EWMA(scores) + k * EWMA_std(scores).
    A score above this bound at a given timestamp is flagged as anomalous.
    This is the "moving bound" that should show up as a non-flat line when
    plotted against reconstruction error — verify that visually before
    trusting it (per the 30-hour plan's Checkpoint 3)."""
    ewma_mean = scores.ewm(span=span, min_periods=min_periods).mean()
    ewma_std = scores.ewm(span=span, min_periods=min_periods).std()
    return ewma_mean + k * ewma_std


def flag_above_threshold(scores: pd.Series, threshold: pd.Series) -> pd.Series:
    """Boolean series: True where scores exceed the (moving) threshold.
    Rows where the threshold itself is still NaN (warm-up period) are
    never flagged — there isn't enough history yet to judge them."""
    return (scores > threshold) & threshold.notna()
