"""src/evaluation.py — the metrics named throughout the plan, all in one
place so no slide number goes unmeasured.

Owned by: Role B. Chronological split is enforced by the CALLER (train.py) —
this module just computes metrics on whatever arrays it's given.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def point_level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred: boolean arrays, same length, same index alignment."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def event_level_f1(true_events: list[tuple], pred_events: list[dict], overlap_required: float = 0.1) -> dict:
    """true_events: list of (start_time, end_time) ground-truth intervals.
    pred_events: list of event dicts from src/events.py (has start_time/end_time).
    A predicted event 'catches' a true event if their time ranges overlap by
    at least overlap_required fraction of the true event's duration — this
    is intentionally lenient (matching the exact boundary isn't the point;
    catching the event at all is)."""
    matched_true = set()
    matched_pred = set()
    for ti, (t_start, t_end) in enumerate(true_events):
        t_start, t_end = pd.Timestamp(t_start), pd.Timestamp(t_end)
        true_dur = max((t_end - t_start).total_seconds(), 1e-6)
        for pi, pe in enumerate(pred_events):
            p_start, p_end = pd.Timestamp(pe["start_time"]), pd.Timestamp(pe["end_time"])
            overlap_start = max(t_start, p_start)
            overlap_end = min(t_end, p_end)
            overlap_s = max((overlap_end - overlap_start).total_seconds(), 0)
            if overlap_s / true_dur >= overlap_required:
                matched_true.add(ti)
                matched_pred.add(pi)

    n_true, n_pred = len(true_events), len(pred_events)
    tp = len(matched_true)
    recall = tp / n_true if n_true else 0.0
    precision = tp / n_pred if n_pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "event_precision": precision, "event_recall": recall, "event_f1": f1,
        "n_true_events": n_true, "n_predicted_events": n_pred, "n_matched": tp,
    }


def false_alarms_per_day(pred_events: list[dict], true_events: list[tuple],
                          total_duration_days: float, overlap_required: float = 0.1) -> float:
    """Predicted events that don't overlap any true event, normalized by
    the total time span evaluated."""
    n_false = 0
    for pe in pred_events:
        p_start, p_end = pd.Timestamp(pe["start_time"]), pd.Timestamp(pe["end_time"])
        matched_any = False
        for (t_start, t_end) in true_events:
            t_start, t_end = pd.Timestamp(t_start), pd.Timestamp(t_end)
            overlap_start = max(t_start, p_start)
            overlap_end = min(t_end, p_end)
            if (overlap_end - overlap_start).total_seconds() > 0:
                matched_any = True
                break
        if not matched_any:
            n_false += 1
    return n_false / max(total_duration_days, 1e-6)


def detection_delay(true_events: list[tuple], pred_events: list[dict]) -> float | None:
    """Mean delay, in seconds, between a true event's start and the earliest
    predicted event that overlaps it. Returns None if no matches."""
    delays = []
    for (t_start, t_end) in true_events:
        t_start = pd.Timestamp(t_start)
        candidates = [pd.Timestamp(pe["start_time"]) for pe in pred_events
                      if pd.Timestamp(pe["start_time"]) <= pd.Timestamp(t_end)
                      and pd.Timestamp(pe["end_time"]) >= t_start]
        if candidates:
            earliest = min(candidates)
            delays.append(max((earliest - t_start).total_seconds(), 0))
    return float(np.mean(delays)) if delays else None


def measure_runtime(fn, *args, **kwargs) -> tuple:
    """Wraps any function call, returns (result, elapsed_seconds). Use this
    around detect() calls to get the 'runtime per window' numbers the deck
    promises to report."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
