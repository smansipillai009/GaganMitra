"""src/events.py — persistence, hysteresis, and grouping to turn noisy
point-level flags into standard event objects.

Owned by: Role B. Output matches the "suggested event object" shape from
the plan's team-workflow section — this IS the shared contract between
Role B (produces events) and Role D (dashboard consumes events, nothing else).
"""
from __future__ import annotations
import uuid
import pandas as pd


def group_into_events(
    flagged: pd.Series,
    scores: pd.Series,
    min_persistence_rows: int = 60,
    hysteresis_ratio: float = 0.8,
    threshold: pd.Series | None = None,
    gap_tolerance_rows: int = 5,
) -> list[dict]:
    """Group a boolean 'flagged' series into discrete events.

    Persistence: an isolated single-row flag does not become an event unless
    it lasts at least min_persistence_rows.
    Hysteresis: once an event starts, it keeps going even if the score dips
    slightly below the threshold (down to hysteresis_ratio * threshold),
    rather than flickering on/off on noise.
    Gap tolerance: two flagged spans separated by fewer than
    gap_tolerance_rows non-flagged rows get merged into one event.
    """
    events = []
    in_event = False
    start_idx = None
    idx_list = flagged.index.tolist()
    n = len(idx_list)

    i = 0
    while i < n:
        is_flagged = bool(flagged.iloc[i])

        if not in_event and is_flagged:
            in_event = True
            start_idx = i

        elif in_event:
            still_active = is_flagged
            if not still_active and threshold is not None and not pd.isna(threshold.iloc[i]):
                # hysteresis: stay in the event if score is still elevated,
                # even if it dipped below the original threshold
                still_active = scores.iloc[i] > (hysteresis_ratio * threshold.iloc[i])

            if not still_active:
                # check gap tolerance before closing the event
                lookahead = flagged.iloc[i:i + gap_tolerance_rows]
                if lookahead.any():
                    pass  # small gap, keep the event open
                else:
                    end_idx = i - 1
                    duration_rows = end_idx - start_idx + 1
                    if duration_rows >= min_persistence_rows:
                        events.append(_make_event(idx_list, scores, start_idx, end_idx))
                    in_event = False
                    start_idx = None
        i += 1

    if in_event and start_idx is not None:
        end_idx = n - 1
        duration_rows = end_idx - start_idx + 1
        if duration_rows >= min_persistence_rows:
            events.append(_make_event(idx_list, scores, start_idx, end_idx))

    return events


def _make_event(idx_list: list, scores: pd.Series, start_idx: int, end_idx: int) -> dict:
    window_scores = scores.iloc[start_idx:end_idx + 1]
    return {
        "event_id": f"EVT-{uuid.uuid4().hex[:8].upper()}",
        "start_time": str(idx_list[start_idx]),
        "end_time": str(idx_list[end_idx]),
        "duration_rows": end_idx - start_idx + 1,
        "peak_score": float(window_scores.max()),
        "mean_score": float(window_scores.mean()),
        # filled in later by severity.py / explainability.py / alert_bundle.py:
        "severity": None,
        "affected_channels": [],
        "contributing_channels": [],
        "possible_subsystem": None,
        "recommended_investigation": None,
        "model_version": None,
        "preprocessing_version": None,
        "data_provenance": None,
        "mission_id": None,
        "source": None,
    }
