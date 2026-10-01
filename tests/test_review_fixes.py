"""
Tests for the fixes made during code review:
1. Real, data-driven contributing-channel ranking (not canned templates).
2. Mission-agnostic mode-awareness (works on ESA-ADB config too, not just
   hardcoded synthetic channel names).
3. The genuinely moving dynamic threshold option in thresholding.py.
"""
from run_pipeline import run_pipeline
from src.thresholding import compute_moving_threshold
from src.event_manager import EventManager
import pandas as pd
import numpy as np


def test_events_have_real_contributing_channel_ranking():
    _, _, events, _ = run_pipeline(
        source="synthetic", scenario="power_fault", detector_type="statistical",
        output_dir="/tmp/gm_test_output_1",
    )
    assert len(events) >= 1
    for evt in events:
        contributing = evt.metadata.get("contributing_channels", [])
        assert evt.metadata.get("attribution_method") == "max_score_ranking"
        assert len(contributing) == len(evt.channels)
        # must be sorted descending by contribution_score
        scores = [c["contribution_score"] for c in contributing]
        assert scores == sorted(scores, reverse=True)
        # every entry has both the raw channel id and its standardized name
        for c in contributing:
            assert "channel" in c and "standard_name" in c and "contribution_score" in c


def test_events_have_no_leftover_root_cause_wording():
    _, _, events, _ = run_pipeline(
        source="synthetic", scenario="power_fault", detector_type="statistical",
        output_dir="/tmp/gm_test_output_2",
    )
    for evt in events:
        assert "root_cause_hypothesis" not in evt.metadata
        # subsystem_hypothesis is the renamed, safer-worded key (added by
        # ExplainabilityEngine downstream of EventManager in run_pipeline)


def test_mode_awareness_generalizes_to_esa_config_without_crashing():
    """The old version hardcoded EPS_SOLAR_I/EPS_BATT_I — literal strings that
    only exist in the synthetic mission config. This confirms the new
    standard_name-based version runs cleanly against the ESA-ADB config too,
    even though ESA-ADB has no solar-current-equivalent channel at all (so
    the suppression simply never has anything to suppress there — safe,
    not a crash)."""
    _, _, events, _ = run_pipeline(
        source="esa", scenario="power_fault", detector_type="statistical",
        output_dir="/tmp/gm_test_output_3",
    )
    assert len(events) >= 1  # pipeline still runs end-to-end on the ESA path


def test_mode_awareness_still_suppresses_real_diurnal_case_on_synthetic():
    """Direct unit test of the suppression logic itself, independent of the
    full pipeline, using the synthetic mission's real channel config."""
    import yaml
    with open("configs/missions/synthetic_eo_sat.yaml") as f:
        mission_config = yaml.safe_load(f)

    mgr = EventManager(mission_config=mission_config, anomaly_threshold=0.5)

    # Find the actual raw channel IDs for battery_current / solar_array_current
    battery_current_ch = next(
        ch for ch, cfg in mission_config["channels"].items()
        if cfg["standard_name"] == "battery_current"
    )
    solar_current_ch = next(
        ch for ch, cfg in mission_config["channels"].items()
        if cfg["standard_name"] == "solar_array_current"
    )

    n = 200
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1s")
    raw_df = pd.DataFrame({"timestamp": timestamps})
    df_scores = pd.DataFrame({"timestamp": timestamps})
    for ch in mission_config["channels"]:
        raw_df[ch] = 1.0
        df_scores[ch] = 0.0

    # Inject a brief (60s), low-criticality current blip — should be suppressed
    df_scores.loc[50:56, battery_current_ch] = 0.9
    df_scores.loc[50:56, solar_current_ch] = 0.9
    raw_df[battery_current_ch] = 1.0
    raw_df[solar_current_ch] = 1.0

    events = mgr.correlate_events(raw_df, df_scores)
    assert events == [], "Expected the brief low-criticality current blip to be suppressed as a normal terminator transition"


def test_compute_moving_threshold_actually_moves():
    """Confirms the new dynamic threshold is genuinely non-flat, addressing
    the review finding that the old 'dynamic thresholding' claim was really
    a fixed 0.70 cutoff."""
    idx = pd.date_range("2026-01-01", periods=300, freq="1s")
    # scores that drift upward over time, well above any floor so the floor
    # clip doesn't mask whether the EWMA computation itself is moving
    scores = pd.Series(np.linspace(0.5, 3.0, 300) + np.random.default_rng(0).normal(0, 0.05, 300), index=idx)
    bound = compute_moving_threshold(scores, span=50, k=2.0, min_periods=10, floor=0.0)
    # the bound should not be a single constant value once past the warm-up period
    assert bound.iloc[100:].nunique() > 1
    # and it should trend upward along with the drifting scores, not stay flat
    assert bound.iloc[-1] > bound.iloc[100]


def test_real_esa_data_path_runs_without_crashing():
    """The esa-real path is honestly documented as under-detecting right now
    (see README's Real-Data Validation Status). This test only confirms it
    runs end-to-end without crashing — NOT that detection quality is good.
    If this starts failing, something broke the real-data ingestion itself,
    which is a different, more urgent problem than the known detection gap."""
    from run_pipeline import run_pipeline
    clean_df, df_scores, events, eval_res = run_pipeline(
        source="esa-real", scenario="power_fault", detector_type="statistical",
        output_dir="/tmp/gm_test_output_esa_real",
    )
    assert len(clean_df) > 0
    assert clean_df.shape[1] >= 2  # timestamp + at least one real channel
    # deliberately NOT asserting on event_recall/precision — see docstring
