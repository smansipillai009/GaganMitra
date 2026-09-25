"""Unit tests for the boilerplate's core modules. These test structure and
logic correctness, not detection quality — calibration/quality is separate,
ongoing work (see pipeline.py's Day-10 smoke test and its documented finding
about baseline detector calibration)."""
import numpy as np
import pandas as pd
import pytest
from src import simulator, data_loader, mission_adapter, windowing, events as events_mod, severity


def test_simulator_reproducible_with_seed():
    a = simulator.generate_normal_telemetry(duration_minutes=10, seed=42)
    b = simulator.generate_normal_telemetry(duration_minutes=10, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_simulator_different_seed_differs():
    a = simulator.generate_normal_telemetry(duration_minutes=10, seed=1)
    b = simulator.generate_normal_telemetry(duration_minutes=10, seed=2)
    assert not a["BAT_V"].equals(b["BAT_V"])


def test_inject_spike_changes_only_target_window():
    normal = simulator.generate_normal_telemetry(duration_minutes=10, seed=42)
    faulted, label = simulator.inject_spike(normal, "BAT_V", at_frac=0.5, magnitude=5.0, duration_s=5)
    # outside the injected window, values should be unchanged
    assert faulted["BAT_V"].iloc[0] == normal["BAT_V"].iloc[0]
    assert len(label) == 1
    assert label.iloc[0]["fault_type"] == "spike"


def test_loader_rejects_missing_timestamp_column():
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(data_loader.LoaderError):
        data_loader.load_dataframe(df)


def test_loader_sorts_chronologically():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-02", "2026-01-01"]),
        "value": [2, 1],
    })
    sorted_df, meta = data_loader.load_dataframe(df)
    assert sorted_df["value"].tolist() == [1, 2]


def test_mission_adapter_produces_required_columns():
    normal = simulator.generate_normal_telemetry(duration_minutes=5, seed=42)
    mc = mission_adapter.load_mission_config("synthetic_demo_sat", missions_dir="configs/missions")
    common = mission_adapter.to_common_schema(normal, mc, satellite_id="test_sat")
    assert list(common.columns) == mission_adapter.REQUIRED_COMMON_COLUMNS
    assert mission_adapter.validate_common_schema(common) == []


def test_windowing_never_shuffles():
    idx = pd.date_range("2026-01-01", periods=100, freq="1s")
    wide = pd.DataFrame({"a": np.arange(100), "b": np.arange(100) * 2}, index=idx)
    windows = windowing.make_windows(wide, sequence_length=10, stride=10)
    # first window's first value should be the earliest data, not shuffled
    assert windows["X"][0][0][0] == 0
    assert windows["X"][1][0][0] == 10  # second window starts where first left off


def test_windowing_raises_if_too_short():
    idx = pd.date_range("2026-01-01", periods=5, freq="1s")
    wide = pd.DataFrame({"a": np.arange(5)}, index=idx)
    with pytest.raises(ValueError):
        windowing.make_windows(wide, sequence_length=10)


def test_group_into_events_finds_persistent_run():
    idx = pd.date_range("2026-01-01", periods=100, freq="1s")
    flagged = pd.Series([False] * 40 + [True] * 35 + [False] * 25, index=idx)
    scores = pd.Series([1.0] * 40 + [5.0] * 35 + [1.0] * 25, index=idx)
    evs = events_mod.group_into_events(flagged, scores, min_persistence_rows=30)
    assert len(evs) == 1
    assert evs[0]["duration_rows"] == 35


def test_group_into_events_ignores_short_blip():
    idx = pd.date_range("2026-01-01", periods=100, freq="1s")
    flagged = pd.Series([False] * 50 + [True] * 3 + [False] * 47, index=idx)
    scores = pd.Series([1.0] * 100, index=idx)
    evs = events_mod.group_into_events(flagged, scores, min_persistence_rows=30)
    assert len(evs) == 0


def test_severity_formula_monotonic_in_score():
    weights = {
        "score_weight": 0.5, "persistence_weight": 0.3, "channel_count_weight": 0.2,
        "thresholds": {"low": 0.3, "medium": 0.6, "high": 0.85},
    }
    _, low_sev = severity.compute_severity(1.1, 1.0, 10, 200, 1, 8, weights)
    _, high_sev = severity.compute_severity(5.0, 1.0, 10, 200, 1, 8, weights)
    assert high_sev > low_sev
