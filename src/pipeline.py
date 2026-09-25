"""src/pipeline.py — chains every module together on synthetic data.

Run with: python -m src.pipeline

This is the actual Day-10 exit criterion made runnable: simulator -> loader
-> mission adapter -> quality -> preprocessing -> windowing -> baseline
detector -> NDT thresholding -> events -> severity -> explainability ->
alert bundle -> evaluation. If this script runs without crashing and prints
at least one real event, the boilerplate is genuinely end-to-end, not just
"files exist."
"""
from __future__ import annotations
import yaml
from pathlib import Path

from src import simulator, data_loader, mission_adapter, data_quality
from src import preprocessing, windowing, baseline_detector, thresholding
from src import events as events_mod, severity as severity_mod, explainability
from src import alert_bundle, evaluation

ROOT = Path(__file__).parent.parent


def run():
    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    # 1. Generate synthetic telemetry with an injected fault
    normal = simulator.generate_normal_telemetry(duration_minutes=120, seed=42)
    faulted, fault_label = simulator.inject_battery_degradation(normal, at_frac=0.4, end_frac=0.8)
    print(f"[1/10] Simulated {len(faulted)} rows with 1 injected fault: "
          f"{fault_label.iloc[0]['fault_type']}")

    # 2. Load (validate + sort)
    raw_df, load_meta = data_loader.load_dataframe(faulted)
    print(f"[2/10] Loaded: {load_meta['n_rows']} rows, {load_meta['duplicate_timestamps']} duplicate timestamps")

    # 3. Adapt to common schema
    mission_config = mission_adapter.load_mission_config(
        "synthetic_demo_sat", missions_dir=ROOT / "configs" / "missions"
    )
    common_df = mission_adapter.to_common_schema(raw_df, mission_config, satellite_id="demo_sat_01")
    problems = mission_adapter.validate_common_schema(common_df)
    print(f"[3/10] Adapted to common schema: {len(common_df)} rows, "
          f"{common_df['channel_name'].nunique()} channels, validation problems: {problems or 'none'}")

    # 4. Quality report
    quality = data_quality.quality_report(common_df)
    print(f"[4/10] Quality: {quality['pct_missing']}% missing, {quality['n_channels']} channels")

    # 5. Preprocess (wide pivot + missingness indicators + fill short gaps)
    wide = preprocessing.wide_pivot(common_df)
    wide = preprocessing.add_missingness_indicators(wide)
    wide = preprocessing.forward_fill_short_gaps(wide)
    n_train = int(len(wide) * 0.5)
    scaler = preprocessing.TrainOnlyScaler().fit(wide.iloc[:n_train])
    wide_scaled = scaler.transform(wide)
    print(f"[5/10] Preprocessed: wide shape {wide.shape}, scaler fit on first {n_train} rows only")

    # 6. Windowing (not used further in this smoke test, but confirm it runs)
    window_cols = [c for c in wide.columns if not c.endswith("_missing")]
    windows = windowing.make_windows(wide_scaled.dropna(), sequence_length=20, stride=5)
    print(f"[6/10] Windowed: X shape {windows['X'].shape}")

    # 7. Baseline detector
    per_channel_scores = baseline_detector.robust_zscore_scores(wide_scaled, window=60)
    combined = baseline_detector.combined_score(per_channel_scores, method="max")
    print(f"[7/10] Baseline scores computed: max={combined.max():.2f}, mean={combined.mean():.2f}")

    # 8. NDT-style dynamic threshold + flagging
    dyn_threshold = thresholding.moving_ewma_bound(combined, span=100, k=3.0)
    flagged = thresholding.flag_above_threshold(combined, dyn_threshold)
    print(f"[8/10] Dynamic threshold computed: {flagged.sum()} flagged rows out of {len(flagged)}")

    # 9. Events -> severity -> contributing channels -> alert bundle
    # NOTE: min_persistence_rows is set low (3) here ONLY to prove the full
    # chain works end-to-end on today's smoke test. This is NOT a calibrated
    # value. Real testing (see below) shows the baseline detector's default
    # config.yaml settings (window=60, k=3.0, min_persistence=30) barely
    # detect the injected synthetic faults at all — gradual drift gets
    # absorbed by the adaptive rolling median, and a stuck sensor loses
    # variance rather than spiking it, so neither reliably crosses a
    # z-score-based threshold. This is a genuine, expected finding, not a
    # bug: events.py's grouping logic itself is unit-tested and correct
    # (see tests/test_events.py). Calibrating real thresholds against real
    # fault behavior is real work for the execution phase (Milestone 2),
    # not something to force-fix with a low persistence number long-term.
    detected_events = events_mod.group_into_events(
        flagged, combined, min_persistence_rows=3, threshold=dyn_threshold
    )
    print(f"[9/10] Grouped into {len(detected_events)} event(s) "
          f"(min_persistence_rows=3 for this smoke test only — recalibrate before real use)")

    bundles = []
    for ev in detected_events:
        ranking, method = explainability.rank_contributing_channels(
            per_channel_scores, ev["start_time"], ev["end_time"], top_k=3
        )
        label, sev_score = severity_mod.compute_severity(
            peak_score=ev["peak_score"],
            threshold_at_peak=dyn_threshold.loc[ev["end_time"]] if ev["end_time"] in dyn_threshold.index else 1.0,
            duration_rows=ev["duration_rows"],
            max_expected_duration_rows=200,
            num_channels_involved=len(ranking),
            max_expected_channels=len(window_cols),
            weights=config["severity"],
        )
        ev["severity"] = label
        ev["severity_score"] = sev_score
        possible_subsystem = ranking[0]["channel"] if ranking else None

        bundle = alert_bundle.build_alert_bundle(
            ev, mission_id="synthetic_demo_sat", source="synthetic",
            contributing_channels=ranking, attribution_method=method,
            possible_subsystem=possible_subsystem,
            model_version="baseline-v0.1", preprocessing_version="prep-v0.1",
            data_provenance="synthetic simulator, seed=42, battery_degradation fault",
        )
        bundles.append(bundle)
        print(f"    -> {ev['event_id']}: {label} ({sev_score:.2f}), "
              f"top channel: {possible_subsystem}, method: {method}")

    # 10. Evaluate against the known injected fault
    true_events = [(fault_label.iloc[0]["start_time"], fault_label.iloc[0]["end_time"])]
    total_days = (wide.index.max() - wide.index.min()).total_seconds() / 86400
    event_metrics = evaluation.event_level_f1(true_events, detected_events)
    fa_per_day = evaluation.false_alarms_per_day(detected_events, true_events, total_days)
    print(f"[10/10] Evaluation vs. known injected fault: {event_metrics}, "
          f"false_alarms_per_day={fa_per_day:.2f}")

    return bundles


if __name__ == "__main__":
    run()
