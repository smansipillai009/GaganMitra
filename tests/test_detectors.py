"""
Unit Tests for Detectors
"""

from src.config import get_mission_config
from src.simulator import SpacecraftSimulator
from src.detectors.statistical import RobustStatisticalDetector
from src.detectors.isolation_forest import MultivariateIsolationForestDetector


def test_statistical_detector():
    cfg = get_mission_config("synthetic_eo_sat")
    sim = SpacecraftSimulator()
    df, _ = sim.generate_telemetry(duration_hours=2.0, scenario="power_fault")

    train_df = df.iloc[:int(len(df) * 0.3)]
    det = RobustStatisticalDetector(mission_config=cfg)
    det.fit(train_df)
    df_scores, scores = det.score(df)

    assert "EPS_BATT_V" in df_scores.columns
    assert len(scores) > 0


def test_isolation_forest_detector():
    cfg = get_mission_config("synthetic_eo_sat")
    sim = SpacecraftSimulator()
    df, _ = sim.generate_telemetry(duration_hours=2.0, scenario="power_fault")

    train_df = df.iloc[:int(len(df) * 0.3)]
    det = MultivariateIsolationForestDetector(mission_config=cfg)
    det.fit(train_df)
    df_scores, scores = det.score(df)

    assert len(scores) > 0
