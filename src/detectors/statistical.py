"""
GaganMitra Robust Statistical Anomaly Detector
Leverages Median Absolute Deviation (MAD), Exponentially Weighted Moving Average (EWMA),
and dynamic rolling z-scores to detect telemetry excursions with high explainability.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from src.detectors.base import BaseDetector
from src.schemas import AnomalyScore


class RobustStatisticalDetector(BaseDetector):
    def __init__(
        self,
        mission_config: Dict[str, Any],
        window_size: int = 30,
        z_threshold: float = 3.2,
        use_mad: bool = True,
        ewma_alpha: float = 0.15
    ):
        super().__init__(name="RobustStatisticalDetector", mission_config=mission_config)
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.use_mad = use_mad
        self.ewma_alpha = ewma_alpha
        self.nominal_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, train_df: pd.DataFrame) -> "RobustStatisticalDetector":
        """Learns baseline nominal centers and robust dispersion per channel."""
        channel_names = [c for c in train_df.columns if c in self.channels_cfg]

        for ch in channel_names:
            series = train_df[ch].dropna().astype(float)
            med = float(series.median())
            if self.use_mad:
                mad = float((series - med).abs().median())
                scale = 1.4826 * mad if mad > 1e-5 else 1.0
            else:
                scale = float(series.std()) if float(series.std()) > 1e-5 else 1.0

            self.nominal_stats[ch] = {
                "median": med,
                "scale": scale,
                "min": float(series.min()),
                "max": float(series.max())
            }

        return self

    def score(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[AnomalyScore]]:
        """
        Calculates per-channel rolling deviation scores.
        A score of 1.0 means exact threshold reached; >1.0 indicates severe anomaly.
        """
        df_scores = pd.DataFrame({"timestamp": df["timestamp"]})
        anomaly_scores: List[AnomalyScore] = []
        channel_names = [c for c in df.columns if c in self.channels_cfg]

        for ch in channel_names:
            series = df[ch].astype(float)
            stats = self.nominal_stats.get(ch, {
                "median": float(series.median()),
                "scale": max(1e-5, float(series.std())),
            })

            # Calculate EWMA to smooth high-frequency telemetry jitter
            ewma = series.ewm(alpha=self.ewma_alpha, adjust=False).mean()

            # Dynamic rolling median and rolling MAD
            roll_med = series.rolling(window=self.window_size, min_periods=5).median().fillna(stats["median"])
            roll_abs_dev = (series - roll_med).abs()

            # Physical scale floor from nominal range to avoid tiny-denominator noise blowup
            ch_cfg = self.channels_cfg.get(ch, {})
            nom_min = ch_cfg.get("nominal_min")
            nom_max = ch_cfg.get("nominal_max")
            min_scale_floor = 1e-4
            if nom_min is not None and nom_max is not None:
                min_scale_floor = max(min_scale_floor, 0.05 * (nom_max - nom_min))
            min_scale_floor = max(min_scale_floor, 0.05 * stats["scale"])

            roll_mad = roll_abs_dev.rolling(window=self.window_size, min_periods=5).median().fillna(stats["scale"])
            roll_mad = np.maximum(roll_mad, min_scale_floor)

            # Robust Z-score: combination of local window deviation + baseline drift
            local_z = (series - roll_med).abs() / (1.4826 * roll_mad)
            baseline_z = (ewma - stats["median"]).abs() / max(stats["scale"], min_scale_floor)

            # If within nominal min/max, suppress false baseline alarms caused by normal orbital cycles.
            # IMPORTANT: only do this for non-critical channels. For critical channels (e.g. battery
            # voltage/current), a slow drift can stay fully inside the physical nominal_min/max band for
            # most of its ramp (that's exactly what a "gradual degradation" fault looks like), so dampening
            # baseline_z here previously delayed detection until the very end of the fault window instead
            # of catching it early. See tests/test_review_fixes.py::test_slow_drift_detected_early.
            criticality = ch_cfg.get("criticality", "medium")
            in_nominal_bounds = pd.Series(True, index=series.index)
            if nom_min is not None and nom_max is not None and criticality not in ("critical", "high"):
                in_nominal_bounds = (series >= nom_min) & (series <= nom_max)
                # Dampen baseline_z when comfortably within operational range
                baseline_z = np.where(in_nominal_bounds, baseline_z * 0.35, baseline_z)

            # Blended score: local sudden excursion + baseline drift
            blended_z = 0.7 * local_z + 0.3 * baseline_z

            # Normalize so threshold maps to 0.70; 1.0+ is definite anomaly
            normalized_score = np.clip(blended_z / self.z_threshold * 0.7, 0.0, 2.0)
            df_scores[ch] = normalized_score

            subsystem = self.channels_cfg.get(ch, {}).get("subsystem", "unknown")

            for idx, val in enumerate(normalized_score):
                is_cand = val >= 0.70
                if is_cand or val >= 0.50:  # record suspicious and anomalous timestamps
                    anomaly_scores.append(
                        AnomalyScore(
                            timestamp=df["timestamp"].iloc[idx],
                            channel_name=ch,
                            subsystem=subsystem,
                            raw_value=float(series.iloc[idx]),
                            anomaly_score=round(float(val), 4),
                            detector=self.name,
                            is_candidate=bool(is_cand),
                            threshold=0.70,
                        )
                    )

        # Compute aggregate max score across all channels per timestamp
        score_cols = [c for c in df_scores.columns if c != "timestamp"]
        df_scores["max_anomaly_score"] = df_scores[score_cols].max(axis=1)

        return df_scores, anomaly_scores
