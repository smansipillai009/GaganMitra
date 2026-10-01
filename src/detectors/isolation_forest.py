"""
GaganMitra Multivariate Isolation Forest Detector
Learns multi-channel nominal correlation manifolds and detects subtle multivariate
spacecraft anomalies with channel attribution.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.detectors.base import BaseDetector
from src.schemas import AnomalyScore
from src.preprocessing import RobustTelemetryScaler


class MultivariateIsolationForestDetector(BaseDetector):
    def __init__(
        self,
        mission_config: Dict[str, Any],
        contamination: float = 0.03,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        super().__init__(name="MultivariateIsolationForest", mission_config=mission_config)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = RobustTelemetryScaler(use_mad=True)
        self.model: Optional[IsolationForest] = None
        self.channels: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "MultivariateIsolationForestDetector":
        """Fits multivariate Isolation Forest on scaled nominal telemetry."""
        self.channels = [c for c in train_df.columns if c in self.channels_cfg]
        if not self.channels:
            raise ValueError("No recognized mission channels found in training data.")

        train_scaled = self.scaler.fit(train_df, channels=self.channels).transform(train_df, channels=self.channels)
        X = train_scaled[self.channels].fillna(0.0).values

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        return self

    def score(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[AnomalyScore]]:
        """
        Calculates multivariate anomaly scores and attributes channel contributions.
        """
        if self.model is None:
            raise RuntimeError("Model must be fitted before scoring.")

        channels = [c for c in df.columns if c in self.channels_cfg]
        df_scaled = self.scaler.transform(df, channels=channels)
        X = df_scaled[channels].fillna(0.0).values

        # Isolation Forest decision_function: lower score = more anomalous
        raw_decision = self.model.decision_function(X)
        # Invert and normalize to [0, 1.5]
        # Typically decision_function ranges from -0.3 (extreme outlier) to +0.2 (very nominal)
        # Shift so 0.0 is anomaly boundary, map to normalized score
        multivariate_score = np.clip(-raw_decision * 3.0 + 0.5, 0.0, 2.0)

        df_scores = pd.DataFrame({"timestamp": df["timestamp"]})
        anomaly_scores: List[AnomalyScore] = []

        # Channel contribution attribution via individual feature deviation from median
        deviations = np.abs(X)  # Since X is scaled around median = 0, std = 1
        sum_devs = np.sum(deviations, axis=1, keepdims=True) + 1e-6
        weights = deviations / sum_devs  # relative channel responsibility

        for i, ch in enumerate(channels):
            # Per-channel score is proportional to multivariate score * channel weight
            ch_score = np.clip(multivariate_score * (weights[:, i] * len(channels)), 0.0, 2.0)
            df_scores[ch] = np.round(ch_score, 4)

            subsystem = self.channels_cfg.get(ch, {}).get("subsystem", "unknown")

            for idx, val in enumerate(ch_score):
                if val >= 0.70 or multivariate_score[idx] >= 0.75:
                    anomaly_scores.append(
                        AnomalyScore(
                            timestamp=df["timestamp"].iloc[idx],
                            channel_name=ch,
                            subsystem=subsystem,
                            raw_value=float(df[ch].iloc[idx]),
                            anomaly_score=round(float(val), 4),
                            detector=self.name,
                            is_candidate=bool(val >= 0.70),
                            threshold=0.70,
                        )
                    )

        df_scores["max_anomaly_score"] = np.round(multivariate_score, 4)
        return df_scores, anomaly_scores
