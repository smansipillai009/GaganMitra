"""
GaganMitra Detector Base Class
Defines the standard interface for all anomaly detection algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pandas as pd
from src.schemas import AnomalyScore


class BaseDetector(ABC):
    def __init__(self, name: str, mission_config: Dict[str, Any]):
        self.name = name
        self.mission_config = mission_config
        self.channels_cfg = mission_config.get("channels", {})

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseDetector":
        """Calibrates/trains detector on nominal or historical telemetry."""
        pass

    @abstractmethod
    def score(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[AnomalyScore]]:
        """
        Calculates anomaly scores for each timestamp and channel.
        Returns:
            df_scores: DataFrame with timestamp and channel anomaly scores.
            anomaly_scores: Flat list of AnomalyScore records.
        """
        pass
