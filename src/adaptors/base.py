"""
GaganMitra Adapter Base Class
Defines the standard contract for converting any mission or ground station
telemetry format into GaganMitra's normalized multi-channel schema.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from src.config import get_mission_config


class BaseTelemetryAdapter(ABC):
    def __init__(self, mission_id: str):
        self.mission_id = mission_id
        self.mission_config = get_mission_config(mission_id)
        self.channels_cfg = self.mission_config.get("channels", {})

    @abstractmethod
    def load_telemetry(self, source: Any) -> pd.DataFrame:
        """Loads raw telemetry from CSV, file path, dict, or DataFrame."""
        pass

    def standardize(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Maps raw mission telemetry into the standard schema:
        - Ensures 'timestamp' is parsed as datetime
        - Maps raw channel names to standard definitions
        - Returns (standardized_df, mission_metadata)
        """
        df = raw_df.copy()
        if "timestamp" not in df.columns:
            # Check for alternative timestamp column names
            time_cols = [c for c in df.columns if c.lower() in ["time", "datetime", "utc", "date_time", "epoch"]]
            if time_cols:
                df = df.rename(columns={time_cols[0]: "timestamp"})
            else:
                raise ValueError(f"Could not locate a timestamp column in telemetry: {list(df.columns)}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Retain known channels defined in mission config
        available_channels = [c for c in df.columns if c in self.channels_cfg]
        if not available_channels:
            # Try matching by standard_name or lower-case
            name_map = {}
            for ch_key, ch_val in self.channels_cfg.items():
                for col in df.columns:
                    if col == ch_key or col.lower() == ch_key.lower() or col == ch_val.get("standard_name"):
                        name_map[col] = ch_key
            if name_map:
                df = df.rename(columns=name_map)
                available_channels = [c for c in df.columns if c in self.channels_cfg]

        metadata = {
            "mission_id": self.mission_id,
            "spacecraft_name": self.mission_config.get("spacecraft_name", self.mission_id),
            "channels": {ch: self.channels_cfg[ch] for ch in available_channels},
            "channel_count": len(available_channels),
            "start_time": str(df["timestamp"].min()),
            "end_time": str(df["timestamp"].max()),
            "record_count": len(df),
        }

        # Keep timestamp + valid telemetry channels
        keep_cols = ["timestamp"] + available_channels
        df_standard = df[keep_cols]

        return df_standard, metadata
