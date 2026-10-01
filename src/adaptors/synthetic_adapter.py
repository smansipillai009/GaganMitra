"""
GaganMitra Synthetic Mission Adapter
Ingests telemetry from the simulator or saved CSV files and adapts it
to the common schema.
"""

from pathlib import Path
from typing import Any, Tuple, Dict
import pandas as pd
from src.adapters.base import BaseTelemetryAdapter


class SyntheticMissionAdapter(BaseTelemetryAdapter):
    def __init__(self, mission_id: str = "synthetic_eo_sat"):
        super().__init__(mission_id=mission_id)

    def load_telemetry(self, source: Any) -> pd.DataFrame:
        """Accepts either a pandas DataFrame or path to a CSV file."""
        if isinstance(source, pd.DataFrame):
            return source.copy()
        elif isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"Telemetry source file not found: {p}")
            return pd.read_csv(p)
        else:
            raise TypeError(f"Unsupported telemetry source type: {type(source)}")
