"""src/mission_adapter.py — converts a raw telemetry DataFrame + mission
config into the common schema defined in configs/common_schema.md.

Owned by: Role C. This is THE piece that makes "one engine, many missions"
true. Every source (ESA-ADB, synthetic, future Indian data) must go through
this before touching anything downstream.
"""
from __future__ import annotations
import pandas as pd
import yaml
from pathlib import Path

REQUIRED_COMMON_COLUMNS = [
    "timestamp", "satellite_id", "mission_id", "channel_name",
    "value", "unit", "subsystem", "operating_mode", "criticality", "source",
]


def load_mission_config(mission_id: str, missions_dir: str | Path = "configs/missions") -> dict:
    path = Path(missions_dir) / f"{mission_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No mission config found for '{mission_id}' at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def to_common_schema(
    raw_df: pd.DataFrame,
    mission_config: dict,
    satellite_id: str = "default_sat",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Reshape a wide raw telemetry DataFrame (one column per channel) into
    the long common-schema format (one row per timestamp+channel).

    This is the adapter interface named in the plan:
        load_source(file, config) -> standard_telemetry_dataframe
    split here into load (data_loader.py) + adapt (this function), so the two
    concerns don't get tangled.
    """
    mission_id = mission_config["mission_id"]
    source = mission_config.get("source", "unknown")
    mode_signal_cfg = mission_config.get("mode_signal") or {}
    mode_col = mode_signal_cfg.get("source_name")

    rows = []
    for standard_name, ch_cfg in mission_config.get("channels", {}).items():
        src_col = ch_cfg.get("source_name")
        if src_col is None or src_col not in raw_df.columns:
            # Known-unavailable channel — still worth logging, never silently
            # dropped from the mission config's perspective.
            continue
        sub_df = pd.DataFrame({
            "timestamp": raw_df[timestamp_col],
            "satellite_id": satellite_id,
            "mission_id": mission_id,
            "channel_name": standard_name,
            "value": raw_df[src_col].astype(float),
            "unit": ch_cfg.get("unit", "unknown"),
            "subsystem": ch_cfg.get("subsystem", "unknown"),
            "operating_mode": raw_df[mode_col] if (mode_col and mode_col in raw_df.columns) else None,
            "criticality": ch_cfg.get("criticality", "medium"),
            "source": source,
        })
        rows.append(sub_df)

    if not rows:
        raise ValueError(
            f"No channels from mission config '{mission_id}' matched columns in "
            f"raw_df ({list(raw_df.columns)}). Check source_name values in the "
            f"mission config — this is exactly the kind of thing esa_adb.yaml's "
            f"TODO placeholder needs fixed for before this will work on real data."
        )

    result = pd.concat(rows, ignore_index=True)
    result = result[REQUIRED_COMMON_COLUMNS]
    return result.sort_values("timestamp").reset_index(drop=True)


def validate_common_schema(df: pd.DataFrame) -> list[str]:
    """Returns a list of problems found (empty list = valid). Cheap
    stand-in for Pydantic validation — upgrade to Pydantic later only if
    this becomes unwieldy, per the requirements.txt note."""
    problems = []
    missing_cols = [c for c in REQUIRED_COMMON_COLUMNS if c not in df.columns]
    if missing_cols:
        problems.append(f"Missing required columns: {missing_cols}")
        return problems  # can't check further without the columns
    if df["value"].isna().all():
        problems.append("All 'value' entries are null — adapter likely misconfigured")
    if df["channel_name"].isna().any():
        problems.append("Some rows have null channel_name")
    return problems
