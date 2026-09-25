"""src/simulator.py — synthetic Indian-style spacecraft telemetry generator.

Owned by: Role B (ML/evaluation), per the plan's "simulator fault logic" line.
Produces two things: a normal telemetry DataFrame and a fault-label DataFrame
(ground truth for evaluating whether the detector actually catches what we
injected). Uses a fixed seed so results are reproducible across runs and
across teammates' machines.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_SEED = 42
DEFAULT_DURATION_MIN = 240  # 4 hours of 1-second telemetry
DEFAULT_FREQ_S = 1


def generate_normal_telemetry(
    duration_minutes: int = DEFAULT_DURATION_MIN,
    freq_seconds: int = DEFAULT_FREQ_S,
    seed: int = DEFAULT_SEED,
    start_time: str = "2026-01-01T00:00:00",
) -> pd.DataFrame:
    """Generate normal (fault-free) synthetic telemetry.

    Returns a DataFrame with columns matching synthetic_demo_sat.yaml's
    source_name values: timestamp, BAT_V, BAT_I, BAT_TEMP, BUS_I, TEMP_INT,
    WHEEL_RPM, ATT_ERR, COMM_TEMP, OP_MODE.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_minutes * 60 / freq_seconds)
    t = pd.date_range(start=start_time, periods=n, freq=f"{freq_seconds}s")

    # Simple orbit-driven eclipse pattern: ~90 min period, ~35 min eclipse
    orbit_phase = (np.arange(n) * freq_seconds / 60) % 90
    in_eclipse = orbit_phase < 35
    op_mode = np.where(in_eclipse, "eclipse", "nominal")

    # Battery drains slightly in eclipse, recharges in sunlight — normal behavior
    bat_v = 28.0 + 0.3 * np.sin(np.arange(n) / 500) - 0.5 * in_eclipse + rng.normal(0, 0.02, n)
    bat_i = 2.4 + 0.3 * in_eclipse + rng.normal(0, 0.05, n)
    bat_temp = 22.0 + 0.5 * np.sin(np.arange(n) / 3000) + rng.normal(0, 0.1, n)
    bus_i = 1.8 + 0.2 * in_eclipse + rng.normal(0, 0.05, n)
    temp_int = 21.0 + 1.0 * np.sin(np.arange(n) / 4000) + rng.normal(0, 0.15, n)
    wheel_rpm = 3100 + 50 * np.sin(np.arange(n) / 2000) + rng.normal(0, 5, n)
    att_err = np.abs(rng.normal(0, 0.05, n))
    comm_temp = 18.0 + rng.normal(0, 0.2, n)

    return pd.DataFrame({
        "timestamp": t,
        "BAT_V": bat_v, "BAT_I": bat_i, "BAT_TEMP": bat_temp, "BUS_I": bus_i,
        "TEMP_INT": temp_int, "WHEEL_RPM": wheel_rpm, "ATT_ERR": att_err,
        "COMM_TEMP": comm_temp, "OP_MODE": op_mode,
    })


# ---------------------------------------------------------------------------
# Fault injection — each function mutates a copy of the telemetry and returns
# (telemetry_with_fault, fault_label_df) where fault_label_df has columns:
# start_time, end_time, fault_type, affected_channels, subsystem.
# ---------------------------------------------------------------------------

def inject_spike(df: pd.DataFrame, channel: str, at_frac: float = 0.5,
                  magnitude: float = 5.0, duration_s: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    n = len(df)
    idx = int(n * at_frac)
    end_idx = min(idx + duration_s, n)
    df.loc[idx:end_idx, channel] += magnitude
    label = pd.DataFrame([{
        "start_time": df["timestamp"].iloc[idx], "end_time": df["timestamp"].iloc[end_idx],
        "fault_type": "spike", "affected_channels": [channel], "subsystem": None,
    }])
    return df, label


def inject_drift(df: pd.DataFrame, channel: str, at_frac: float = 0.4,
                  end_frac: float = 0.9, total_drift: float = -3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gradual linear drift — the classic 'slow battery degradation' scenario."""
    df = df.copy()
    n = len(df)
    start_idx, end_idx = int(n * at_frac), int(n * end_frac)
    ramp = np.linspace(0, total_drift, end_idx - start_idx)
    df.loc[start_idx:end_idx - 1, channel] += ramp
    df.loc[end_idx:, channel] += total_drift
    label = pd.DataFrame([{
        "start_time": df["timestamp"].iloc[start_idx], "end_time": df["timestamp"].iloc[end_idx - 1],
        "fault_type": "drift", "affected_channels": [channel], "subsystem": None,
    }])
    return df, label


def inject_stuck(df: pd.DataFrame, channel: str, at_frac: float = 0.5,
                  duration_s: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    n = len(df)
    idx = int(n * at_frac)
    end_idx = min(idx + duration_s, n)
    stuck_value = df[channel].iloc[idx]
    df.loc[idx:end_idx, channel] = stuck_value
    label = pd.DataFrame([{
        "start_time": df["timestamp"].iloc[idx], "end_time": df["timestamp"].iloc[end_idx],
        "fault_type": "stuck", "affected_channels": [channel], "subsystem": None,
    }])
    return df, label


def inject_dropout(df: pd.DataFrame, channel: str, at_frac: float = 0.6,
                    duration_s: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    n = len(df)
    idx = int(n * at_frac)
    end_idx = min(idx + duration_s, n)
    df.loc[idx:end_idx, channel] = np.nan
    label = pd.DataFrame([{
        "start_time": df["timestamp"].iloc[idx], "end_time": df["timestamp"].iloc[end_idx],
        "fault_type": "dropout", "affected_channels": [channel], "subsystem": None,
    }])
    return df, label


def inject_battery_degradation(df: pd.DataFrame, at_frac: float = 0.5,
                                end_frac: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-channel fault: voltage drifts down, current gets noisier, temp rises.
    This is the demo scenario named in the plan's 'Demo Story' section."""
    df = df.copy()
    n = len(df)
    start_idx, end_idx = int(n * at_frac), int(n * end_frac)
    span = end_idx - start_idx
    df.loc[start_idx:end_idx - 1, "BAT_V"] += np.linspace(0, -2.5, span)
    df.loc[start_idx:end_idx - 1, "BAT_I"] += np.random.default_rng(1).normal(0, 0.3, span)
    df.loc[start_idx:end_idx - 1, "BAT_TEMP"] += np.linspace(0, 4.0, span)
    label = pd.DataFrame([{
        "start_time": df["timestamp"].iloc[start_idx], "end_time": df["timestamp"].iloc[end_idx - 1],
        "fault_type": "battery_degradation",
        "affected_channels": ["BAT_V", "BAT_I", "BAT_TEMP"], "subsystem": "power",
    }])
    return df, label


def inject_reaction_wheel_degradation(df: pd.DataFrame, at_frac: float = 0.5,
                                       end_frac: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    n = len(df)
    start_idx, end_idx = int(n * at_frac), int(n * end_frac)
    span = end_idx - start_idx
    df.loc[start_idx:end_idx - 1, "WHEEL_RPM"] += np.linspace(0, -400, span)
    df.loc[start_idx:end_idx - 1, "ATT_ERR"] += np.linspace(0, 0.8, span)
    label = pd.DataFrame([{
        "start_time": df["timestamp"].iloc[start_idx], "end_time": df["timestamp"].iloc[end_idx - 1],
        "fault_type": "reaction_wheel_degradation",
        "affected_channels": ["WHEEL_RPM", "ATT_ERR"], "subsystem": "adcs",
    }])
    return df, label


if __name__ == "__main__":
    # Quick manual check: python -m src.simulator
    normal = generate_normal_telemetry()
    faulted, label = inject_battery_degradation(normal)
    print(f"Generated {len(normal)} rows, {normal.shape[1]} columns")
    print(label)
