"""
Unit Tests for Spacecraft Mission Simulator
"""

from src.simulator import SpacecraftSimulator


def test_simulator_nominal():
    sim = SpacecraftSimulator()
    df, faults = sim.generate_telemetry(duration_hours=1.0, scenario=None)
    assert len(df) > 0
    assert len(faults) == 0
    assert "EPS_BATT_V" in df.columns
    assert "TCS_BATT_TEMP" in df.columns


def test_simulator_fault_injection():
    sim = SpacecraftSimulator()
    df, faults = sim.generate_telemetry(duration_hours=2.0, scenario="power_fault")
    assert len(faults) == 1
    assert faults[0].subsystem == "power"
    assert "EPS_BATT_V" in faults[0].affected_channels
