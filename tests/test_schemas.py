"""
Unit Tests for GaganMitra Data Contracts & Schemas
"""

from datetime import datetime, timezone
import pytest
from src.schemas import TelemetryRecord, AnomalyScore, TelemetryEvent, GroundTruthFault


def test_telemetry_record():
    rec = TelemetryRecord(
        timestamp=datetime.now(timezone.utc),
        mission_id="synthetic_eo_sat",
        spacecraft_id="GaganMitra-EOS1",
        channel_name="EPS_BATT_V",
        standard_name="battery_voltage",
        value=28.4,
        unit="V",
        subsystem="power",
        criticality="critical"
    )
    assert rec.value == 28.4
    assert rec.subsystem == "power"


def test_telemetry_event():
    now = datetime.now(timezone.utc)
    event = TelemetryEvent(
        event_id="EVT-POW-001",
        start_time=now,
        end_time=now,
        duration_seconds=60.0,
        severity="HIGH",
        subsystem="power",
        channels=["EPS_BATT_V", "EPS_BATT_I"],
        peak_channel="EPS_BATT_V",
        evidence=["Voltage drop observed"],
        confidence=0.92,
        status="NEW"
    )
    assert event.severity == "HIGH"
    assert len(event.channels) == 2
