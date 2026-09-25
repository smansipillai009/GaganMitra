"""src/alert_bundle.py — assembles the final downloadable alert bundle.

Owned by: Role C (schema/provenance) + Role D (download wiring in app.py).
This is the single source of truth for the alert JSON shape — the dashboard
should only ever read this shape, never model internals directly.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone

REQUIRED_FIELDS = [
    "event_id", "mission_id", "source", "start_time", "end_time",
    "peak_score", "severity", "severity_score", "duration_rows",
    "affected_channels", "contributing_channels", "attribution_method",
    "possible_subsystem", "recommended_investigation",
    "model_version", "preprocessing_version", "data_provenance",
    "generated_at",
]

CAUTIOUS_TERMS_REMINDER = (
    "Use 'contributing telemetry channels' and 'possible affected subsystem' — "
    "never 'confirmed root cause' or 'guaranteed failure'. See gaganmitra_01_"
    "team_roles_and_learning.txt Section 5 for the full wording rules."
)


def build_alert_bundle(event: dict, mission_id: str, source: str,
                        contributing_channels: list[dict], attribution_method: str,
                        possible_subsystem: str | None, model_version: str,
                        preprocessing_version: str, data_provenance: str) -> dict:
    affected = [c["channel"] for c in contributing_channels]
    bundle = {
        "event_id": event["event_id"],
        "mission_id": mission_id,
        "source": source,
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "peak_score": event["peak_score"],
        "severity": event.get("severity"),
        "severity_score": event.get("severity_score"),
        "duration_rows": event["duration_rows"],
        "affected_channels": affected,
        "contributing_channels": contributing_channels,
        "attribution_method": attribution_method,
        "possible_subsystem": possible_subsystem,
        "recommended_investigation": (
            f"Inspect {possible_subsystem or 'the affected'} subsystem telemetry; "
            f"compare against operating-mode history and standard diagnostic procedure."
        ),
        "model_version": model_version,
        "preprocessing_version": preprocessing_version,
        "data_provenance": data_provenance,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    missing = [f for f in REQUIRED_FIELDS if f not in bundle]
    if missing:
        raise ValueError(f"Alert bundle missing required fields: {missing}")
    return bundle


def to_json(bundle: dict) -> str:
    return json.dumps(bundle, indent=2, default=str)


def to_html(bundle: dict) -> str:
    """Simple, judge-readable HTML export — no framework dependency."""
    rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in bundle.items())
    return f"""<html><head><title>GaganMitra Alert — {bundle['event_id']}</title></head>
<body style="font-family: sans-serif;">
<h2>GaganMitra Alert Bundle: {bundle['event_id']}</h2>
<p style="color:#666;font-size:0.85em">{CAUTIOUS_TERMS_REMINDER}</p>
<table border="1" cellpadding="6" cellspacing="0">{rows}</table>
</body></html>"""
