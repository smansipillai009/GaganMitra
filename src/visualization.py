"""src/visualization.py — Plotly chart builders, kept separate from app.py
so the dashboard file itself stays about layout, not chart internals.

Owned by: Role D.
"""
from __future__ import annotations
import plotly.graph_objects as go
import pandas as pd


def telemetry_replay_chart(wide_df: pd.DataFrame, channels: list[str],
                            events: list[dict] | None = None) -> go.Figure:
    fig = go.Figure()
    for ch in channels:
        if ch in wide_df.columns:
            fig.add_trace(go.Scatter(x=wide_df.index, y=wide_df[ch], name=ch, mode="lines"))
    if events:
        for e in events:
            fig.add_vrect(
                x0=e["start_time"], x1=e["end_time"],
                fillcolor="red", opacity=0.15, line_width=0,
                annotation_text=e.get("severity", "EVENT"), annotation_position="top left",
            )
    fig.update_layout(title="Telemetry Replay", xaxis_title="Time", yaxis_title="Value",
                       template="plotly_dark", height=400)
    return fig


def anomaly_score_chart(scores: pd.Series, threshold: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores.index, y=scores.values, name="Anomaly score", mode="lines"))
    if threshold is not None:
        fig.add_trace(go.Scatter(x=threshold.index, y=threshold.values, name="Dynamic threshold (NDT)",
                                  mode="lines", line=dict(dash="dash", color="orange")))
    fig.update_layout(title="Anomaly Score vs. Dynamic Threshold", xaxis_title="Time",
                       yaxis_title="Score", template="plotly_dark", height=300)
    return fig


def contributing_channels_chart(ranking: list[dict]) -> go.Figure:
    channels = [r["channel"] for r in ranking]
    values = [r["contribution_score"] for r in ranking]
    fig = go.Figure(go.Bar(x=values, y=channels, orientation="h"))
    fig.update_layout(title="Contributing Channels", xaxis_title="Contribution score",
                       template="plotly_dark", height=300)
    return fig


def severity_badge_color(severity: str) -> str:
    return {"HIGH": "#e74c3c", "MEDIUM": "#e8a23d", "LOW": "#f1c40f", "MINIMAL": "#4fd6c4"}.get(
        severity, "#888888"
    )
