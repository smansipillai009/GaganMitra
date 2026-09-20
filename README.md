# GaganMitra 

Explainable spacecraft telemetry anomaly detection and health monitoring.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Exit criterion for Day 1: the command above opens a browser tab showing the
GaganMitra title, a mission selector in the sidebar, and the selected
mission's config printed as JSON. No real data or detection yet — that
starts Day 2.

## Project structure

```
gaganmitra/
├── app.py                  # Streamlit dashboard (Role D)
├── configs/
│   ├── config.yaml          # main settings (windowing, thresholds, severity rule)
│   ├── common_schema.md     # the shared data shape every source must match
│   └── missions/
│       ├── synthetic_demo_sat.yaml   # Scenario B mission config
│       └── esa_adb.yaml              # Scenario A mission config (PLACEHOLDER —
│                                        needs real channel names after Day 2's
│                                        data audit, see the TODO inside it)
├── data/{raw,processed,demo}
├── models/{checkpoints,scalers}
├── src/                     # empty for now — Days 2-10 fill this in
├── outputs/{metrics,plots,alerts}
└── tests/
```

## Two things flagged before Day 2 — decide these as a team first

**1. Role split is currently lopsided.** The plan's two-person version gives
Person 2 (Priyamvada) three roles' worth of work — backend/adapters/
integration (Role C) AND UI/product (Role D) AND presentation — while
Person 1 (Mansi) covers two (Role A + Role B). Before committing to this,
decide together whether some of Role C (schema validation, source adapters)
should split across both people, since it's genuinely backend/data work that
overlaps with Role A's territory.

**2. Scope trim for the 10 days.** `requirements.txt` intentionally leaves
Pydantic, DuckDB, PyArrow/Parquet, PyTorch Lightning, Optuna, ruptures, and
ONNX Runtime as "optional / later" — install one only when a specific task
actually needs it, not up front. This matches the project's own rule (don't
start with heavy infrastructure) — the v2 plan's expanded tech list drifted
from that a little; this README is where that gets corrected back.

## Known placeholder needing real work before it can be trusted

`configs/missions/esa_adb.yaml` has fake channel names (`TBD_AFTER_DATA_AUDIT`).
This MUST be filled in with real ESA-ADB column names during Day 2's data
audit — do not let this placeholder silently survive past Day 2.
