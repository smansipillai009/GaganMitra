# 🛰️ GaganMitra (गगनमित्र) -- Spacecraft-Health Monitoring & Decision-Support System **[Practice Model]** {Team Mavericks} 

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

## What's actually working right now (not just scaffolded)

Every module below has been run and tested, not just written:

- `src/simulator.py` — generates reproducible synthetic telemetry with a
  real orbit/eclipse pattern, and 6 fault-injection functions (spike, drift,
  stuck, dropout, battery degradation, reaction-wheel degradation).
- `src/data_loader.py`, `src/mission_adapter.py`, `src/data_quality.py`,
  `src/preprocessing.py`, `src/windowing.py` — the full data pipeline runs
  end to end (see `src/pipeline.py`).
- `src/baseline_detector.py` + `src/thresholding.py` — a real robust
  z-score/EWMA detector with NDT-style moving threshold.
- `src/events.py` — persistence/hysteresis/grouping logic, unit-tested
  directly with a hand-crafted signal (confirmed correct independent of
  whether the detector's default settings catch a given fault).
- `src/severity.py`, `src/explainability.py`, `src/alert_bundle.py` — the
  full severity formula, contributing-channel ranking (working
  reconstruction-deviation method; Gradient SHAP left as an explicit stub
  for the execution phase, since it needs a trained model to attribute
  against), and alert JSON assembly with full provenance fields.
- `src/lstm_autoencoder.py` — the CONDITIONED LSTM autoencoder architecture
  is real and tested: forward pass works, conditioning on mode/telecommand
  channels works, reconstruction error computes, and a minimal training
  loop actually reduces loss over epochs. Not yet trained on real data —
  that's execution-phase work.
- `src/evaluation.py` — point-level and event-level metrics, false
  alarms/day, detection delay, all implemented and callable.
- 14 passing tests in `tests/`.

## An honest finding from running the full pipeline (`python -m src.pipeline`)

Running the whole chain on synthetic data with the default `config.yaml`
settings (z-score window=60, k=3.0, min_persistence_rows=30) produces
**zero detected events** against injected battery-degradation and
stuck-sensor faults. This is NOT a bug — `test_group_into_events_finds_
persistent_run` proves the grouping logic itself is correct on a clean
signal. The real reason: a rolling-median z-score baseline adapts to slow
drift as it happens (so gradual degradation stops looking anomalous
quickly), and a stuck sensor *loses* variance rather than spiking it, so
neither reliably crosses a z-score threshold with these defaults.

This is genuine, useful information, not a failure to hide: it's a real
argument for why the LSTM autoencoder (Milestone 3 of the execution plan)
matters — it can be trained to notice these subtler patterns in a way a
short-window statistical baseline structurally can't. Recalibrating the
baseline's window/threshold/persistence values against real fault behavior
is real, necessary work for the execution phase (see Milestone 2), not
something to force-fix with arbitrary parameter tweaks now.

`src/pipeline.py` currently uses `min_persistence_rows=3` for its own
smoke test specifically to prove the full chain produces a bundle — this
value is explicitly commented in the code as NOT calibrated and must be
revisited with real data.
