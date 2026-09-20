"""GaganMitra — Day 1 blank dashboard.

Exit criterion for today: `streamlit run app.py` opens without errors and
shows this page. Everything below is a placeholder for Role D to build out
starting Day 2 — do not add real logic here yet.
"""
import streamlit as st
import yaml
from pathlib import Path

st.set_page_config(page_title="GaganMitra", page_icon="🛰️", layout="wide")

CONFIG_PATH = Path(__file__).parent / "configs" / "config.yaml"
MISSIONS_DIR = Path(__file__).parent / "configs" / "missions"


@st.cache_data
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_data
def list_missions():
    return sorted(p.stem for p in MISSIONS_DIR.glob("*.yaml"))


config = load_config()
missions = list_missions()

st.title("🛰️ GaganMitra")
st.caption("Explainable Spacecraft Health Monitoring — Day 1 boilerplate")

with st.sidebar:
    st.header("Mission / Source")
    selected_mission = st.selectbox("Select mission config", missions, index=0)
    st.caption(f"Project version: {config['project']['version']}")

st.info(
    "This is the Day 1 skeleton. No real telemetry, detection, or alerts "
    "yet — that's Days 2-10. This page just confirms the app launches "
    "cleanly and can read mission configs."
)

st.subheader("Loaded mission config")
mission_path = MISSIONS_DIR / f"{selected_mission}.yaml"
with open(mission_path) as f:
    mission_config = yaml.safe_load(f)
st.json(mission_config)

st.subheader("What gets built here, day by day")
st.markdown(
    """
    - **Days 2-4** (Role A/B): real ESA-ADB + synthetic data, channel audit, fault injection
    - **Days 5-6** (Role C): source adapters mapping raw data to the common schema
    - **Day 7** (Role B): baseline detector (z-score/EWMA, Isolation Forest)
    - **Day 8** (Role B): events, severity, alert bundles
    - **Day 9** (Role D): this page becomes the real dashboard — telemetry
      charts, active anomalies, event detail, contributing channels
    - **Day 10** (both): full offline end-to-end test
    """
)
