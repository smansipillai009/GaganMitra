"""
Unit Tests for Data Quality Layer
"""

import pandas as pd
from datetime import datetime, timedelta
from src.config import get_mission_config
from src.data_quality import DataQualityAuditor


def test_data_quality_audit():
    cfg = get_mission_config("synthetic_eo_sat")
    auditor = DataQualityAuditor(mission_config=cfg)

    t0 = datetime(2026, 9, 26, 12, 0, 0)
    times = [t0 + timedelta(seconds=i * 5) for i in range(50)]
    df = pd.DataFrame({
        "timestamp": times,
        "EPS_BATT_V": [28.0] * 50,
        "EPS_BATT_I": [1.0] * 50
    })

    # Inject glitch spike
    df.loc[25, "EPS_BATT_V"] = 99.0

    clean_df, issues, summary = auditor.audit(df)
    assert len(issues) > 0
    assert summary["overall_quality_grade"] in ["NOMINAL", "DEGRADED", "COMPROMISED"]
