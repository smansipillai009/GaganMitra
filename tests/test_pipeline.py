"""
End-to-End Pipeline Integration Test
"""

from run_pipeline import run_pipeline


def test_full_pipeline_synthetic(tmp_path):
    clean_df, df_scores, events, eval_res = run_pipeline(
        source="synthetic",
        scenario="power_fault",
        detector_type="statistical",
        output_dir=str(tmp_path)
    )
    assert len(clean_df) > 0
    assert len(events) >= 1
    assert eval_res.event_recall > 0.0


def test_full_pipeline_esa(tmp_path):
    clean_df, df_scores, events, eval_res = run_pipeline(
        source="esa",
        scenario="power_fault",
        detector_type="statistical",
        output_dir=str(tmp_path)
    )
    assert len(clean_df) > 0
    assert len(events) >= 1
