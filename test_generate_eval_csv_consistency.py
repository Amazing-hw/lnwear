import json
from pathlib import Path

import pandas as pd
import pytest

import s06_deploy_eval as s06
import s08_run_pipeline as s08


def _write_eval_payload(artifact_dir: Path, payload: dict) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "end_to_end_eval_test_prob_mean.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _base_payload() -> dict:
    details = [
        {
            "sample_name": "mixed-pass",
            "target": 1,
            "pred": 1,
            "fallback": False,
            "window_preds": [0, 1, 1],
            "window_states": [0, 1, 1],
            "window_targets": [0, 1, 1],
            "window_probs": [0.1, 0.9, 0.9],
        },
        {
            "sample_name": "negative",
            "target": 0,
            "pred": 0,
            "fallback": False,
            "window_preds": [1],
            "window_states": [0],
            "window_targets": [0],
            "window_probs": [0.9],
        },
        {
            "sample_name": "fallback-pos",
            "target": 1,
            "pred": 0,
            "fallback": True,
            "window_preds": [1],
            "window_states": [1],
            "window_targets": [1],
            "window_probs": [0.8],
        },
        {
            "sample_name": "no-window",
            "target": 0,
            "pred": 0,
            "fallback": False,
            "window_preds": [],
            "window_states": [],
            "window_targets": [],
            "window_probs": [],
        },
    ]
    return {
        "summary": {
            "accuracy": 0.75,
            "confusion_matrix": {"TN": 2, "FP": 0, "FN": 1, "TP": 1},
        },
        "window_model_summary": {
            "warmup_frames": 0,
            "total_windows": 4,
            "accuracy": 0.75,
            "confusion_matrix": {"TN": 1, "FP": 1, "FN": 0, "TP": 2},
        },
        "window_stream_summary": {
            "warmup_frames": 1,
            "skipped_warmup_windows": 1,
            "total_windows": 2,
            "accuracy": 1.0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 2},
        },
        "details": details,
    }


def test_generate_eval_csv_matches_official_metric_summaries(tmp_path):
    _write_eval_payload(tmp_path, _base_payload())

    s08.generate_eval_csv(tmp_path)

    xgb = pd.read_csv(tmp_path / "per_sample_xgboost_windows.csv")
    sm = pd.read_csv(tmp_path / "per_sample_statemachine_windows.csv")
    sm_detail = pd.read_csv(tmp_path / "statemachine_window_details.csv")
    sample = pd.read_csv(tmp_path / "per_sample_final_prediction.csv")
    all_samples = pd.read_csv(tmp_path / "per_sample_inference_summary_test_prob_mean.csv")

    assert xgb["total_windows"].sum() == 4
    assert xgb["correct_windows"].sum() == 3
    assert set(xgb["sample_name"]) == {"mixed-pass", "negative"}

    assert sm["raw_windows"].sum() == 4
    assert sm["skipped_warmup_windows"].sum() == 2
    assert sm["total_windows"].sum() == 2
    assert sm["correct_windows"].sum() == 2
    assert sm["output_valid_windows"].sum() == 2
    assert set(sm["sample_name"]) == {"mixed-pass", "negative"}
    assert sm_detail["output_valid"].tolist() == [0, 1, 1, 0]
    assert sm_detail["state_output"].isna().tolist() == [True, False, False, True]

    assert len(sample) == 4
    assert sample["is_correct"].sum() == 3
    fallback = sample.loc[sample["sample_name"] == "fallback-pos"].iloc[0]
    assert fallback["pred"] == 0
    assert fallback["is_fallback"] == 1
    assert set(all_samples["sample_name"]) == {
        "mixed-pass", "negative", "fallback-pos", "no-window"
    }


def test_generate_eval_csv_rejects_summary_mismatch(tmp_path):
    payload = _base_payload()
    payload["window_model_summary"]["total_windows"] = 999
    _write_eval_payload(tmp_path, payload)

    with pytest.raises(AssertionError, match="per_sample_xgboost_windows"):
        s08.generate_eval_csv(tmp_path)


def test_prob_mean_exports_the_independent_stream_trace_used_by_csv(tmp_path, monkeypatch):
    """The XGBoost sample method must not erase the parallel stream trace."""
    monkeypatch.setattr(s06, "_BUNDLE", None)
    cfg = dict(s06.DEFAULT_POSTPROCESS_CONFIG)
    results = [{
        "sample_name": "prob-mean-sample",
        "target": 1,
        "mode": 0,
        "fallback": False,
        "window_probs": [0.1, 0.9, 0.9],
        "window_preds": [0, 1, 1],
        "window_targets": [0, 1, 1],
        "quality_metas": [None, None, None],
    }]

    sample_summary, details = s06.compute_sample_metrics(
        results,
        method="prob_mean",
        cfg=cfg,
        model_threshold=0.5,
    )
    assert details[0]["window_states"] == []

    stream_summary = s06.compute_window_stream_metrics(
        details,
        cfg,
        warmup_frames=0,
        model_threshold=0.5,
    )
    assert len(details[0]["stage2_states"]) == 3

    payload = {
        "summary": sample_summary,
        "window_model_summary": s06.compute_window_model_metrics(results),
        "window_stream_summary": stream_summary,
        "details": details,
    }
    _write_eval_payload(tmp_path, payload)

    s08.generate_eval_csv(tmp_path)

    stream_csv = pd.read_csv(tmp_path / "per_sample_statemachine_windows.csv")
    assert stream_csv["total_windows"].sum() == stream_summary["total_windows"]
