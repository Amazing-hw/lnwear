import json
from pathlib import Path

import pandas as pd
import pytest

import s08_run_pipeline as s08


def _write_eval_payload(artifact_dir: Path, payload: dict) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "end_to_end_eval_test_state_machine.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _base_payload() -> dict:
    details = [
        {
            "sample_name": "mixed-pass",
            "target": 1,
            "pred": 1,
            "stage1_pass": True,
            "fallback": False,
            "window_preds": [0, 1, 1],
            "window_states": [0, 1, 1],
            "stage2_states": [0, 1, 1],
            "window_targets": [0, 1, 1],
            "window_probs": [0.1, 0.9, 0.9],
        },
        {
            "sample_name": "stage1-fail",
            "target": 0,
            "pred": 0,
            "stage1_pass": False,
            "fallback": False,
            "window_preds": [1],
            "window_states": [0],
            "stage2_states": [1],
            "window_targets": [0],
            "window_probs": [0.9],
        },
        {
            "sample_name": "fallback-pos",
            "target": 1,
            "pred": 0,
            "stage1_pass": True,
            "fallback": True,
            "window_preds": [1],
            "window_states": [1],
            "stage2_states": [1],
            "window_targets": [1],
            "window_probs": [0.8],
        },
        {
            "sample_name": "no-window",
            "target": 0,
            "pred": 0,
            "stage1_pass": True,
            "fallback": False,
            "window_preds": [],
            "window_states": [],
            "stage2_states": [],
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

    assert xgb["total_windows"].sum() == 4
    assert xgb["correct_windows"].sum() == 3
    assert set(xgb["sample_name"]) == {"mixed-pass", "stage1-fail"}

    assert sm["raw_windows"].sum() == 4
    assert sm["skipped_warmup_windows"].sum() == 2
    assert sm["total_windows"].sum() == 2
    assert sm["correct_windows"].sum() == 2
    assert sm["output_valid_windows"].sum() == 2
    assert set(sm["sample_name"]) == {"mixed-pass", "stage1-fail"}
    assert sm_detail["output_valid"].tolist() == [0, 1, 1, 0]
    assert sm_detail["state_output"].isna().tolist() == [True, False, False, True]

    assert len(sample) == 4
    assert sample["is_correct"].sum() == 3
    fallback = sample.loc[sample["sample_name"] == "fallback-pos"].iloc[0]
    assert fallback["pred"] == 0
    assert fallback["is_fallback"] == 1


def test_generate_eval_csv_rejects_summary_mismatch(tmp_path):
    payload = _base_payload()
    payload["window_model_summary"]["total_windows"] = 999
    _write_eval_payload(tmp_path, payload)

    with pytest.raises(AssertionError, match="per_sample_xgboost_windows"):
        s08.generate_eval_csv(tmp_path)
