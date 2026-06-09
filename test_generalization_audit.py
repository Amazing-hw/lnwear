import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def _write_minimal_audit_artifacts(artifact_dir):
    rows = [
        {
            "sample_name": "neg/a",
            "h5_file": "file_a.h5",
            "target": 0,
            "pred_raw": 1,
            "error_type": "FP",
            "is_error": 1,
            "window_index": 3,
            "window_start_sec": 3.0,
            "window_end_sec": 6.0,
            "prob_raw": 0.91,
            "prob_bin": "prob>=0.8",
            "time_bin": "0-10s",
            "mode": 2,
            "stage2_enabled": 1,
            "ood_rate": 0.45,
            "ood_bin": "ood>0.3",
            "quality_bin": "low_quality",
        },
        {
            "sample_name": "pos/b",
            "h5_file": "file_b.h5",
            "target": 1,
            "pred_raw": 0,
            "error_type": "FN",
            "is_error": 1,
            "window_index": 4,
            "window_start_sec": 4.0,
            "window_end_sec": 7.0,
            "prob_raw": 0.12,
            "prob_bin": "prob<0.2",
            "time_bin": "0-10s",
            "mode": 1,
            "stage2_enabled": 1,
            "ood_rate": 0.35,
            "ood_bin": "ood>0.3",
            "quality_bin": "low_quality",
        },
        {
            "sample_name": "pos/c",
            "h5_file": "file_c.h5",
            "target": 1,
            "pred_raw": 1,
            "error_type": "TP",
            "is_error": 0,
            "window_index": 20,
            "window_start_sec": 20.0,
            "window_end_sec": 23.0,
            "prob_raw": 0.88,
            "prob_bin": "prob>=0.8",
            "time_bin": ">=20s",
            "mode": 1,
            "stage2_enabled": 1,
            "ood_rate": 0.0,
            "ood_bin": "ood<=0.3",
            "quality_bin": "ok",
        },
    ]
    pd.DataFrame(rows).to_csv(
        artifact_dir / "window_error_analysis_test_state_machine.csv",
        index=False,
    )
    (artifact_dir / "hard_negatives_test_state_machine.json").write_text(
        json.dumps({
            "false_positives": [
                {"sample_name": "neg/a", "max_prob": 0.91, "mean_prob": 0.91, "mode": 2}
            ],
            "high_risk_negatives": [
                {"sample_name": "neg/a", "max_prob": 0.91, "mean_prob": 0.91, "mode": 2}
            ],
        }),
        encoding="utf-8",
    )
    (artifact_dir / "end_to_end_eval_test_state_machine.json").write_text(
        json.dumps({
            "summary": {
                "accuracy": 0.8,
                "confusion_matrix": {"TN": 8, "FP": 2, "FN": 1, "TP": 9},
            },
            "window_model_summary": {
                "accuracy": 0.67,
                "precision": 0.5,
                "recall": 0.5,
                "confusion_matrix": {"TN": 0, "FP": 1, "FN": 1, "TP": 1},
            },
            "details": [
                {"sample_name": "neg/a", "target": 0, "pred": 1, "mode": 2},
                {"sample_name": "pos/b", "target": 1, "pred": 0, "mode": 1},
                {"sample_name": "pos/c", "target": 1, "pred": 1, "mode": 1,
                 "first_worn_output_sec": 5.0},
            ],
        }),
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "is_default_params": True,
            "mean_cv_accuracy": 0.95,
            "std_cv_accuracy": 0.01,
            "final_total_nodes": 200,
            "chosen_reason": "default_params_baseline_not_beaten",
        }
    ]).to_csv(artifact_dir / "model_search_results.csv", index=False)
    (artifact_dir / "final_model_config.json").write_text(
        json.dumps({"model_search": {"strategy": "staged_group_cv"}}),
        encoding="utf-8",
    )


def test_generalization_audit_exports_strata_and_action_items(tmp_path):
    _write_minimal_audit_artifacts(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s10_generalization_audit.py"),
            "--artifact_dir",
            str(tmp_path),
            "--split",
            "test",
            "--method",
            "state_machine",
            "--min_support",
            "5",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    out_dir = tmp_path / "generalization_audit"
    assert "generalization_audit" in result.stdout
    for name in [
        "summary.json",
        "summary.md",
        "window_strata.csv",
        "sample_strata.csv",
        "action_items.csv",
    ]:
        assert (out_dir / name).exists()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["missing_optional_dimensions"] == ["subject_id", "device_id", "session_id"]
    assert summary["window_metrics"]["fp_rate"] > 0
    assert summary["sample_metrics"]["false_worn_event_rate"] > 0
    assert summary["sample_metrics"]["first_worn_latency_p95_sec"] == 5.0

    window_strata = pd.read_csv(out_dir / "window_strata.csv")
    assert "low_support" in window_strata.columns
    assert window_strata["low_support"].any()
    assert {"mode", "quality_bin", "ood_bin"}.issubset(set(window_strata["dimension"]))

    action_items = pd.read_csv(out_dir / "action_items.csv")
    assert {
        "priority", "issue_type", "stratum", "evidence_metric", "n_samples", "suggested_action"
    }.issubset(action_items.columns)
    assert "hard_negative_fp_cluster" in set(action_items["issue_type"])
    assert "fn_low_quality_or_ood" in set(action_items["issue_type"])


def test_sample_latency_can_be_derived_from_state_windows():
    from s10_generalization_audit import summarize_sample_metrics

    metrics = summarize_sample_metrics(pd.DataFrame([
        {
            "sample_name": "pos/late",
            "target": 1,
            "pred": 1,
            "window_states": [0, 0, 1],
            "window_end_sec": [3.0, 4.0, 5.0],
        },
        {
            "sample_name": "pos/missed",
            "target": 1,
            "pred": 0,
            "window_states": [0, 0, 0],
            "window_end_sec": [3.0, 4.0, 5.0],
        },
        {
            "sample_name": "neg/ignore",
            "target": 0,
            "pred": 0,
            "window_states": [0, 1],
            "window_end_sec": [3.0, 4.0],
        },
    ]))

    assert metrics["first_worn_latency_p50_sec"] == 5.0
    assert metrics["first_worn_latency_p95_sec"] == 5.0


def test_s08_dry_run_can_insert_generalization_audit_after_eval():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--run_generalization_audit",
            "--stop_after",
            "s10_audit",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout + result.stderr
    eval_pos = output.index("s06_deploy_eval.py")
    audit_pos = output.index("s10_generalization_audit.py")
    assert eval_pos < audit_pos
    assert "--split test" in output
    assert "--method state_machine" in output
