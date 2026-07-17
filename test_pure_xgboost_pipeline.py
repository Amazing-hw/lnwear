import subprocess
import sys
from pathlib import Path

import s06_deploy_eval as s06


ROOT = Path(__file__).resolve().parent


def test_sample_metrics_are_not_masked_by_legacy_stage1_flags():
    results = [
        {
            "sample_name": "negative_predicted_positive",
            "target": 0,
            "window_probs": [0.8, 0.9],
            "window_preds": [1, 1],
            "stage1_gate_flags": [0, 0],
            "fallback": False,
        },
        {
            "sample_name": "positive_predicted_positive",
            "target": 1,
            "window_probs": [0.85, 0.95],
            "window_preds": [1, 1],
            "stage1_gate_flags": [0, 0],
            "fallback": False,
        },
    ]

    summary, details = s06.compute_sample_metrics(
        results,
        method="prob_mean",
        cfg={},
        model_threshold=0.5,
    )

    assert summary["confusion_matrix"] == {"TN": 0, "FP": 1, "FN": 0, "TP": 1}
    assert summary["evaluation_semantics"] == "xgboost_only_v1"
    assert "stage1_only" not in summary
    assert "fused_output" not in summary
    assert [row["pred"] for row in details] == [1, 1]
    assert all("stage1_pass" not in row for row in details)


def test_active_pipeline_no_longer_schedules_stage1(tmp_path):
    assert not (ROOT / "s02_ir_dc_threshold.py").exists()

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dataset_dir",
            str(tmp_path / "dataset"),
            "--artifact_dir",
            str(tmp_path / "artifacts"),
            "--dry_run",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "s02_ir_dc_threshold.py" not in output
    assert "stage1_threshold.json" not in output


def test_active_scripts_do_not_reference_removed_stage1_runtime():
    for filename in (
        "s03_extract_feature_pool.py",
        "s06_deploy_eval.py",
        "s08_run_pipeline.py",
        "pipeline_acceptance.py",
    ):
        source = (ROOT / filename).read_text(encoding="utf-8").lower()
        assert "stage1_threshold.json" not in source, filename
        assert "fuse_stage1_stage2_states" not in source, filename
        assert "stage1streaminggate" not in source, filename
