import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import s06_deploy_eval as s06
import s07_postprocess_optimize as s07


ROOT = Path(__file__).resolve().parent


def test_window_error_analysis_exports_fp_fn_strata(tmp_path):
    details = [
        {
            "sample_name": "neg/a",
            "target": 0,
            "mode": 2,
            "stage1_pass": True,
            "window_probs": [0.2, 0.85],
            "window_preds": [0, 1],
            "stage2_enabled_flags": [1, 1],
            "window_start_sec": [3.0, 4.0],
            "window_end_sec": [6.0, 7.0],
            "quality_metas": [{"Ambient_std": 2.0}, {"Ambient_std": 9.0}],
            "window_ood_scores": [0.0, 0.4],
        },
        {
            "sample_name": "pos/b",
            "target": 1,
            "mode": 1,
            "stage1_pass": True,
            "window_probs": [0.1],
            "window_preds": [0],
            "stage2_enabled_flags": [1],
            "window_start_sec": [3.0],
            "window_end_sec": [6.0],
            "quality_metas": [{}],
            "window_ood_scores": [0.0],
        },
    ]

    report = s06.compute_window_error_analysis(details)
    csv_path, json_path = s06.export_window_error_analysis(report, tmp_path, "valid", "state_machine")

    assert report["summary"]["confusion_matrix"] == {"TN": 1, "FP": 1, "FN": 1, "TP": 0}
    assert report["summary"]["total_windows"] == 3
    assert report["summary"]["error_windows"] == 2
    assert report["strata"]["error_type"]["FP"]["n_windows"] == 1
    assert report["strata"]["error_type"]["FN"]["n_windows"] == 1
    assert report["strata"]["prob_bin"]["prob>=0.8"]["fp"] == 1
    assert Path(csv_path).exists()
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["summary"]["total_windows"] == 3


def test_window_stream_metrics_uses_per_window_targets_for_grouped_h5():
    results = [
        {
            "sample_name": "mixed_record",
            "target": 1,
            "stage1_pass": True,
            "fallback": False,
            "window_probs": [0.1, 0.9, 0.9],
            "window_targets": [0, 1, 1],
            "quality_metas": [{}, {}, {}],
        }
    ]
    cfg = {
        "alpha": 1.0,
        "median_k": 1,
        "T_on": 0.5,
        "T_off": 0.5,
        "K_on": 1,
        "K_off": 1,
        "cooldown_sec": 0.0,
    }

    metrics = s06.compute_window_stream_metrics(results, cfg, warmup_frames=0)

    assert metrics["confusion_matrix"] == {"TN": 1, "FP": 0, "FN": 0, "TP": 2}
    assert metrics["accuracy"] == 1.0


def test_window_stream_metrics_reports_state_machine_output_after_warmup():
    results = [
        {
            "sample_name": "warmup_raw_only_positive",
            "target": 1,
            "stage1_pass": True,
            "fallback": False,
            "window_probs": [0.60, 0.60, 0.60],
            "window_preds": [1, 1, 1],
            "window_targets": [1, 1, 1],
            "quality_metas": [{}, {}, {}],
        }
    ]
    cfg = {
        "alpha": 1.0,
        "median_k": 1,
        "T_on": 0.70,
        "T_off": 0.30,
        "K_on": 2,
        "K_off": 1,
        "cooldown_sec": 0.0,
    }

    metrics = s06.compute_window_stream_metrics(results, cfg, warmup_frames=2)

    assert metrics["total_windows"] == 1
    assert metrics["confusion_matrix"] == {"TN": 0, "FP": 0, "FN": 1, "TP": 0}
    assert metrics["skipped_warmup_windows"] == 2


def test_postprocess_replay_records_valid_selection_and_test_metrics():
    valid_cache = {
        "sample_name": "valid-pos",
        "target": 1,
        "window_end_sec": np.array([3.0, 4.0, 5.0]),
        "stage1_enabled": np.array([1, 1, 1]),
        "prob_raw": np.array([0.9, 0.9, 0.9]),
        "quality": np.ones(3),
        "stride_sec": 1.0,
    }
    test_cache = {
        "sample_name": "test-neg",
        "target": 0,
        "window_end_sec": np.array([3.0, 4.0, 5.0]),
        "stage1_enabled": np.array([1, 1, 1]),
        "prob_raw": np.array([0.1, 0.1, 0.1]),
        "quality": np.ones(3),
        "stride_sec": 1.0,
    }
    params = {
        "ema_alpha": 1.0,
        "median_k": 1,
        "T_on": 0.8,
        "T_off": 0.3,
        "K_on": 2,
        "K_off": 1,
        "cooldown_sec": 0.0,
    }

    payload = s07.build_replay_report(
        best_params=params,
        selection_split="valid",
        selection_caches=[valid_cache],
        replay_split="test",
        replay_caches=[test_cache],
    )

    assert payload["selection"]["split"] == "valid"
    assert payload["replay"]["split"] == "test"
    assert payload["selection"]["metrics"]["sample_accuracy"] == 1.0
    assert payload["replay"]["metrics"]["sample_accuracy"] == 1.0
    assert payload["best_params"]["T_on"] == 0.8


def test_window_cache_preserves_window_indices_and_targets(tmp_path):
    result = {
        "sample_name": "record_a",
        "target": 1,
        "mode": 0,
        "window_probs": [0.2, 0.8],
        "window_preds": [0, 1],
        "stage2_enabled_flags": [1, 1],
        "window_start_sec": [3.0, 20.0],
        "window_end_sec": [6.0, 23.0],
        "window_indices": [3, 20],
        "window_targets": [0, 1],
        "quality_metas": [{}, {}],
    }

    path = s06.write_window_cache_npz(
        result,
        tmp_path,
        window_sec=3,
        stride_sec=1,
        model_threshold=0.5,
    )
    cache = s07.load_window_cache_npz(path)

    assert cache["window_indices"].tolist() == [3, 20]
    assert cache["window_targets"].tolist() == [0, 1]


def test_s08_dry_run_exports_replay_cache_before_postprocess():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--stop_after",
            "s07_post",
            "--export_window_cache",
            "--optimize_postprocess",
            "--postprocess_split",
            "valid",
            "--split",
            "test",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout + result.stderr
    valid_cache_pos = output.index("--split valid")
    replay_cache_pos = output.index("--split test")
    post_pos = output.index("s07_postprocess_optimize.py")
    assert valid_cache_pos < replay_cache_pos < post_pos
    assert "--replay_split test" in output
