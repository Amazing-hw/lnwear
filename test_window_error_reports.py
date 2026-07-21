import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import s06_deploy_eval as s06
import s07_postprocess_optimize as s07
from stage2_feature_catalog import FEATURE_POOL_VERSION


ROOT = Path(__file__).resolve().parent


def test_window_model_metrics_include_all_xgboost_windows():
    metrics = s06.compute_window_model_metrics([
        {
            "sample_name": "computed",
            "target": 1,
            "fallback": False,
            "window_preds": [1, 1],
            "window_targets": [1, 1],
        }
    ])

    assert metrics["total_windows"] == 2
    assert metrics["accuracy"] == 1.0


def test_state_machine_output_is_not_masked_by_legacy_gate_fields():
    cfg = {
        **s06.DEFAULT_POSTPROCESS_CONFIG,
        "alpha": 1.0,
        "median_k": 1,
        "T_on": 0.5,
        "T_off": 0.2,
        "K_on": 2,
        "K_off": 2,
        "cooldown_sec": 0,
    }
    summary, details = s06.compute_sample_metrics(
        [{
            "sample_name": "parallel",
            "target": 1,
            "fallback": False,
            "window_probs": [0.9, 0.9, 0.9],
            "window_preds": [1, 1, 1],
            "quality_metas": [{}, {}, {}],
            "stage1_gate_flags": [0, 0, 0],
        }],
        method="state_machine",
        cfg=cfg,
        model_threshold=0.5,
        stride_sec=1.0,
    )

    detail = details[0]
    assert detail["window_states"] == [0, 1, 1]
    assert detail["pred"] == 1
    assert summary["accuracy"] == 1.0
    assert summary["evaluation_semantics"] == "xgboost_postprocess_only_v1"


def test_legacy_closed_gate_cannot_mask_xgboost_output():
    cfg = {
        **s06.DEFAULT_POSTPROCESS_CONFIG,
        "alpha": 1.0,
        "T_on": 0.5,
        "T_off": 0.2,
        "K_on": 1,
        "K_off": 1,
        "cooldown_sec": 0,
    }
    _summary, details = s06.compute_sample_metrics(
        [{
            "sample_name": "closed",
            "target": 1,
            "fallback": False,
            "window_probs": [0.9, 0.9],
            "window_preds": [1, 1],
            "quality_metas": [{}, {}],
            "stage1_gate_flags": [0, 0],
        }],
        method="state_machine",
        cfg=cfg,
        model_threshold=0.5,
    )

    assert details[0]["pred"] == 1
    assert details[0]["window_states"] == [1, 1]


def test_sample_metrics_preserve_partial_window_feature_failures():
    summary, details = s06.compute_sample_metrics(
        [{
            "sample_name": "partial_failure",
            "target": 1,
            "fallback": False,
            "window_probs": [0.9],
            "window_preds": [1],
            "quality_metas": [{}],
            "window_feature_failure_count": 2,
            "window_feature_failure_examples": [
                {"window_position": 0, "error": "RuntimeError: failed"},
            ],
        }],
        method="prob_mean",
        cfg=s06.DEFAULT_POSTPROCESS_CONFIG,
        model_threshold=0.5,
    )

    assert summary["samples_with_window_feature_failures"] == 1
    assert summary["total_window_feature_failures"] == 2
    assert details[0]["window_feature_failure_count"] == 2
    assert details[0]["window_feature_failure_examples"][0]["window_position"] == 0


class _DeployTestBooster:
    def get_dump(self, with_stats=True):
        return ["0:leaf=0.0,cover=1.0"]

    def trees_to_data_frame(self):
        return pd.DataFrame([
            {
                "Tree": 0,
                "Node": 0,
                "ID": "0-0",
                "Feature": "Leaf",
                "Split": np.nan,
                "Yes": np.nan,
                "No": np.nan,
                "Missing": np.nan,
                "Gain": 0.0,
                "Cover": 1.0,
            }
        ])


class _DeployTestModel:
    n_estimators = 1

    def get_booster(self):
        return _DeployTestBooster()

    def get_params(self):
        return {"n_estimators": 1}


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
    assert "stage1_gate" not in cache
    assert "stage1_enabled" not in cache


def test_window_cache_export_removes_only_top_level_obsolete_npz(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    split_dir = artifact_dir / "window_outputs" / "valid"
    nested_dir = split_dir / "nested"
    test_dir = artifact_dir / "window_outputs" / "test"
    nested_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (split_dir / "stale.npz").write_bytes(b"stale")
    (split_dir / "keep.txt").write_text("keep", encoding="utf-8")
    (nested_dir / "nested-stale.npz").write_bytes(b"nested")
    (test_dir / "other-split.npz").write_bytes(b"other")
    result = {
        "sample_name": "current/sample",
        "target": 1,
        "mode": 0,
        "window_probs": [0.9],
        "window_preds": [1],
        "stage2_enabled_flags": [1],
        "quality_metas": [{}],
    }

    s06.export_window_cache(
        [result],
        artifact_dir,
        "valid",
        window_sec=5.0,
        stride_sec=1.0,
        model_threshold=0.5,
        metadata={
            "model_fingerprint": {"source": "test"},
            "feature_names": ["GREEN_AC_RMS"],
        },
    )

    assert (split_dir / "current_sample.npz").exists()
    assert not (split_dir / "stale.npz").exists()
    assert (split_dir / "keep.txt").exists()
    assert (nested_dir / "nested-stale.npz").exists()
    assert (test_dir / "other-split.npz").exists()


def test_window_cache_export_does_not_silently_skip_failed_sample(
        tmp_path, monkeypatch):
    def fail_write(*_args, **_kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(s06, "write_window_cache_npz", fail_write)

    with pytest.raises(RuntimeError, match="broken_sample.*synthetic write failure"):
        s06.export_window_cache(
            [{"sample_name": "broken_sample", "target": 0}],
            tmp_path / "artifacts",
            "valid",
            window_sec=5.0,
            stride_sec=1.0,
            model_threshold=0.5,
        )


@pytest.mark.parametrize(
    ("cache_root", "split", "error_pattern"),
    [
        ("../escaped", "valid", "cache_root.*artifact_dir"),
        ("window_outputs", "../escaped", "split.*cache_root"),
    ],
)
def test_window_cache_export_rejects_path_escape(
        tmp_path, cache_root, split, error_pattern):
    artifact_dir = tmp_path / "artifacts"

    with pytest.raises(ValueError, match=error_pattern):
        s06.export_window_cache(
            [],
            artifact_dir,
            split,
            window_sec=5.0,
            stride_sec=1.0,
            model_threshold=0.5,
            cache_root=cache_root,
        )

    assert not (tmp_path / "escaped").exists()


def test_export_deploy_artifacts_writes_selected_stage2_contracts(
        tmp_path, monkeypatch):
    feature_names = ["GREEN_AC_RMS"]
    bundle = {
        "feature_pool_version": FEATURE_POOL_VERSION,
        "feature_names": feature_names,
        "fill_values": {"GREEN_AC_RMS": 0.0},
        "model": _DeployTestModel(),
        "raw_model": _DeployTestModel(),
        "threshold": 0.5,
        "threshold_policy": {},
        "quality_thresholds": {},
        "feature_quantiles": {},
        "fingerprint": {"source": "test"},
        "meta": {
            "fs_ppg": 25.0,
            "win_sec": 5.0,
            "step_sec": 1.0,
            "use_stage2_ir": False,
        },
    }
    monkeypatch.setattr(s06.joblib, "load", lambda _path: bundle)

    s06.export_deploy_artifacts(tmp_path)

    deploy_dir = tmp_path / "deploy_package"
    catalog = json.loads(
        (deploy_dir / "stage2_feature_catalog.json").read_text(encoding="utf-8"))
    c_contract = json.loads(
        (deploy_dir / "stage2_c_contract.json").read_text(encoding="utf-8"))
    assert catalog["feature_order"] == feature_names
    assert list(catalog["features"]) == feature_names
    assert c_contract["feature_order"] == feature_names
    assert c_contract["sample_rate_hz"] == 25.0
    assert c_contract["window_samples"] == 125
    assert "sum_squares" in c_contract["operator_inventory"]


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
            "--feature_selection_mode",
            "auto",
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
