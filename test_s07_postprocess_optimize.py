import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

import s06_deploy_eval as s06
import s07_postprocess_optimize as s07


ROOT = Path(__file__).resolve().parent


def _write_cache(path, sample_name, target, probs, **contract_overrides):
    probs = np.asarray(probs, dtype=float)
    n = probs.size
    starts = np.arange(n, dtype=float)
    ends = starts + 5.0
    pred_raw = (probs >= 0.5).astype(np.int32)
    payload = {
        "sample_name": np.asarray(sample_name),
        "target": np.asarray(target, dtype=np.int32),
        "window_start_sec": starts,
        "window_end_sec": ends,
        "stage1_enabled": np.ones(n, dtype=np.int32),
        "prob_raw": probs,
        "pred_raw": pred_raw,
        "quality": np.ones(n, dtype=float),
        "ood_rate": np.zeros(n, dtype=float),
        "mode": np.asarray(0, dtype=np.int32),
        "fallback": np.asarray(0, dtype=np.int32),
        "model_threshold": np.asarray(
            contract_overrides.get("model_threshold", 0.5), dtype=float),
        "window_sec": np.asarray(
            contract_overrides.get("window_sec", 5.0), dtype=float),
        "stride_sec": np.asarray(
            contract_overrides.get("stride_sec", 1.0), dtype=float),
        "cache_schema_version": np.asarray(
            contract_overrides.get("cache_schema_version", "window_outputs_v2")),
        "model_fingerprint_json": np.asarray(json.dumps(
            contract_overrides.get("model_fingerprint", {"source": "test"}))),
        "feature_names_json": np.asarray(json.dumps(
            contract_overrides.get("feature_names", ["GREEN_AC_RMS"]))),
        "skip_initial_windows": np.asarray(
            contract_overrides.get("skip_initial_windows", 3), dtype=np.int32),
        "window_indices": np.arange(n, dtype=np.int32),
        "window_targets": np.full(n, target, dtype=np.int32),
    }
    np.savez(path, **payload)


def test_s06_window_contract_accepts_numerically_close_values():
    bundle = {"meta": {"win_sec": 5.0, "step_sec": 0.2}}

    resolved = s06.validate_inference_window_contract(
        bundle, window_sec=5.0 + 1e-10, stride_sec=0.2 - 1e-10)

    assert resolved == (5.0 + 1e-10, 0.2 - 1e-10)


def test_s06_window_contract_rejects_cli_mismatch():
    bundle = {"meta": {"win_sec": 5.0, "step_sec": 1.0}}

    with pytest.raises(ValueError, match="window_sec.*model_bundle.meta.win_sec"):
        s06.validate_inference_window_contract(
            bundle, window_sec=3.0, stride_sec=1.0)


@pytest.mark.parametrize(
    ("field", "override"),
    [
        ("cache_schema_version", "window_outputs_v1"),
        ("model_fingerprint", {"source": "other"}),
        ("feature_names", ["GREEN_DC_MEAN"]),
        ("model_threshold", 0.6),
        ("window_sec", 3.0),
        ("stride_sec", 2.0),
        ("skip_initial_windows", 4),
    ],
)
def test_s07_rejects_mixed_cache_contracts(tmp_path, field, override):
    artifact_dir = tmp_path / "artifacts"
    cache_dir = artifact_dir / "window_outputs" / "valid"
    cache_dir.mkdir(parents=True)
    _write_cache(cache_dir / "a.npz", "a", 0, [0.1])
    _write_cache(cache_dir / "b.npz", "b", 1, [0.9], **{field: override})

    with pytest.raises(ValueError, match=field):
        s07.load_split_caches(artifact_dir, "window_outputs", "valid")


def test_s07_rejects_uniformly_obsolete_cache_schema(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    cache_dir = artifact_dir / "window_outputs" / "valid"
    cache_dir.mkdir(parents=True)
    _write_cache(
        cache_dir / "old.npz",
        "old",
        0,
        [0.1],
        cache_schema_version="window_outputs_v1",
    )

    with pytest.raises(ValueError, match="window_outputs_v2"):
        s07.load_split_caches(artifact_dir, "window_outputs", "valid")


def test_s07_rejects_empty_cache_directory_and_cli_exits_nonzero(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    (artifact_dir / "window_outputs" / "valid").mkdir(parents=True)

    with pytest.raises(ValueError, match="no NPZ"):
        s07.load_split_caches(artifact_dir, "window_outputs", "valid")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s07_postprocess_optimize.py"),
            "--artifact_dir",
            str(artifact_dir),
            "--split",
            "valid",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0, result.stdout + result.stderr


def test_s07_rejects_test_split_for_parameter_selection(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s07_postprocess_optimize.py"),
            "--artifact_dir",
            str(tmp_path / "artifacts"),
            "--split",
            "test",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "selection split must be 'valid'" in result.stdout + result.stderr


def test_s07_rejects_entire_batch_when_one_cache_is_malformed(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    cache_dir = artifact_dir / "window_outputs" / "valid"
    cache_dir.mkdir(parents=True)
    _write_cache(cache_dir / "valid.npz", "valid", 0, [0.1])
    np.savez(cache_dir / "malformed.npz", sample_name=np.asarray("bad"))

    with pytest.raises(ValueError, match="malformed.npz.*missing required keys"):
        s07.load_split_caches(artifact_dir, "window_outputs", "valid")


def test_s07_rejects_cache_provenance_that_differs_from_available_bundle(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    cache_dir = artifact_dir / "window_outputs" / "valid"
    cache_dir.mkdir(parents=True)
    _write_cache(cache_dir / "sample.npz", "sample", 0, [0.1])
    joblib.dump(
        {
            "fingerprint": {"source": "current-model"},
            "feature_names": ["GREEN_AC_RMS"],
        },
        artifact_dir / "model_bundle.pkl",
    )

    with pytest.raises(ValueError, match="model_fingerprint.*model_bundle"):
        s07.load_split_caches(artifact_dir, "window_outputs", "valid")


def test_s07_rejects_cache_features_that_differ_from_available_bundle(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    cache_dir = artifact_dir / "window_outputs" / "valid"
    cache_dir.mkdir(parents=True)
    _write_cache(cache_dir / "sample.npz", "sample", 0, [0.1])
    joblib.dump(
        {
            "fingerprint": {"source": "test"},
            "feature_names": ["GREEN_DC_MEAN"],
        },
        artifact_dir / "model_bundle.pkl",
    )

    with pytest.raises(ValueError, match="feature_names.*model_bundle"):
        s07.load_split_caches(artifact_dir, "window_outputs", "valid")


def test_s07_parallel_grid_search_initializes_worker_caches(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    cache_dir = artifact_dir / "window_outputs" / "valid"
    cache_dir.mkdir(parents=True)
    _write_cache(cache_dir / "neg.npz", "neg", 0, [0.05, 0.05, 0.10, 0.05, 0.10])
    _write_cache(cache_dir / "pos.npz", "pos", 1, [0.80, 0.85, 0.90, 0.90, 0.95])

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s07_postprocess_optimize.py"),
            "--artifact_dir",
            str(artifact_dir),
            "--split",
            "valid",
            "--n_workers",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "worker caches not initialized" not in output
    assert "UnboundLocalError" not in output
    optimized_path = artifact_dir / "postprocess_opt" / "postprocess_optimized.json"
    assert optimized_path.exists()
    optimized = json.loads(optimized_path.read_text(encoding="utf-8"))
    final_config = json.loads(
        (artifact_dir / "final_model_config.json").read_text(encoding="utf-8"))
    provenance = optimized["cache_provenance"]
    assert provenance["cache_schema_version"] == "window_outputs_v2"
    assert provenance["model_fingerprint"] == {"source": "test"}
    assert provenance["feature_names"] == ["GREEN_AC_RMS"]
    assert provenance["model_threshold"] == 0.5
    assert provenance["window_sec"] == 5.0
    assert provenance["stride_sec"] == 1.0
    assert provenance["skip_initial_windows"] == 3
    assert final_config["postprocess_optimization"]["cache_provenance"] == provenance


def test_s07_budgeted_grid_keeps_representative_candidates_and_caps_runtime():
    full_grid = list(s07.iter_param_grid())

    budgeted = s07.select_postprocess_search_grid(full_grid, search_budget=48)

    assert len(budgeted) == 48
    assert full_grid[0] in budgeted
    assert any(p["T_on"] >= 0.70 and p["K_on"] >= 5 for p in budgeted)
    assert any(p["median_k"] == 3 for p in budgeted)


def test_s07_zero_budget_keeps_full_grid():
    full_grid = list(s07.iter_param_grid())

    assert s07.select_postprocess_search_grid(full_grid, search_budget=0) == full_grid


def test_s07_grid_compares_final_majority_and_any_worn_voting_strategies():
    strategies = {item["sample_pred_strategy"] for item in s07.iter_param_grid()}

    assert strategies == {
        "final_state",
        "majority_state_after_warmup",
        "any_worn_after_warmup",
    }


def test_s07_default_warmup_matches_deploy_eval_default():
    assert s07.DEFAULT_WARMUP_FRAMES == 5


def test_s07_window_accuracy_uses_per_window_targets_from_cache():
    cache = {
        "sample_name": "mixed-record",
        "target": 1,
        "window_end_sec": np.array([1.0, 2.0, 3.0]),
        "stage1_enabled": np.array([1, 1, 1]),
        "prob_raw": np.array([0.1, 0.9, 0.9]),
        "quality": np.ones(3),
        "stride_sec": 1.0,
        "window_targets": np.array([0, 1, 1]),
    }
    params = {
        "ema_alpha": 1.0,
        "median_k": 1,
        "T_on": 0.5,
        "T_off": 0.5,
        "K_on": 1,
        "K_off": 1,
        "cooldown_sec": 0.0,
    }

    _details, metrics = s07.evaluate_postprocess_on_caches([cache], params)

    assert metrics["window_accuracy"] == 1.0


def test_s07_window_accuracy_can_skip_state_machine_warmup_windows():
    cache = {
        "sample_name": "warmup-pos",
        "target": 1,
        "window_end_sec": np.array([1.0, 2.0, 3.0]),
        "stage1_enabled": np.array([1, 1, 1]),
        "prob_raw": np.array([0.9, 0.9, 0.9]),
        "quality": np.ones(3),
        "stride_sec": 1.0,
        "window_targets": np.array([1, 1, 1]),
    }
    params = {
        "ema_alpha": 1.0,
        "median_k": 1,
        "T_on": 0.5,
        "T_off": 0.5,
        "K_on": 3,
        "K_off": 1,
        "cooldown_sec": 0.0,
    }

    _details, cold_metrics = s07.evaluate_postprocess_on_caches(
        [cache], params, warmup_frames=0)
    _details, warm_metrics = s07.evaluate_postprocess_on_caches(
        [cache], params, warmup_frames=2)

    assert cold_metrics["window_accuracy"] == 1 / 3
    assert warm_metrics["window_accuracy"] == 1.0
    assert warm_metrics["skipped_warmup_windows"] == 2


def test_s07_first_worn_latency_is_relative_to_first_valid_stage2_probability():
    cache = {
        "sample_name": "relative-latency",
        "target": 1,
        "window_end_sec": np.array([5.0, 6.0, 7.0]),
        "stage1_enabled": np.array([1, 1, 1]),
        "prob_raw": np.array([0.9, 0.9, 0.9]),
        "quality": np.ones(3),
        "stride_sec": 1.0,
        "window_targets": np.ones(3, dtype=int),
    }
    params = {
        "ema_alpha": 1.0,
        "median_k": 1,
        "T_on": 0.5,
        "T_off": 0.5,
        "K_on": 2,
        "K_off": 1,
        "cooldown_sec": 0.0,
        "sample_pred_strategy": "any_worn_after_warmup",
    }

    detail = s07.run_postprocess_on_cache(cache, params)
    _details, metrics = s07.evaluate_postprocess_on_caches([cache], params)

    assert detail["first_valid_probability_sec"] == 5.0
    assert detail["positive_output_latency_sec"] == 1.0
    assert metrics["first_worn_output_p95_sec"] == 1.0


def test_postprocess_candidate_selection_maximizes_accuracy_under_fpr_and_latency():
    candidates = [
        {"params": "fast", "window_accuracy": 0.95, "window_fp_rate": 0.005,
         "first_worn_output_p95_sec": 1.0, "state_flip_count": 3, "parameter_complexity": 4},
        {"params": "accurate", "window_accuracy": 0.98, "window_fp_rate": 0.009,
         "first_worn_output_p95_sec": 2.5, "state_flip_count": 2, "parameter_complexity": 5},
        {"params": "too_slow", "window_accuracy": 0.99, "window_fp_rate": 0.0,
         "first_worn_output_p95_sec": 3.1, "state_flip_count": 0, "parameter_complexity": 1},
    ]

    decision = s07.select_postprocess_candidate(
        candidates, max_window_fp_rate=0.01, max_added_latency_sec=3.0
    )

    assert decision["selected_params"] == "accurate"
    assert decision["deployment_acceptance"] is True


def test_postprocess_candidate_selection_marks_analysis_only_and_tie_breaks():
    candidates = [
        {"params": "noisy", "window_accuracy": 0.97, "window_fp_rate": 0.02,
         "first_worn_output_p95_sec": 2.0, "state_flip_count": 4, "parameter_complexity": 4},
        {"params": "stable", "window_accuracy": 0.97, "window_fp_rate": 0.02,
         "first_worn_output_p95_sec": 1.5, "state_flip_count": 1, "parameter_complexity": 3},
    ]

    decision = s07.select_postprocess_candidate(candidates)

    assert decision["selected_params"] == "stable"
    assert decision["deployment_acceptance"] is False
    assert decision["status"] == "analysis_only"


def test_wear_state_machine_first_window_respects_quality():
    sm = s06.WearStateMachine(
        alpha=0.4,
        T_on=0.7,
        T_off=0.3,
        K_on=1,
        K_off=1,
        cooldown_sec=0.0,
    )

    state, score = sm.update(0.95, quality=0.0, stride_sec=1.0)

    assert state == 0
    assert score == 0.0


def test_default_postprocess_threshold_allows_sustained_low_confidence_positive():
    cfg = dict(s06.DEFAULT_POSTPROCESS_CONFIG)
    cfg.update({
        "alpha": 1.0,
        "median_k": 1,
        "K_on": 2,
        "K_off": 1,
        "cooldown_sec": 0.0,
        "sample_pred_strategy": "any_worn_after_warmup",
    })

    final, states, _preds, scores = s06.apply_postprocess(
        [0.60] * 6,
        [{}] * 6,
        "state_machine",
        cfg,
        model_threshold=0.5,
    )

    assert final == 1
    assert states[-1] == 1
    assert max(scores) == 0.60


def test_s06_apply_postprocess_uses_actual_stride_for_cooldown():
    cfg = {
        "alpha": 1.0,
        "median_k": 1,
        "T_on": 0.5,
        "T_off": 0.5,
        "K_on": 1,
        "K_off": 1,
        "cooldown_sec": 4.0,
        "sample_pred_strategy": "final_state",
    }

    final, states, _preds, _scores = s06.apply_postprocess(
        [0.9, 0.1, 0.1],
        [{}, {}, {}],
        "state_machine",
        cfg,
        model_threshold=0.5,
        stride_sec=2.0,
    )

    assert states == [1, 1, 0]
    assert final == 0


def test_explicit_high_t_on_is_preserved_for_conservative_configs():
    cfg = dict(s06.DEFAULT_POSTPROCESS_CONFIG)
    cfg.update({
        "alpha": 1.0,
        "median_k": 1,
        "T_on": 0.70,
        "K_on": 2,
        "K_off": 1,
        "cooldown_sec": 0.0,
        "sample_pred_strategy": "any_worn_after_warmup",
    })

    final, states, _preds, _scores = s06.apply_postprocess(
        [0.66] * 6,
        [{}] * 6,
        "state_machine",
        cfg,
        model_threshold=0.5,
    )

    assert final == 0
    assert states == [0, 0, 0, 0, 0, 0]


def test_s07_any_worn_strategy_keeps_positive_sample_after_late_drop():
    cache = {
        "sample_name": "late-drop-pos",
        "target": 1,
        "window_end_sec": np.arange(1, 7, dtype=float),
        "stage1_enabled": np.ones(6, dtype=int),
        "prob_raw": np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1]),
        "quality": np.ones(6),
        "stride_sec": 1.0,
        "window_targets": np.ones(6, dtype=int),
    }
    params = {
        "ema_alpha": 1.0,
        "median_k": 1,
        "T_on": 0.5,
        "T_off": 0.5,
        "K_on": 2,
        "K_off": 2,
        "cooldown_sec": 0.0,
        "sample_pred_strategy": "any_worn_after_warmup",
        "sample_pred_warmup_frames": 0,
    }

    detail = s07.run_postprocess_on_cache(cache, params)

    assert detail["states"] == [0, 1, 1, 1, 0, 0]
    assert detail["pred"] == 1


def test_s07_stage2_state_warms_while_stage1_masks_output():
    cache = {
        "sample_name": "gate-opens-after-warmup",
        "target": 1,
        "window_end_sec": np.arange(1, 4, dtype=float),
        "stage1_enabled": np.array([0, 0, 1], dtype=int),
        "prob_raw": np.array([0.9, 0.9, 0.9]),
        "quality": np.ones(3),
        "stride_sec": 1.0,
        "window_targets": np.ones(3, dtype=int),
    }
    params = {
        "ema_alpha": 1.0,
        "median_k": 1,
        "T_on": 0.5,
        "T_off": 0.5,
        "K_on": 2,
        "K_off": 1,
        "cooldown_sec": 0.0,
        "sample_pred_strategy": "final_state",
        "sample_pred_warmup_frames": 0,
    }

    detail = s07.run_postprocess_on_cache(cache, params)

    assert detail["stage2_states"] == [0, 1, 1]
    assert detail["states"] == [0, 0, 1]
    assert detail["stage2_pred"] == 1
    assert detail["pred"] == 1


def test_s07_postprocess_handles_empty_window_cache_without_unbound_state():
    params = {
        "ema_alpha": 1.0,
        "median_k": 1,
        "T_on": 0.5,
        "T_off": 0.5,
        "K_on": 1,
        "K_off": 1,
        "cooldown_sec": 0.0,
    }
    cache = {
        "sample_name": "empty",
        "target": 1,
        "window_end_sec": np.array([], dtype=float),
        "stage1_enabled": np.array([], dtype=int),
        "prob_raw": np.array([], dtype=float),
        "quality": np.array([], dtype=float),
        "stride_sec": 1.0,
        "window_targets": np.array([], dtype=int),
    }

    detail = s07.run_postprocess_on_cache(cache, params)

    assert detail["pred"] == 0
    assert detail["states"] == []
    assert detail["scores"] == []
    assert detail["window_targets"] == []


def test_s07_postprocess_state_machine_matches_s06_leaky_counter_semantics():
    probs = [0.8, 0.8, 0.6, 0.8, 0.8]
    params = {
        "ema_alpha": 1.0,
        "median_k": 1,
        "T_on": 0.7,
        "T_off": 0.3,
        "K_on": 3,
        "K_off": 1,
        "cooldown_sec": 0.0,
    }
    sm = s06.WearStateMachine(
        alpha=params["ema_alpha"],
        T_on=params["T_on"],
        T_off=params["T_off"],
        K_on=params["K_on"],
        K_off=params["K_off"],
        cooldown_sec=params["cooldown_sec"],
    )
    expected_states = [sm.update(p, quality=1.0, stride_sec=1.0)[0] for p in probs]
    cache = {
        "sample_name": "pos",
        "target": 1,
        "window_end_sec": np.arange(1, len(probs) + 1, dtype=float),
        "stage1_enabled": np.ones(len(probs), dtype=int),
        "prob_raw": np.asarray(probs, dtype=float),
        "quality": np.ones(len(probs), dtype=float),
        "stride_sec": 1.0,
    }

    actual = s07.run_postprocess_on_cache(cache, params)

    assert actual["states"] == expected_states
