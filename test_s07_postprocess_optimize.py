import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import s06_deploy_eval as s06
import s07_postprocess_optimize as s07


ROOT = Path(__file__).resolve().parent


def _write_cache(path, sample_name, target, probs):
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
        "model_threshold": np.asarray(0.5, dtype=float),
        "window_sec": np.asarray(5.0, dtype=float),
        "stride_sec": np.asarray(1.0, dtype=float),
        "cache_schema_version": np.asarray("test_v1"),
        "model_fingerprint_json": np.asarray(json.dumps({"source": "test"})),
        "feature_names_json": np.asarray(json.dumps(["GREEN_AC_RMS"])),
        "skip_initial_windows": np.asarray(3, dtype=np.int32),
        "window_indices": np.arange(n, dtype=np.int32),
        "window_targets": np.full(n, target, dtype=np.int32),
    }
    np.savez(path, **payload)


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
    assert (artifact_dir / "postprocess_opt" / "postprocess_optimized.json").exists()


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
