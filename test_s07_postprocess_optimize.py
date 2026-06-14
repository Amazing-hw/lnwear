import json
import subprocess
import sys
from pathlib import Path

import numpy as np


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
