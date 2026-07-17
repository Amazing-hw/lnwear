import os
import subprocess
import sys
from pathlib import Path

import pytest

import s01_data_split as s01
import s03_extract_feature_pool as s03
import s04_feature_selection as s04
import s05_train_final_model as s05
import s06_deploy_eval as s06
import s07_postprocess_optimize as s07
import s08_run_pipeline as s08


ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize("module", [s01, s03, s06])
def test_sample_level_stages_use_requested_bounded_workers(module, monkeypatch):
    monkeypatch.delenv("WL_FORCE_SERIAL", raising=False)

    assert module.resolve_n_workers(4, n_items=8) == 4
    assert module.resolve_n_workers(4, n_items=3) <= 3
    assert module.resolve_n_workers(4, n_items=2) <= 2


@pytest.mark.parametrize("module", [s01, s03, s04, s06, s07])
def test_process_stages_support_force_serial_fallback(module, monkeypatch):
    monkeypatch.setenv("WL_FORCE_SERIAL", "1")

    assert module.resolve_n_workers(4, n_items=20) == 1


def test_s04_parallelizes_fold_batches_but_avoids_tiny_pool_overhead(monkeypatch):
    monkeypatch.delenv("WL_FORCE_SERIAL", raising=False)

    assert s04.resolve_n_workers(4, n_items=10) == 4
    assert s04.resolve_n_workers(4, n_items=4) == 1
    assert s04.resolve_n_workers(8, n_items=5) == 5


def test_s07_postprocess_workers_do_not_exceed_grid_size(monkeypatch):
    monkeypatch.delenv("WL_FORCE_SERIAL", raising=False)

    assert s07.resolve_n_workers(4, n_items=2) == 2


def test_xgboost_inner_jobs_default_to_one(monkeypatch):
    monkeypatch.delenv("WL_INNER_N_JOBS", raising=False)

    assert s04.get_inner_n_jobs() == 1
    assert s05.get_inner_n_jobs() == 1


def test_s08_caps_nested_numeric_threads_by_default(monkeypatch):
    for name in s08.THREAD_ENV_DEFAULTS:
        monkeypatch.delenv(name, raising=False)

    resolved = s08.configure_thread_env()

    assert resolved == {name: "1" for name in s08.THREAD_ENV_DEFAULTS}


def test_s08_propagates_global_workers_to_enabled_pipeline_stages():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--feature_selection_mode",
            "auto",
            "--n_workers",
            "98",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    output = result.stdout + result.stderr
    # s02 阈值阶段已删除；其余六个支持 worker 的步骤必须全部继承全局值。
    assert output.count("--n_workers 98") >= 6
    assert "--model_search_n_workers 98" in output
    assert "OMP_NUM_THREADS=1" in output
    assert "MKL_NUM_THREADS=1" in output


def test_parallel_executor_implementations_remain_present():
    process_modules = [s01, s03, s04, s06]

    assert all(hasattr(module, "ProcessPoolExecutor") for module in process_modules)
    assert hasattr(s05, "ThreadPoolExecutor")
    assert os.path.exists(ROOT / "s07_postprocess_optimize.py")
