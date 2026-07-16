import json
import pickle
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

import s02_ir_dc_threshold as s02
import s03_extract_feature_pool as s03
import s04_feature_selection as s04
import s06_deploy_eval as s06
from stage2_feature_catalog import FEATURE_POOL_VERSION


ROOT = Path(__file__).resolve().parent


def test_s02_uses_fixed_stage1_deploy_threshold_constants():
    args = Namespace(
        fixed_dc_threshold=1.0,
        fixed_ac_dc_threshold=9.0,
    )

    dc, acdc = s02.resolve_fixed_deploy_thresholds(args)

    assert dc == 0.1e6
    assert acdc == 1.0


def test_s02_train_gate_is_derived_from_new_deploy_dc_threshold():
    deploy_dc, deploy_acdc = s02.resolve_fixed_deploy_thresholds()
    train_gate = s02.make_train_threshold(deploy_dc, deploy_acdc)

    assert train_gate["dc_threshold"] == 0.09e6
    assert train_gate["ac_dc_threshold"] == 1.1


def test_s02_cli_does_not_expose_stage1_threshold_args():
    result = subprocess.run(
        [sys.executable, str(ROOT / "s02_ir_dc_threshold.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout + result.stderr
    assert "--fixed_dc_threshold" not in output
    assert "--fixed_ac_dc_threshold" not in output


def test_s02_default_min_duration_matches_3s_decision_window(monkeypatch):
    ppg = np.zeros((300, 6), dtype=float)
    ppg[:, 0] = 4.0e6

    monkeypatch.setattr(s02, "load_ppg", lambda _sample: ppg)

    df = s02.extract_stage1_windows(
        [{
            "sample_name": "three_second_sample", "h5_file": "x.h5",
            "target": 1, "frequency": 100,
        }],
        n_workers=1,
    )

    assert len(df) == 3
    assert df["decision_sec"].unique().tolist() == [3.0]


def test_stage1_ambient_check_uses_configured_ratio_threshold():
    ppg = np.zeros((300, 6), dtype=float)
    ppg[:, 0] = 1000.0
    ppg[:, 1] = 900.0

    assert s03.stage1_ambient_check(ppg, ambient_ratio_threshold=0.8) is False
    assert s03.stage1_ambient_check(ppg, ambient_ratio_threshold=1.0) is True


def test_stage2_training_extraction_uses_all_windows_independent_of_stage1(monkeypatch):
    ppg = np.zeros((3, 300, 6), dtype=float)
    ppg[:, :, 0] = 2.0e6
    ppg[:, :, 1] = 1.9e6
    sample = {
        "sample_name": "ambient_near_ir", "h5_file": "synthetic.h5",
        "target": 1, "frequency": 25, "ppg_config": 0,
    }

    monkeypatch.setattr(s03, "load_ppg", lambda _sample: ppg)
    monkeypatch.setattr(s03, "load_acc", lambda _sample: None)
    monkeypatch.setattr(
        s03,
        "extract_stage2_window",
        lambda *_args, **_kwargs: (
            {"GREEN_AC_RMS": 1.0},
            {"feature_pool_version": FEATURE_POOL_VERSION},
            {},
        ),
    )
    rows = s03._extract_rows_for_sample(
        sample,
        dc_threshold=1.0e12,
        ac_dc_threshold=1.0,
        window_len=75,
        stride_len=25,
        fs=25,
        target_aware_stride=False,
        stride_neg=25,
        stride_pos=25,
        skip_initial_windows=0,
    )

    assert len(rows) == 3


def test_process_pool_worker_hooks_are_pickleable():
    for fn in (
        s04._init_stab_worker,
        s04._run_one_fold,
        s06._init_worker,
        s06._worker_infer,
    ):
        assert pickle.loads(pickle.dumps(fn)) is fn


def test_s06_summarizes_stage1_target1_pass_rate():
    summary = s06.summarize_stage1_target1_pass_rate([
        {"target": 1, "stage1_pass": True, "fallback": False},
        {"target": 1, "stage1_pass": False, "fallback": False},
        {"target": 1, "stage1_pass": True, "fallback": True},
        {"target": 0, "stage1_pass": True, "fallback": False},
    ])

    assert summary == {
        "target1_total_samples": 3,
        "target1_stage1_pass_samples": 1,
        "target1_stage1_pass_rate": 1 / 3,
    }


def test_s08_dry_run_does_not_expose_stage1_threshold_tuning_args():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--stop_after",
            "s02",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "--fixed_dc_threshold" not in output
    assert "--fixed_ac_dc_threshold" not in output

    help_result = subprocess.run(
        [sys.executable, str(ROOT / "s08_run_pipeline.py"), "--help"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    help_output = help_result.stdout + help_result.stderr
    assert "--s02_dc" not in help_output
    assert "--s02_acdc" not in help_output
    assert "--fixed_dc_threshold" not in help_output
    assert "--fixed_ac_dc_threshold" not in help_output
