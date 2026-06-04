import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import s02_ir_dc_threshold as s02


ROOT = Path(__file__).resolve().parent


def test_s02_uses_fixed_stage1_deploy_threshold_constants():
    args = Namespace(
        fixed_dc_threshold=1.0,
        fixed_ac_dc_threshold=9.0,
    )

    dc, acdc = s02.resolve_fixed_deploy_thresholds(args)

    assert dc == 3.6e6
    assert acdc == 0.35


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
