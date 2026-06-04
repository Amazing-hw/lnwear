import s04_feature_selection as s04
import pandas as pd
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_estimate_s04_workload_flags_vif_before_first_step():
    estimates = s04.estimate_s04_workload(
        n_train_rows=5000,
        n_valid_rows=1000,
        n_features=120,
        n_samples=80,
        n_workers=4,
        skip_vif=False,
        shap_available=True,
        run_subset_search=False,
    )

    assert estimates[0]["step"] == "STEP 1 clean/VIF"
    assert estimates[0]["risk"] == "high"
    assert "before STEP 1 completes" in estimates[0]["note"]


def test_estimate_s04_workload_marks_vif_low_when_skipped():
    estimates = s04.estimate_s04_workload(
        n_train_rows=5000,
        n_valid_rows=1000,
        n_features=120,
        n_samples=80,
        n_workers=4,
        skip_vif=True,
        shap_available=False,
        run_subset_search=True,
    )

    clean = next(item for item in estimates if item["step"] == "STEP 1 clean/VIF")
    assert clean["risk"] == "low"
    assert "skip_vif=True" in clean["note"]


def test_diagnostics_can_reuse_cleaning_result_without_reclean(monkeypatch):
    df_train = pd.DataFrame({
        "sample_name": ["a", "a", "b", "b"],
        "target": [0, 0, 1, 1],
        "f_keep": [0.1, 0.2, 0.8, 0.9],
        "f_removed": [1.0, 1.0, 1.0, 1.0],
    })
    df_valid = pd.DataFrame({
        "sample_name": ["v0", "v1"],
        "target": [0, 1],
        "f_keep": [0.15, 0.85],
        "f_removed": [1.0, 1.0],
    })

    def fail_if_reclean(*_args, **_kwargs):
        raise AssertionError("diagnostics should reuse existing cleaning results")

    monkeypatch.setattr(s04, "clean_features_by_train", fail_if_reclean)
    diag = s04.compute_all_feature_diagnostics(
        df_train,
        df_valid,
        ["f_keep", "f_removed"],
        kept_features=["f_keep"],
        removed_map={"low_variance": ["f_removed"]},
        fill_values={"f_keep": 0.2, "f_removed": 1.0},
    )

    removed = diag.set_index("feature").loc["f_removed"]
    assert int(removed["removed"]) == 1
    assert removed["removed_reason"] == "low_variance"


def test_s08_can_forward_skip_vif_to_s04():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--stop_after",
            "s04",
            "--skip_vif",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout + result.stderr
    s04_cmd = output.split("s04_feature_selection.py")[-1]
    assert "--skip_vif" in s04_cmd
