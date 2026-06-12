import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent


def _write_synthetic_grouped_h5(dataset_dir):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    h5_path = dataset_dir / "synthetic_grouped.h5"

    rng = np.random.default_rng(20260612)
    fs = 100
    n_time = 300
    n_channels = 40
    t = np.arange(n_time) / fs
    window_order = [5, 0, 4, 1, 3, 2, 8, 6, 7]

    with h5py.File(h5_path, "w") as f:
        for rec in range(40):
            label = 1 if rec % 2 == 0 else 0
            grp = f.create_group(f"record_{rec:02d}_{label}")
            grp.create_dataset("ppg_config", data=np.array(65, dtype=np.int32))
            for w in window_order:
                child = grp.create_group(f"rec{rec:02d}_w{w}_{label}")
                ppg = np.zeros((n_channels, n_time), dtype=np.float32)
                phase = 0.13 * rec + 0.09 * w
                ppg[0, :] = (
                    4_200_000
                    + 900 * np.sin(2 * np.pi * 1.1 * t + phase)
                    + rng.normal(0, 40, n_time)
                )
                ppg[1, :] = (
                    120_000
                    + 150 * np.sin(2 * np.pi * 0.3 * t + phase)
                    + rng.normal(0, 15, n_time)
                )
                base = 1_700_000 if label else 1_350_000
                amp = 9_000 if label else 3_000
                for c, off in zip([6, 7, 8, 9, 10, 11, 12, 13, 14], np.linspace(0, 0.8, 9)):
                    ppg[c, :] = (
                        base
                        + amp * np.sin(2 * np.pi * (1.15 + 0.02 * c) * t + phase + off)
                        + rng.normal(0, 250, n_time)
                    )
                for c in range(n_channels):
                    if not np.any(ppg[c, :]):
                        ppg[c, :] = 50_000 + rng.normal(0, 20, n_time)

                acc = np.zeros((3, n_time), dtype=np.float32)
                acc[0, :] = (0.02 if label else 0.15) * np.sin(2 * np.pi * 3 * t + phase)
                acc[0, :] += rng.normal(0, 0.005, n_time)
                acc[1, :] = (0.02 if label else 0.12) * np.cos(2 * np.pi * 2 * t + phase)
                acc[1, :] += rng.normal(0, 0.005, n_time)
                acc[2, :] = 1.0 + rng.normal(0, 0.004 if label else 0.02, n_time)

                child.create_dataset("ppg", data=ppg)
                child.create_dataset("acc", data=acc)


def test_readme_does_not_document_empty_model_search_feature_counts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert '--model_search_feature_counts ""' not in readme


def test_s08_synthetic_grouped_h5_smoke_exports_consistent_deploy_artifacts(tmp_path):
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "artifacts"
    _write_synthetic_grouped_h5(dataset_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dataset_dir",
            str(dataset_dir),
            "--artifact_dir",
            str(artifact_dir),
            "--n_workers",
            "1",
            "--no-model_search",
            "--model_search_feature_counts",
            "10",
            "--max_features",
            "10",
            "--skip_vif",
            "--no-plot_errors",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )

    output = result.stdout + result.stderr
    assert "No deploy formula registered" not in output
    assert "deploy artifacts are consistent with model_bundle.pkl" in output

    expected = [
        "model_bundle.pkl",
        "final_model_config.json",
        "final_model.json",
        "deploy_feature_extractor.py",
        "deploy_selected_feature_formulas.json",
        "deploy_xgboost.json",
        "deploy_cookbook.json",
        "golden_vectors.json",
        "deploy_package/model_params.json",
    ]
    for rel_path in expected:
        assert (artifact_dir / rel_path).exists(), rel_path

    bundle = joblib.load(artifact_dir / "model_bundle.pkl")
    config = json.loads((artifact_dir / "final_model_config.json").read_text(encoding="utf-8"))
    formulas = json.loads((artifact_dir / "deploy_selected_feature_formulas.json").read_text(encoding="utf-8"))
    model_params = json.loads((artifact_dir / "deploy_package" / "model_params.json").read_text(encoding="utf-8"))

    selected = list(bundle["feature_names"])
    assert selected
    assert config["selected_features"] == selected
    assert list(formulas.keys()) == selected
    assert model_params["selected_features"] == selected
    assert float(config["window_model_threshold"]) == float(bundle["threshold"])
    assert float(model_params["window_threshold"]) == float(bundle["threshold"])
    assert config["use_stage2_ir"] is False
    assert bundle["meta"]["use_stage2_ir"] is False

    script_path = artifact_dir / "deploy_feature_extractor.py"
    script = script_path.read_text(encoding="utf-8")
    for forbidden in ["s03_extract_feature_pool", "from s0", "import s0", "sys.path"]:
        assert forbidden not in script

    spec = importlib.util.spec_from_file_location("deploy_feature_extractor_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.FEATURE_ORDER == selected
    assert module.USE_STAGE2_IR is False
