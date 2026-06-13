import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

import s04_feature_selection as s04
import s08_run_pipeline as s08


ROOT = Path(__file__).resolve().parent


def test_deployment_feature_filter_removes_complex_operators():
    features = [
        "GREEN_AC_RMS",
        "G_TOP2_CORR_MIN",
        "GTOP2_BAND_ENERGY_RATIO",
        "GTOP2_DOM_FREQ",
        "AMB_BAND_ENERGY_RATIO",
        "ACC_PPG_coherence_mean",
        "GREEN_Entropy_SampEn",
        "GREEN_Temporal_peak_prominence",
        "ACC_TREMOR_PEAK_FREQ",
        "G_bp_lag_std",
    ]

    filtered = s04.filter_features_for_deployment(features)

    assert "GREEN_AC_RMS" in filtered
    assert "G_TOP2_CORR_MIN" in filtered
    assert "GTOP2_BAND_ENERGY_RATIO" in filtered
    assert "GTOP2_DOM_FREQ" in filtered
    assert "AMB_BAND_ENERGY_RATIO" not in filtered
    assert "ACC_PPG_coherence_mean" not in filtered
    assert "GREEN_Entropy_SampEn" not in filtered
    assert "GREEN_Temporal_peak_prominence" not in filtered
    assert "ACC_TREMOR_PEAK_FREQ" not in filtered
    assert "G_bp_lag_std" not in filtered


def test_deployment_feature_cost_summary_counts_green_top2_fft_only():
    summary = s04.summarize_deployment_feature_costs(
        ["GREEN_AC_RMS", "GTOP2_BAND_ENERGY_RATIO", "AMB_BAND_ENERGY_RATIO"]
    )

    assert summary["feature_set"] == "deployment_friendly"
    assert summary["fft_source_count"] == 2
    assert summary["fft_sources"] == ["green_top2", "ambient"]
    assert summary["forbidden_selected"] == ["AMB_BAND_ENERGY_RATIO"]


def test_s08_dry_run_does_not_expose_feature_pool_switches():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--dataset_dir",
            "dataset",
            "--artifact_dir",
            "artifacts",
            "--stop_after",
            "s05",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout + result.stderr

    assert ("feature" + "_profile") not in output
    assert "fft_budget" not in output
    assert "filter_mode" not in output
    assert "representative model search" in output


def test_deploy_extractor_has_no_complex_library_dependency_without_profile_meta(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    selected = ["GREEN_AC_RMS", "G_TOP2_CORR_MIN", "GTOP2_BAND_ENERGY_RATIO"]
    bundle = {
        "feature_names": selected,
        "fill_values": {name: 0.0 for name in selected},
        "clip_bounds": {name: [-10.0, 10.0] for name in selected},
        "threshold": 0.5,
        "meta": {
            "fs_ppg": 25.0,
            "win_sec": 5.0,
            "use_stage2_ir": False,
        },
    }
    joblib.dump(bundle, artifact_dir / "model_bundle.pkl")
    (artifact_dir / "final_model_config.json").write_text(
        json.dumps({
            "selected_features": selected,
            "window_model_threshold": 0.5,
            "fill_values": {name: 0.0 for name in selected},
            "clip_bounds": {name: [-10.0, 10.0] for name in selected},
        }),
        encoding="utf-8",
    )

    s08.export_feature_extractor_script(str(artifact_dir))
    text = (artifact_dir / "deploy_feature_extractor.py").read_text(encoding="utf-8")

    assert "scipy" not in text
    assert "coherence" not in text
    assert "find_peaks" not in text
    assert "Entropy" not in text
    assert "SampEn" not in text
    assert "FEATURE_ORDER" in text


def test_model_search_rows_include_deployment_cost_metadata():
    import s05_train_final_model as s05

    rows = s05.build_model_search_result_rows([{
        "rank_input_order": 0,
        "eligible": True,
        "score": 0.9,
        "fp_rate": 0.01,
        "size_ratio": 0.1,
        "total_nodes": 100,
        "avg_nodes_per_tree": 5,
        "feature_count": 12,
        "deployment_feature_cost_summary": {
            "feature_set": "deployment_friendly",
            "fft_source_count": 1,
            "forbidden_selected_count": 0,
        },
        "params": {"n_estimators": 20},
    }])

    assert rows[0]["feature_set"] == "deployment_friendly"
    assert rows[0]["deployment_fft_source_count"] == 1
    assert rows[0]["deployment_forbidden_selected_count"] == 0


def test_deploy_extractor_import_and_feature_vector_are_finite(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    selected = ["GREEN_AC_RMS", "G_TOP2_CORR_MIN", "GTOP2_BAND_ENERGY_RATIO", "ACC_MAG_MEAN"]
    bundle = {
        "feature_names": selected,
        "fill_values": {name: 0.0 for name in selected},
        "clip_bounds": {name: [-1e6, 1e6] for name in selected},
        "threshold": 0.5,
        "meta": {"fs_ppg": 25.0, "win_sec": 5.0, "use_stage2_ir": False},
    }
    joblib.dump(bundle, artifact_dir / "model_bundle.pkl")
    (artifact_dir / "final_model_config.json").write_text(
        json.dumps({"selected_features": selected, "window_model_threshold": 0.5}),
        encoding="utf-8",
    )

    script = Path(s08.export_feature_extractor_script(str(artifact_dir)))
    spec = __import__("importlib.util").util.spec_from_file_location("deploy_feature_extractor_test", script)
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.default_rng(7)
    n = 125
    vec = module.extract_features(
        rng.normal(800, 2, size=n),
        rng.normal(500, 3, size=n),
        rng.normal(1000, 5, size=n),
        rng.normal(1003, 5, size=n),
        rng.normal(998, 5, size=n),
        acc=rng.normal(0, 0.1, size=(n, 3)),
        fs=25,
    )

    assert len(vec) == len(selected)
    assert np.all(np.isfinite(vec))
