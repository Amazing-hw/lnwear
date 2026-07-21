import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

import s03_extract_feature_pool as s03
import s04_feature_selection as s04
import s08_run_pipeline as s08
import stage2_feature_catalog as catalog


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
    assert "AMB_BAND_ENERGY_RATIO" in filtered  # amb FFT now allowed
    assert "GREEN_Entropy_SampEn" not in filtered
    assert "GREEN_Temporal_peak_prominence" not in filtered
    assert "ACC_TREMOR_PEAK_FREQ" not in filtered
    assert "G_bp_lag_std" not in filtered
    assert "ACC_PPG_coherence_mean" not in filtered


def test_deployment_feature_cost_summary_counts_reused_fft_sources():
    summary = s04.summarize_deployment_feature_costs(
        [
            "GREEN_AC_RMS",
            "GTOP2_BAND_ENERGY_RATIO",
            "AMB_BAND_ENERGY_RATIO",
            "GMEDIAN_FFT_PEAK_MEDIAN_RATIO",
            "G_ZONE_PHASE_CONCENTRATION",
        ]
    )

    assert summary["feature_set"] == "deployment_friendly"
    assert summary["fft_source_count"] == 4
    assert summary["fft_sources"] == [
        "green_top2", "ambient", "green_median", "green_3zone"
    ]
    assert summary["forbidden_selected"] == []  # AMB_BAND_ENERGY_RATIO now allowed


def test_subset_evaluation_uses_each_feature_catalog_cost(monkeypatch):
    class FakeClassifier:
        def fit(self, _x, _y, verbose=False):
            return self

        def predict_proba(self, x):
            probability = np.linspace(0.2, 0.8, len(x))
            return np.column_stack([1.0 - probability, probability])

    monkeypatch.setattr(s04.xgb, "XGBClassifier", lambda **_kwargs: FakeClassifier())
    features = ["G_TOP2_CORR_MIN", "G_ZONE_HR_SUPPORT_RATIO"]
    train = __import__("pandas").DataFrame({
        "target": [0, 1, 0, 1],
        features[0]: [0.1, 0.9, 0.2, 0.8],
        features[1]: [0.0, 1.0, 0.2, 0.8],
    })
    valid = train.copy()

    result = s04.evaluate_feature_subsets(
        train,
        valid,
        {"cost_contract": {"features": features, "description": "test"}},
    )

    expected = np.mean([
        catalog.feature_record(name)["deployment_cost"] for name in features
    ])
    assert result.iloc[0]["deployment_cost_mean"] == pytest.approx(expected)


def test_s03_and_s04_deployment_fft_allow_lists_match():
    features = [
        "GTOP2_BAND_ENERGY_RATIO",
        "GTOP2_FFT_PEAK_MEDIAN_RATIO",
        "GTOP2_DOM_FREQ",
        "GREEN_BAND_ENERGY_RATIO",
        "GREEN_FFT_PEAK_MEDIAN_RATIO",
        "GREEN_DOM_FREQ",
        "FFT_PEAK_MEDIAN_RATIO",
        "AMB_BAND_ENERGY_RATIO",
        "AMB_FFT_PEAK_MEDIAN_RATIO",
        "AMB_DOM_FREQ",
        "AMBX_FFT_PEAK_MEDIAN_RATIO",
        "AMBX_DOM_FREQ",
    ]

    s03_allowed = s03.filter_deployment_friendly_stage2_features(features)
    s04_allowed = s04.filter_features_for_deployment(features)

    assert s03_allowed == s04_allowed


def test_s04_deployment_policy_delegates_to_s03_source_of_truth():
    features = [
        "GREEN_CORR",
        "GREEN_AC",
        "AMB_AC",
        "ACC_YSUM",
        "GREEN_DC",
        "AMB_DC",
        "GREEN_XCORR",
        "FFT_PEAK_MEDIAN_RATIO",
        "ACC_PPG_coherence_mean",
        "IRX_bp_skewness",
    ]

    for name in features:
        assert s04.is_deployment_allowed_feature(name) == s03.is_deployment_friendly_stage2_feature(name)

    # The retained commercial feature-name mapping is not silently aliased into
    # the governed Stage2 deployment surface.
    assert s04.filter_features_for_deployment(s03.COMMERCIAL_8_FEATURE_NAMES) == [
        "GREEN_CORR"
    ]


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
            "--feature_selection_mode",
            "auto",
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


def test_deploy_extractor_is_self_contained_without_profile_meta(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    selected = ["GREEN_AC_RMS", "G_TOP2_CORR_MIN", "GTOP2_BAND_ENERGY_RATIO"]
    bundle = {
        "feature_pool_version": catalog.FEATURE_POOL_VERSION,
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

    assert "from s03_extract_feature_pool" not in text
    assert "S03_SOURCE_DIR" not in text
    assert "sys.path.insert" not in text
    assert "from scipy.signal import resample_poly" not in text
    assert "ppg_25 = source_ppg[::4]" in text
    assert "acc_25 = source_acc[::4]" in text
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


def test_export_deploy_cookbook_writes_performance_profile(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    selected = [
        "GREEN_AC_RMS",
        "GREEN_SEG_ACDC_CV",
        "G_2OF3_AC_SUPPORT",
        "G_TOP2_CORR_MIN",
        "GTOP2_BAND_ENERGY_RATIO",
        "ACC_REL_MOTION",
    ]
    model = __import__("xgboost").XGBClassifier(
        n_estimators=4,
        max_depth=2,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=0,
    )
    model.fit(np.asarray([[0.0] * len(selected), [1.0] * len(selected)], dtype=float), np.asarray([0, 1]))
    joblib.dump(
        {
            "feature_pool_version": catalog.FEATURE_POOL_VERSION,
            "feature_names": selected,
            "fill_values": {name: 0.0 for name in selected},
            "clip_bounds": {name: [-1.0, 1.0] for name in selected},
            "threshold": 0.37,
            "model": model,
            "raw_model": model,
            "meta": {
                "fs_ppg": 25.0,
                "win_sec": 5.0,
                "step_sec": 1.0,
                "use_stage2_ir": False,
            },
        },
        artifact_dir / "model_bundle.pkl",
    )
    s08.export_feature_extractor_script(str(artifact_dir))
    s08.export_deploy_cookbook(str(artifact_dir))

    profile_path = artifact_dir / "deploy_performance_profile.json"
    assert profile_path.exists()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile["feature_names"] == selected
    assert profile["feature_cost_summary"]["feature_count"] == len(selected)
    assert profile["feature_cost_summary"]["fft_source_count"] >= 1
    assert profile["model_summary"]["n_estimators"] == 4
    assert profile["model_summary"]["total_nodes"] >= 1
    assert profile["model_summary"]["avg_nodes_per_tree"] > 0
    assert profile["reliability_feature_summary"]["selected_count"] >= 3
    assert "GREEN_SEG_ACDC_CV" in profile["reliability_feature_summary"]["selected_features"]
    assert profile["deployment_targets"]["window_model_threshold"] == 0.37


def test_s04_generates_accuracy_beam_subset_candidates():
    summary = []
    for i, feature in enumerate([
        "GREEN_AC_MAD",
        "GTOP2_BAND_ENERGY_RATIO",
        "G_TOP2_CORR_MIN",
        "G_TOP2_RANK_STABILITY",
        "ACC_GREEN_BP_CORR",
        "ACC_REL_MOTION",
        "AMBX_AC_MAD",
        "GREEN_AMB_BP_CORR",
    ]):
        summary.append({
            "feature": feature,
            "combined_score": 1.0 - i * 0.03,
            "deployment_score": 1.0 - i * 0.02,
            "group": s04.feature_to_group(feature),
            "deployment_cost": 1.0,
            "fp_proxy_sample_fp_rate": 0.1 + i * 0.01,
        })

    candidates = s04.generate_feature_subset_candidates(
        summary,
        [item["feature"] for item in summary],
        max_features=6,
    )

    beam_names = [name for name in candidates if name.startswith("accuracy_beam_")]
    assert beam_names
    for name in beam_names:
        assert len(candidates[name]["features"]) <= 6
        assert "accuracy-first beam" in candidates[name]["description"]


def test_s04_accuracy_first_group_limits_allow_more_high_signal_features():
    summary = []
    for i, feature in enumerate([
        "ACC_GREEN_BP_CORR",
        "ACC_REL_MOTION",
        "ACC_MAG_MAD",
        "G_TOP2_CORR_MIN",
        "G_TOP2_TO_ALL_AC_RATIO",
        "G_TOP2_RANK_STABILITY",
        "GTOP2_BAND_ENERGY_RATIO",
        "GTOP2_DOM_FREQ",
        "GREEN_AC_MAD",
    ]):
        summary.append({
            "feature": feature,
            "combined_score": 1.0 - i * 0.01,
            "deployment_score": 1.0 - i * 0.01,
            "group": s04.feature_to_group(feature),
        })

    default_selected, _ = s04.select_by_group_from_combined(
        summary,
        max_features=8,
        group_limits=s04.GROUP_LIMITS_DEFAULT,
    )
    accuracy_selected, _ = s04.select_by_group_from_combined(
        summary,
        max_features=8,
        group_limits=s04.group_limits_for_ranking_objective("window_accuracy"),
    )

    acc_groups = {"acc_motion", "acc_green_coupling"}
    default_acc = sum(s04.feature_to_group(f) in acc_groups for f in default_selected)
    accuracy_acc = sum(s04.feature_to_group(f) in acc_groups for f in accuracy_selected)
    assert 1 <= default_acc <= 3
    assert 1 <= accuracy_acc <= 3
    assert len(accuracy_selected) <= 8


def test_s05_threshold_objective_accuracy_prefers_max_window_accuracy():
    import s05_train_final_model as s05

    y_true = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.10, 0.40, 0.60, 0.55, 0.58, 0.90])

    best = s05.select_threshold_from_probs(
        y_true,
        probs,
        objective="accuracy",
        beta=0.5,
        min_precision=0.95,
    )

    assert best["objective"] == "accuracy"
    assert best["accuracy"] >= 5 / 6
    assert "fp_rate" in best


def test_hard_negative_mining_preserves_object_worn_context():
    import pandas as pd
    import s05_train_final_model as s05

    df = pd.DataFrame({
        "sample_name": ["n_obj", "n_skin", "p1", "n_obj2"],
        "h5_file": ["a.h5", "b.h5", "c.h5", "d.h5"],
        "window_index": [4, 5, 6, 7],
        "target": [0, 0, 1, 0],
        "mode": [1, 1, 1, 1],
        "negative_type": ["object_worn", "skin_off", None, "object_worn"],
        "scene_type": ["object_worn", "off_wrist", None, "object_worn_reflective"],
        "subject_type": ["non_human", "human", "human", "non_human"],
    })
    probs = np.array([0.92, 0.20, 0.80, 0.88])

    weights, report, summary = s05.build_hard_negative_training_weights_from_oof(
        df,
        probs,
        min_probability=0.8,
        top_percentile=0.5,
        hard_negative_weight=4.0,
    )

    assert weights[0] == 4.0
    assert "negative_type" in report.columns
    assert "object_worn" in set(report["negative_type"])
    assert summary["object_worn_hard_negatives"] == 2
    assert summary["object_worn_fraction"] == 1.0


def test_s06_action_items_prioritize_object_worn_false_positives():
    import pandas as pd
    import s06_deploy_eval as s06

    action_items = s06.build_action_items(
        window_strata=pd.DataFrame(),
        sample_strata=pd.DataFrame(),
        hard_payload={
            "false_positives": [
                {"sample_name": "obj1", "negative_type": "object_worn", "subject_type": "non_human"},
                {"sample_name": "obj2", "scene_type": "object_worn_reflective"},
            ]
        },
        model_search_df=pd.DataFrame(),
        window_metrics={"accuracy": 0.95, "n": 2},
        min_support=1,
    )

    assert "object_worn_false_positive_cluster" in set(action_items["issue_type"])
    row = action_items[action_items["issue_type"] == "object_worn_false_positive_cluster"].iloc[0]
    assert row["priority"] == "P0"
    assert "object_worn" in row["stratum"]


def test_deploy_extractor_import_and_feature_vector_are_finite(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    selected = ["GREEN_AC_RMS", "G_TOP2_CORR_MIN", "GTOP2_BAND_ENERGY_RATIO", "ACC_MAG_MEAN"]
    bundle = {
        "feature_pool_version": catalog.FEATURE_POOL_VERSION,
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
