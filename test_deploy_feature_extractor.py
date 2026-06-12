from pathlib import Path
import importlib.util
import json
import py_compile
import subprocess
import sys

import joblib
import numpy as np
import pytest
from xgboost import XGBClassifier

import s06_deploy_eval as s06
import s03_extract_feature_pool as s03
import s05_train_final_model as s05
import s08_run_pipeline as s08


def test_ambx_bp_shape_features_have_deploy_formulas():
    selected = ["AMBX_bp_skewness", "AMBX_bp_kurtosis"]

    formulas = s08.build_selected_feature_formulas(selected)

    assert "amb_bp" in formulas["AMBX_bp_skewness"].get("intermediate_signals", {})
    assert "amb_bp" in formulas["AMBX_bp_kurtosis"].get("intermediate_signals", {})
    assert "std" in formulas["AMBX_bp_skewness"]["formula"]
    assert "std" in formulas["AMBX_bp_kurtosis"]["formula"]


def test_all_s03_window_features_have_deploy_formulas():
    fs = 25
    n = 125
    t = np.arange(n, dtype=float) / fs
    ir = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t)
    ambient = 1.0e5 + 500.0 * np.sin(2 * np.pi * 0.4 * t)
    g1 = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t + 0.01)
    g2 = 2.1e6 + 7.5e3 * np.sin(2 * np.pi * 1.2 * t + 0.03)
    g3 = 1.9e6 + 8.5e3 * np.sin(2 * np.pi * 1.2 * t - 0.02)

    features = list(s03.extract_feature_pool_from_window(ir, ambient, g1, g2, g3, fs=fs).keys())

    formulas = s08.build_selected_feature_formulas(features)

    assert not [name for name in features if s03.is_stage2_ir_feature(name)]
    assert "G_consensus_AC_MAD_range" in formulas
    assert len(formulas) == len(features)
    forbidden_dep_tokens = ("ir_raw", "ir_bp", "ir_dc")
    for name, info in formulas.items():
        deps = info.get("depends", []) + list(info.get("intermediate_signals", {}).keys())
        formula = str(info.get("formula", ""))
        assert not s03.is_stage2_ir_feature(name), name
        assert not any(token in deps for token in forbidden_dep_tokens), name
        assert "IR_" not in formula and "ir_" not in formula, name


def test_green_reliability_features_capture_three_channel_failure_modes():
    fs = 25
    n = 125
    t = np.arange(n, dtype=float) / fs
    ambient = 1.0e5 + 200.0 * np.sin(2 * np.pi * 0.3 * t)
    ir = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t)
    base = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t)
    aligned = s03.extract_feature_pool_from_window(
        ir, ambient,
        base,
        2.0e6 + 7.8e3 * np.sin(2 * np.pi * 1.2 * t + 0.03),
        2.0e6 + 8.2e3 * np.sin(2 * np.pi * 1.2 * t - 0.02),
        fs=fs,
    )
    single_channel_fake = s03.extract_feature_pool_from_window(
        ir, ambient,
        base,
        np.full_like(base, 2.0e6),
        np.full_like(base, 2.0e6),
        fs=fs,
    )
    tilted = s03.extract_feature_pool_from_window(
        ir, ambient,
        base,
        2.0e6 + 7.8e3 * np.sin(2 * np.pi * 1.2 * t + 0.02),
        2.0e6 + 1.0e3 * np.sin(2 * np.pi * 1.2 * t - 0.04),
        fs=fs,
    )
    all_weak = s03.extract_feature_pool_from_window(
        ir, ambient,
        np.full_like(base, 2.0e6),
        np.full_like(base, 2.0e6),
        np.full_like(base, 2.0e6),
        fs=fs,
    )

    expected = [
        "G_2OF3_AC_SUPPORT",
        "G_TOP2_TO_ALL_AC_RATIO",
        "G_TOP2_CORR_MIN",
        "G_WEAK_CHANNEL_GAP",
        "G_SPATIAL_STABILITY_SCORE",
    ]
    for name in expected:
        assert name in aligned
        assert np.isfinite(aligned[name])

    assert aligned["G_2OF3_AC_SUPPORT"] == 1.0
    assert aligned["G_TOP2_CORR_MIN"] > 0.95
    assert aligned["G_WEAK_CHANNEL_GAP"] < 0.15
    assert aligned["G_SPATIAL_STABILITY_SCORE"] > single_channel_fake["G_SPATIAL_STABILITY_SCORE"]

    assert single_channel_fake["G_2OF3_AC_SUPPORT"] < 1.0
    assert single_channel_fake["G_TOP2_CORR_MIN"] < aligned["G_TOP2_CORR_MIN"]
    assert tilted["G_2OF3_AC_SUPPORT"] >= 2.0 / 3.0
    assert tilted["G_TOP2_CORR_MIN"] > 0.90
    assert all_weak["G_SPATIAL_STABILITY_SCORE"] == 0.0


def test_green_reliability_features_have_deploy_formulas():
    selected = [
        "G_2OF3_AC_SUPPORT",
        "G_TOP2_TO_ALL_AC_RATIO",
        "G_TOP2_CORR_MIN",
        "G_WEAK_CHANNEL_GAP",
        "G_SPATIAL_STABILITY_SCORE",
    ]

    formulas = s08.build_selected_feature_formulas(selected)

    assert set(formulas) == set(selected)
    for info in formulas.values():
        text = json.dumps(info, ensure_ascii=False)
        assert "ir_" not in text
        assert "IR_" not in text


def test_all_s03_window_features_with_acc_export_deploy_script(tmp_path):
    fs = 25
    n = 125
    t = np.arange(n, dtype=float) / fs
    ir = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t)
    ambient = 1.0e5 + 500.0 * np.sin(2 * np.pi * 0.4 * t)
    g1 = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t + 0.01)
    g2 = 2.1e6 + 7.5e3 * np.sin(2 * np.pi * 1.2 * t + 0.03)
    g3 = 1.9e6 + 8.5e3 * np.sin(2 * np.pi * 1.2 * t - 0.02)
    ppg = np.column_stack([ir, ambient, g1, g2, g3, np.zeros(n)])
    acc = np.column_stack([
        0.01 * np.sin(2 * np.pi * 1.0 * t),
        0.02 * np.cos(2 * np.pi * 0.5 * t),
        1.0 + 0.01 * np.sin(2 * np.pi * 0.8 * t),
    ])
    selected = list(s03.extract_window_features(ppg, fs=fs, acc_window=acc).keys())

    joblib.dump(
        {
            "feature_names": selected,
            "fill_values": {name: 0.0 for name in selected},
            "clip_bounds": {},
            "threshold": 0.37,
            "meta": {
                "fs_ppg": float(fs),
                "win_sec": 5.0,
                "step_sec": 1.0,
                "use_stage2_ir": False,
            },
        },
        tmp_path / "model_bundle.pkl",
    )

    out_path = s08.export_feature_extractor_script(str(tmp_path))
    py_compile.compile(str(out_path), doraise=True)
    spec = importlib.util.spec_from_file_location("deploy_feature_extractor_all_features", out_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    vector = module.extract_features(ir, ambient, g1, g2, g3, acc=acc, fs=float(fs))

    assert not [name for name in selected if s03.is_stage2_ir_feature(name)]
    assert list(module.FEATURE_ORDER) == selected
    assert "G_consensus_AC_MAD_range" in module.FEATURE_ORDER
    assert len(vector) == len(selected)


def test_rendered_deploy_feature_extractor_is_project_source_independent():
    selected = [
        "GREEN_CORR",
        "GREEN_AC",
        "AMB_AC",
        "ACC_YSUM",
        "GREEN_DC",
        "AMB_DC",
        "GREEN_XCORR",
        "FFT_PEAK_MEDIAN_RATIO",
        "mode",
    ]
    formulas = {name: {"formula": name, "intermediate_signals": {}} for name in selected}

    script = s08._render_selected_feature_extractor(
        selected,
        {name: 0.0 for name in selected},
        {},  # clip_bounds (empty for this test)
        formulas,
        scripts_dir=Path(__file__).resolve().parent,
    )

    forbidden = [
        "s03_extract_feature_pool",
        "sys.path",
        "_CANDIDATE_CODE_DIRS",
        "SCRIPTS_DIR",
        "from s03",
    ]
    for token in forbidden:
        assert token not in script
    assert "FEATURE_ORDER" in script
    assert "CLIP_BOUNDS" in script
    assert "WINDOW_MODEL_THRESHOLD" in script
    assert "def classify_probability(" in script
    assert "def extract_features(" in script


def test_rendered_deploy_feature_extractor_compiles_and_runs_standalone(tmp_path):
    selected = [
        "GREEN_CORR",
        "GREEN_AC",
        "AMB_AC",
        "ACC_YSUM",
        "GREEN_DC",
        "AMB_DC",
        "GREEN_XCORR",
        "FFT_PEAK_MEDIAN_RATIO",
        "mode",
    ]
    formulas = {name: {"formula": name, "intermediate_signals": {}} for name in selected}
    script = s08._render_selected_feature_extractor(
        selected,
        {name: 0.0 for name in selected},
        {},  # clip_bounds (empty for this test)
        formulas,
        scripts_dir=Path(__file__).resolve().parent,
        window_model_threshold=0.42,
    )
    script_path = tmp_path / "deploy_feature_extractor.py"
    script_path.write_text(script, encoding="utf-8")

    py_compile.compile(str(script_path), doraise=True)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(tmp_path),
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Feature vector: 9 values" in result.stdout
    assert "GREEN_CORR" in result.stdout

    spec = importlib.util.spec_from_file_location("deploy_feature_extractor", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.WINDOW_MODEL_THRESHOLD == 0.42
    assert module.classify_probability(0.41) == 0
    assert module.classify_probability(0.42) == 1


def test_export_feature_extractor_script_embeds_bundle_threshold(tmp_path):
    selected = [
        "GREEN_CORR",
        "GREEN_AC",
        "AMB_AC",
        "ACC_YSUM",
        "GREEN_DC",
        "AMB_DC",
        "GREEN_XCORR",
        "FFT_PEAK_MEDIAN_RATIO",
    ]
    joblib.dump(
        {
            "feature_names": selected,
            "fill_values": {name: 0.0 for name in selected},
            "threshold": 0.37,
        },
        tmp_path / "model_bundle.pkl",
    )

    out_path = s08.export_feature_extractor_script(str(tmp_path))
    script = Path(out_path).read_text(encoding="utf-8")

    assert "WINDOW_MODEL_THRESHOLD = 0.37" in script
    assert "s03_extract_feature_pool" not in script
    assert "sys.path" not in script


def test_s03_stage2_ir_disabled_omits_ir_features_before_selection():
    fs = 25
    n = 125
    t = np.arange(n, dtype=float) / fs
    ir = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t)
    ambient = 1.0e5 + 500.0 * np.sin(2 * np.pi * 0.4 * t)
    g1 = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t + 0.01)
    g2 = 2.1e6 + 7.5e3 * np.sin(2 * np.pi * 1.2 * t + 0.03)
    g3 = 1.9e6 + 8.5e3 * np.sin(2 * np.pi * 1.2 * t - 0.02)
    ppg = np.column_stack([ir, ambient, g1, g2, g3, np.zeros(n)])
    acc = np.column_stack([
        0.01 * np.sin(2 * np.pi * 1.0 * t),
        0.02 * np.cos(2 * np.pi * 0.5 * t),
        1.0 + 0.01 * np.sin(2 * np.pi * 0.8 * t),
    ])

    features = s03.extract_window_features(ppg, fs=fs, acc_window=acc, use_stage2_ir=False)
    ppg_with_different_ir = ppg.copy()
    ppg_with_different_ir[:, 0] = 1.0e6 + 5.0e5 * np.sin(2 * np.pi * 0.7 * t)
    features_with_different_ir = s03.extract_window_features(
        ppg_with_different_ir, fs=fs, acc_window=acc, use_stage2_ir=False
    )

    forbidden = [
        "IR_mean",
        "IRX_bp_skewness",
        "GREEN_IR_BP_CORR",
        "ACC_IR_BP_CORR",
        "AMB_STAGE1_RATIO",
        "IR_DC_LEVEL",
        "corr_Ambient_IR",
    ]
    for name in forbidden:
        assert name not in features
    assert features.keys() == features_with_different_ir.keys()
    for name in features:
        assert np.isclose(features[name], features_with_different_ir[name], equal_nan=True), name
    assert "GREEN_CORR" in features
    assert "AMB_DC" in features


def test_s03_feature_pool_source_keys_are_stage2_ir_free_even_with_acc():
    fs = 25
    n = 125
    t = np.arange(n, dtype=float) / fs
    ir = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t)
    ambient = 1.0e5 + 500.0 * np.sin(2 * np.pi * 0.4 * t)
    g1 = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t + 0.01)
    g2 = 2.1e6 + 7.5e3 * np.sin(2 * np.pi * 1.2 * t + 0.03)
    g3 = 1.9e6 + 8.5e3 * np.sin(2 * np.pi * 1.2 * t - 0.02)
    ppg = np.column_stack([ir, ambient, g1, g2, g3, np.zeros(n)])
    acc = np.column_stack([
        0.01 * np.sin(2 * np.pi * 1.0 * t),
        0.02 * np.cos(2 * np.pi * 0.5 * t),
        1.0 + 0.01 * np.sin(2 * np.pi * 0.8 * t),
    ])

    pool = s03.extract_feature_pool_from_window(ir, ambient, g1, g2, g3, fs=fs)
    window_features = s03.extract_window_features(ppg, fs=fs, acc_window=acc)

    assert not [name for name in pool if s03.is_stage2_ir_feature(name)]
    assert not [name for name in window_features if s03.is_stage2_ir_feature(name)]
    assert not any("IR" in name for name in window_features)
    assert "GREEN_SEG_ACDC_CV" in pool
    assert "GREEN_FFT_harmonic_ratio" in pool
    assert "ACC_GREEN_BP_CORR" in window_features


def test_s05_filters_stale_ir_features_before_training():
    features = [
        "IRX_bp_skewness",
        "GREEN_CORR",
        "corr_Ambient_IR",
        "AMBX_bp_skewness",
    ]

    assert s05.enforce_no_stage2_ir_features(features, "unit-test") == [
        "GREEN_CORR",
        "AMBX_bp_skewness",
    ]


def test_s06_rejects_stage2_ir_features_in_loaded_bundle():
    with pytest.raises(ValueError, match="IR-derived Stage2 features"):
        s06.assert_no_stage2_ir_features(
            ["GREEN_CORR", "IR_mean"],
            "unit-test",
        )


def test_s06_quality_fallback_does_not_use_ir_mean():
    assert s06.compute_quality({"Ambient_std": 0.0, "G_mean_mean": 1.0, "IR_mean": 0.0}) == 1.0


def test_export_feature_extractor_rejects_ir_features_in_stage2_bundle(tmp_path):
    selected = [
        "GREEN_CORR",
        "IRX_bp_skewness",
        "AMBX_bp_skewness",
    ]
    joblib.dump(
        {
            "feature_names": selected,
            "fill_values": {name: 0.0 for name in selected},
            "threshold": 0.37,
            "meta": {"use_stage2_ir": False},
        },
        tmp_path / "model_bundle.pkl",
    )

    with pytest.raises(ValueError, match="IR-derived Stage2 features"):
        s08.export_feature_extractor_script(str(tmp_path))


def test_deploy_feature_extractor_ignores_ir_for_stage2_features(tmp_path):
    selected = ["GREEN_AC", "AMB_AC"]
    joblib.dump(
        {
            "feature_names": selected,
            "fill_values": {name: 0.0 for name in selected},
            "threshold": 0.37,
            "meta": {
                "fs_ppg": 25.0,
                "win_sec": 5.0,
                "step_sec": 1.0,
                "use_stage2_ir": False,
            },
        },
        tmp_path / "model_bundle.pkl",
    )

    out_path = s08.export_feature_extractor_script(str(tmp_path))
    spec = importlib.util.spec_from_file_location("deploy_feature_extractor", out_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    n = 125
    t = np.arange(n, dtype=float) / 25.0
    ir_a = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t)
    ir_b = 1.0e6 + 5.0e5 * np.sin(2 * np.pi * 0.7 * t)
    ambient = 1.0e5 + np.zeros(n)
    g1 = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t)
    g2 = 2.1e6 + 7.5e3 * np.sin(2 * np.pi * 1.2 * t)
    g3 = 1.9e6 + 8.5e3 * np.sin(2 * np.pi * 1.2 * t)

    assert not hasattr(module, "USE_STAGE2_IR")
    assert module.DEFAULT_WINDOW_SEC == 5.0
    assert module.extract_features(ir_a, ambient, g1, g2, g3, fs=25.0) == module.extract_features(
        ir_b, ambient, g1, g2, g3, fs=25.0
    )


def test_s08_deploy_feature_script_and_xgboost_metadata_share_bundle_source(tmp_path):
    selected = [
        "GREEN_CORR",
        "GREEN_AC",
        "AMB_AC",
        "ACC_YSUM",
        "GREEN_DC",
        "AMB_DC",
        "GREEN_XCORR",
        "FFT_PEAK_MEDIAN_RATIO",
    ]
    model = XGBClassifier(
        n_estimators=1,
        max_depth=1,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=0,
    )
    model.fit(
        np.asarray([[0.0] * len(selected), [1.0] * len(selected)], dtype=float),
        np.asarray([0, 1]),
    )
    joblib.dump(
        {
            "feature_names": selected,
            "fill_values": {name: float(i) for i, name in enumerate(selected)},
            "model": model,
            "raw_model": model,
            "threshold": 0.37,
            "quality_thresholds": {},
            "meta": {
                "fs_ppg": 25.0,
                "win_sec": 3.0,
                "step_sec": 1.0,
                "use_stage2_ir": False,
            },
        },
        tmp_path / "model_bundle.pkl",
    )
    (tmp_path / "stage1_threshold.json").write_text(
        '{"deploy_stage1_threshold":{"dc_threshold":3600000,"ac_dc_threshold":0.35}}',
        encoding="utf-8",
    )

    script_path = s08.export_feature_extractor_script(str(tmp_path))
    s08.export_deploy_cookbook(str(tmp_path))

    script = Path(script_path).read_text(encoding="utf-8")
    deploy_xgb = __import__("json").loads((tmp_path / "deploy_xgboost.json").read_text(encoding="utf-8"))

    assert deploy_xgb["feature_order"] == selected
    assert deploy_xgb["threshold"] == 0.37
    assert "FEATURE_ORDER = [\n  \"GREEN_CORR\"" in script
    assert "WINDOW_MODEL_THRESHOLD = 0.37" in script
    assert '"GREEN_AC": 1.0' in script


def test_s06_deploy_package_uses_bundle_features_over_stale_selected_features(tmp_path):
    bundle_features = ["GREEN_CORR", "GREEN_AC"]
    stale_features = ["AMB_AC"]
    model = XGBClassifier(
        n_estimators=1,
        max_depth=1,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=0,
    )
    model.fit(np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float), np.asarray([0, 1]))

    (tmp_path / "stage1_threshold.json").write_text(
        '{"deploy_stage1_threshold":{"dc_threshold":3600000,"ac_dc_threshold":0.35}}',
        encoding="utf-8",
    )
    (tmp_path / "selected_features.json").write_text(
        '{"selected_features":["AMB_AC"]}',
        encoding="utf-8",
    )
    joblib.dump(
        {
            "feature_names": bundle_features,
            "fill_values": {name: 0.0 for name in bundle_features},
            "model": model,
            "raw_model": model,
            "threshold": 0.37,
            "threshold_policy": {"selection_data": "valid_threshold_split"},
            "quality_thresholds": {},
            "feature_quantiles": {},
            "fingerprint": {"unit": "test"},
            "meta": {
                "fs_ppg": 25.0,
                "win_sec": 3.0,
                "step_sec": 1.0,
                "use_stage2_ir": False,
            },
        },
        tmp_path / "model_bundle.pkl",
    )

    s06.export_deploy_artifacts(str(tmp_path), skip_initial_windows=3)

    model_params = (tmp_path / "deploy_package" / "model_params.json").read_text(encoding="utf-8")
    feature_formulas = (tmp_path / "deploy_package" / "feature_formulas.json").read_text(encoding="utf-8")
    deploy_config = (tmp_path / "deploy_package" / "deploy_config.json").read_text(encoding="utf-8")

    assert '"selected_features": [\n    "GREEN_CORR",\n    "GREEN_AC"\n  ]' in model_params
    assert '"n_selected_features": 2' in feature_formulas
    assert '"names": [\n      "GREEN_CORR",\n      "GREEN_AC"\n    ]' in deploy_config
    assert "AMB_AC" not in model_params
    assert "AMB_AC" not in deploy_config


def test_validate_deploy_artifact_consistency_passes_and_catches_feature_drift(tmp_path):
    selected = ["GREEN_CORR", "GREEN_AC"]
    model = XGBClassifier(
        n_estimators=1,
        max_depth=1,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=0,
    )
    model.fit(np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float), np.asarray([0, 1]))

    (tmp_path / "stage1_threshold.json").write_text(
        '{"deploy_stage1_threshold":{"dc_threshold":3600000,"ac_dc_threshold":0.35}}',
        encoding="utf-8",
    )
    joblib.dump(
        {
            "feature_names": selected,
            "fill_values": {name: float(i) for i, name in enumerate(selected)},
            "clip_bounds": {"GREEN_AC": [-1.0, 1.0]},
            "model": model,
            "raw_model": model,
            "threshold": 0.37,
            "quality_thresholds": {},
            "feature_quantiles": {},
            "meta": {
                "fs_ppg": 25.0,
                "win_sec": 3.0,
                "step_sec": 1.0,
                "use_stage2_ir": False,
            },
        },
        tmp_path / "model_bundle.pkl",
    )
    s08.export_feature_extractor_script(str(tmp_path))
    s08.export_deploy_cookbook(str(tmp_path))
    s06.export_deploy_artifacts(str(tmp_path), skip_initial_windows=3)

    report = s08.validate_deploy_artifact_consistency(str(tmp_path))
    assert report["feature_names"] == selected
    assert report["threshold"] == 0.37

    xgb_path = tmp_path / "deploy_xgboost.json"
    deploy_xgb = json.loads(xgb_path.read_text(encoding="utf-8"))
    deploy_xgb["feature_order"] = list(reversed(selected))
    xgb_path.write_text(json.dumps(deploy_xgb), encoding="utf-8")

    with pytest.raises(ValueError, match="deploy_xgboost.json feature_order"):
        s08.validate_deploy_artifact_consistency(str(tmp_path))


def test_export_golden_vectors_and_validate_feature_order(tmp_path):
    selected = [
        "GREEN_CORR",
        "GREEN_AC",
        "AMB_AC",
        "ACC_YSUM",
        "GREEN_DC",
        "AMB_DC",
        "GREEN_XCORR",
        "FFT_PEAK_MEDIAN_RATIO",
    ]
    model = XGBClassifier(
        n_estimators=1,
        max_depth=1,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=0,
    )
    model.fit(
        np.asarray([[0.0] * len(selected), [1.0] * len(selected)], dtype=float),
        np.asarray([0, 1]),
    )
    (tmp_path / "stage1_threshold.json").write_text(
        '{"deploy_stage1_threshold":{"dc_threshold":3600000,"ac_dc_threshold":0.35}}',
        encoding="utf-8",
    )
    joblib.dump(
        {
            "feature_names": selected,
            "fill_values": {name: 0.0 for name in selected},
            "clip_bounds": {},
            "model": model,
            "raw_model": model,
            "threshold": 0.37,
            "quality_thresholds": {},
            "feature_quantiles": {},
            "meta": {
                "fs_ppg": 25.0,
                "win_sec": 3.0,
                "step_sec": 1.0,
                "use_stage2_ir": False,
            },
        },
        tmp_path / "model_bundle.pkl",
    )
    s08.export_feature_extractor_script(str(tmp_path))
    s08.export_deploy_cookbook(str(tmp_path))
    s06.export_deploy_artifacts(str(tmp_path), skip_initial_windows=3)

    golden_path = s08.export_golden_vectors(str(tmp_path))
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))

    assert golden["feature_order"] == selected
    assert golden["vectors"]
    assert golden["vectors"][0]["feature_vector_length"] == len(selected)
    assert 0.0 <= golden["vectors"][0]["probability"] <= 1.0
    s08.validate_deploy_artifact_consistency(str(tmp_path))

    golden["feature_order"] = list(reversed(selected))
    Path(golden_path).write_text(json.dumps(golden), encoding="utf-8")
    with pytest.raises(ValueError, match="golden_vectors.json feature_order"):
        s08.validate_deploy_artifact_consistency(str(tmp_path))
