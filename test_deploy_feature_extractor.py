from pathlib import Path
import importlib.util
import py_compile
import subprocess
import sys

import joblib
import numpy as np
from xgboost import XGBClassifier

import s06_deploy_eval as s06
import s08_run_pipeline as s08


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
