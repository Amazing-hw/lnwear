import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import s05_train_final_model as s05


ROOT = Path(__file__).resolve().parent


def _model_search_args(**overrides):
    values = {
        "model_search_n_estimators": "20,25,30,35,40,45,50,55,60,70,80",
        "model_search_max_depth": "2,3,4",
        "model_search_learning_rate": "0.025,0.03,0.04,0.05,0.06,0.08,0.10",
        "model_search_min_child_weight": "10,15,20,25,30,40,50",
        "model_search_reg_lambda": "5,8,10,12,16,20,30",
        "model_search_reg_alpha": "0,0.5,1,1.5,2,3",
        "model_search_subsample": "0.70,0.75,0.80,0.85,0.90",
        "model_search_colsample_bytree": "0.70,0.75,0.80,0.85,0.90",
        "model_search_max_candidates": 600,
        "model_search_random_state": 42,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _sample_name_series(values):
    try:
        return pd.Series(values, dtype="string[pyarrow]")
    except (ImportError, TypeError, ValueError):
        return pd.Series(values, dtype="string")


def _grouped_valid_frame():
    rows = []
    for group_idx in range(8):
        for window_idx in range(2):
            rows.append(
                {
                    "sample_name": f"sample_{group_idx}",
                    "target": group_idx % 2,
                    "window_index": window_idx,
                    "feature": float(group_idx + window_idx),
                }
            )
    df = pd.DataFrame(rows)
    df["sample_name"] = _sample_name_series(df["sample_name"].tolist())
    return df


def test_group_split_accepts_arrow_string_sample_names():
    df_valid = _grouped_valid_frame()

    df_calib, df_threshold, split_meta = s05.split_valid_for_calibration_threshold(
        df_valid,
        threshold_fraction=0.5,
        random_state=7,
    )
    df_model_select, df_search_calib, search_meta = s05.split_calibration_for_model_search(
        df_valid,
        search_fraction=0.5,
        random_state=7,
    )

    assert split_meta["fallback"] is False
    assert search_meta["fallback"] is False
    assert set(df_calib["sample_name"]).isdisjoint(set(df_threshold["sample_name"]))
    assert set(df_model_select["sample_name"]).isdisjoint(set(df_search_calib["sample_name"]))


def test_prepare_valid_calibration_threshold_data_keeps_disjoint_groups_for_multi_k():
    df_valid = _grouped_valid_frame()

    prepared = s05.prepare_valid_calibration_threshold_data(
        df_valid,
        ["feature"],
        {"feature": 0.0},
        threshold_fraction=0.5,
        random_state=7,
    )

    split = prepared["split"]
    assert split["fallback"] is False
    assert set(split["calibration_groups"]).isdisjoint(set(split["threshold_groups"]))
    assert len(prepared["X_calib"]) == len(prepared["y_calib"]) == len(prepared["df_calib"])
    assert len(prepared["X_threshold"]) == len(prepared["y_threshold"]) == len(prepared["df_threshold"])
    assert set(prepared["df_calib"]["sample_name"]).isdisjoint(
        set(prepared["df_threshold"]["sample_name"])
    )


def test_model_search_stability_summary_flags_close_top_candidates_and_default_strength():
    df = pd.DataFrame([
        {
            "eligible": True,
            "mean_cv_accuracy": 0.9820,
            "std_cv_accuracy": 0.004,
            "mean_cv_fp_rate": 0.02,
            "feature_count": 10,
            "final_total_nodes": 180,
            "is_default_params": False,
        },
        {
            "eligible": True,
            "mean_cv_accuracy": 0.9815,
            "std_cv_accuracy": 0.003,
            "mean_cv_fp_rate": 0.015,
            "feature_count": 15,
            "final_total_nodes": 240,
            "is_default_params": True,
        },
    ])

    summary = s05.summarize_model_search_stability(df)

    assert summary["available"] is True
    assert summary["top_accuracy_margin"] == pytest.approx(0.0005)
    assert summary["close_top_candidate_count"] == 2
    assert summary["default_params_rank"] == 2
    assert summary["is_unstable"] is True


def test_parse_model_search_grid_exposes_adjustable_values():
    args = argparse.Namespace(
        model_search_n_estimators="20,30,40",
        model_search_max_depth="2,3",
        model_search_learning_rate="0.03,0.05",
        model_search_min_child_weight="20,50",
        model_search_reg_lambda="10,20",
        model_search_reg_alpha="1,2",
        model_search_subsample="0.7,0.8",
        model_search_colsample_bytree="0.7,0.9",
    )

    grid = s05.build_model_search_grid(args, scale_pos_weight=1.25)

    assert len(grid) == 3 * 2 * 2 * 2 * 2 * 2 * 2 * 2 + 1
    assert {
        "n_estimators": 20,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 20,
        "reg_lambda": 10.0,
        "reg_alpha": 1.0,
        "scale_pos_weight": 1.25,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
    } in grid
    assert any(s05.is_default_xgb_params(params) for params in grid)


def test_default_model_search_axes_have_fine_n_estimators():
    args = _model_search_args()

    axes = s05.build_model_search_axes(args)
    n_estimators = set(axes["n_estimators"])

    assert {25, 35, 45, 55}.issubset(n_estimators)


def test_model_search_grid_force_includes_default_params_when_custom_grid_excludes_it():
    args = _model_search_args(
        model_search_n_estimators="20",
        model_search_max_depth="2",
        model_search_learning_rate="0.03",
        model_search_min_child_weight="30",
        model_search_reg_lambda="20",
        model_search_reg_alpha="2",
        model_search_subsample="0.7",
        model_search_colsample_bytree="0.7",
        model_search_max_candidates=10,
    )

    grid = s05.build_model_search_grid(args, scale_pos_weight=1.0)

    assert any(s05.is_default_xgb_params(params) for params in grid)


def test_model_search_grid_sampling_is_deterministic_and_keeps_default():
    args = _model_search_args(model_search_max_candidates=25, model_search_random_state=7)

    grid1 = s05.build_model_search_grid(args, scale_pos_weight=1.0)
    grid2 = s05.build_model_search_grid(args, scale_pos_weight=1.0)

    assert len(grid1) == 25
    assert grid1 == grid2
    assert any(s05.is_default_xgb_params(params) for params in grid1)


def test_model_search_score_penalizes_fp_and_oversized_models():
    good = s05.score_model_search_candidate(
        {"accuracy": 0.98, "confusion_matrix": {"TN": 95, "FP": 5, "FN": 2, "TP": 98}},
        total_nodes=260,
        max_model_nodes=400,
        fp_cost=2.0,
        size_cost=0.1,
    )
    fp_heavy = s05.score_model_search_candidate(
        {"accuracy": 0.99, "confusion_matrix": {"TN": 80, "FP": 20, "FN": 1, "TP": 99}},
        total_nodes=260,
        max_model_nodes=400,
        fp_cost=2.0,
        size_cost=0.1,
    )
    oversized = s05.score_model_search_candidate(
        {"accuracy": 0.99, "confusion_matrix": {"TN": 95, "FP": 5, "FN": 1, "TP": 99}},
        total_nodes=401,
        max_model_nodes=400,
        fp_cost=2.0,
        size_cost=0.1,
    )

    assert good["eligible"] is True
    assert fp_heavy["eligible"] is True
    assert good["score"] > fp_heavy["score"]
    assert oversized["eligible"] is False
    assert oversized["score"] == float("-inf")


def test_accuracy_first_model_search_prefers_smaller_model_within_tolerance():
    records = [
        {
            "eligible": True,
            "selection_accuracy": 0.9820,
            "selection_fp_rate": 0.02,
            "total_nodes": 390,
            "is_default_params": False,
        },
        {
            "eligible": True,
            "selection_accuracy": 0.9805,
            "selection_fp_rate": 0.02,
            "total_nodes": 180,
            "is_default_params": False,
        },
        {
            "eligible": True,
            "selection_accuracy": 0.9760,
            "selection_fp_rate": 0.0,
            "total_nodes": 80,
            "is_default_params": False,
        },
    ]

    chosen = s05.choose_accuracy_first_model_search_record(records, accuracy_tolerance=0.002)

    assert chosen["selection_accuracy"] == 0.9805
    assert chosen["total_nodes"] == 180
    assert chosen["chosen_reason"] == "within_accuracy_tolerance_smallest_model"


def test_accuracy_first_model_search_default_strictly_prefers_best_accuracy():
    records = [
        {
            "eligible": True,
            "selection_accuracy": 0.9820,
            "selection_fp_rate": 0.02,
            "total_nodes": 390,
            "is_default_params": False,
        },
        {
            "eligible": True,
            "selection_accuracy": 0.9805,
            "selection_fp_rate": 0.02,
            "total_nodes": 180,
            "is_default_params": False,
        },
    ]

    chosen = s05.choose_accuracy_first_model_search_record(records)

    assert chosen["selection_accuracy"] == 0.9820
    assert chosen["total_nodes"] == 390
    assert chosen["chosen_reason"] == "max_accuracy"


def test_accuracy_first_model_search_keeps_default_when_smaller_model_is_worse():
    records = [
        {
            "eligible": True,
            "selection_accuracy": 0.9810,
            "selection_fp_rate": 0.03,
            "total_nodes": 360,
            "is_default_params": True,
        },
        {
            "eligible": True,
            "selection_accuracy": 0.9760,
            "selection_fp_rate": 0.01,
            "total_nodes": 90,
            "is_default_params": False,
        },
    ]

    chosen = s05.choose_accuracy_first_model_search_record(records, accuracy_tolerance=0.002)

    assert chosen["is_default_params"] is True
    assert chosen["selection_accuracy"] == 0.9810
    assert chosen["chosen_reason"] == "max_accuracy"


def test_cv_model_search_keeps_default_until_candidate_beats_it():
    records = [
        {
            "eligible": True,
            "mean_cv_accuracy": 0.982,
            "std_cv_accuracy": 0.01,
            "mean_cv_fp_rate": 0.03,
            "final_total_nodes": 360,
            "is_default_params": True,
        },
        {
            "eligible": True,
            "mean_cv_accuracy": 0.982,
            "std_cv_accuracy": 0.005,
            "mean_cv_fp_rate": 0.01,
            "final_total_nodes": 120,
            "is_default_params": False,
        },
    ]

    chosen = s05.choose_cv_model_search_record(records, accuracy_tolerance=0.0)

    assert chosen["is_default_params"] is True
    assert chosen["chosen_reason"] == "default_params_baseline_not_beaten"


def test_cv_model_search_prefers_better_candidate_and_filters_oversized():
    records = [
        {
            "eligible": True,
            "mean_cv_accuracy": 0.982,
            "std_cv_accuracy": 0.01,
            "mean_cv_fp_rate": 0.03,
            "final_total_nodes": 360,
            "is_default_params": True,
        },
        {
            "eligible": False,
            "mean_cv_accuracy": 0.990,
            "std_cv_accuracy": 0.01,
            "mean_cv_fp_rate": 0.01,
            "final_total_nodes": 999,
            "is_default_params": False,
        },
        {
            "eligible": True,
            "mean_cv_accuracy": 0.984,
            "std_cv_accuracy": 0.02,
            "mean_cv_fp_rate": 0.04,
            "final_total_nodes": 300,
            "is_default_params": False,
        },
    ]

    chosen = s05.choose_cv_model_search_record(records, accuracy_tolerance=0.0)

    assert chosen["is_default_params"] is False
    assert chosen["mean_cv_accuracy"] == 0.984
    assert chosen["beats_default_params"] is True
    assert chosen["chosen_reason"] == "best_cv_accuracy"


def test_model_search_result_rows_include_accuracy_first_fields():
    rows = s05.build_model_search_result_rows([
        {
            "rank_input_order": 1,
            "eligible": True,
            "score": 0.98,
            "fp_rate": 0.02,
            "size_ratio": 0.5,
            "total_nodes": 200,
            "final_total_nodes": 200,
            "avg_nodes_per_tree": 5.0,
            "selection_threshold": 0.42,
            "selection_accuracy": 0.981,
            "selection_fp_rate": 0.01,
            "mean_cv_accuracy": 0.982,
            "std_cv_accuracy": 0.003,
            "mean_cv_fp_rate": 0.01,
            "mean_cv_precision": 0.96,
            "mean_cv_recall": 0.97,
            "cv_folds_completed": 6,
            "beats_default_params": True,
            "chosen_reason": "max_accuracy",
            "is_default_params": True,
            "metrics": {"accuracy": 0.9, "confusion_matrix": {"TN": 9, "FP": 1, "FN": 0, "TP": 10}},
            "selection_metrics": {"precision": 0.95, "recall": 1.0},
            "params": {"n_estimators": 40},
        }
    ])

    assert rows[0]["selection_threshold"] == 0.42
    assert rows[0]["selection_accuracy"] == 0.981
    assert rows[0]["selection_fp_rate"] == 0.01
    assert rows[0]["mean_cv_accuracy"] == 0.982
    assert rows[0]["std_cv_accuracy"] == 0.003
    assert rows[0]["final_total_nodes"] == 200
    assert rows[0]["cv_folds_completed"] == 6
    assert rows[0]["beats_default_params"] is True
    assert rows[0]["is_default_params"] is True
    assert rows[0]["chosen_reason"] == "max_accuracy"


def test_s08_dry_run_exposes_model_search_params_to_s05():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--stop_after",
            "s05",
            "--model_search",
            "--max_features",
            "12",
            "--max_model_nodes",
            "260",
            "--model_search_n_estimators",
            "20,30",
            "--model_search_max_depth",
            "2",
            "--model_search_learning_rate",
            "0.05",
            "--model_search_min_child_weight",
            "30,50",
            "--model_search_reg_lambda",
            "20",
            "--model_search_reg_alpha",
            "2",
            "--model_search_subsample",
            "0.8",
            "--model_search_colsample_bytree",
            "0.7,0.8",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "--max_features 12" in output
    assert " --model_search " in output
    assert "--no-model_search" not in output
    assert "--max_model_nodes 260" in output
    assert "--model_search_strategy staged_group_cv" in output
    assert "--model_search_max_candidates 600" in output
    assert "--model_search_stage2_top_k 80" in output
    assert "--model_search_cv_folds 3" in output
    assert "--model_search_cv_repeats 2" in output
    assert "--model_search_random_state 42" in output
    assert "--model_search_accuracy_tolerance 0.0 " in output
    assert "--model_search_accuracy_tolerance 0.002 " not in output
    assert '--model_search_n_estimators "20,30"' in output
    assert '--model_search_max_depth "2"' in output
    assert '--model_search_colsample_bytree "0.7,0.8"' in output


def test_s08_default_includes_model_search_but_skips_npz_and_postprocess_search():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert " --model_search " in output
    assert "--model_search_strategy staged_group_cv" in output
    assert " --optimize " not in output
    assert "s07_postprocess_optimize.py" not in output
    assert "--export_window_cache" not in output
    assert "s09_commercial_compare.py" not in output


def test_s08_model_search_can_explicitly_run_npz_and_postprocess_search():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--model_search",
            "--export_window_cache",
            "--optimize_postprocess",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "s06_deploy_eval.py" in output
    assert "--export_window_cache" in output
    assert output.count("--export_window_cache") == 2
    assert "s07_postprocess_optimize.py" in output
    assert "--replay_split test" in output
    assert "s09_commercial_compare.py" not in output


def test_s08_model_search_can_be_disabled_for_fast_dry_runs():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--no-model_search",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "--no-model_search" in output
    assert " --model_search " not in output
