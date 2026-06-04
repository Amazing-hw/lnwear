import argparse
import subprocess
import sys
from pathlib import Path

import s05_train_final_model as s05


ROOT = Path(__file__).resolve().parent


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

    assert len(grid) == 3 * 2 * 2 * 2 * 2 * 2 * 2 * 2
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
    assert "--model_search" in output
    assert "--max_model_nodes 260" in output
    assert '--model_search_n_estimators "20,30"' in output
    assert '--model_search_max_depth "2"' in output
    assert '--model_search_colsample_bytree "0.7,0.8"' in output


def test_s08_model_search_full_pipeline_stops_before_commercial_compare_by_default():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "s08_run_pipeline.py"),
            "--dry_run",
            "--model_search",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    output = result.stdout + result.stderr
    assert "--model_search" in output
    assert "s07_postprocess_optimize.py" in output
    assert "--replay_split test" in output
    assert "s09_commercial_compare.py" not in output
