# s08_run_pipeline.py
# -*- coding: utf-8 -*-
"""
主控脚本：人工可审计特征流程与显式无人值守训练流程。

默认命令运行到 s04，导出完整排序后等待人工固化特征：
    python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts

人工只修改 manual_feature_selection.csv 的 selected 列后恢复训练：
    python s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04

恢复流程严格使用 CSV 中的特征名称、顺序和数量，执行 XGBoost 搜参、
train OOF hard-negative 候选、评估和部署导出；不搜索特征数量。

显式无人值守流程（含 XGBoost 搜参与部署导出）使用：
    python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --feature_selection_mode auto

无人值守流程会运行到 s06_cb：
    s01 数据切分
    s02 Stage1 固定阈值
    s03 Stage2 特征窗口（默认 5s/1s，也支持 3s/1s）
    s04 特征筛选 + s04_search 候选子集搜索
    s05 XGBoost 训练；默认执行节点预算约束下的 staged group-CV 搜参
    s06_eval 用当前已固化/默认状态机做 test 端到端评估
    s06_xpt/s06_feat/s06_plot/s06_cb 导出部署产物、特征脚本、错误图和部署配方
    s06_cb 后自动校验部署特征顺序、阈值、fill/clip 与 model_bundle.pkl 完全一致
    s06_cb 后同时导出 golden_vectors.json，供端侧实现做特征向量和概率对齐

legacy s06 状态机优化、NPZ 缓存导出和 s07 后处理搜参很耗时，默认不跑；需要时显式运行：
    python s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04 --optimize
    python s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04 --with_postprocess

泛化审计只读取已有评估 artifacts，不重新训练；需要时显式运行：
    python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --run_generalization_audit --stop_after s06_audit

如果 --stop_after 直接指向 s07_post/s06_audit，脚本会视为显式请求并自动打开对应可选步骤。

Full optimization shortcut:
    python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --full_optimize

Stage2 no-IR policy:
    IR is reserved for Stage1 DC/ACDC gating only. Stage2 feature selection,
    XGBoost training, s06/s07 evaluation, deploy_feature_extractor.py, and
    golden_vectors.json use only ambient, green, and ACC features.

Preflight gates before a real-data run:
    python -m py_compile s01_data_split.py s02_ir_dc_threshold.py s03_extract_feature_pool.py s04_feature_selection.py s05_train_final_model.py s06_deploy_eval.py s07_postprocess_optimize.py s08_run_pipeline.py
    python -m pytest test_deploy_feature_extractor.py test_end_to_end_pipeline_guard.py -q --basetemp .pytest_tmp_deploy_guard

Do not pass an empty string to --model_search_feature_counts. For a fixed feature
count in auto mode, pass one explicit value, for example:
    python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --feature_selection_mode auto --max_features 15 --model_search_feature_counts 15

用法:
    # 第一阶段：生成排序 CSV 后暂停
    python new/s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts

    # 第二阶段：人工编辑 CSV 后恢复训练、评估和部署导出
    python new/s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04

    # 需要时再显式打开 NPZ 缓存导出和 s07 后处理搜参
    python new/s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04 --with_postprocess

    # 跳过某些步骤
    python new/s08_run_pipeline.py --skip s02,s03

    # 只跑到特征筛选
    python new/s08_run_pipeline.py --stop_after s04

    # 只复用已有 artifacts 做评估和部署导出
    python new/s08_run_pipeline.py --artifact_dir artifacts --skip s01,s02,s03,s04,s04_search,s05

流程:
    s01: 数据扫描 & train/valid/test 切分
    s02: Stage1 IR DC/ACDC 固定阈值配置
    s03: Stage2 特征池提取（预切窗直接使用；连续时序按 window_sec/stride_sec 滑窗）
    s04: 稳定性特征筛选
    s05: XGBoost 最终模型训练
    s06_opt:  legacy 状态机参数网格搜索（默认不跑；需 --optimize）
    s06_cache: 导出 valid 逐窗缓存（默认不跑；需 --export_window_cache）
    s07_post: FP 敏感后处理搜参（默认不跑；需 --optimize_postprocess）
    s06_eval: 端到端评估
    s06_audit: 泛化审计（默认不跑；需 --run_generalization_audit，由 s06_deploy_eval 内嵌执行）
    s06_xpt: 导出部署产物 (--export_deploy)
"""

import argparse
import glob
import os
import json
import logging
import re
import shlex
import subprocess
import sys
import time
import joblib

from s03_extract_feature_pool import is_stage2_ir_feature
from s04_feature_selection import (
    is_deployment_allowed_feature,
    summarize_deployment_feature_costs,
)
from stage2_feature_catalog import (
    FEATURE_CATALOG,
    FEATURE_POOL_VERSION,
    build_selected_feature_contract,
    feature_record as stage2_feature_record,
    model_candidate_names as stage2_model_candidate_names,
    selected_catalog as selected_stage2_catalog,
)
import ast
import importlib.util
from datetime import timedelta


THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def configure_thread_env():
    """Cap inherited BLAS/OpenMP threads unless the caller already set them."""
    for name, value in THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)
    return {name: os.environ.get(name) for name in THREAD_ENV_DEFAULTS}


def dataset_has_h5_files(dataset_dir):
    patterns = [os.path.join(dataset_dir, "*.h5")]
    if not os.path.isabs(dataset_dir):
        patterns.append(os.path.join("..", dataset_dir, "*.h5"))
    return any(glob.glob(pattern) for pattern in patterns)


def _read_s05_quick_k_score(artifact_dir):
    """Read the window-accuracy score written by a quick s05 run."""
    config_path = os.path.join(artifact_dir, "final_model_config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None

    for section in (
        "valid_best_threshold_metrics",
        "threshold_split_best_metrics",
        "valid_default_threshold_metrics",
    ):
        metrics = config.get(section) or {}
        try:
            value = metrics.get("accuracy")
            if value is not None:
                return float(value)
        except Exception:
            continue

    try:
        value = config.get("model_search", {}).get("feature_search", {}).get("best_score")
        if value is not None:
            return float(value)
    except Exception:
        pass
    return None


AUTO_E2E_SCORE_WEIGHTS = {
    "sample_accuracy": 0.30,
    "sample_recall": 0.20,
    "window_accuracy": 0.15,
    "sample_fp_rate": -0.25,
    "false_worn_event_rate": -0.20,
    "latency_penalty": -0.10,
    "deploy_cost": -0.05,
}


def _finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if value != value or value in (float("inf"), float("-inf")):
        return float(default)
    return value


def pipeline_acceptance_exit_code(report, stopped_at=None):
    """Return a failing exit code only for a completed, rejected pipeline."""
    if stopped_at:
        return 0
    return 0 if bool((report or {}).get("overall_passed")) else 2


def score_auto_e2e_metrics(metrics, deploy_cost=0.0, max_first_worn_output_p95_sec=6.0):
    """Product-oriented score for automatic end-to-end model selection."""
    metrics = metrics or {}
    max_latency = max(_finite_float(max_first_worn_output_p95_sec, 6.0), 1e-6)
    latency = _finite_float(metrics.get("first_worn_output_p95_sec"), max_latency)
    latency_penalty = max(0.0, latency / max_latency)
    return float(
        AUTO_E2E_SCORE_WEIGHTS["sample_accuracy"] * _finite_float(metrics.get("sample_accuracy"))
        + AUTO_E2E_SCORE_WEIGHTS["sample_recall"] * _finite_float(metrics.get("sample_recall"))
        + AUTO_E2E_SCORE_WEIGHTS["window_accuracy"] * _finite_float(metrics.get("window_accuracy"))
        + AUTO_E2E_SCORE_WEIGHTS["sample_fp_rate"] * _finite_float(metrics.get("sample_fp_rate"))
        + AUTO_E2E_SCORE_WEIGHTS["false_worn_event_rate"] * _finite_float(metrics.get("false_worn_event_rate"))
        + AUTO_E2E_SCORE_WEIGHTS["latency_penalty"] * latency_penalty
        + AUTO_E2E_SCORE_WEIGHTS["deploy_cost"] * _finite_float(deploy_cost)
    )


def auto_e2e_constraints_pass(metrics, baseline_window_accuracy=None, constraints=None):
    constraints = constraints or {}
    metrics = metrics or {}
    if _finite_float(metrics.get("sample_fp_rate"), float("inf")) > _finite_float(constraints.get("max_sample_fp_rate"), float("inf")):
        return False
    if _finite_float(metrics.get("false_worn_event_rate"), float("inf")) > _finite_float(constraints.get("max_false_worn_event_rate"), float("inf")):
        return False
    if _finite_float(metrics.get("first_worn_output_p95_sec"), float("inf")) > _finite_float(constraints.get("max_first_worn_output_p95_sec"), float("inf")):
        return False
    if baseline_window_accuracy is not None:
        min_delta = _finite_float(constraints.get("min_window_accuracy_delta"), -0.01)
        min_window = _finite_float(baseline_window_accuracy) + min_delta
        if _finite_float(metrics.get("window_accuracy"), 0.0) < min_window:
            return False
    return True


def select_auto_e2e_candidate(candidates, baseline_window_accuracy=None, constraints=None):
    """Select the best E2E candidate, preferring candidates that satisfy hard constraints."""
    constraints = constraints or {}
    scored = []
    for item in candidates or []:
        enriched = dict(item)
        metrics = dict(enriched.get("valid_metrics") or enriched.get("metrics") or {})
        deploy_cost = _finite_float(enriched.get("deploy_cost"), 0.0)
        enriched["valid_metrics"] = metrics
        enriched["deploy_cost"] = deploy_cost
        enriched["constraint_pass"] = auto_e2e_constraints_pass(
            metrics,
            baseline_window_accuracy=baseline_window_accuracy,
            constraints=constraints,
        )
        enriched["auto_score"] = score_auto_e2e_metrics(
            metrics,
            deploy_cost=deploy_cost,
            max_first_worn_output_p95_sec=constraints.get("max_first_worn_output_p95_sec", 6.0),
        )
        scored.append(enriched)
    if not scored:
        raise ValueError("select_auto_e2e_candidate requires at least one candidate")
    scored.sort(
        key=lambda item: (
            bool(item.get("constraint_pass", False)),
            _finite_float(item.get("auto_score"), float("-inf")),
            _finite_float(item.get("valid_metrics", {}).get("sample_accuracy"), 0.0),
        ),
        reverse=True,
    )
    return scored[0]


def _extract_replay_metrics(payload, selection_split="valid", replay_split="test"):
    if not isinstance(payload, dict):
        return None, None
    selection = payload.get("selection") or {}
    replay = payload.get("replay") or {}
    valid_metrics = selection.get("metrics") if selection else None
    test_metrics = replay.get("metrics") if replay else None
    return valid_metrics, test_metrics


def export_auto_e2e_summary(artifact_dir, postprocess_split="valid", split="test",
                            constraints=None, dry_run=False):
    """Write the automatic E2E selection summary from the current pipeline artifacts."""
    out_dir = os.path.join(os.fspath(artifact_dir), "auto_optimize")
    summary_path = os.path.join(out_dir, "auto_optimization_summary.json")
    candidate_scores_path = os.path.join(out_dir, "candidate_scores.csv")
    candidate_manifest_path = os.path.join(out_dir, "candidate_manifest.json")
    if dry_run:
        print(f"[auto_e2e] would write {summary_path}")
        return None

    replay_path = os.path.join(
        os.fspath(artifact_dir),
        "postprocess_opt",
        f"postprocess_replay_{postprocess_split}_to_{split}.json",
    )
    if not os.path.exists(replay_path):
        replay_path = os.path.join(
            os.fspath(artifact_dir),
            "postprocess_opt",
            f"postprocess_replay_{postprocess_split}.json",
        )
    replay_payload = _load_json_if_exists(replay_path) or {}
    valid_metrics, test_metrics = _extract_replay_metrics(
        replay_payload,
        selection_split=postprocess_split,
        replay_split=split,
    )
    if not valid_metrics:
        print(f"[auto_e2e] replay metrics not found, skip summary: {replay_path}")
        return None

    eval_payload = _load_json_if_exists(
        os.path.join(os.fspath(artifact_dir), f"end_to_end_eval_{split}_state_machine.json")
    ) or {}
    baseline_window = (eval_payload.get("window_model_summary") or {}).get("accuracy")
    perf = _load_json_if_exists(os.path.join(os.fspath(artifact_dir), "deploy_performance_profile.json")) or {}
    feature_cost = perf.get("feature_cost_summary") or {}
    deploy_cost = _finite_float(feature_cost.get("deployment_cost_mean"), 0.0)
    candidate = {
        "candidate": "auto_e2e_current_best",
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "deploy_cost": deploy_cost,
        "source": replay_path,
    }
    selected = select_auto_e2e_candidate(
        [candidate],
        baseline_window_accuracy=baseline_window,
        constraints=constraints,
    )
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "source": "s08_run_pipeline --auto_optimize_e2e",
        "selection_policy": "product_metrics_first",
        "constraints": constraints or {},
        "score_weights": AUTO_E2E_SCORE_WEIGHTS,
        "baseline_window_accuracy": baseline_window,
        "selected_candidate": selected,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(candidate_scores_path, "w", encoding="utf-8") as f:
        f.write("candidate,constraint_pass,auto_score,sample_accuracy,sample_recall,window_accuracy,sample_fp_rate,false_worn_event_rate,first_worn_output_p95_sec,deploy_cost\n")
        m = selected.get("valid_metrics", {})
        f.write(
            f"{selected.get('candidate')},{selected.get('constraint_pass')},"
            f"{selected.get('auto_score')},"
            f"{m.get('sample_accuracy')},{m.get('sample_recall')},{m.get('window_accuracy')},"
            f"{m.get('sample_fp_rate')},{m.get('false_worn_event_rate')},"
            f"{m.get('first_worn_output_p95_sec')},{selected.get('deploy_cost')}\n"
        )
    manifest = {
        "source": "s08_run_pipeline --auto_optimize_e2e",
        "selected_candidate": selected.get("candidate"),
        "constraint_pass": bool(selected.get("constraint_pass", False)),
        "artifacts": {
            "summary": summary_path,
            "candidate_scores": candidate_scores_path,
            "postprocess_replay": replay_path,
            "end_to_end_eval": os.path.join(
                os.fspath(artifact_dir), f"end_to_end_eval_{split}_state_machine.json"
            ),
            "deploy_performance_profile": os.path.join(
                os.fspath(artifact_dir), "deploy_performance_profile.json"
            ),
        },
    }
    with open(candidate_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[OK] auto E2E summary -> {summary_path}")
    print(f"[OK] auto E2E candidate scores -> {candidate_scores_path}")
    print(f"[OK] auto E2E candidate manifest -> {candidate_manifest_path}")
    return summary_path



def _json_float_map(values, feature_names):
    out = {}
    for name in feature_names:
        value = values.get(name, 0.0)
        out[name] = float(0.0 if value is None else value)
    return out


def _json_clip_map(clip_bounds, feature_names):
    """Convert clip_bounds dict to a JSON-safe form with only selected features."""
    out = {}
    for name in feature_names:
        bound = clip_bounds.get(name)
        if bound is not None and isinstance(bound, (list, tuple)) and len(bound) == 2:
            out[name] = [float(bound[0]), float(bound[1])]
    return out


def _stage2_ir_selected_features(feature_names):
    return [name for name in feature_names if is_stage2_ir_feature(name)]


def build_selected_feature_formulas(selected_features):
    ir_features = _stage2_ir_selected_features(selected_features)
    if ir_features:
        raise ValueError(
            "Stage2 deployment features must not include IR-derived features. "
            "Please rerun s03-s05 after the no-IR Stage2 policy update. Offending features: "
            + ", ".join(ir_features)
        )
    selected = selected_stage2_catalog(selected_features)
    return {
        name: {
            "formula": record["formula"],
            "preprocessing": record["preprocessing"],
            "signal_source": record["signal_source"],
            "unit": record["unit"],
            "numerical_guard": record["numerical_guard"],
            "c_operators": list(record["c_operators"]),
            "c_abs_tolerance": float(record["c_abs_tolerance"]),
            "c_rel_tolerance": float(record["c_rel_tolerance"]),
            "intermediate_signals": {},
        }
        for name, record in selected.items()
    }


def _assert_selected_features_deployment_allowed(selected_features):
    forbidden_features = [
        name for name in selected_features
        if not is_deployment_allowed_feature(name)
    ]
    if forbidden_features:
        raise ValueError(
            "Selected features contain non deployment-friendly operators. "
            "Deployment export will not silently crop model inputs; rerun s04-s05 so "
            "feature filtering happens before model training. Offending features: "
            + ", ".join(forbidden_features)
        )


def _render_deployment_feature_extractor(selected_features, fill_values, clip_bounds, formulas,
                                         window_model_threshold=0.5,
                                         default_fs=25.0,
                                         window_sec=5.0):
    """Render a standalone Python extractor with the governed s03 engine inlined."""
    order_json = json.dumps(selected_features, ensure_ascii=False, indent=2)
    fill_json = json.dumps(fill_values, ensure_ascii=False, indent=2)
    clip_json = json.dumps(clip_bounds, ensure_ascii=False, indent=2)
    formulas_json = json.dumps(formulas, ensure_ascii=False, indent=2)
    threshold_json = json.dumps(float(window_model_threshold), ensure_ascii=False)
    default_fs_py = repr(float(default_fs))
    window_sec_py = repr(float(window_sec))
    feature_engine = _standalone_feature_engine_source()

    return f'''# -*- coding: utf-8 -*-
"""Auto-generated standalone deployment feature extractor.

The governed training feature engine is embedded at export time. This file has
no project-source dependency and can be deployed together with final_model.json.
Do not edit generated formulas by hand.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict

import numpy as np
from scipy.signal import resample_poly


FEATURE_ORDER = {order_json}
FILL_VALUES = {fill_json}
CLIP_BOUNDS = {clip_json}
FEATURE_FORMULAS = {formulas_json}
WINDOW_MODEL_THRESHOLD = {threshold_json}
DEFAULT_FS = {default_fs_py}
DEFAULT_WINDOW_SEC = {window_sec_py}


{feature_engine}


def _finite_1d(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    if mask.all():
        return arr
    fill = float(np.median(arr[mask])) if mask.any() else 0.0
    return np.where(mask, arr, fill).astype(float)


def _clean_value(name, value):
    if value is None or not np.isfinite(value):
        return float(FILL_VALUES.get(name, 0.0))
    v = float(value)
    bound = CLIP_BOUNDS.get(name)
    if bound is not None and isinstance(bound, (list, tuple)) and len(bound) == 2:
        lo, hi = float(bound[0]), float(bound[1])
        v = min(max(v, lo), hi)
    return v


def _prepare_acc(acc, n):
    if acc is None:
        return None
    arr = np.asarray(acc, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 3:
        pad = np.zeros((arr.shape[0], 3 - arr.shape[1]), dtype=float)
        arr = np.hstack([arr, pad])
    arr = arr[:n, :3]
    if arr.shape[0] == 0:
        return None
    mask = np.isfinite(arr)
    if not mask.all():
        arr = np.where(mask, arr, 0.0).astype(float)
    return arr


def extract_raw_feature_dict(ir, ambient, g1, g2, g3, acc=None, fs=25, ppg_config=0):
    ir = _finite_1d(ir)
    ambient = _finite_1d(ambient)
    g1 = _finite_1d(g1)
    g2 = _finite_1d(g2)
    g3 = _finite_1d(g3)
    n = min(ir.size, ambient.size, g1.size, g2.size, g3.size)
    if n <= 0:
        raw = {{}}
    else:
        ppg = np.column_stack([
            ir[:n],
            ambient[:n],
            np.zeros(n, dtype=float),
            g1[:n],
            g2[:n],
            g3[:n],
        ])
        raw = extract_window_features(
            ppg,
            fs=float(fs),
            acc_window=_prepare_acc(acc, n),
            use_stage2_ir=False,
            ppg_config=0,
        )
        raw["mode"] = float(ppg_config)
    missing = [name for name in FEATURE_ORDER if name not in raw]
    if missing:
        raise KeyError(
            "Selected deployment features missing from governed feature pool: "
            + ", ".join(missing)
        )
    return {{name: float(raw[name]) for name in FEATURE_ORDER}}


def extract_feature_dict(ir, ambient, g1, g2, g3, acc=None, fs=25, ppg_config=0):
    raw = extract_raw_feature_dict(
        ir, ambient, g1, g2, g3, acc=acc, fs=fs, ppg_config=ppg_config
    )
    return {{name: _clean_value(name, raw[name]) for name in FEATURE_ORDER}}


def extract_features(ir, ambient, g1, g2, g3, acc=None, fs=25, ppg_config=0):
    feature_dict = extract_feature_dict(
        ir, ambient, g1, g2, g3, acc=acc, fs=fs, ppg_config=ppg_config
    )
    return [feature_dict[name] for name in FEATURE_ORDER]


def extract_features_from_ppg(ppg, acc=None, *, frequency, ppg_config):
    """Extract the model vector directly from raw multi-channel PPG."""
    raw_ppg = np.asarray(ppg, dtype=float)
    if raw_ppg.ndim != 2:
        raise ValueError(f"ppg must have shape (T, C), got {{raw_ppg.shape}}")
    try:
        frequency = int(frequency)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("frequency must be 25 or 100") from exc
    if frequency not in (25, 100):
        raise ValueError(f"frequency must be 25 or 100, got {{frequency}}")

    raw_acc = None if acc is None else np.asarray(acc, dtype=float)
    if frequency == 100:
        raw_ppg = resample_poly(raw_ppg, 1, 4, axis=0)
        if raw_acc is not None and raw_acc.size:
            raw_acc = resample_poly(raw_acc, 1, 4, axis=0)
    ir, ambient, g1, g2, g3 = get_channels_from_window(raw_ppg, ppg_config)
    return extract_features(
        ir, ambient, g1, g2, g3,
        acc=raw_acc, fs=25, ppg_config=ppg_config,
    )


def classify_probability(probability):
    return int(float(probability) >= float(WINDOW_MODEL_THRESHOLD))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = max(32, int(round(DEFAULT_FS * DEFAULT_WINDOW_SEC)))
    t = np.arange(n, dtype=float) / DEFAULT_FS
    ir = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t)
    ambient = 1.0e5 + 500.0 * np.sin(2 * np.pi * 0.4 * t)
    g1 = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t + 0.01)
    g2 = 2.1e6 + 7.5e3 * np.sin(2 * np.pi * 1.2 * t + 0.03)
    g3 = 1.9e6 + 8.5e3 * np.sin(2 * np.pi * 1.2 * t - 0.02)
    acc = rng.normal(0, 0.01, (n, 3))
    vec = extract_features(ir, ambient, g1, g2, g3, acc=acc, fs=DEFAULT_FS)
    print(f"Feature vector: {{len(vec)}} values")
    for i, (name, value) in enumerate(zip(FEATURE_ORDER, vec)):
        print(f"{{i:02d}} {{name}} = {{value:.8g}}")
'''

def _standalone_feature_engine_source():
    """Inline s03 source so deploy script is standalone yet numerically identical."""
    s03_path = os.path.join(SCRIPTS_DIR, "s03_extract_feature_pool.py")
    with open(s03_path, "r", encoding="utf-8") as f:
        source = f.read()
    engine_start = source.index("def apply_stage2_ir_policy")
    try:
        engine_end = source.index("def _downsample_ppg", engine_start)
    except ValueError:
        engine_end = len(source)
    engine = source[engine_start:engine_end].strip()
    catalog_records = {
        name: {
            "group": record.get("group"),
            "fft": bool(record.get("fft")),
            "model_candidate": name in set(stage2_model_candidate_names()),
        }
        for name, record in FEATURE_CATALOG.items()
    }
    catalog_source = repr(catalog_records)
    candidate_names_source = repr(stage2_model_candidate_names())
    commercial_names_source = repr([
        "GREEN_CORR",
        "GREEN_AC",
        "AMB_AC",
        "ACC_YSUM",
        "GREEN_DC",
        "AMB_DC",
        "GREEN_XCORR",
        "FFT_PEAK_MEDIAN_RATIO",
    ])
    return f'''EPS = 1e-12
MIN_ZONE_RELATIVE_AC_RMS = 1e-8
MIN_ZONE_ABSOLUTE_AC_RMS = 1e-9
DEFAULT_USE_STAGE2_IR = False
FEATURE_FS = 25
STAGE2_FEATURE_POOL_VERSION = {FEATURE_POOL_VERSION!r}
COMMERCIAL_8_FEATURE_NAMES = {commercial_names_source}
FEATURE_CATALOG = {catalog_source}


def stage2_feature_record(name):
    return FEATURE_CATALOG[name]


def stage2_model_candidate_names():
    return list({candidate_names_source})


def is_stage2_model_candidate(name):
    return bool(FEATURE_CATALOG.get(name, {{}}).get("model_candidate"))


def validate_stage2_candidate_names(names):
    unknown = [name for name in names if not is_stage2_model_candidate(name)]
    if unknown:
        raise ValueError("unknown Stage2 model candidates: " + ", ".join(unknown))
    return list(names)


{engine}'''


def _render_selected_feature_extractor(selected_features, fill_values, clip_bounds, formulas,
                                       window_model_threshold=0.5,
                                       use_stage2_ir=False,
                                       default_fs=25.0,
                                       window_sec=5.0):
    """Generate the standalone deploy script used by tests and golden vectors."""
    return _render_deployment_feature_extractor(
        selected_features,
        fill_values,
        clip_bounds,
        formulas,
        window_model_threshold=window_model_threshold,
        default_fs=default_fs,
        window_sec=window_sec,
    )


def export_feature_extractor_script(artifact_dir):
    """Export a compact extractor for the actual selected deployment features."""
    bp = os.path.join(artifact_dir, "model_bundle.pkl")
    if not os.path.exists(bp):
        print("[WARN] model_bundle.pkl not found, skip feature extractor script")
        return None

    bundle = joblib.load(bp)
    selected = list(bundle["feature_names"])
    meta = bundle.get("meta", {}) or {}
    ir_features = _stage2_ir_selected_features(selected)
    if ir_features:
        raise ValueError(
            "model_bundle.pkl contains IR-derived Stage2 features. "
            "Deployment export will not silently crop model inputs; rerun s03-s05 "
            "so the model is trained with ambient/green/ACC features only. Offending features: "
            + ", ".join(ir_features)
        )
    _assert_selected_features_deployment_allowed(selected)

    fill_values = _json_float_map(bundle.get("fill_values", {}), selected)
    clip_bounds = _json_clip_map(bundle.get("clip_bounds", {}), selected)
    formulas = build_selected_feature_formulas(selected)
    script_text = _render_selected_feature_extractor(
        selected,
        fill_values,
        clip_bounds,
        formulas,
        window_model_threshold=float(bundle.get("threshold", 0.5)),
        use_stage2_ir=bool(meta.get("use_stage2_ir", False)),
        default_fs=float(meta.get("fs_ppg", 25.0)),
        window_sec=float(meta.get("win_sec", 5.0)),
    )

    out_path = os.path.join(artifact_dir, "deploy_feature_extractor.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(script_text)

    formula_path = os.path.join(artifact_dir, "deploy_selected_feature_formulas.json")
    with open(formula_path, "w", encoding="utf-8") as f:
        json.dump(formulas, f, indent=2, ensure_ascii=False)

    export_stage2_feature_contracts(artifact_dir)

    print(f"[OK] selected deploy feature extractor -> {out_path}")
    print(f"[OK] selected deploy formulas -> {formula_path}")
    return out_path


def export_stage2_feature_contracts(artifact_dir):
    """Export selected catalog metadata and the firmware operator contract."""
    bundle_path = os.path.join(artifact_dir, "model_bundle.pkl")
    if not os.path.exists(bundle_path):
        raise ValueError(f"model_bundle.pkl missing: {bundle_path}")
    bundle = joblib.load(bundle_path)
    selected = list(bundle["feature_names"])
    version = bundle.get("feature_pool_version", FEATURE_POOL_VERSION)
    if version != FEATURE_POOL_VERSION:
        raise ValueError(
            f"model_bundle feature_pool_version={version!r} does not match "
            f"{FEATURE_POOL_VERSION}; rerun s03-s05 before deployment export."
        )
    meta = bundle.get("meta", {}) or {}
    fs = float(meta.get("fs_ppg", 25.0))
    win_sec = float(meta.get("win_sec", 5.0))
    contract = build_selected_feature_contract(
        selected,
        fs=fs,
        window_samples=max(1, int(round(fs * win_sec))),
    )
    catalog_payload = {
        "feature_pool_version": FEATURE_POOL_VERSION,
        "feature_order": selected,
        "features": selected_stage2_catalog(selected),
    }
    catalog_path = os.path.join(artifact_dir, "stage2_feature_catalog.json")
    contract_path = os.path.join(artifact_dir, "stage2_c_contract.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog_payload, f, indent=2, ensure_ascii=False)
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)
    deploy_package = os.path.join(artifact_dir, "deploy_package")
    if os.path.isdir(deploy_package):
        for name, payload in (
            ("stage2_feature_catalog.json", catalog_payload),
            ("stage2_c_contract.json", contract),
        ):
            with open(os.path.join(deploy_package, name), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[OK] selected Stage2 catalog -> {catalog_path}")
    print(f"[OK] Stage2 C contract -> {contract_path}")
    return {"catalog": catalog_path, "c_contract": contract_path}


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_deploy_script_constants(path):
    if not os.path.exists(path):
        raise ValueError(f"deploy_feature_extractor.py missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    wanted = {
        "FEATURE_ORDER",
        "FILL_VALUES",
        "CLIP_BOUNDS",
        "WINDOW_MODEL_THRESHOLD",
        "DEFAULT_FS",
        "DEFAULT_WINDOW_SEC",
    }
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    missing = sorted(wanted - set(values))
    if missing:
        raise ValueError("deploy_feature_extractor.py missing constants: " + ", ".join(missing))
    return values


def _normalize_clip_bounds(bounds):
    out = {}
    for name, bound in (bounds or {}).items():
        if bound is None or not isinstance(bound, (list, tuple)) or len(bound) != 2:
            continue
        out[name] = [float(bound[0]), float(bound[1])]
    return out


def _assert_same(label, actual, expected):
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _assert_float_same(label, actual, expected, tol=1e-12):
    if abs(float(actual) - float(expected)) > tol:
        raise ValueError(f"{label} mismatch: expected {float(expected)!r}, got {float(actual)!r}")


def _json_safe_number(value):
    value = float(value)
    if not (value == value) or value in (float("inf"), float("-inf")):
        return 0.0
    return value


def export_golden_vectors(artifact_dir, n_vectors=1):
    """Export deterministic deployment golden vectors for endpoint parity checks."""
    bundle_path = os.path.join(artifact_dir, "model_bundle.pkl")
    script_path = os.path.join(artifact_dir, "deploy_feature_extractor.py")
    if not os.path.exists(bundle_path):
        print("[WARN] model_bundle.pkl not found, skip golden vectors")
        return None
    if not os.path.exists(script_path):
        print("[WARN] deploy_feature_extractor.py not found, skip golden vectors")
        return None

    bundle = joblib.load(bundle_path)
    selected = list(bundle["feature_names"])
    threshold = float(bundle.get("threshold", 0.5))
    spec = importlib.util.spec_from_file_location("_lnwear_deploy_feature_extractor", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if list(module.FEATURE_ORDER) != selected:
        raise ValueError("deploy_feature_extractor.py FEATURE_ORDER mismatch before golden export")

    fs = float(bundle.get("meta", {}).get("fs_ppg", 25.0))
    win_sec = float(bundle.get("meta", {}).get("win_sec", 5.0))
    n = max(32, int(round(fs * win_sec)))
    rng = __import__("numpy").random.default_rng(12345)
    np = __import__("numpy")
    vectors = []
    for idx in range(int(max(1, n_vectors))):
        t = np.arange(n, dtype=float) / fs
        phase = 0.15 * idx
        ir = 4.0e6 + 1.0e4 * np.sin(2 * np.pi * 1.2 * t + phase)
        ambient = 1.0e5 + 500.0 * np.sin(2 * np.pi * 0.4 * t + phase)
        g1 = 2.0e6 + 8.0e3 * np.sin(2 * np.pi * 1.2 * t + 0.01 + phase)
        g2 = 2.1e6 + 7.5e3 * np.sin(2 * np.pi * 1.2 * t + 0.03 + phase)
        g3 = 1.9e6 + 8.5e3 * np.sin(2 * np.pi * 1.2 * t - 0.02 + phase)
        acc = rng.normal(0, 0.01, (n, 3))
        raw_feature_dict = module.extract_raw_feature_dict(
            ir, ambient, g1, g2, g3, acc=acc, fs=fs
        )
        feature_dict = module.extract_feature_dict(
            ir, ambient, g1, g2, g3, acc=acc, fs=fs
        )
        feature_vector = [feature_dict[name] for name in selected]
        feature_dict = dict(zip(selected, feature_vector))
        X = np.asarray([feature_vector], dtype=float)
        model = bundle.get("model") or bundle.get("raw_model")
        probability = float(model.predict_proba(X)[:, 1][0])
        label = int(probability >= threshold)
        vectors.append({
            "id": f"synthetic_{idx}",
            "fs": fs,
            "window_sec": win_sec,
            "n_samples": int(n),
            "mode": 0,
            "feature_vector_length": int(len(feature_vector)),
            "features_raw": {
                name: _json_safe_number(raw_feature_dict[name])
                for name in selected
            },
            "features_after_fill_clip": {
                name: _json_safe_number(feature_dict[name])
                for name in selected
            },
            "feature_vector": [_json_safe_number(v) for v in feature_vector],
            "tolerances": {
                name: {
                    "abs": float(stage2_feature_record(name)["c_abs_tolerance"]),
                    "rel": float(stage2_feature_record(name)["c_rel_tolerance"]),
                }
                for name in selected
            },
            "probability": _json_safe_number(probability),
            "threshold": threshold,
            "window_label": label,
        })

    payload = {
        "version": 1,
        "source": "s08_run_pipeline.export_golden_vectors",
        "feature_order": selected,
        "threshold": threshold,
        "n_vectors": int(len(vectors)),
        "vectors": vectors,
    }
    out_path = os.path.join(artifact_dir, "golden_vectors.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[OK] golden vectors -> {out_path}")
    return out_path


def validate_deploy_artifact_consistency(artifact_dir):
    """Fail fast when deployment files drift from model_bundle.pkl."""
    bundle_path = os.path.join(artifact_dir, "model_bundle.pkl")
    if not os.path.exists(bundle_path):
        raise ValueError(f"model_bundle.pkl missing: {bundle_path}")

    bundle = joblib.load(bundle_path)
    selected = list(bundle["feature_names"])
    threshold = float(bundle.get("threshold", 0.5))
    expected_fill = _json_float_map(bundle.get("fill_values", {}), selected)
    expected_clip = _json_clip_map(bundle.get("clip_bounds", {}), selected)
    expected_meta = {
        key: bundle.get("meta", {}).get(key)
        for key in ("fs_ppg", "win_sec", "step_sec")
        if key in bundle.get("meta", {})
    }

    script_constants = _parse_deploy_script_constants(
        os.path.join(artifact_dir, "deploy_feature_extractor.py")
    )
    _assert_same("deploy_feature_extractor.py FEATURE_ORDER", list(script_constants["FEATURE_ORDER"]), selected)
    _assert_same("deploy_feature_extractor.py FILL_VALUES", script_constants["FILL_VALUES"], expected_fill)
    _assert_same("deploy_feature_extractor.py CLIP_BOUNDS", script_constants["CLIP_BOUNDS"], expected_clip)
    _assert_float_same("deploy_feature_extractor.py WINDOW_MODEL_THRESHOLD",
                       script_constants["WINDOW_MODEL_THRESHOLD"], threshold)
    if "fs_ppg" in expected_meta:
        _assert_float_same("deploy_feature_extractor.py DEFAULT_FS",
                           script_constants["DEFAULT_FS"], expected_meta["fs_ppg"])
    if "win_sec" in expected_meta:
        _assert_float_same("deploy_feature_extractor.py DEFAULT_WINDOW_SEC",
                           script_constants["DEFAULT_WINDOW_SEC"], expected_meta["win_sec"])

    formulas = _load_json_if_exists(os.path.join(artifact_dir, "deploy_selected_feature_formulas.json"))
    if formulas is not None:
        missing_formulas = [name for name in selected if name not in formulas]
        if missing_formulas:
            raise ValueError("deploy_selected_feature_formulas.json missing formulas: "
                             + ", ".join(missing_formulas))

    selected_catalog = _load_json_if_exists(
        os.path.join(artifact_dir, "stage2_feature_catalog.json")
    )
    if selected_catalog is not None:
        _assert_same(
            "stage2_feature_catalog.json feature_pool_version",
            selected_catalog.get("feature_pool_version"),
            FEATURE_POOL_VERSION,
        )
        _assert_same(
            "stage2_feature_catalog.json feature_order",
            list(selected_catalog.get("feature_order", [])),
            selected,
        )

    c_contract = _load_json_if_exists(os.path.join(artifact_dir, "stage2_c_contract.json"))
    if c_contract is not None:
        _assert_same(
            "stage2_c_contract.json feature_pool_version",
            c_contract.get("feature_pool_version"),
            FEATURE_POOL_VERSION,
        )
        _assert_same(
            "stage2_c_contract.json feature_order",
            list(c_contract.get("feature_order", [])),
            selected,
        )

    deploy_xgb = _load_json_if_exists(os.path.join(artifact_dir, "deploy_xgboost.json"))
    if deploy_xgb is not None:
        _assert_same("deploy_xgboost.json feature_order", list(deploy_xgb.get("feature_order", [])), selected)
        _assert_same("deploy_xgboost.json feature_names", list(deploy_xgb.get("feature_names", [])), selected)
        _assert_same("deploy_xgboost.json fill_values",
                     {name: float(deploy_xgb.get("fill_values", {}).get(name, 0.0)) for name in selected},
                     expected_fill)
        _assert_same("deploy_xgboost.json clip_bounds",
                     _normalize_clip_bounds(deploy_xgb.get("clip_bounds", {})),
                     _normalize_clip_bounds(expected_clip))
        _assert_float_same("deploy_xgboost.json threshold", deploy_xgb.get("threshold", 0.5), threshold)

    cookbook = _load_json_if_exists(os.path.join(artifact_dir, "deploy_cookbook.json"))
    if cookbook is not None:
        cb_features = cookbook.get("B_selected_features", {}).get("feature_order", [])
        _assert_same("deploy_cookbook.json B_selected_features.feature_order", list(cb_features), selected)
        cb_inf = cookbook.get("C_xgboost_inference", {})
        _assert_same("deploy_cookbook.json C_xgboost_inference.fill_values",
                     {name: float(cb_inf.get("fill_values", {}).get(name, 0.0)) for name in selected},
                     expected_fill)
        _assert_same("deploy_cookbook.json C_xgboost_inference.clip_bounds",
                     _normalize_clip_bounds(cb_inf.get("clip_bounds", {})),
                     _normalize_clip_bounds(expected_clip))
        _assert_float_same("deploy_cookbook.json C_xgboost_inference.model_threshold",
                           cb_inf.get("model_threshold", 0.5), threshold)

    model_params = _load_json_if_exists(os.path.join(artifact_dir, "deploy_package", "model_params.json"))
    if model_params is not None:
        _assert_same("deploy_package/model_params.json selected_features",
                     list(model_params.get("selected_features", [])), selected)
        _assert_same("deploy_package/model_params.json fill_values",
                     {name: float(model_params.get("fill_values", {}).get(name, 0.0)) for name in selected},
                     expected_fill)
        _assert_same("deploy_package/model_params.json clip_bounds",
                     _normalize_clip_bounds(model_params.get("clip_bounds", {})),
                     _normalize_clip_bounds(expected_clip))
        _assert_float_same("deploy_package/model_params.json window_threshold",
                           model_params.get("window_threshold", 0.5), threshold)
        for key, expected in expected_meta.items():
            if key in model_params.get("meta", {}):
                _assert_same(f"deploy_package/model_params.json meta.{key}",
                             model_params["meta"][key], expected)

    golden = _load_json_if_exists(os.path.join(artifact_dir, "golden_vectors.json"))
    if golden is not None:
        _assert_same("golden_vectors.json feature_order", list(golden.get("feature_order", [])), selected)
        _assert_float_same("golden_vectors.json threshold", golden.get("threshold", 0.5), threshold)
        for idx, vec in enumerate(golden.get("vectors", [])):
            _assert_same(
                f"golden_vectors.json vectors[{idx}].feature_vector_length",
                int(vec.get("feature_vector_length", -1)),
                len(selected),
            )
            _assert_same(
                f"golden_vectors.json vectors[{idx}].features_raw order",
                list(vec.get("features_raw", {})),
                selected,
            )
            expected_tolerances = {
                name: {
                    "abs": float(stage2_feature_record(name)["c_abs_tolerance"]),
                    "rel": float(stage2_feature_record(name)["c_rel_tolerance"]),
                }
                for name in selected
            }
            _assert_same(
                f"golden_vectors.json vectors[{idx}].tolerances",
                vec.get("tolerances", {}),
                expected_tolerances,
            )
            _assert_same(
                f"golden_vectors.json vectors[{idx}].feature_vector length",
                len(vec.get("feature_vector", [])),
                len(selected),
            )

    report = {
        "feature_names": selected,
        "n_features": len(selected),
        "threshold": threshold,
        "meta": expected_meta,
        "checked_files": [
            name for name, exists in [
                ("deploy_feature_extractor.py", True),
                ("deploy_selected_feature_formulas.json", formulas is not None),
                ("stage2_feature_catalog.json", selected_catalog is not None),
                ("stage2_c_contract.json", c_contract is not None),
                ("deploy_xgboost.json", deploy_xgb is not None),
                ("deploy_cookbook.json", cookbook is not None),
                ("deploy_package/model_params.json", model_params is not None),
                ("golden_vectors.json", golden is not None),
            ] if exists
        ],
    }
    print("[OK] deploy artifacts are consistent with model_bundle.pkl")
    return report
import sys
import time
from datetime import timedelta

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def _script_path(name):
    return os.path.join(SCRIPTS_DIR, f"{name}.py")


RUNTIME_PROFILES = {
    "fast": {
        "model_search_max_candidates": 120,
        "model_search_stage2_top_k": 16,
        "feature_search_swap_max_candidates": 4,
        "model_search_full_top_k": 1,
        "postprocess_search_budget": 240,
    },
    "balanced": {
        "model_search_max_candidates": 180,
        "model_search_stage2_top_k": 24,
        "feature_search_swap_max_candidates": 8,
        "model_search_full_top_k": 1,
        "postprocess_search_budget": 240,
    },
    "thorough": {
        "model_search_max_candidates": 360,
        "model_search_stage2_top_k": 48,
        "feature_search_swap_max_candidates": 12,
        "model_search_full_top_k": 2,
        "postprocess_search_budget": 2000,
    },
}


def _arg_was_provided(argv, opt):
    prefix = opt + "="
    return any(item == opt or item.startswith(prefix) for item in argv)


def apply_runtime_profile(args, raw_argv):
    profile = RUNTIME_PROFILES.get(args.runtime_profile, RUNTIME_PROFILES["balanced"])
    option_map = {
        "model_search_max_candidates": "--model_search_max_candidates",
        "model_search_stage2_top_k": "--model_search_stage2_top_k",
        "feature_search_swap_max_candidates": "--feature_search_swap_max_candidates",
        "model_search_full_top_k": "--model_search_full_top_k",
        "postprocess_search_budget": "--postprocess_search_budget",
    }
    for attr, opt in option_map.items():
        if not _arg_was_provided(raw_argv, opt):
            setattr(args, attr, profile[attr])


def _record_runtime(runtime_events, name, elapsed, dry_run=False):
    if runtime_events is not None:
        runtime_events.append({
            "name": str(name),
            "elapsed": float(elapsed),
            "dry_run": bool(dry_run),
        })


def _print_runtime_summary(runtime_events):
    if not runtime_events:
        return
    print("\n[RUNTIME] step elapsed summary")
    total = 0.0
    for item in runtime_events:
        elapsed = float(item.get("elapsed", 0.0))
        total += elapsed
        suffix = " (dry-run)" if item.get("dry_run") else ""
        print(f"  - {item['name']}: {timedelta(seconds=int(elapsed))}{suffix}")
    print(f"  total measured: {timedelta(seconds=int(total))}")


def _run(name, cmd, dry_run=False, runtime_events=None):
    """执行一个子步骤。返回 True/False。"""
    print(f"\n{'─' * 70}")
    print(f"[RUN] {name}")
    print(f"  {cmd}")
    if dry_run:
        print("  (dry-run, skipped)")
        _record_runtime(runtime_events, name, 0.0, dry_run=True)
        return True
    t0 = time.time()
    rc = subprocess.run(shlex.split(cmd), check=False).returncode
    dt = time.time() - t0
    _record_runtime(runtime_events, name, dt, dry_run=False)
    if rc == 0:
        print(f"[OK] {name}  [{timedelta(seconds=int(dt))}]")
        return True
    else:
        print(f"[FAIL] {name}  FAILED (exit={rc})")
        return False


def _parse_csv_strings(value):
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _parse_csv_ints(value):
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def run_embedded_feature_embedding_report(args):
    import s04_feature_selection as feature_selection

    try:
        result = feature_selection.run_embedding_report(
            artifact_dir=args.artifact_dir,
            methods=_parse_csv_strings("pca,tsne"),
            dims=_parse_csv_ints("2,3"),
            formats=_parse_csv_strings("png"),
            max_points=0,
            perplexity=30.0,
            random_state=42,
            dpi=600,
        )
    except Exception as exc:
        print(f"[WARN] skip feature embedding report: {exc}")
        return None
    print(f"[OK] feature embedding report -> {result['report_path']}")
    return result


def run_embedded_generalization_audit(args):
    import s06_deploy_eval as deploy_eval

    result = deploy_eval.run_audit(
        args.artifact_dir,
        split=args.split,
        method="state_machine",
        min_support=args.audit_min_support,
    )
    print(f"[OK] generalization_audit -> {result['out_dir']}")
    print(json.dumps(result["paths"], indent=2, ensure_ascii=False))


RELIABILITY_FEATURE_HINTS = (
    "GREEN_SEG_ACDC_CV",
    "AMB_SEG_ACDC_CV",
    "GREEN_AMB_LEAK_STABILITY",
    "GREEN_AMB_SEG_CORR_RANGE",
    "G_2OF3_AC_SUPPORT",
    "G_TOP2_TO_ALL_AC_RATIO",
    "G_TOP2_CORR_MIN",
    "G_WEAK_CHANNEL_GAP",
    "G_SPATIAL_STABILITY_SCORE",
    "G_SPATIAL_VMAG_RANGE",
    "ACC_TO_GTOP2_AC_RATIO",
    "ACC_STILL_X_GREEN_STABILITY",
    "ACC_STILL_GREEN_MISMATCH",
    "ACC_GREEN_BP_CORR",
    "GREEN_AMB_BP_CORR",
    "GREEN_AMB_ENV_CORR",
    "GREEN_AMB_LEAK",
)


def _count_xgb_nodes(raw_model):
    try:
        booster = raw_model.get_booster()
        dump = booster.get_dump(with_stats=True)
    except Exception:
        return 0
    return int(sum(1 for tree in dump for line in tree.splitlines() if line.strip()))


def _summarize_reliability_features(selected):
    features = [
        name for name in selected
        if any(hint in str(name) for hint in RELIABILITY_FEATURE_HINTS)
    ]
    return {
        "selected_count": int(len(features)),
        "selected_features": features,
        "feature_cost_summary": summarize_deployment_feature_costs(features),
    }


def _load_eval_details(artifact_dir, split="test", method="state_machine"):
    import json as _json
    import os as _os
    eval_path = _os.path.join(artifact_dir, f"end_to_end_eval_{split}_{method}.json")
    if not _os.path.exists(eval_path):
        return None
    with open(eval_path, "r", encoding="utf-8") as f:
        return _json.load(f).get("details", [])


def _build_full_feature_recipe(selected_features):
    """
    为每个入选特征构建完整的自包含计算配方。
    从 25Hz 原始窗口 → 预处理 → 中间信号 → 特征值，全链路。
    """
    selected_features = list(selected_features)
    catalog_formulas = build_selected_feature_formulas(selected_features)

    channel_extract = {
        "ir": "window[:, 0]",
        "ambient": "window[:, 1]",
        "g1": "ppg_config==0 ? ch3 : ppg_config==1 ? (ch3+ch9)/2 : (ch6+ch9+ch12)/3",
        "g2": "ppg_config==0 ? ch4 : ppg_config==1 ? (ch4+ch10)/2 : (ch7+ch10+ch13)/3",
        "g3": "ppg_config==0 ? ch5 : ppg_config==1 ? (ch5+ch11)/2 : (ch8+ch11+ch14)/3",
    }
    config_source = (
        "read required H5 ppg_config in {0,1,2}; do not infer from signal values"
    )
    preprocess_steps = [
        (
            "finite_replacement",
            "replace NaN/Inf by the finite channel median; use zero only when no finite sample exists",
        ),
        (
            "isolated_spike_repair",
            "remove_burr(x,k=6): replace a point only when both neighbour differences exceed max(6*MAD(diff(x)),1e-12)",
        ),
        (
            "rolling_median_detrend",
            "baseline=rolling_median(contact_raw, nearest odd max(3,round(0.8*fs))); pulse=contact_raw-baseline",
        ),
        (
            "optional_short_smoothing",
            "if round(0.04*fs)>=2, pulse=moving_average(pulse,round(0.04*fs)); at 25Hz this step is inactive",
        ),
    ]
    preprocess_output = {
        "contact_raw": "finite replacement followed by isolated-spike repair; used for DC/contact features",
        "pulse": "contact_raw minus its 0.8s rolling-median baseline, with optional 0.04s smoothing",
        "dc": "median(contact_raw)",
    }
    composite = {
        "g_mean_raw": "mean(g1_contact_raw,g2_contact_raw,g3_contact_raw)",
        "g_mean_pulse": "mean(g1_pulse,g2_pulse,g3_pulse)",
        "g_top2_raw": "tie-aware average of every equally maximal two-zone AC-RMS pair mean",
        "g_top2_pulse": "tie-aware average of every equally maximal two-zone AC-RMS pair pulse mean",
        "g_median_raw": "pointwise_median(g1_contact_raw,g2_contact_raw,g3_contact_raw)",
        "g_median_pulse": "pointwise_median(g1_pulse,g2_pulse,g3_pulse)",
        "green_pair_views": "the three unordered two-zone mean raw/pulse signals",
        "ambient_residuals": "each zone pulse after guarded linear ambient projection removal",
        "acc_magnitude": "sqrt(acc_x^2+acc_y^2+acc_z^2)",
        "acc_motion": "acc_magnitude minus its 0.8s rolling-median baseline, then guarded clipping",
    }
    utils = {
        "guarded_ratio": "finite numerator/denominator with a signal-scale denominator floor",
        "guarded_corr": "finite correlation; return zero if either relative standard deviation is too small",
        "robust_mad": "median(abs(x-median(x)))",
        "fft_cache": "Hamming-window rFFT over 0.5-5Hz; a zero-energy band has dom_freq=0",
        "frequency_validity_gate": "zone pulse RMS>max(1e-8*median(abs(zone_raw)),1e-9), dom_freq>0, spectral peak/median ratio>=3, and 40-180bpm autocorrelation peak>=0.20",
        "top2_tie_rule": "include all pairs whose pair AC sum is within max(1e-9*best,1e-12) of the best pair",
    }
    recipes = {}
    for name in selected_features:
        info = dict(catalog_formulas[name])
        info["intermediate_signals"] = {
            "signal_source": info["signal_source"],
            "preprocessing": info["preprocessing"],
        }
        recipes[name] = info
    return (
        recipes,
        channel_extract,
        config_source,
        preprocess_steps,
        preprocess_output,
        composite,
        utils,
    )


def export_deploy_cookbook(artifact_dir):
    """
    导出完整部署配方: 每个入选特征从 25Hz 原始窗口 → 预处理 → 中间信号 → 特征值的完整链。
    嵌入式工程师可直接翻译为 C/Rust，无需查任何其他文件。
    """
    import os as _os
    import json as _json
    import joblib as _joblib

    bundle_path = _os.path.join(artifact_dir, "model_bundle.pkl")
    th_path = _os.path.join(artifact_dir, "stage1_threshold.json")

    if not _os.path.exists(bundle_path):
        print("[WARN] model_bundle.pkl not found, skip deploy cookbook")
        return

    bundle = _joblib.load(bundle_path)
    selected = bundle["feature_names"]
    fill_values = bundle["fill_values"]
    clip_bounds = bundle.get("clip_bounds", {})
    threshold = float(bundle["threshold"])
    meta = bundle.get("meta", {}) or {}
    model = bundle["model"]
    raw = bundle.get("raw_model", model)
    booster = raw.get_booster()
    n_estimators = raw.n_estimators
    total_nodes = _count_xgb_nodes(raw)
    feature_cost_summary = summarize_deployment_feature_costs(selected)
    reliability_feature_summary = _summarize_reliability_features(selected)
    avg_nodes_per_tree = float(total_nodes) / float(max(int(n_estimators), 1))

    # 生成完整特征配方
    recipe, ch_extract, config_source, preproc_steps, preproc_out, composite, utils = _build_full_feature_recipe(selected)

    # 组装输出
    cookbook = {
        "_title": "手表佩戴活体检测 — 部署配方 (Deployment Cookbook)",
        "_for": "嵌入式/工程化部署工程师。本文件自包含，无需查任何其他文件。",
        "_input": "PPG/ACC窗口 + frequency in {25,100} + ppg_config in {0,1,2}; 100Hz先降到25Hz",
        "A_deployment_operator_budget": {
            "feature_set": "deployment_friendly",
            "selected_feature_cost_summary": feature_cost_summary,
            "reliability_feature_summary": reliability_feature_summary,
            "model_node_summary": {
                "n_estimators": int(n_estimators),
                "total_nodes": int(total_nodes),
                "avg_nodes_per_tree": avg_nodes_per_tree,
            },
            "operator_notes": (
                "Stage2 uses scalar statistics, ratio/MAD/IQR features, simple correlations, "
                "and explicitly cataloged reusable FFT sources for deployment-friendly implementation."
            ),
        },

        # ---- Section A: 公共计算（所有特征共用） ----
        "A_channel_extraction": {
            "_note": "从 PPG 窗口提取 ir/ambient/g1/g2/g3",
            "frequency_rule": "frequency==25: direct; frequency==100: polyphase downsample to 25Hz",
            "ppg_config_source": config_source,
            "channels": ch_extract,
        },
        "A_preprocessing": {
            "_note": "对 ambient/g1/g2/g3 执行以下管线，产出 contact_raw、pulse 和 dc；IR 仅供 Stage1",
            "pipeline": [{"step": i+1, "name": name, "formula": formula} for i, (name, formula) in enumerate(preproc_steps)],
            "outputs": preproc_out,
        },
        "A_composite_signals": {
            "_note": "从各通道预处理结果合成复合信号",
            "signals": composite,
        },
        "A_utility_functions": {
            "_note": "以下所有公式中引用的工具函数",
            "definitions": utils,
        },

        # ---- Section B: 入选特征完整配方 ----
        "B_selected_features": {
            "_note": f"共 {len(selected)} 个特征，按此顺序组成 XGBoost 输入向量 feature_vec[0..{len(selected)-1}]",
            "feature_order": selected,
            "recipes": {f: recipe[f] for f in selected},
        },

        # ---- Section C: XGBoost 推理 ----
        "C_xgboost_inference": {
            "_note": "拿到 feature_vec 后的推理步骤",
            "fill_values": fill_values,
            "clip_bounds": clip_bounds,
            "preprocess_order": [
                "1. select feature_order",
                "2. fill NaN/inf with fill_values",
                "3. clip each selected feature by clip_bounds",
            ],
            "fill_rule": "feature_vec[i] 为 NaN/inf 时用 fill_values[feature_name] 替换",
            "clip_rule": "fill 后对每个入选特征执行 clip(lower, upper)，边界来自训练集 IQR（s05 clip_outliers k=1.5）",
            "model_threshold": threshold,
            "n_estimators": n_estimators,
            "model_json": _json.loads(booster.save_config()),
            "inference": [
                "1. feature_vec = [compute_feature(f) for f in feature_order]",
                "2. for i, v in enumerate(feature_vec): if isnan(v) or isinf(v): feature_vec[i] = fill_values[feature_order[i]]",
                "3. for i, v in enumerate(feature_vec): feature_vec[i] = clamp(v, clip_bounds[feature_order[i]][0], clip_bounds[feature_order[i]][1])",
                "4. proba = xgboost_predict(model, feature_vec)  // → float in [0,1]",
                "5. window_pred = 1 if proba >= threshold else 0",
            ],
        },

        # ---- Section D: Stage1 & Stage3 ----
        "D_stage1_gate": {
            "_note": "IR 快速门控与 Stage2 持续推理并行运行；这里只屏蔽最终对外输出",
            "ir_5hz": "resample(ir_raw_100Hz -> 5Hz)",
            "primitive_window": "1s stride=1s (5 points @5Hz)",
            "decision_window": "3 consecutive primitive decisions",
            "dc_formula": "min(neighbor_mean) where neighbor_mean[i]=(x[i]+x[i+1])/2",
            "ac_formula": "median(|diff(x)|)",
            "rule": "dc > dc_thresh AND ac/|dc| < acdc_thresh",
            "streaming_gate_rule": "set stage1_gate=1 after 3 consecutive pass primitives; set it to 0 after 3 consecutive fail primitives",
            "fusion_rule": "output_state[t] = stage1_gate[t] AND stage2_state[t]",
            "parallel_semantics_version": "stage1_mask_stage2_continuous_v1",
            "thresholds": {},
        },
        "D_stage3_postprocess": {
            "_note": "对 XGBoost 输出的逐窗概率做时序平滑",
            "algorithm": "EMA + hysteresis + cooldown",
            "params": {"alpha": 0.4, "T_on": 0.75, "T_off": 0.35, "K_on": 5, "K_off": 3, "cooldown_sec": 5.0},
            "pseudocode": [
                "score[t] = alpha * quality[t] * proba[t] + (1-alpha*quality[t]) * score[t-1]",
                "IF state==0 and count(score>T_on) >= K_on and cooldown_expired: state=1, reset counter",
                "IF state==1 and count(score<T_off) >= K_off and cooldown_expired: state=0, reset counter",
                "output_state[t] = stage1_gate[t] AND state[t]  // gate never pauses or resets Stage2",
                "quality[t] from the feature-specific train thresholds stored in bundle['quality_thresholds']",
            ],
            "quality_thresholds": bundle.get("quality_thresholds", {}),
        },
    }

    # Stage1 阈值
    if _os.path.exists(th_path):
        with open(th_path, "r", encoding="utf-8") as f:
            th_data = _json.load(f)
        dth = th_data.get("deploy_stage1_threshold", {})
        cookbook["D_stage1_gate"]["thresholds"] = {
            "dc_threshold": float(dth.get("dc_threshold", 0)),
            "ac_dc_threshold": float(dth.get("ac_dc_threshold", 0)),
        }

    out_path = _os.path.join(artifact_dir, "deploy_cookbook.json")
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(cookbook, f, indent=2, ensure_ascii=False)

    xgb_path = _os.path.join(artifact_dir, "deploy_xgboost.json")
    with open(xgb_path, "w", encoding="utf-8") as f:
        _json.dump({
            "feature_names": selected,
            "feature_order": selected,
            "fill_values": fill_values,
            "clip_bounds": clip_bounds,
            "preprocess_order": [
                "select feature_order",
                "fill NaN/inf with fill_values",
                "clip by clip_bounds",
            ],
            "n_estimators": n_estimators,
            "threshold": threshold,
            "model": _json.loads(booster.save_config()),
        }, f, indent=2, ensure_ascii=False)

    profile = {
        "_title": "Deployment performance profile",
        "feature_names": selected,
        "feature_cost_summary": feature_cost_summary,
        "reliability_feature_summary": reliability_feature_summary,
        "model_summary": {
            "n_estimators": int(n_estimators),
            "total_nodes": int(total_nodes),
            "avg_nodes_per_tree": avg_nodes_per_tree,
            "max_model_nodes": int(meta.get("max_model_nodes", 0) or 0),
        },
        "operator_reuse_plan": {
            "shared_preprocessing": [
                "one preprocessing pass per raw channel",
                "reuse g_top2_raw/g_top2_pulse for green reliability, FFT, and ACC coupling features",
                "reuse one FFT result per source listed in feature_cost_summary.fft_sources",
            ],
            "fft_sources": feature_cost_summary.get("fft_sources", []),
            "fft_source_count": feature_cost_summary.get("fft_source_count", 0),
        },
        "deployment_targets": {
            "window_model_threshold": threshold,
            "fs_ppg": float(meta.get("fs_ppg", 25.0)),
            "window_sec": float(meta.get("win_sec", 5.0)),
            "step_sec": float(meta.get("step_sec", 1.0)),
            "use_stage2_ir": bool(meta.get("use_stage2_ir", False)),
        },
    }
    profile_path = _os.path.join(artifact_dir, "deploy_performance_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        _json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"[OK] deploy_cookbook.json -> {out_path}")
    print(f"[OK] deploy_xgboost.json  -> {xgb_path}")
    print(f"[OK] deploy_performance_profile.json -> {profile_path}")

def generate_eval_csv(artifact_dir, split="test", method="state_machine"):
    """生成三份逐样本 CSV，统计口径与 s06 官方准确率指标完全一致：

    1. per_sample_xgboost_windows.csv   — Stage2 单窗 XGBoost 准确率
       与 compute_window_model_metrics 一致：
         - 仅跳过 fallback / 无窗口的样本
         - 所有合法 Stage2 窗口参与，不按 Stage1 过滤

    2. per_sample_statemachine_windows.csv — 状态机后处理窗口准确率
       与 compute_window_stream_metrics 一致：
         - 仅跳过 fallback 的样本，并使用独立 Stage2 状态
         - 按 warmup_frames 跳过每条样本前 N 个窗口（消除冷启动）
         - 逐窗比较 state[i] vs window_targets[i]

    3. per_sample_final_prediction.csv  — 样本级最终预测结果
       与 compute_sample_metrics 一致：
         - pred 是 Stage1 门控与持续 Stage2 状态融合后的最终输出
    """
    import os as _os
    import json as _json
    details = _load_eval_details(artifact_dir, split, method)
    if not details:
        print("[WARN] 评估结果为空，跳过 CSV")
        return

    # 读取 warmup_frames —— 与 compute_window_stream_metrics 保持一致
    warmup_frames = 5  # 默认值 K_on，确保状态机预填充充分
    eval_payload = {}
    eval_path = _os.path.join(artifact_dir, f"end_to_end_eval_{split}_{method}.json")
    if _os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_payload = _json.load(f)
        warmup_frames = int((eval_payload.get("window_stream_summary") or {}).get("warmup_frames", 5))

    # ---- CSV 1: XGBoost 单窗预测 ----
    # 与 compute_window_model_metrics 一致：Stage2 独立窗口指标不按 Stage1 过滤。
    rows_xgb = []
    for d in details:
        name = d.get("sample_name", "")
        wpreds = d.get("window_preds", [])
        if d.get("fallback", False) or len(wpreds) == 0:
            continue
        wtargs = d.get("window_targets", [])
        sample_target = int(d.get("target", 0))
        n_win = len(wpreds)
        n_correct = 0
        for i in range(n_win):
            t = int(wtargs[i]) if i < len(wtargs) else sample_target
            if int(wpreds[i]) == t:
                n_correct += 1
        n_wrong = n_win - n_correct
        acc = n_correct / n_win if n_win > 0 else 0.0
        rows_xgb.append((name, n_win, n_correct, n_wrong, round(acc, 6)))

    csv1 = _os.path.join(artifact_dir, "per_sample_xgboost_windows.csv")
    with open(csv1, "w", encoding="utf-8") as f:
        f.write("sample_name,total_windows,correct_windows,wrong_windows,accuracy\n")
        for row in rows_xgb:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")
    total_xgb_wins = sum(r[1] for r in rows_xgb)
    total_xgb_correct = sum(r[2] for r in rows_xgb)
    total_xgb_wrong = sum(r[3] for r in rows_xgb)
    xgb_global_acc = total_xgb_correct / total_xgb_wins if total_xgb_wins > 0 else 0.0
    print(f"[OK] XGBoost 逐样本窗口 CSV: {csv1} "
          f"(n_samples={len(rows_xgb)}, total_windows={total_xgb_wins}, "
          f"global_acc={round(xgb_global_acc, 6) if total_xgb_wins > 0 else 0})")

    # ---- CSV 2: 状态机后处理窗口预测 ----
    # 与 compute_window_stream_metrics 一致：
    #   前 warmup_frames 窗只预热状态机，不对外输出有效状态，也不计入准确率
    #   后续窗逐窗比较 state[i] vs window_targets[i]
    rows_sm = []
    rows_sm_detail = []
    for d in details:
        name = d.get("sample_name", "")
        states = d.get("stage2_states", d.get("window_states", []))
        if d.get("fallback", False) or len(states) == 0:
            continue
        wtargs = d.get("window_targets", [])
        sample_target = int(d.get("target", 0))
        skipped = min(max(0, int(warmup_frames)), len(states))
        raw_win = len(states)
        n_win = raw_win - skipped
        n_correct = 0
        for i in range(skipped, raw_win):
            t = int(wtargs[i]) if i < len(wtargs) else sample_target
            if int(states[i]) == t:
                n_correct += 1
        for i in range(raw_win):
            t = int(wtargs[i]) if i < len(wtargs) else sample_target
            output_valid = 1 if i >= skipped else 0
            state_internal = int(states[i])
            state_output = state_internal if output_valid else ""
            is_correct = int(state_internal == t) if output_valid else ""
            rows_sm_detail.append((
                name, i, t, state_internal, state_output, output_valid, is_correct
            ))
        n_wrong = n_win - n_correct
        acc = n_correct / n_win if n_win > 0 else 0.0
        rows_sm.append((name, raw_win, skipped, n_win, n_correct, n_wrong, round(acc, 6)))

    csv2 = _os.path.join(artifact_dir, "per_sample_statemachine_windows.csv")
    with open(csv2, "w", encoding="utf-8") as f:
        f.write("sample_name,raw_windows,skipped_warmup_windows,total_windows,output_valid_windows,correct_windows,wrong_windows,accuracy\n")
        for row in rows_sm:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[3]},{row[4]},{row[5]},{row[6]}\n")
    csv2_detail = _os.path.join(artifact_dir, "statemachine_window_details.csv")
    with open(csv2_detail, "w", encoding="utf-8") as f:
        f.write("sample_name,window_index,target,state_internal,state_output,output_valid,is_correct\n")
        for row in rows_sm_detail:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n")
    total_sm_wins = sum(r[3] for r in rows_sm)
    total_sm_correct = sum(r[4] for r in rows_sm)
    total_sm_wrong = sum(r[5] for r in rows_sm)
    sm_global_acc = total_sm_correct / total_sm_wins if total_sm_wins > 0 else 0.0
    print(f"[OK] 状态机逐样本窗口 CSV: {csv2} "
          f"(n_samples={len(rows_sm)}, total_windows={total_sm_wins}, "
          f"warmup_frames={warmup_frames} skipped, "
          f"global_acc={round(sm_global_acc, 6) if total_sm_wins > 0 else 0})")

    # ---- CSV 3: 样本级最终预测 ----
    # 与 compute_sample_metrics 一致：pred 是 Stage1 AND Stage2 的最终融合输出。
    rows_sample = []
    for d in details:
        name = d.get("sample_name", "")
        target = int(d.get("target", 0))
        pred = int(d.get("pred", -1))
        is_correct = 1 if pred == target else 0
        n_win = len(d.get("window_probs", []))
        stage1_ok = 1 if d.get("stage1_pass", False) else 0
        fallback = 1 if d.get("fallback", False) else 0
        rows_sample.append((name, target, pred, is_correct, n_win, stage1_ok, fallback))

    csv3 = _os.path.join(artifact_dir, "per_sample_final_prediction.csv")
    with open(csv3, "w", encoding="utf-8") as f:
        f.write("sample_name,target,pred,is_correct,total_windows,stage1_pass,is_fallback\n")
        for row in rows_sample:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n")
    total_correct = sum(r[3] for r in rows_sample)
    sample_acc = total_correct / len(rows_sample) if rows_sample else 0.0
    print(f"[OK] 样本级最终预测 CSV: {csv3} "
          f"(n_samples={len(rows_sample)}, "
          f"accuracy={round(sample_acc, 6) if rows_sample else 0})")

    def _assert_equal(csv_name, field, actual, expected):
        if expected is None:
            return
        assert int(actual) == int(expected), (
            f"{csv_name} mismatch for {field}: csv={actual}, official={expected}"
        )

    def _assert_close(csv_name, field, actual, expected, tol=1e-9):
        if expected is None:
            return
        assert abs(float(actual) - float(expected)) <= tol, (
            f"{csv_name} mismatch for {field}: csv={actual}, official={expected}"
        )

    def _cm_total_errors(summary):
        cm = (summary or {}).get("confusion_matrix") or {}
        return int(cm.get("FP", 0)) + int(cm.get("FN", 0))

    window_model_summary = eval_payload.get("window_model_summary") or {}
    _assert_equal(
        "per_sample_xgboost_windows.csv",
        "total_windows",
        total_xgb_wins,
        window_model_summary.get("total_windows"),
    )
    _assert_equal(
        "per_sample_xgboost_windows.csv",
        "wrong_windows",
        total_xgb_wrong,
        _cm_total_errors(window_model_summary),
    )
    _assert_close(
        "per_sample_xgboost_windows.csv",
        "accuracy",
        xgb_global_acc,
        window_model_summary.get("accuracy"),
    )

    window_stream_summary = eval_payload.get("window_stream_summary") or {}
    _assert_equal(
        "per_sample_statemachine_windows.csv",
        "total_windows",
        total_sm_wins,
        window_stream_summary.get("total_windows"),
    )
    _assert_equal(
        "per_sample_statemachine_windows.csv",
        "wrong_windows",
        total_sm_wrong,
        _cm_total_errors(window_stream_summary),
    )
    _assert_close(
        "per_sample_statemachine_windows.csv",
        "accuracy",
        sm_global_acc,
        window_stream_summary.get("accuracy"),
    )
    _assert_equal(
        "per_sample_statemachine_windows.csv",
        "warmup_frames",
        warmup_frames,
        window_stream_summary.get("warmup_frames"),
    )

    sample_summary = eval_payload.get("summary") or {}
    _assert_equal(
        "per_sample_final_prediction.csv",
        "total_samples",
        len(rows_sample),
        sample_summary.get("total_samples", len(rows_sample)),
    )
    _assert_equal(
        "per_sample_final_prediction.csv",
        "wrong_samples",
        len(rows_sample) - total_correct,
        _cm_total_errors(sample_summary),
    )
    _assert_close(
        "per_sample_final_prediction.csv",
        "accuracy",
        sample_acc,
        sample_summary.get("accuracy"),
    )


def plot_error_samples(artifact_dir, split="test", method="state_machine",
                       window_sec=5, stride_sec=1):
    """
    准确率非 100% 的样本画图。4 子图:
      1. 原始 target (0/1 横线)
      2. 窗口级 XGBoost 概率 (probs)
      3. 窗口级状态机 EMA score
      4. 后处理标签值 (states 0/1)
    """
    import os as _os
    import numpy as _np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except ImportError:
        print("[WARN] matplotlib not installed, skip plotting")
        return

    details = _load_eval_details(artifact_dir, split, method)
    if not details:
        print("[WARN] no eval details found")
        return

    errors = [d for d in details if d["pred"] != d["target"]]
    if not errors:
        print("[OK] all samples correct, no plots needed")
        return

    out_dir = _os.path.join(artifact_dir, "error_plots")
    _os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Wrong predictions: {len(errors)}/{len(details)}")

    for d in errors:
        target = d["target"]
        pred = d["pred"]
        probs = d.get("window_probs", [])
        scores = d.get("window_scores", [])
        states = d.get("window_states", [])
        n_win = d.get("n_windows", len(probs))
        t = _np.arange(n_win) * stride_sec + window_sec / 2.0 if n_win > 0 else _np.array([])

        fig, axes = _plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # 1. Ground truth
        ax = axes[0]
        ax.set_ylabel("Target", fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{d['sample_name']}  target={target}  pred={pred}  "
                     f"s1_pass={d.get('stage1_pass',False)}  fb={d.get('fallback',False)}",
                     fontsize=10)
        if n_win > 0:
            c = "green" if target == 1 else "red"
            ax.axhline(y=target, color=c, linewidth=2, linestyle="--", alpha=0.7)
            ax.fill_between([t[0], t[-1]], target - 0.05, target + 0.05,
                            alpha=0.15, color=c)

        # 2. XGBoost probs (continuous)
        ax = axes[1]
        ax.set_ylabel("Model Probs", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if n_win > 0:
            ax.step(t, probs, where="mid", linewidth=1.5, color="steelblue")
            ax.fill_between(t, 0, _np.array(probs), alpha=0.12, color="steelblue", step="mid")
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

        # 3. State machine EMA scores (continuous)
        ax = axes[2]
        ax.set_ylabel("State Machine Score", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if n_win > 0 and len(scores) > 0:
            ax.plot(t, scores, linewidth=2, color="darkorange", marker=".", markersize=3)
            ax.fill_between(t, 0, _np.array(scores), alpha=0.10, color="darkorange")

        # 4. State machine labels (0/1)
        ax = axes[3]
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("State Labels", fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)
        if n_win > 0 and len(states) > 0:
            ax.step(t, states, where="mid", linewidth=2, color="crimson")
            is_wrong = pred != target
            ax.text(t[len(t) // 2] if len(t) > 0 else 0, 0.5,
                    "WRONG" if is_wrong else "OK",
                    fontsize=28, color="red" if is_wrong else "green",
                    alpha=0.25, weight="bold", ha="center", va="center")

        _plt.tight_layout()
        safe_name = d["sample_name"].replace("/", "_").replace("\\", "_")
        fig.savefig(_os.path.join(out_dir, f"{safe_name}.png"), dpi=600, bbox_inches="tight")
        _plt.close(fig)

    print(f"[OK] {len(errors)} plots -> {out_dir}/")


def main():
    p = argparse.ArgumentParser(
        description="手表佩戴活体检测 — 全流程主控脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --dataset_dir dataset --artifact_dir artifacts
  %(prog)s --stop_after s04
  %(prog)s --skip s02,s03
  %(prog)s --artifact_dir artifacts --export_deploy  (仅评估 + 导出)
""")

    # ── 通用参数 ──
    p.add_argument("--artifact_dir", default="artifacts", help="产物目录")
    p.add_argument("--dataset_dir", default="dataset", help="原始 H5 数据目录")
    p.add_argument("--n_workers", type=int,
                   default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                   help="并行 worker 数")
    p.add_argument("--dry_run", action="store_true", help="只打印命令不执行")
    p.add_argument("--runtime_profile", default="balanced",
                   choices=["fast", "balanced", "thorough"],
                   help="运行预算档：fast 更快，balanced 默认折中，thorough 恢复更重搜参/后处理预算")
    p.add_argument(
        "--feature_selection_mode",
        choices=["manual", "auto"],
        default="manual",
        help="manual 在 s04 导出完整排序后暂停；auto 执行无人值守特征选择与训练",
    )
    p.add_argument(
        "--manual_feature_file",
        default=None,
        help="manual 恢复训练时使用的 CSV 特征文件；默认 artifact_dir/manual_feature_selection.csv",
    )

    # ── 步骤控制 ──
    p.add_argument("--skip", default="", help="跳过的步骤，逗号分隔 (如 s03,s04)")
    p.add_argument("--stop_after", default="s06_cb",
                   help="运行到此步骤后停止；默认 s06_cb，即导出部署配方后停止")

    # ── s03 参数 ──
    p.add_argument("--window_sec", type=int, default=5, choices=[3, 5],
                   help="Stage2 窗口秒数：3s (75点@25Hz) 或 5s (125点@25Hz)")
    p.add_argument("--stride_sec", type=int, default=1, help="Stage2 滑窗步长（秒）")
    p.add_argument("--skip_initial_windows", type=int, default=0,
                   help="optional extra leading-window skip after automatic [3:-3] trimming")
    p.add_argument("--use_stage2_ir", action=argparse.BooleanOptionalAction, default=False,
                   help="legacy compatibility flag; Stage2 model features are always ambient/green/ACC only")

    # ── s04 参数 ──
    p.add_argument(
        "--max_features",
        type=int,
        default=15,
        help=(
            "auto 模式的最终特征数上限；manual 首阶段仅用于 s04 展示，"
            "恢复训练时的实际特征数完全由 CSV 决定"
        ),
    )
    p.add_argument("--min_fold_auc", type=float, default=0.55, help="稳定性选择最低有效 AUC")
    p.add_argument("--skip_vif", action=argparse.BooleanOptionalAction, default=False,
                   help="跳过 s04 VIF 清洗步骤；特征很多或 VIF 卡住时建议开启")
    p.add_argument("--deployment_score_weight", type=float, default=0.25,
                   help="s04 部署导向重排权重。0=保持原始重要性排序")

    # ── s05 参数 ──
    p.add_argument("--fp_cost_weight", type=float, default=0.25,
                   help="s04 sample/state-machine FP cost proxy reranking weight")
    p.add_argument("--fp_proxy_recall_floor", type=float, default=0.95,
                   help="s04 FP proxy positive-window recall floor")
    p.add_argument("--fp_proxy_state_k_on", type=int, default=3,
                   help="s04 FP proxy consecutive windows for state trigger")
    p.add_argument("--ranking_objective", default="balanced",
                   choices=["balanced", "window_accuracy"],
                   help="s04 feature ranking objective; window_accuracy relaxes group caps for raw Stage2 accuracy")
    p.add_argument("--threshold_objective", default="accuracy",
                   choices=["f1", "precision", "recall", "fbeta", "precision_constrained", "accuracy"])
    p.add_argument("--threshold_beta", type=float, default=0.5,
                   help="F-beta 参数 (<1 偏precision)")

    # ── s06 参数 ──
    p.add_argument("--threshold_min_precision", type=float, default=0.95,
                   help="s05 precision_constrained threshold search minimum precision")

    # s05 model-search params
    p.add_argument("--model_search", action=argparse.BooleanOptionalAction, default=True,
                   help="enable s05 XGBoost param search under a node budget; use --no-model_search to disable")
    p.add_argument("--mine_hard_negatives", action=argparse.BooleanOptionalAction, default=None,
                   help="train an OOF train-only hard-negative candidate; defaults on for manual-selection resume")
    p.add_argument("--model_search_strategy", default="staged_group_cv",
                   choices=["staged_group_cv", "single_split"],
                   help="s05 model search strategy")
    p.add_argument("--max_model_nodes", type=int, default=500,
                   help="s05 max total XGBoost nodes for --model_search")
    p.add_argument("--model_search_fp_cost", type=float, default=2.0,
                   help="s05 FP penalty in model-search score")
    p.add_argument("--model_search_size_cost", type=float, default=0.1,
                   help="s05 model-size penalty in model-search score")
    p.add_argument("--model_search_accuracy_tolerance", type=float, default=0.0,
                   help="s05 accuracy gap allowed when preferring a smaller model; default 0.0 means accuracy strictly wins under the node budget")
    p.add_argument("--model_search_valid_fraction", type=float, default=0.5,
                   help="fraction of valid calibration pool used for model search")
    p.add_argument("--model_search_max_candidates", type=int, default=180,
                   help="maximum sampled s05 model-search candidates")
    p.add_argument("--model_search_stage1_top_k", type=int, default=4,
                   help="number of stage-1 structure candidates advanced to stage-2 refine")
    p.add_argument("--model_search_stage2_top_k", type=int, default=24,
                   help="number of stage-A candidates kept for s05 staged_group_cv")
    p.add_argument("--model_search_feature_counts", type=str, default="8,10,12,15,18",
                   help="仅 auto 模式使用的特征数量候选；manual 模式严格使用 CSV，不搜索特征数量")
    p.add_argument("--model_search_full_top_k", type=int, default=1,
                   help="特征数量 quick 评估后，对得分前 N 个 k 做完整模型搜参；quick 最优 k 最后运行，保证最终产物保留最佳候选")
    p.add_argument("--feature_search_local_swap", action=argparse.BooleanOptionalAction, default=True,
                   help="enable fixed-size local feature swaps in s05 feature-count search")
    p.add_argument("--feature_search_swap_tail_size", type=int, default=3,
                   help="number of lowest-ranked selected features eligible for local swaps")
    p.add_argument("--feature_search_swap_pool_size", type=int, default=8,
                   help="number of next-ranked candidate features considered for local swaps")
    p.add_argument("--feature_search_swap_max_candidates", type=int, default=8,
                   help="maximum local-swap feature sets evaluated per k")
    p.add_argument("--model_search_cv_folds", type=int, default=3,
                   help="s05 staged_group_cv folds")
    p.add_argument("--model_search_cv_repeats", type=int, default=2,
                   help="s05 staged_group_cv repeats")
    p.add_argument("--model_search_random_state", type=int, default=42,
                   help="s05 model-search sampling/CV seed")
    p.add_argument("--model_search_n_workers", type=int, default=None,
                   help="parallel s05 model candidates; defaults to --n_workers")
    p.add_argument("--model_search_n_estimators", default="20,25,30,35,40,45,50,55,60",
                   help="comma-separated s05 n_estimators candidates")
    p.add_argument("--model_search_max_depth", default="2,3,4,5",
                   help="comma-separated s05 max_depth candidates")
    p.add_argument("--model_search_learning_rate", default="0.025,0.03,0.04,0.05,0.06,0.08,0.10,0.15,0.20",
                   help="comma-separated s05 learning_rate candidates")
    p.add_argument("--model_search_min_child_weight", default="10,15,20,25,30,40,50",
                   help="comma-separated s05 min_child_weight candidates")
    p.add_argument("--model_search_reg_lambda", default="5,8,10,12,16,20,30",
                   help="comma-separated s05 reg_lambda candidates")
    p.add_argument("--model_search_reg_alpha", default="0,0.5,1,1.5,2,3",
                   help="comma-separated s05 reg_alpha candidates")
    p.add_argument("--model_search_subsample", default="0.70,0.75,0.80,0.85,0.90",
                   help="comma-separated s05 subsample candidates")
    p.add_argument("--model_search_colsample_bytree", default="0.70,0.75,0.80,0.85,0.90",
                   help="comma-separated s05 colsample_bytree candidates")
    # s06 eval / calibration params
    p.add_argument("--calibration_method", default="isotonic", choices=["none", "isotonic"])
    p.add_argument("--threshold_valid_fraction", type=float, default=0.5)
    p.add_argument("--calibration_random_state", type=int, default=42)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"],
                   help="s06 评估用的数据 split")
    p.add_argument("--export_deploy", action=argparse.BooleanOptionalAction, default=True,
                   help="s06 导出部署产物 (--no-export_deploy 跳过)")
    p.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=False,
                   help="s06 运行 legacy 状态机参数优化；默认不跑，需显式 --optimize")
    # 图表分析均为默认行为，无需 CLI 配置
    p.add_argument("--export_window_cache", action=argparse.BooleanOptionalAction, default=False,
                   help="export window-level NPZ cache for s07 postprocess optimization")
    p.add_argument("--optimize_postprocess", action=argparse.BooleanOptionalAction, default=False,
                   help="run s07 FP-sensitive postprocess optimization on cached windows")
    p.add_argument("--full_optimize", action="store_true",
                   help="enable full search loop: model/feature-count search, window cache export, and s07 postprocess optimization")
    p.add_argument("--with_postprocess", action="store_true",
                   help="alias for --export_window_cache --optimize_postprocess")
    p.add_argument("--auto_optimize_e2e", action="store_true",
                   help="product-metric-first automatic E2E optimization using s04/s05/s06/s07")
    p.add_argument("--accuracy_first_optimize", action="store_true",
                   help="只优化 Stage2 raw window accuracy；除非显式指定，否则不自动打开 hard-negative、s07 后处理或泛化审计")
    p.add_argument("--hard_negative_optimize", action="store_true",
                   help="removed legacy shortcut; use --accuracy_first_optimize, --with_postprocess, or --auto_optimize_e2e")
    p.add_argument("--staged_e2e_optimize", action="store_true",
                   help="removed legacy shortcut; run the main flow directly or use --with_postprocess / --auto_optimize_e2e")
    p.add_argument("--postprocess_split", default="valid", metavar="valid",
                   help="split used by s07 postprocess optimization; fixed to valid to protect test isolation")
    p.add_argument("--postprocess_fp_cost", type=float, default=1.5,
                   help="s07 sample false-positive cost")
    p.add_argument("--max_sample_fp_rate", type=float, default=0.02,
                   help="s07 maximum FP / true-negative-sample rate")
    p.add_argument("--max_false_worn_event_rate", type=float, default=0.02,
                   help="s07 maximum negative-sample false-worn event rate")
    p.add_argument("--max_window_fp_rate", type=float, default=0.01,
                   help="s07 maximum streaming window false-positive rate")
    p.add_argument("--max_first_worn_output_p95_sec", type=float, default=3.0,
                   help="s07 maximum P95 added latency from first valid Stage2 probability")
    p.add_argument("--postprocess_search_budget", type=int, default=240,
                   help="maximum s07 postprocess candidates to evaluate; <=0 keeps full grid")
    p.add_argument("--postprocess_warmup_frames", type=int, default=5,
                   help="s07 window-level metrics skip this many leading state-machine windows per sample")
    p.add_argument("--run_generalization_audit", action=argparse.BooleanOptionalAction, default=False,
                   help="run optional generalization audit after s06_eval")
    p.add_argument("--audit_min_support", type=int, default=10,
                   help="minimum sample/window count before a stratum is treated as reliable in generalization audit")
    p.add_argument("--hard_negative_weight", type=float, default=3.0,
                   help="s05 sample weight assigned to train-only mined hard negatives")
    p.add_argument("--hard_negative_top_percentile", type=float, default=0.10,
                   help="s05 fraction of highest-probability train negatives selected as hard negatives")
    p.add_argument("--hard_negative_min_probability", type=float, default=None,
                   help="s05 minimum OOF probability for hard-negative mining; defaults to initial threshold")
    p.add_argument("--export_deploy_cookbook", action=argparse.BooleanOptionalAction, default=True,
                   help="导出部署配方给嵌入式同事 (--no-export_deploy_cookbook 跳过)")

    _raw_argv = sys.argv[1:]
    args = p.parse_args()
    if args.model_search_n_workers is None:
        args.model_search_n_workers = args.n_workers

    if args.postprocess_split != "valid":
        p.error("postprocess_split must be 'valid'; test is reserved for frozen read-only replay")
    apply_runtime_profile(args, _raw_argv)
    if args.hard_negative_optimize:
        print("[ERROR] --hard_negative_optimize has been removed from the recommended pipeline.")
        print("        Use --accuracy_first_optimize for window accuracy, or --with_postprocess for explicit s07 postprocess search.")
        sys.exit(2)
    if args.staged_e2e_optimize:
        print("[ERROR] --staged_e2e_optimize has been removed from the recommended pipeline.")
        print("        Use the main flow directly, or add --with_postprocess when postprocess search is needed.")
        sys.exit(2)
    if args.accuracy_first_optimize and args.hard_negative_optimize:
        print("[ERROR] --accuracy_first_optimize and --hard_negative_optimize target different objectives; run them separately.")
        sys.exit(2)
    if args.model_search_full_top_k < 1:
        print("[ERROR] --model_search_full_top_k must be >= 1")
        sys.exit(2)
    if args.accuracy_first_optimize:
        args.model_search = True
        args.threshold_objective = "accuracy"
        args.ranking_objective = "window_accuracy"
        args.deployment_score_weight = 0.0
        args.fp_cost_weight = 0.0
        if "--model_search_full_top_k" not in _raw_argv:
            args.model_search_full_top_k = max(3, int(args.model_search_full_top_k))
    if args.auto_optimize_e2e:
        args.feature_selection_mode = "auto"
        args.model_search = True
        args.export_window_cache = True
        args.optimize_postprocess = True
        args.ranking_objective = "balanced"
        args.threshold_objective = "precision_constrained"
        if "--threshold_min_precision" not in _raw_argv:
            args.threshold_min_precision = 0.97
        if "--model_search_fp_cost" not in _raw_argv:
            args.model_search_fp_cost = 4.0
        if "--fp_cost_weight" not in _raw_argv:
            args.fp_cost_weight = max(float(args.fp_cost_weight), 0.35)
        if "--postprocess_fp_cost" not in _raw_argv:
            args.postprocess_fp_cost = max(float(args.postprocess_fp_cost), 4.0)
    if args.full_optimize:
        args.feature_selection_mode = "auto"
        args.model_search = True
        args.export_window_cache = True
        args.optimize_postprocess = True
    if args.with_postprocess:
        args.model_search = True
        args.export_window_cache = True
        args.optimize_postprocess = True

    _feature_counts = []
    if args.feature_selection_mode == "auto":
        for _part in str(args.model_search_feature_counts or "").split(","):
            _part = _part.strip()
            if _part.isdigit():
                _k = int(_part)
                if _k <= 18:
                    _feature_counts.append(_k)
        if _feature_counts:
            args.model_search_feature_counts = ",".join(str(k) for k in sorted(set(_feature_counts)))
        if int(args.max_features) > 18:
            print("[WARN] auto-mode max_features capped at 18 for the current search policy")
            args.max_features = 18
    thread_env = configure_thread_env()
    print("[parallel] thread caps inherited by child steps: " +
          ", ".join(f"{k}={v}" for k, v in thread_env.items()))
    # 步骤定义: (key, display_name, 是否默认启用)
    all_steps = [
        ("s01",       "数据扫描 & 切分"),
        ("s02",       "Stage1 固定阈值配置"),
        ("s03",       "特征池提取"),
        ("s04",       "稳定性特征筛选"),
        ("s04_search","候选特征子集搜索"),
        ("s04_embed", "Feature embedding PCA/t-SNE/UMAP report"),
        ("s05",       "XGBoost 模型训练"),
        ("s05_viz",   "ROC/PR 曲线图"),
        ("s06_opt",   "状态机参数优化"),
        ("s06_cache", "导出 valid 逐窗 NPZ 缓存"),
        ("s06_replay_cache", "导出 replay 逐窗 NPZ 缓存"),
        ("s07_post",  "FP 敏感后处理搜参"),
        ("s06_eval",  "端到端评估"),
        ("s06_tree_viz", "XGBoost 树特征使用图"),
        ("s06_audit", "泛化审计"),
        ("s06_xpt",   "导出部署产物"),
        ("s06_feat",  "导出特征提取脚本"),
        ("s06_plot",  "画错误样本图"),
        ("s06_cb",    "导出部署配方"),
    ]

    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
    if "s10_audit" in skip_set:
        skip_set.add("s06_audit")
    raw_stop_after = args.stop_after
    stop_after = raw_stop_after
    if stop_after == "s10_audit":
        stop_after = "s06_audit"
    step_keys = [key for key, _ in all_steps]
    if stop_after not in step_keys:
        print(f"[ERROR] unknown --stop_after={raw_stop_after!r}; choose one of: {','.join(step_keys)}")
        sys.exit(2)

    manual_feature_file = args.manual_feature_file or os.path.join(
        args.artifact_dir, "manual_feature_selection.csv"
    )
    manual_resume = (
        args.feature_selection_mode == "manual"
        and "s04" in skip_set
        and os.path.exists(manual_feature_file)
    )
    if args.feature_selection_mode == "manual" and not manual_resume:
        if step_keys.index(stop_after) > step_keys.index("s04"):
            stop_after = "s04"
            print(
                "[manual] feature ranking is the first phase; stopping after s04. "
                f"Set selected=1 in {manual_feature_file} and save the CSV."
            )
        skip_set.add("s04_search")
    elif args.feature_selection_mode == "manual":
        skip_set.add("s04_search")
        print(
            "[manual] exact CSV feature names/order/count are frozen; "
            "feature-count search is disabled."
        )

    auto_enabled = []
    if stop_after in {"s06_cache", "s06_replay_cache", "s07_post"}:
        if "s06_cache" not in skip_set and not args.export_window_cache:
            args.export_window_cache = True
            auto_enabled.append("--export_window_cache")
    if (stop_after in {"s07_post"} or args.optimize_postprocess):
        if "s07_post" not in skip_set and not args.optimize_postprocess:
            args.optimize_postprocess = True
            auto_enabled.append("--optimize_postprocess")
    if stop_after == "s06_audit" and "s06_audit" not in skip_set and not args.run_generalization_audit:
        args.run_generalization_audit = True
        auto_enabled.append("--run_generalization_audit")
    if auto_enabled:
        print("[auto] enabled for --stop_after target: " + ", ".join(auto_enabled))

    stage2_ir_flag = "--use_stage2_ir" if args.use_stage2_ir else "--no-use_stage2_ir"
    s04_skip_vif_flag = "--skip_vif" if args.skip_vif else ""
    model_search_flag = "--model_search" if args.model_search else "--no-model_search"
    mine_hard_negatives = (
        args.feature_selection_mode == "manual"
        if args.mine_hard_negatives is None
        else bool(args.mine_hard_negatives)
    )
    hard_negative_flag = "--mine_hard_negatives" if mine_hard_negatives else ""
    feature_count_search_arg = (
        f'--model_search_feature_counts "{args.model_search_feature_counts}" '
        if args.feature_selection_mode == "auto"
        else ""
    )
    cache_export_flag = " --export_window_cache" if args.export_window_cache else ""
    if not args.dry_run and "s01" not in skip_set and not dataset_has_h5_files(args.dataset_dir):
        print(f"[ERROR] no .h5 files found in dataset_dir={args.dataset_dir!r}")
        print("        Pass --dataset_dir with the real H5 directory, or use --skip s01 when reusing artifacts.")
        sys.exit(2)

    # ── 构建命令 ──
    commands = {}

    # s01
    if "s01" not in skip_set:
        commands["s01"] = (
            f'"{PYTHON}" "{_script_path("s01_data_split")}" '
            f'--dataset_dir "{args.dataset_dir}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--n_workers {args.n_workers}'
        )

    # s02
    if "s02" not in skip_set:
        commands["s02"] = (
            f'"{PYTHON}" "{_script_path("s02_ir_dc_threshold")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--n_workers {args.n_workers}'
        )

    # s03
    if "s03" not in skip_set:
        commands["s03"] = (
            f'"{PYTHON}" "{_script_path("s03_extract_feature_pool")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--n_workers {args.n_workers}'
        )

    # s04
    if "s04" not in skip_set:
        commands["s04"] = (
            f'"{PYTHON}" "{_script_path("s04_feature_selection")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--max_features {args.max_features} '
            f'--min_fold_auc {args.min_fold_auc} '
            f'--deployment_score_weight {args.deployment_score_weight} '
            f'--fp_cost_weight {args.fp_cost_weight} '
            f'--fp_proxy_recall_floor {args.fp_proxy_recall_floor} '
            f'--fp_proxy_state_k_on {args.fp_proxy_state_k_on} '
            f'--ranking_objective {args.ranking_objective} '
            f'--feature_selection_mode {args.feature_selection_mode} '
            f'{s04_skip_vif_flag} '
            f'--n_workers {args.n_workers}'
        )

    # s04_search: 候选特征子集搜索（可选步骤，覆写 selected_features.json）
    if "s04_search" not in skip_set and args.feature_selection_mode == "auto":
        commands["s04_search"] = (
            f'"{PYTHON}" "{_script_path("s04_feature_selection")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--max_features {args.max_features} '
            f'--min_fold_auc {args.min_fold_auc} '
            f'--deployment_score_weight {args.deployment_score_weight} '
            f'--fp_cost_weight {args.fp_cost_weight} '
            f'--fp_proxy_recall_floor {args.fp_proxy_recall_floor} '
            f'--fp_proxy_state_k_on {args.fp_proxy_state_k_on} '
            f'--ranking_objective {args.ranking_objective} '
            f'--feature_selection_mode auto '
            f'{s04_skip_vif_flag} '
            f'--run_subset_search '
            f'--subset_search_max_features {args.max_features} '
            f'--n_workers {args.n_workers}'
        )

    # s04_embed: 特征空间嵌入可视化报告（默认自动生成）
    if "s04_embed" not in skip_set:
        commands["s04_embed"] = (
            f'__feature_embedding_report__ '
            f'--methods "pca,tsne" '
            f'--dims "2,3" '
            f'--formats "png" '
            f'--max_points 0 '
            f'--perplexity 30.0 '
            f'--random_state 42 '
            f'--dpi 600'
        )

    # s05
    if "s05" not in skip_set:
        selection_args = f'--feature_selection_mode {args.feature_selection_mode} '
        if args.feature_selection_mode == "manual":
            selection_args += f'--manual_feature_file "{manual_feature_file}" '
        local_swap_flag = (
            "--no-feature_search_local_swap"
            if args.feature_selection_mode == "manual"
            else ("--feature_search_local_swap" if args.feature_search_local_swap else "--no-feature_search_local_swap")
        )
        commands["s05"] = (
            f'"{PYTHON}" "{_script_path("s05_train_final_model")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'{selection_args}'
            f'--max_features {args.max_features} '
            f'--threshold_objective {args.threshold_objective} '
            f'--threshold_beta {args.threshold_beta} '
            f'--threshold_min_precision {args.threshold_min_precision} '
            f'--calibration_method {args.calibration_method} '
            f'--threshold_valid_fraction {args.threshold_valid_fraction} '
            f'--calibration_random_state {args.calibration_random_state} '
            f'--window_sec {args.window_sec} '
            f'--step_sec {args.stride_sec} '
            f'{stage2_ir_flag} '
            f'{model_search_flag} '
            f'{hard_negative_flag} '
            f'--model_search_strategy {args.model_search_strategy} '
            f'--max_model_nodes {args.max_model_nodes} '
            f'--model_search_fp_cost {args.model_search_fp_cost} '
            f'--model_search_size_cost {args.model_search_size_cost} '
            f'--model_search_accuracy_tolerance {args.model_search_accuracy_tolerance} '
            f'--model_search_valid_fraction {args.model_search_valid_fraction} '
            f'--model_search_max_candidates {args.model_search_max_candidates} '
            f'--model_search_stage1_top_k {args.model_search_stage1_top_k} '
            f'--model_search_stage2_top_k {args.model_search_stage2_top_k} '
            f'{feature_count_search_arg}'
            f'{local_swap_flag} '
            f'--feature_search_swap_tail_size {args.feature_search_swap_tail_size} '
            f'--feature_search_swap_pool_size {args.feature_search_swap_pool_size} '
            f'--feature_search_swap_max_candidates {args.feature_search_swap_max_candidates} '
            f'--model_search_cv_folds {args.model_search_cv_folds} '
            f'--model_search_cv_repeats {args.model_search_cv_repeats} '
            f'--model_search_random_state {args.model_search_random_state} '
            f'--model_search_n_workers {args.model_search_n_workers} '
            f'--model_search_n_estimators "{args.model_search_n_estimators}" '
            f'--model_search_max_depth "{args.model_search_max_depth}" '
            f'--model_search_learning_rate "{args.model_search_learning_rate}" '
            f'--model_search_min_child_weight "{args.model_search_min_child_weight}" '
            f'--model_search_reg_lambda "{args.model_search_reg_lambda}" '
            f'--model_search_reg_alpha "{args.model_search_reg_alpha}" '
            f'--model_search_subsample "{args.model_search_subsample}" '
            f'--model_search_colsample_bytree "{args.model_search_colsample_bytree}"'
        )

    # s06_opt
    if "s06_opt" not in skip_set and args.optimize:
        commands["s06_opt"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split valid '
            f'--n_workers {args.n_workers} '
            f'--optimize '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag}'
        )
    elif "s06_opt" not in skip_set:
        print("(s06_opt: --no-optimize 跳过)")

    # s06_cache: export the postprocess-search split independently from legacy s06 optimization.
    if "s06_cache" not in skip_set and args.export_window_cache:
        commands["s06_cache"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.postprocess_split} '
            f'--n_workers {args.n_workers} '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--window_output_root window_outputs'
            f'{cache_export_flag}'
        )
    elif "s06_cache" not in skip_set:
        print("(s06_cache: --no-export_window_cache skipped)")

    if ("s06_replay_cache" not in skip_set and args.export_window_cache
            and args.optimize_postprocess and args.split != args.postprocess_split):
        commands["s06_replay_cache"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.split} '
            f'--n_workers {args.n_workers} '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--window_output_root window_outputs'
            f'{cache_export_flag}'
        )
    elif "s06_replay_cache" not in skip_set:
        print("(s06_replay_cache: skipped)")

    # s06_eval
    if "s06_eval" not in skip_set:
        commands["s06_eval"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.split} '
            f'--n_workers {args.n_workers} '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--window_output_root window_outputs'
        )

    # s06_audit: read-only generalization audit over existing artifacts.
    if "s06_audit" not in skip_set and args.run_generalization_audit:
        commands["s06_audit"] = (
            f'__generalization_audit__ '
            f'--split {args.split} '
            f'--method state_machine '
            f'--min_support {args.audit_min_support}'
        )
    elif "s06_audit" not in skip_set:
        print("(s06_audit: --no-run_generalization_audit skipped)")

    # s07_post: tune the richer FP-sensitive state machine on cached windows.
    if "s07_post" not in skip_set and args.optimize_postprocess:
        commands["s07_post"] = (
            f'"{PYTHON}" "{_script_path("s07_postprocess_optimize")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.postprocess_split} '
            f'--cache_root window_outputs '
            f'--fp_cost {args.postprocess_fp_cost} '
            f'--max_sample_fp_rate {args.max_sample_fp_rate} '
            f'--max_false_worn_event_rate {args.max_false_worn_event_rate} '
            f'--max_window_fp_rate {args.max_window_fp_rate} '
            f'--max_first_worn_output_p95_sec {args.max_first_worn_output_p95_sec} '
            f'--search_budget {args.postprocess_search_budget} '
            f'--warmup_frames {args.postprocess_warmup_frames} '
            f'--n_workers {args.n_workers} '
            f'--replay_split {args.split}'
        )
    elif "s07_post" not in skip_set:
        print("(s07_post: --no-optimize_postprocess 跳过)")

    # s06_xpt
    if "s06_xpt" not in skip_set and args.export_deploy:
        commands["s06_xpt"] = (
            f'"{PYTHON}" "{_script_path("s06_deploy_eval")}" '
            f'--artifact_dir "{args.artifact_dir}" '
            f'--split {args.split} '
            f'--n_workers {args.n_workers} '
            f'--window_sec {args.window_sec} '
            f'--stride_sec {args.stride_sec} '
            f'--skip_initial_windows {args.skip_initial_windows} '
            f'{stage2_ir_flag} '
            f'--export_deploy'
        )
    elif "s06_xpt" not in skip_set:
        print("(s06_xpt: --no-export_deploy 跳过)")

    # s06_plot (纯 Python 调用，非子进程，默认自动生成)
    if "s06_plot" not in skip_set:
        commands["s06_plot"] = "__plot__"

    # s06_feat (纯 Python 调用)
    if "s06_feat" not in skip_set and args.export_deploy_cookbook:
        commands["s06_feat"] = "__extractor__"
    elif "s06_feat" not in skip_set:
        print("(s06_feat: --no-export_deploy_cookbook 跳过)")

    # s06_cb (纯 Python 调用)
    if "s06_cb" not in skip_set and args.export_deploy_cookbook:
        commands["s06_cb"] = "__cookbook__"
    elif "s06_cb" not in skip_set:
        print("(s06_cb: --no-export_deploy_cookbook 跳过)")

    # s05_viz — ROC/PR curves (embedded, 默认自动生成)
    if "s05_viz" not in skip_set:
        commands["s05_viz"] = "__s05_viz__"

    # s06_tree_viz — XGBoost tree feature usage (embedded, 默认自动生成)
    if "s06_tree_viz" not in skip_set:
        commands["s06_tree_viz"] = "__s06_tree_viz__"

    # ── 执行 ──
    print("=" * 70)
    print(" 手表佩戴活体检测 — 全流程")
    print("=" * 70)
    print(f"  产物目录:     {args.artifact_dir}")
    print(f"  数据目录:     {args.dataset_dir}")
    print(f"  并行 worker:  {args.n_workers}")
    print(f"  特征选择:     {args.feature_selection_mode}")
    if manual_resume:
        print(f"  人工特征 CSV: {manual_feature_file}（名称、顺序、数量均固定）")
    else:
        print(f"  s04 特征上限: {args.max_features}")
    print(f"  评估 split:   {args.split}")
    print(f"  运行预算档:   {args.runtime_profile}")
    print(f"  导出部署:     {'是' if args.export_deploy else '否'}")
    print(f"  状态机优化:   {'是' if args.optimize else '否'}")
    print(f"  后处理搜参:   {'是' if args.optimize_postprocess else '否'}")
    print("=" * 70)

    total_start = time.time()
    runtime_events = []
    results = {}
    completed_keys = set()
    stopped_at = None

    for key, display_name in all_steps:
        if key in skip_set:
            print(f"\n  [SKIP] {display_name} (--skip)")
            continue

        cmd = commands.get(key)
        if cmd is None:
            continue  # optional step not requested

        special_cmd = (
            cmd in {"__plot__", "__extractor__", "__cookbook__",
                    "__s05_viz__", "__s06_tree_viz__"}
            or str(cmd).startswith("__feature_embedding_report__")
            or str(cmd).startswith("__generalization_audit__")
        )
        if args.dry_run and special_cmd:
            print(f"\n[RUN] {display_name}")
            print(f"  {cmd}")
            print("  (dry-run, skipped)")
            _record_runtime(runtime_events, display_name, 0.0, dry_run=True)
            completed_keys.add(key)
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if cmd == "__plot__":
            t0 = time.time()
            generate_eval_csv(args.artifact_dir, split=args.split)
            plot_error_samples(args.artifact_dir, split=args.split,
                               window_sec=args.window_sec, stride_sec=args.stride_sec)
            dt = time.time() - t0
            _record_runtime(runtime_events, display_name, dt, dry_run=False)
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if cmd == "__extractor__":
            t0 = time.time()
            export_feature_extractor_script(args.artifact_dir)
            dt = time.time() - t0
            _record_runtime(runtime_events, display_name, dt, dry_run=False)
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if cmd == "__cookbook__":
            t0 = time.time()
            export_deploy_cookbook(args.artifact_dir)
            if "s06_feat" in completed_keys:
                export_stage2_feature_contracts(args.artifact_dir)
                export_golden_vectors(args.artifact_dir)
                validate_deploy_artifact_consistency(args.artifact_dir)
            dt = time.time() - t0
            _record_runtime(runtime_events, display_name, dt, dry_run=False)
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if str(cmd).startswith("__feature_embedding_report__"):
            t0 = time.time()
            run_embedded_feature_embedding_report(args)
            dt = time.time() - t0
            _record_runtime(runtime_events, display_name, dt, dry_run=False)
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if str(cmd).startswith("__generalization_audit__"):
            t0 = time.time()
            run_embedded_generalization_audit(args)
            dt = time.time() - t0
            _record_runtime(runtime_events, display_name, dt, dry_run=False)
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if cmd == "__s05_viz__":
            t0 = time.time()
            _bundle_path = os.path.join(args.artifact_dir, "model_bundle.pkl")
            _valid_csv = os.path.join(args.artifact_dir, "feature_pool_valid.csv")
            if os.path.exists(_bundle_path) and os.path.exists(_valid_csv):
                import pandas as _pd
                _bundle = joblib.load(_bundle_path)
                _model = _bundle.get("model") or _bundle.get("raw_model")
                _feat = list(_bundle["feature_names"])
                _fill = _bundle.get("fill_values", {})
                _df = _pd.read_csv(_valid_csv)
                _y = _df["target"].values.astype(int)
                _X = _df[_feat].fillna({f: _fill.get(f, 0.0) for f in _feat}).values.astype(float)
                import s05_train_final_model as _s05
                _s05.export_roc_pr_curves(_model, _X, _y, args.artifact_dir)
            else:
                print(f"[WARN] skip s05_viz: need model_bundle.pkl and feature_pool_valid.csv")
            dt = time.time() - t0
            _record_runtime(runtime_events, display_name, dt, dry_run=False)
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        if cmd == "__s06_tree_viz__":
            t0 = time.time()
            _bundle_path2 = os.path.join(args.artifact_dir, "model_bundle.pkl")
            if os.path.exists(_bundle_path2):
                import s06_deploy_eval as _s06
                _s06.export_tree_feature_usage_plot(args.artifact_dir)
            else:
                print("[WARN] skip s06_tree_viz: model_bundle.pkl not found")
            dt = time.time() - t0
            _record_runtime(runtime_events, display_name, dt, dry_run=False)
            completed_keys.add(key)
            print(f"[OK] {display_name}  [{timedelta(seconds=int(dt))}]")
            if key == stop_after:
                stopped_at = key
                print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
                break
            continue

        # 特征数量搜参：先快速评估各 k（无 model_search），再对 top-N k 做完整搜参
        if (
            key == "s05"
            and args.feature_selection_mode == "auto"
            and args.model_search_feature_counts
        ):
            _counts = [int(x.strip()) for x in args.model_search_feature_counts.split(",") if x.strip()]
            _counts = sorted(set(_counts))
            if len(_counts) > 1:
                print(f"\n[特征数量搜参] 快速评估 k = {_counts}（使用默认参数，无 model_search）")
                _best_k, _best_acc = None, -1.0
                _quick_scores = []
                # 构建无 model_search 的命令模板
                _cmd_no_search = cmd.replace(" --model_search ", " --no-model_search ")
                _cmd_quick_template = _cmd_no_search
                for _k in _counts:
                    _cmd_k = re.sub(r' --max_features \d+', f' --max_features {_k}', _cmd_quick_template)
                    _cmd_k = re.sub(r'--model_search_feature_counts "[^"]*"',
                                    f'--model_search_feature_counts "{_k}"', _cmd_k)
                    print(f"\n  --- k={_k} ---")
                    _ok = _run(
                        f'{display_name} (k={_k}, quick)',
                        _cmd_k,
                        dry_run=args.dry_run,
                        runtime_events=runtime_events,
                    )
                    if not _ok:
                        continue
                    # 读取 CV 结果判断最优 k
                    if not args.dry_run:
                        _acc = _read_s05_quick_k_score(args.artifact_dir)
                        if _acc is not None:
                            print(f"  [k={_k}] quick valid accuracy={_acc:.6f}")
                            _quick_scores.append((_k, _acc))
                            if _acc > _best_acc:
                                _best_k, _best_acc = _k, _acc
                if _best_k is not None:
                    _top_n = max(1, min(int(args.model_search_full_top_k), len(_quick_scores)))
                    _selected_scores = sorted(
                        _quick_scores,
                        key=lambda item: (item[1], -item[0]),
                        reverse=True,
                    )[:_top_n]
                    _selected_scores = sorted(_selected_scores, key=lambda item: (item[1], -item[0]))
                    print(
                        f"\n[feature-count search] full search top {_top_n}: "
                        + ", ".join(f"k={_k} acc={_acc:.4f}" for _k, _acc in _selected_scores)
                    )
                    ok = True
                    for _idx, (_full_k, _full_acc) in enumerate(_selected_scores, start=1):
                        _cmd_best = re.sub(r' --max_features \d+', f' --max_features {_full_k}', cmd)
                        _cmd_best = re.sub(r'--model_search_feature_counts "[^"]*"',
                                           f'--model_search_feature_counts "{_full_k}"', _cmd_best)
                        ok = _run(
                            f'{display_name} (k={_full_k}, full search #{_idx}/{_top_n})',
                            _cmd_best,
                            dry_run=args.dry_run,
                            runtime_events=runtime_events,
                        )
                        if not ok:
                            break
                elif args.dry_run:
                    _top_n = max(1, min(int(args.model_search_full_top_k), len(_counts)))
                    _dry_counts = _counts[-_top_n:]
                    ok = True
                    for _idx, _dry_k in enumerate(_dry_counts, start=1):
                        _cmd_best = re.sub(r' --max_features \d+', f' --max_features {_dry_k}', cmd)
                        _cmd_best = re.sub(r'--model_search_feature_counts "[^"]*"',
                                           f'--model_search_feature_counts "{_dry_k}"', _cmd_best)
                        _label = f'{display_name} (representative model search #{_idx}/{_top_n})'
                        ok = _run(_label, _cmd_best, dry_run=True, runtime_events=runtime_events)
                        if not ok:
                            break
                else:
                    print(
                        "\n[FAIL] feature-count quick search finished, but no quick "
                        "accuracy was found in final_model_config.json; refusing to "
                        "skip the required representative model search."
                    )
                    ok = False
            else:
                ok = _run(display_name, cmd, dry_run=args.dry_run, runtime_events=runtime_events)
        else:
            ok = _run(display_name, cmd, dry_run=args.dry_run, runtime_events=runtime_events)
        results[key] = ok
        if ok:
            completed_keys.add(key)
        if not ok:
            print(f"\n[FAIL] 流水线中断于: {display_name}")
            sys.exit(1)

        if key == stop_after:
            stopped_at = key
            print(f"\n[STOP] 已达到 --stop_after={stop_after}，停止")
            break

    total_dt = time.time() - total_start
    _print_runtime_summary(runtime_events)
    print(f"\n{'=' * 70}")
    if stopped_at:
        print(f"[OK] 流水线已按 --stop_after={stopped_at} 结束  [{timedelta(seconds=int(total_dt))}]")
    else:
        print(f"[RUN] 执行步骤完成，开始最终验收  [{timedelta(seconds=int(total_dt))}]")
    print(f"{'=' * 70}")

    acceptance_exit_code = 0
    if not args.dry_run:
        try:
            from scientific_figures import export_pipeline_scientific_overview
            overview_paths = export_pipeline_scientific_overview(args.artifact_dir)
            print(f"[OK] scientific overview -> {overview_paths['png']}")
        except Exception as exc:
            print(f"[WARN] scientific overview incomplete: {exc}")
        try:
            from pipeline_acceptance import build_pipeline_acceptance_report
            acceptance = build_pipeline_acceptance_report(args.artifact_dir)
            acceptance_exit_code = pipeline_acceptance_exit_code(acceptance, stopped_at)
            acceptance_label = "OK" if acceptance["overall_passed"] else "FAIL"
            print(
                f"[{acceptance_label}] pipeline acceptance -> "
                f"overall_passed={acceptance['overall_passed']}"
            )
        except Exception as exc:
            acceptance_exit_code = pipeline_acceptance_exit_code({}, stopped_at)
            print(f"[FAIL] pipeline acceptance report failed: {exc}")

    if args.auto_optimize_e2e:
        export_auto_e2e_summary(
            args.artifact_dir,
            postprocess_split=args.postprocess_split,
            split=args.split,
            constraints={
                "max_sample_fp_rate": float(args.max_sample_fp_rate),
                "max_false_worn_event_rate": float(args.max_false_worn_event_rate),
                "max_first_worn_output_p95_sec": float(args.max_first_worn_output_p95_sec),
                "min_window_accuracy_delta": -0.01,
            },
            dry_run=args.dry_run,
        )

    if args.export_deploy and "s06_xpt" in completed_keys and not args.dry_run:
        pkg = os.path.join(args.artifact_dir, "deploy_package")
        print(f"\n部署产物: {pkg}/")
        if os.path.isdir(pkg):
            for f in sorted(os.listdir(pkg)):
                sz = os.path.getsize(os.path.join(pkg, f))
                print(f"  {f}  ({sz:,} bytes)")

    if acceptance_exit_code:
        sys.exit(acceptance_exit_code)


if __name__ == "__main__":
    main()
