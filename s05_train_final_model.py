
# s05_train_final_model.py
# -*- coding: utf-8 -*-

"""
步骤5：最终模型训练，防过拟合版

原则：
1. train 训练模型
2. valid 只选择窗口概率阈值和确认指标
3. test 完全不参与
4. 缺失填充值只从 train 计算
"""

import os
import json
import argparse
import hashlib
import logging
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# from sklearn.preprocessing import StandardScaler  # 去掉归一化
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, StratifiedGroupKFold, train_test_split

from s03_extract_feature_pool import filter_stage2_ir_features, is_stage2_ir_feature
from s04_feature_selection import (
    filter_features_for_deployment,
    summarize_deployment_feature_costs,
    validate_feature_pool_frames,
)
from stage2_feature_catalog import (
    FEATURE_POOL_VERSION,
)
from manual_feature_selection import load_manual_selection_csv
from scientific_figures import save_scientific_figure

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def enforce_no_stage2_ir_features(feature_names, context):
    """Filter stale IR-derived features before Stage2 model training/export."""
    original = list(feature_names)
    filtered = filter_stage2_ir_features(original)
    dropped = [f for f in original if is_stage2_ir_feature(f)]
    if dropped:
        suffix = "..." if len(dropped) > 20 else ""
        logger.warning(
            "%s: removed %d IR-derived Stage2 features: %s",
            context, len(dropped), ", ".join(dropped[:20]) + suffix,
        )
    if not filtered:
        raise ValueError(f"{context}: no non-IR Stage2 features remain after filtering")
    return filtered


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manual_feature_selection(manual_path, ranking_path, df_train, df_valid):
    """Validate CSV selection and freeze its exact feature order for audit."""
    manual_path = Path(manual_path)
    ranking_path = Path(ranking_path)
    if not manual_path.exists():
        raise ValueError(
            f"manual feature file missing: {manual_path}; "
            "manual mode does not fall back to automatic selection."
        )
    if not ranking_path.exists():
        raise ValueError(
            f"full ranking file missing: {ranking_path}; run s04 before manual training."
        )

    validate_feature_pool_frames(df_train, df_valid)
    selected, provenance = load_manual_selection_csv(
        manual_path,
        ranking_path,
        train_columns=set(df_train.columns),
        valid_columns=set(df_valid.columns),
    )
    frozen_path = manual_path.parent / "manual_selected_features.json"
    frozen_payload = {
        "schema_version": 1,
        "feature_pool_version": FEATURE_POOL_VERSION,
        "ranking_source": ranking_path.name,
        "selected_features": selected,
        "selection_notes": {},
        "selection_provenance": provenance,
    }
    tmp_path = frozen_path.with_suffix(frozen_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(frozen_payload, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, frozen_path)
    provenance["frozen_manual_feature_file"] = str(frozen_path.resolve())
    provenance["frozen_manual_feature_file_sha256"] = _sha256_file(frozen_path)
    return selected, provenance


DEFAULT_XGB_PARAMS = {
    "n_estimators": 40,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_lambda": 10,
    "reg_alpha": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}


def select_model_candidate(records, max_nodes=500, max_fpr=0.01):
    """Select one validation candidate under explicit deployment constraints."""
    leaderboard = []
    eligible = []
    for source in records:
        record = dict(source)
        reasons = []
        if not bool(record.get("finite_predictions", True)):
            reasons.append("non_finite_predictions")
        nodes = int(record.get("total_nodes", 0) or 0)
        if int(max_nodes) > 0 and nodes > int(max_nodes):
            reasons.append("node_budget")
        record["rejection_reasons"] = reasons
        record["node_budget_pass"] = "node_budget" not in reasons
        record["finite_predictions_pass"] = "non_finite_predictions" not in reasons
        record["fpr_constraint_pass"] = float(record.get("valid_fp_rate", 1.0)) <= float(max_fpr)
        leaderboard.append(record)
        if not reasons:
            eligible.append(record)
    if not eligible:
        raise ValueError("no model candidate has finite predictions within the node budget")

    feasible = [record for record in eligible if record["fpr_constraint_pass"]]
    if feasible:
        selected = min(
            feasible,
            key=lambda record: (
                -float(record.get("valid_accuracy", 0.0)),
                float(record.get("valid_fp_rate", 1.0)),
                -float(record.get("valid_recall", 0.0)),
                int(record.get("total_nodes", 0) or 0),
                float(record.get("cv_accuracy_std", float("inf"))),
                str(record.get("candidate", "")),
            ),
        )
        accepted = True
        status = "deployment_candidate"
        reason = "max_valid_accuracy_within_fpr_and_node_constraints"
    else:
        selected = min(
            eligible,
            key=lambda record: (
                float(record.get("valid_fp_rate", 1.0)),
                -float(record.get("valid_accuracy", 0.0)),
                -float(record.get("valid_recall", 0.0)),
                int(record.get("total_nodes", 0) or 0),
                float(record.get("cv_accuracy_std", float("inf"))),
                str(record.get("candidate", "")),
            ),
        )
        accepted = False
        status = "analysis_only"
        reason = "no_candidate_met_valid_fpr_constraint"
    leaderboard.sort(key=lambda record: record.get("candidate") != selected.get("candidate"))
    return {
        "selected_candidate": str(selected.get("candidate")),
        "deployment_acceptance": accepted,
        "status": status,
        "selection_reason": reason,
        "max_nodes": int(max_nodes),
        "max_valid_fp_rate": float(max_fpr),
        "leaderboard": leaderboard,
    }


def accept_hard_negative_candidate(reference, candidate, tolerance=1e-12):
    """Accept train-OOF hard-negative retraining only without valid regression."""
    reference_accuracy = float(reference.get("valid_accuracy", 0.0))
    candidate_accuracy = float(candidate.get("valid_accuracy", 0.0))
    reference_fpr = float(reference.get("valid_fp_rate", 1.0))
    candidate_fpr = float(candidate.get("valid_fp_rate", 1.0))
    if candidate_accuracy < reference_accuracy - float(tolerance):
        accepted = False
        reason = "valid_accuracy_decreased"
    elif candidate_fpr > reference_fpr + float(tolerance):
        accepted = False
        reason = "valid_false_positive_rate_increased"
    else:
        accepted = True
        reason = "accuracy_not_lower_and_fpr_not_higher"
    return {
        "accepted": accepted,
        "reason": reason,
        "reference_candidate": str(reference.get("candidate")),
        "hard_negative_candidate": str(candidate.get("candidate")),
        "selected_candidate": str(
            candidate.get("candidate") if accepted else reference.get("candidate")
        ),
        "valid_accuracy_delta": candidate_accuracy - reference_accuracy,
        "valid_fp_rate_delta": candidate_fpr - reference_fpr,
    }


def _candidate_record(name, model, metrics):
    cm = metrics.get("confusion_matrix") or {}
    tn = int(cm.get("TN", 0))
    fp = int(cm.get("FP", 0))
    return {
        "candidate": str(name),
        "valid_accuracy": float(metrics.get("accuracy", 0.0)),
        "valid_fp_rate": float(fp / max(tn + fp, 1)),
        "valid_recall": float(metrics.get("recall", 0.0)),
        "total_nodes": int(count_xgb_nodes(model)),
        "cv_accuracy_std": 0.0,
        "finite_predictions": True,
    }


def _atomic_json_dump(payload, path):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

MODEL_SEARCH_PARAM_KEYS = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
    "reg_alpha",
]

DEFAULT_MODEL_SEARCH_SPACE = {
    "n_estimators": [20, 25, 30, 35, 40, 45, 50, 55, 60],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20],
    "min_child_weight": [10, 15, 20, 25, 30, 40, 50],
    "reg_lambda": [5, 8, 10, 12, 16, 20, 30],
    "reg_alpha": [0, 0.5, 1, 1.5, 2, 3],
    "subsample": [0.70, 0.75, 0.80, 0.85, 0.90],
    "colsample_bytree": [0.70, 0.75, 0.80, 0.85, 0.90],
}


def _model_search_default_csv(name):
    return ",".join(str(v) for v in DEFAULT_MODEL_SEARCH_SPACE[name])


def get_inner_n_jobs(default=1):
    """Thread count for XGBoost training; keep conservative on shared servers."""
    try:
        return max(1, int(os.environ.get("WL_INNER_N_JOBS", default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_model_search_workers(n_workers=None, n_items=None, cap=4):
    """Resolve bounded outer-search concurrency while preserving a serial escape hatch."""
    if _env_flag("WL_FORCE_SERIAL"):
        return 1
    if n_workers is None:
        n_workers = 1
    try:
        resolved = max(1, min(int(n_workers), int(cap)))
    except (TypeError, ValueError):
        resolved = 1
    if n_items is not None:
        resolved = min(resolved, max(1, int(n_items)))
    return resolved


def ordered_thread_map(fn, items, n_workers=1):
    """Evaluate independent jobs concurrently and return results in input order."""
    items = list(items)
    workers = resolve_model_search_workers(n_workers, n_items=len(items))
    if workers == 1 or len(items) <= 1:
        return [fn(item) for item in items]

    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def group_split_arrays(group_df):
    """Return sklearn-safe arrays even when pandas stores strings as pyarrow arrays."""
    group_names = group_df["sample_name"].astype("object").to_numpy()
    targets = group_df["target"].to_numpy()
    stratify = targets if group_df["target"].value_counts().min() >= 2 else None
    return group_names, stratify


def split_valid_for_calibration_threshold(df_valid, threshold_fraction=0.5, random_state=42):
    """Split valid by sample group so calibration and threshold selection differ."""
    if "sample_name" not in df_valid.columns or "target" not in df_valid.columns:
        return df_valid.copy(), df_valid.copy(), {
            "fallback": True,
            "reason": "missing_group_or_target",
            "calibration_groups": None,
            "threshold_groups": None,
        }

    group_df = (
        df_valid[["sample_name", "target"]]
        .drop_duplicates("sample_name")
        .reset_index(drop=True)
    )
    if len(group_df) < 4 or group_df["target"].nunique() < 2:
        return df_valid.copy(), df_valid.copy(), {
            "fallback": True,
            "reason": "insufficient_valid_groups",
            "n_groups": int(len(group_df)),
            "calibration_groups": None,
            "threshold_groups": None,
        }

    test_size = float(max(0.2, min(0.8, threshold_fraction)))
    group_names, stratify = group_split_arrays(group_df)
    try:
        calib_groups, threshold_groups = train_test_split(
            group_names,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        calib_groups, threshold_groups = train_test_split(
            group_names,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    calib_set = set(calib_groups)
    threshold_set = set(threshold_groups)
    df_calib = df_valid[df_valid["sample_name"].isin(calib_set)].copy()
    df_threshold = df_valid[df_valid["sample_name"].isin(threshold_set)].copy()
    if df_calib.empty or df_threshold.empty:
        return df_valid.copy(), df_valid.copy(), {
            "fallback": True,
            "reason": "empty_split",
            "calibration_groups": None,
            "threshold_groups": None,
        }

    return df_calib, df_threshold, {
        "fallback": False,
        "reason": "group_split",
        "threshold_fraction": test_size,
        "random_state": int(random_state),
        "n_calibration_groups": int(len(calib_set)),
        "n_threshold_groups": int(len(threshold_set)),
        "calibration_groups": sorted(map(str, calib_set)),
        "threshold_groups": sorted(map(str, threshold_set)),
    }


def split_calibration_for_model_search(df_calib_pool, search_fraction=0.5, random_state=42):
    if "sample_name" not in df_calib_pool.columns or "target" not in df_calib_pool.columns:
        return df_calib_pool.copy(), df_calib_pool.copy(), {
            "fallback": True,
            "reason": "missing_group_or_target",
            "model_selection_groups": None,
            "calibration_groups": None,
        }

    group_df = (
        df_calib_pool[["sample_name", "target"]]
        .drop_duplicates("sample_name")
        .reset_index(drop=True)
    )
    if len(group_df) < 4 or group_df["target"].nunique() < 2:
        return df_calib_pool.copy(), df_calib_pool.copy(), {
            "fallback": True,
            "reason": "insufficient_valid_groups",
            "n_groups": int(len(group_df)),
            "model_selection_groups": None,
            "calibration_groups": None,
        }

    test_size = float(max(0.2, min(0.8, search_fraction)))
    group_names, stratify = group_split_arrays(group_df)
    try:
        calib_groups, search_groups = train_test_split(
            group_names,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        calib_groups, search_groups = train_test_split(
            group_names,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    calib_set = set(calib_groups)
    search_set = set(search_groups)
    df_model_select = df_calib_pool[df_calib_pool["sample_name"].isin(search_set)].copy()
    df_calib = df_calib_pool[df_calib_pool["sample_name"].isin(calib_set)].copy()
    if df_model_select.empty or df_calib.empty:
        return df_calib_pool.copy(), df_calib_pool.copy(), {
            "fallback": True,
            "reason": "empty_split",
            "model_selection_groups": None,
            "calibration_groups": None,
        }

    return df_model_select, df_calib, {
        "fallback": False,
        "reason": "group_split",
        "search_fraction": test_size,
        "random_state": int(random_state),
        "n_model_selection_groups": int(len(search_set)),
        "n_calibration_groups": int(len(calib_set)),
        "model_selection_groups": sorted(map(str, search_set)),
        "calibration_groups": sorted(map(str, calib_set)),
    }


# =========================================================
# 异常值Clipping
# =========================================================

def prepare_valid_calibration_threshold_data(df_valid, selected_features, fill_values,
                                             threshold_fraction=0.5, random_state=42):
    """Prepare disjoint valid splits for probability calibration and threshold locking."""
    df_calib, df_threshold, split_meta = split_valid_for_calibration_threshold(
        df_valid,
        threshold_fraction=threshold_fraction,
        random_state=random_state,
    )
    X_calib, y_calib, _ = prepare_xy(df_calib, selected_features, fill_values=fill_values)
    X_threshold, y_threshold, _ = prepare_xy(df_threshold, selected_features, fill_values=fill_values)
    return {
        "df_calib": df_calib,
        "df_threshold": df_threshold,
        "X_calib": X_calib,
        "y_calib": y_calib,
        "X_threshold": X_threshold,
        "y_threshold": y_threshold,
        "split": split_meta,
    }


def learn_clip_bounds(df, columns, k=1.5):
    """Learn IQR clip bounds once from train rows for the requested numeric columns."""
    cols = [c for c in columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return {}

    sub = df[cols]
    q1 = sub.quantile(0.25)
    q3 = sub.quantile(0.75)
    iqr = q3 - q1
    valid_cols = [c for c in cols if iqr.get(c, 0.0) > 1e-10]
    if not valid_cols:
        return {}

    lower = q1[valid_cols] - k * iqr[valid_cols]
    upper = q3[valid_cols] + k * iqr[valid_cols]
    return {c: (float(lower[c]), float(upper[c])) for c in valid_cols}


def _log_clip_summary(clipped_cols, total_features=None, k=1.5, log_top_n=20):
    if not clipped_cols:
        return
    log_top_n = max(0, int(log_top_n))
    clipped_cols = sorted(
        clipped_cols,
        key=lambda item: max(
            abs(float(item.get("min_overshoot", 0.0))),
            abs(float(item.get("max_overshoot", 0.0))),
        ),
        reverse=True,
    )
    shown = clipped_cols[:log_top_n]
    logger.info(
        "异常值裁剪统计 (k=%s): %d/%d features clipped%s",
        k,
        len(clipped_cols),
        int(total_features if total_features is not None else len(clipped_cols)),
        f"; showing top {len(shown)}" if shown else "",
    )
    for item in shown:
        logger.info(
            "  %s: lower=%.4f, upper=%.4f, clipped_min=%s, clipped_max=%s",
            item["column"],
            item["lower"],
            item["upper"],
            item["clipped_min"],
            item["clipped_max"],
        )
    omitted = len(clipped_cols) - len(shown)
    if omitted > 0:
        logger.info("  ... %d more clipped features omitted", omitted)


def clip_outliers(df, columns, k=1.5, bounds=None, return_bounds=False, log_top_n=20):
    """
    基于 IQR 的异常值裁剪（向量化版）。

    参数:
        df: 输入 DataFrame
        columns: 需要裁剪的列名列表
        k: IQR 倍数，默认 1.5
        bounds: dict[col -> (lower, upper)] 预先计算好的裁剪边界。给定时跳过 IQR 估计直接应用。
        return_bounds: 是否同时返回 bounds 字典（仅 bounds=None 时有意义）。
        log_top_n: 最多打印多少个被裁剪特征；其余只汇总，避免搜参日志刷屏。

    返回:
        若 return_bounds=False：DataFrame
        若 return_bounds=True：  (DataFrame, bounds_dict)
    """
    df = df.copy()

    cols = [c for c in columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return (df, {}) if return_bounds else df

    if bounds is not None:
        # 直接应用预算好的边界（多用于 valid/test）
        valid_cols = [c for c in cols if c in bounds]
        if not valid_cols:
            return (df, {}) if return_bounds else df
        lower = pd.Series({c: bounds[c][0] for c in valid_cols})
        upper = pd.Series({c: bounds[c][1] for c in valid_cols})
        df[valid_cols] = df[valid_cols].clip(lower=lower, upper=upper, axis=1)
        subset_bounds = {c: (float(bounds[c][0]), float(bounds[c][1])) for c in valid_cols}
        return (df, subset_bounds) if return_bounds else df

    learned_bounds = learn_clip_bounds(df, cols, k=k)
    if not learned_bounds:
        return (df, {}) if return_bounds else df

    valid_cols = [c for c in cols if c in learned_bounds]
    lower = pd.Series({c: learned_bounds[c][0] for c in valid_cols})
    upper = pd.Series({c: learned_bounds[c][1] for c in valid_cols})

    before_min = df[valid_cols].min()
    before_max = df[valid_cols].max()

    df[valid_cols] = df[valid_cols].clip(lower=lower, upper=upper, axis=1)

    clipped_cols = []
    for c in valid_cols:
        bmin = before_min[c]
        bmax = before_max[c]
        lo = lower[c]
        hi = upper[c]
        if bmin < lo or bmax > hi:
            clipped_cols.append({
                "column": c,
                "lower": float(lo),
                "upper": float(hi),
                "clipped_min": bool(bmin < lo),
                "clipped_max": bool(bmax > hi),
                "min_overshoot": float(lo - bmin) if bmin < lo else 0.0,
                "max_overshoot": float(bmax - hi) if bmax > hi else 0.0,
            })

    _log_clip_summary(clipped_cols, total_features=len(valid_cols), k=k, log_top_n=log_top_n)

    return (df, learned_bounds) if return_bounds else df


def prepare_fill_values(df_train, selected_features):
    fill_values = {}
    for c in selected_features:
        x = df_train[c].replace([np.inf, -np.inf], np.nan)
        med = x.median()
        if not np.isfinite(med):
            med = 0.0
        fill_values[c] = float(med)
    return fill_values


def apply_fill(df, selected_features, fill_values):
    df = df.copy()
    for c in selected_features:
        if c not in df.columns:
            df[c] = fill_values.get(c, 0.0)
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(fill_values.get(c, 0.0))
    return df


def prepare_xy(df, selected_features, fill_values):
    df = apply_fill(df, selected_features, fill_values)

    X = df[selected_features].values.astype(float)
    y = df["target"].values.astype(int)

    return X, y, None


def eval_model(model, X, y, threshold=0.5):
    p = model.predict_proba(X)[:, 1]
    pred = (p >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y, p))
    except Exception:
        metrics["auc"] = None

    try:
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        metrics["confusion_matrix"] = {
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        }
    except Exception:
        metrics["confusion_matrix"] = None

    return metrics


def _fbeta(precision, recall, beta):
    """F-beta；beta<1 偏 precision，beta>1 偏 recall。"""
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def select_threshold_from_probs(y_true, probs, objective="f1", beta=0.5, min_precision=None):
    """Select a window threshold from probabilities using the same policy as s05."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    best = None
    best_key = None
    if objective == "precision_constrained" and min_precision is None:
        min_precision = 0.95
    for th in np.linspace(0.05, 0.95, 181):
        pred = (probs >= th).astype(int)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        accuracy = float(accuracy_score(y_true, pred))
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fp_rate = float(fp) / float(max(tn + fp, 1))
        if objective == "accuracy":
            score = accuracy
            key = (accuracy, -fp_rate, f1, precision, recall)
        elif objective == "precision":
            score = precision
            key = (score, accuracy, recall, -fp_rate)
        elif objective == "recall":
            score = recall
            key = (score, accuracy, -fp_rate, precision)
        elif objective == "fbeta":
            score = _fbeta(precision, recall, beta)
            key = (score, accuracy, -fp_rate, precision, recall)
        elif objective == "precision_constrained":
            score = recall if precision >= float(min_precision) else -1.0 + precision
            key = (score, accuracy, -fp_rate, precision, recall)
        else:
            score = f1
            key = (score, accuracy, -fp_rate, precision, recall)
        item = {
            "threshold": float(th),
            "score": float(score),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta": float(_fbeta(precision, recall, beta)),
            "fp_rate": fp_rate,
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        }
        if best_key is None or key > best_key:
            best = item
            best_key = key
    if best is not None:
        best["objective"] = objective
        if objective == "fbeta":
            best["beta"] = float(beta)
        if objective == "precision_constrained":
            best["min_precision"] = float(min_precision)
    return best


def search_threshold_by_valid(model, X_valid, y_valid, objective="f1",
                               beta=0.5, min_precision=None):
    """
    在 valid 上搜窗口阈值。

    objective:
      - "accuracy"           : 默认，最大化窗口准确率
      - "f1"                 : F1
      - "precision"          : 仅 precision
      - "recall"             : 仅 recall
      - "fbeta"              : F-beta (beta=0.5 偏 precision)
      - "precision_constrained" : 在 precision >= min_precision 约束下最大化 recall

    默认优先单窗口 accuracy；FP 风险专项优化时再显式切到
    fbeta(beta=0.5) 或 precision_constrained。
    """
    probs = model.predict_proba(X_valid)[:, 1]
    if objective == "accuracy":
        return select_threshold_from_probs(
            y_valid, probs, objective=objective, beta=beta, min_precision=min_precision
        )
    best = None
    best_score = -np.inf

    for th in np.linspace(0.05, 0.95, 181):
        pred = (probs >= th).astype(int)
        precision = float(precision_score(y_valid, pred, zero_division=0))
        recall = float(recall_score(y_valid, pred, zero_division=0))
        f1 = float(f1_score(y_valid, pred, zero_division=0))

        if objective == "f1":
            score = f1
        elif objective == "recall":
            score = recall
        elif objective == "precision":
            score = precision
        elif objective == "fbeta":
            score = _fbeta(precision, recall, beta)
        elif objective == "precision_constrained":
            if min_precision is None:
                min_precision = 0.95
            # 满足约束时按 recall 排序；不满足时给极小分但记录最差精度补偿值用于 fallback
            score = recall if precision >= min_precision else -1.0 + precision
        else:
            score = f1

        item = {
            "threshold": float(th),
            "score": float(score),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta": float(_fbeta(precision, recall, beta)),
        }

        if score > best_score:
            best = item
            best_score = score

    if best is not None:
        best["objective"] = objective
        if objective == "fbeta":
            best["beta"] = float(beta)
        if objective == "precision_constrained":
            best["min_precision"] = float(min_precision if min_precision is not None else 0.95)

    return best


def parse_model_search_values(raw, cast, name):
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = cast(part)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains invalid value {part!r}") from exc
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def build_default_xgb_params(scale_pos_weight=1.0):
    params = dict(DEFAULT_XGB_PARAMS)
    params["scale_pos_weight"] = float(scale_pos_weight)
    return params


def build_model_search_axes(args):
    return {
        "n_estimators": parse_model_search_values(
            args.model_search_n_estimators, int, "model_search_n_estimators"),
        "max_depth": parse_model_search_values(
            args.model_search_max_depth, int, "model_search_max_depth"),
        "learning_rate": parse_model_search_values(
            args.model_search_learning_rate, float, "model_search_learning_rate"),
        "subsample": parse_model_search_values(
            args.model_search_subsample, float, "model_search_subsample"),
        "colsample_bytree": parse_model_search_values(
            args.model_search_colsample_bytree, float, "model_search_colsample_bytree"),
        "min_child_weight": parse_model_search_values(
            args.model_search_min_child_weight, int, "model_search_min_child_weight"),
        "reg_lambda": parse_model_search_values(
            args.model_search_reg_lambda, float, "model_search_reg_lambda"),
        "reg_alpha": parse_model_search_values(
            args.model_search_reg_alpha, float, "model_search_reg_alpha"),
    }


def _model_search_combo_count(axes):
    total = 1
    for values in axes.values():
        total *= len(values)
    return int(total)


def _params_identity(params):
    ident = []
    for key in MODEL_SEARCH_PARAM_KEYS + ["scale_pos_weight"]:
        value = params.get(key)
        if isinstance(value, float):
            ident.append((key, round(float(value), 12)))
        else:
            ident.append((key, value))
    return tuple(ident)


def _params_from_combo_index(axes, index, scale_pos_weight=1.0):
    params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
    keys = list(axes.keys())
    idx = int(index)
    chosen = {}
    for key in reversed(keys):
        values = axes[key]
        chosen[key] = values[idx % len(values)]
        idx //= len(values)
    params.update({key: chosen[key] for key in keys})
    return params


def _dedupe_model_search_grid(grid):
    seen = set()
    out = []
    for params in grid:
        ident = _params_identity(params)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(params)
    return out


def _ensure_default_params_in_grid(grid, scale_pos_weight=1.0, max_candidates=0):
    if any(is_default_xgb_params(params) for params in grid):
        return _dedupe_model_search_grid(grid)
    default_params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
    cap = int(max_candidates or 0)
    if cap > 0 and len(grid) >= cap:
        grid = list(grid[:max(cap - 1, 0)])
    else:
        grid = list(grid)
    grid.append(default_params)
    return _dedupe_model_search_grid(grid)


def build_model_search_grid(args, scale_pos_weight=1.0):
    axes = build_model_search_axes(args)
    max_candidates = int(getattr(args, "model_search_max_candidates", 0) or 0)
    random_state = int(getattr(args, "model_search_random_state", 42))
    total_combinations = _model_search_combo_count(axes)
    if max_candidates > 0 and total_combinations > max_candidates:
        random_state_class = getattr(np.random, "RandomState")
        rng = random_state_class(random_state)
        combo_indices = sorted(
            rng.choice(total_combinations, size=max_candidates, replace=False).tolist()
        )
        grid = [
            _params_from_combo_index(axes, idx, scale_pos_weight=scale_pos_weight)
            for idx in combo_indices
        ]
    else:
        grid = []
        for values in product(*(axes[k] for k in axes.keys())):
            params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
            params.update(dict(zip(axes.keys(), values)))
            grid.append(params)
    return _ensure_default_params_in_grid(
        grid,
        scale_pos_weight=scale_pos_weight,
        max_candidates=max_candidates,
    )


def is_default_xgb_params(params):
    default = build_default_xgb_params(scale_pos_weight=params.get("scale_pos_weight", 1.0))
    keys = [
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha",
        "objective", "eval_metric", "random_state", "scale_pos_weight",
    ]
    for key in keys:
        if key not in params:
            return False
        left = params[key]
        right = default[key]
        if isinstance(right, float):
            if not np.isclose(float(left), float(right)):
                return False
        else:
            if left != right:
                return False
    return True


def count_xgb_nodes(model):
    booster = model.get_booster()
    tree_dumps = booster.get_dump()
    return int(sum(
        len([line for line in tree.split("\n")
             if line.strip() and not line.strip().startswith("booster")])
        for tree in tree_dumps
    ))


def _window_fp_rate_from_metrics(metrics):
    cm = metrics.get("confusion_matrix") or {}
    tn = int(cm.get("TN", 0))
    fp = int(cm.get("FP", 0))
    n_neg = max(tn + fp, 1)
    return float(fp) / float(n_neg)


def score_model_search_candidate(metrics, total_nodes, max_model_nodes=400,
                                 fp_cost=2.0, size_cost=0.1):
    total_nodes = int(total_nodes)
    max_model_nodes = int(max_model_nodes)
    if max_model_nodes > 0 and total_nodes > max_model_nodes:
        return {
            "eligible": False,
            "score": float("-inf"),
            "fp_rate": _window_fp_rate_from_metrics(metrics),
            "size_ratio": float(total_nodes) / float(max(max_model_nodes, 1)),
        }
    fp_rate = _window_fp_rate_from_metrics(metrics)
    size_ratio = float(total_nodes) / float(max(max_model_nodes, 1)) if max_model_nodes > 0 else 0.0
    score = (
        float(metrics.get("accuracy", 0.0))
        - float(fp_cost) * fp_rate
        - float(size_cost) * size_ratio
    )
    return {
        "eligible": True,
        "score": float(score),
        "fp_rate": float(fp_rate),
        "size_ratio": float(size_ratio),
    }


def evaluate_accuracy_first_threshold(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    best = None
    best_key = None
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (probs >= threshold).astype(int)
        accuracy = float(accuracy_score(y, pred))
        precision = float(precision_score(y, pred, zero_division=0))
        recall = float(recall_score(y, pred, zero_division=0))
        f1 = float(f1_score(y, pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        fp_rate = float(fp) / float(max(tn + fp, 1))
        item = {
            "threshold": float(threshold),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_rate": fp_rate,
            "confusion_matrix": {
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
            },
        }
        key = (accuracy, -fp_rate, f1, precision, recall)
        if best_key is None or key > best_key:
            best = item
            best_key = key
    try:
        best["auc"] = float(roc_auc_score(y, probs))
    except Exception:
        best["auc"] = None
    return best


def choose_accuracy_first_model_search_record(records, accuracy_tolerance=0.0):
    eligible = [r for r in records if r.get("eligible")]
    if not eligible:
        return None
    tolerance = max(0.0, float(accuracy_tolerance))
    best_accuracy = max(float(r.get("selection_accuracy", 0.0)) for r in eligible)
    candidates = [
        r for r in eligible
        if float(r.get("selection_accuracy", 0.0)) >= best_accuracy - tolerance
    ]
    chosen = min(
        candidates,
        key=lambda r: (
            int(r.get("total_nodes", 0)),
            float(r.get("selection_fp_rate", 1.0)),
            -float(r.get("selection_accuracy", 0.0)),
            not bool(r.get("is_default_params", False)),
            int(r.get("rank_input_order", 0)),
        ),
    )
    chosen["chosen_reason"] = (
        "within_accuracy_tolerance_smallest_model"
        if float(chosen.get("selection_accuracy", 0.0)) < best_accuracy
        else "max_accuracy"
    )
    return chosen


def build_repeated_group_cv_splits(y, groups=None, n_folds=3, n_repeats=2, random_state=42):
    y = np.asarray(y, dtype=int)
    n_samples = len(y)
    if n_samples < 2:
        idx = np.arange(n_samples)
        return [(idx, idx)], {
            "fallback": True,
            "reason": "insufficient_rows",
            "n_splits": 1,
            "n_repeats": 1,
        }

    n_folds = max(2, int(n_folds))
    n_repeats = max(1, int(n_repeats))
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
    if groups is not None and len(groups) == n_samples:
        group_df = pd.DataFrame({"group": groups, "target": y}).drop_duplicates("group")
        group_counts = group_df["target"].value_counts()
        if len(group_df) >= 2 and group_df["target"].nunique() >= 2 and group_counts.min() >= 2:
            effective_folds = min(n_folds, int(group_counts.min()), int(len(group_df)))
            splits = []
            for repeat in range(n_repeats):
                cv = StratifiedGroupKFold(
                    n_splits=effective_folds,
                    shuffle=True,
                    random_state=int(random_state) + repeat,
                )
                splits.extend(list(cv.split(np.zeros(n_samples), y, groups)))
            return splits, {
                "fallback": False,
                "reason": "stratified_group_kfold",
                "n_splits": int(effective_folds),
                "n_repeats": int(n_repeats),
                "n_groups": int(len(group_df)),
            }

    class_counts = pd.Series(y).value_counts()
    if y.size >= 4 and len(class_counts) >= 2 and int(class_counts.min()) >= 2:
        effective_folds = min(n_folds, int(class_counts.min()), n_samples)
        splits = []
        for repeat in range(n_repeats):
            cv = StratifiedKFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=int(random_state) + repeat,
            )
            splits.extend(list(cv.split(np.zeros(n_samples), y)))
        return splits, {
            "fallback": True,
            "reason": "row_stratified_kfold",
            "n_splits": int(effective_folds),
            "n_repeats": int(n_repeats),
        }

    effective_folds = min(n_folds, n_samples)
    cv = KFold(n_splits=effective_folds, shuffle=True, random_state=int(random_state))
    return list(cv.split(np.zeros(n_samples))), {
        "fallback": True,
        "reason": "row_kfold",
        "n_splits": int(effective_folds),
        "n_repeats": 1,
    }


def build_hard_negative_oof_splits(y, groups=None, n_folds=3, random_state=42):
    """Build train-only OOF splits for hard-negative mining without sample-group leakage."""
    y = np.asarray(y, dtype=int)
    n_samples = len(y)
    if n_samples < 2:
        idx = np.arange(n_samples)
        return [(idx, idx)], {
            "source_split": "train_only",
            "fallback": True,
            "reason": "insufficient_rows",
            "n_splits": 1,
            "n_groups": 0,
        }

    n_folds = max(2, int(n_folds))
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
    if groups is not None and len(groups) == n_samples:
        unique_groups = pd.unique(groups)
        if len(unique_groups) >= 2:
            group_df = pd.DataFrame({"group": groups, "target": y}).drop_duplicates("group")
            group_counts = group_df["target"].value_counts()
            if group_df["target"].nunique() >= 2 and int(group_counts.min()) >= 2:
                effective_folds = min(n_folds, int(group_counts.min()), int(len(group_df)))
                cv = StratifiedGroupKFold(
                    n_splits=effective_folds,
                    shuffle=True,
                    random_state=int(random_state),
                )
                return list(cv.split(np.zeros(n_samples), y, groups)), {
                    "source_split": "train_only",
                    "fallback": False,
                    "reason": "stratified_group_kfold",
                    "n_splits": int(effective_folds),
                    "n_groups": int(len(group_df)),
                }
            effective_folds = min(n_folds, int(len(unique_groups)))
            cv = GroupKFold(n_splits=effective_folds)
            return list(cv.split(np.zeros(n_samples), y, groups)), {
                "source_split": "train_only",
                "fallback": False,
                "reason": "group_kfold",
                "n_splits": int(effective_folds),
                "n_groups": int(len(unique_groups)),
            }

    class_counts = pd.Series(y).value_counts()
    if y.size >= 4 and len(class_counts) >= 2 and int(class_counts.min()) >= 2:
        effective_folds = min(n_folds, int(class_counts.min()), n_samples)
        cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=int(random_state))
        return list(cv.split(np.zeros(n_samples), y)), {
            "source_split": "train_only",
            "fallback": True,
            "reason": "row_stratified_kfold",
            "n_splits": int(effective_folds),
            "n_groups": 0,
        }

    effective_folds = min(n_folds, n_samples)
    cv = KFold(n_splits=effective_folds, shuffle=True, random_state=int(random_state))
    return list(cv.split(np.zeros(n_samples))), {
        "source_split": "train_only",
        "fallback": True,
        "reason": "row_kfold",
        "n_splits": int(effective_folds),
        "n_groups": 0,
    }


def build_hard_negative_training_weights_from_oof(
        df_train, oof_probs, min_probability=0.5, top_percentile=0.10,
        hard_negative_weight=3.0):
    """Select train-only negative windows with high OOF probabilities and build sample weights."""
    n_rows = len(df_train)
    probs = np.asarray(oof_probs, dtype=float)
    if len(probs) != n_rows:
        raise ValueError("oof_probs length must match df_train rows")

    targets = df_train["target"].to_numpy(dtype=int)
    neg_mask = targets == 0
    finite_neg = neg_mask & np.isfinite(probs)
    weights = np.ones(n_rows, dtype=float)
    context_cols = [
        "negative_type", "scene_type", "subject_type", "record",
        "device_id", "session_id", "subject_id",
    ]
    report_cols = [
        "sample_name", "h5_file", "window_index", "target", "mode", "quality_bin",
        *[c for c in context_cols if c in df_train.columns],
        "prob_oof", "selected_reason",
    ]

    if not np.any(finite_neg):
        empty = pd.DataFrame(columns=report_cols)
        return weights, empty, {
            "enabled": True,
            "source_split": "train_only",
            "n_train_rows": int(n_rows),
            "n_negative_rows": int(np.sum(neg_mask)),
            "n_hard_negatives": 0,
            "object_worn_hard_negatives": 0,
            "object_worn_fraction": 0.0,
            "hard_negative_weight": float(hard_negative_weight),
            "min_probability": float(min_probability),
            "top_percentile": float(top_percentile),
            "probability_cutoff": None,
        }

    top_percentile = min(max(float(top_percentile), 0.0), 1.0)
    if top_percentile > 0:
        percentile_cutoff = float(np.quantile(probs[finite_neg], 1.0 - top_percentile))
    else:
        percentile_cutoff = float("inf")
    min_probability = float(min_probability)
    cutoff = min(min_probability, percentile_cutoff)
    selected_mask = finite_neg & ((probs >= min_probability) | (probs >= percentile_cutoff))
    weights[selected_mask] = float(hard_negative_weight)

    report = df_train.loc[selected_mask].copy()
    report["prob_oof"] = probs[selected_mask]
    reasons = []
    for p in report["prob_oof"].astype(float).to_numpy():
        hit_min = p >= min_probability
        hit_top = p >= percentile_cutoff
        if hit_min and hit_top:
            reasons.append("probability_and_top_percentile")
        elif hit_min:
            reasons.append("probability")
        else:
            reasons.append("top_percentile")
    report["selected_reason"] = reasons
    for col in report_cols:
        if col not in report.columns:
            report[col] = None
    report = report[report_cols].sort_values("prob_oof", ascending=False).reset_index(drop=True)
    context_text = pd.Series("", index=report.index, dtype=object)
    for col in ["negative_type", "scene_type", "subject_type", "sample_name", "h5_file"]:
        if col in report.columns:
            context_text = context_text + " " + report[col].fillna("").astype(str).str.lower()
    object_mask = context_text.str.contains("object_worn|object-worn|non_human|non-human|reflective|物体|非人体", regex=True)
    object_count = int(object_mask.sum())

    return weights, report, {
        "enabled": True,
        "source_split": "train_only",
        "n_train_rows": int(n_rows),
        "n_negative_rows": int(np.sum(neg_mask)),
        "n_hard_negatives": int(np.sum(selected_mask)),
        "object_worn_hard_negatives": object_count,
        "object_worn_fraction": float(object_count / max(1, int(np.sum(selected_mask)))),
        "hard_negative_weight": float(hard_negative_weight),
        "min_probability": float(min_probability),
        "top_percentile": float(top_percentile),
        "probability_cutoff": float(cutoff) if np.isfinite(cutoff) else None,
        "percentile_cutoff": float(percentile_cutoff) if np.isfinite(percentile_cutoff) else None,
    }


def mine_hard_negative_training_weights(
        df_train, X_train, y_train, groups, params, min_probability=0.5,
        top_percentile=0.10, hard_negative_weight=3.0, n_folds=3,
        random_state=42):
    """Run train-only group-aware OOF mining and return final sample weights plus audit report."""
    splits, split_meta = build_hard_negative_oof_splits(
        y_train, groups=groups, n_folds=n_folds, random_state=random_state)
    oof_sum = np.zeros(len(y_train), dtype=float)
    oof_count = np.zeros(len(y_train), dtype=int)
    for train_idx, valid_idx in splits:
        if len(train_idx) == 0 or len(valid_idx) == 0 or len(np.unique(y_train[train_idx])) < 2:
            continue
        fold_model = train_xgb_with_params(params, X_train[train_idx], y_train[train_idx])
        oof_sum[valid_idx] += fold_model.predict_proba(X_train[valid_idx])[:, 1]
        oof_count[valid_idx] += 1
    oof_probs = np.full(len(y_train), np.nan, dtype=float)
    covered = oof_count > 0
    oof_probs[covered] = oof_sum[covered] / oof_count[covered]
    weights, report, summary = build_hard_negative_training_weights_from_oof(
        df_train,
        oof_probs,
        min_probability=min_probability,
        top_percentile=top_percentile,
        hard_negative_weight=hard_negative_weight,
    )
    summary["oof_split"] = split_meta
    summary["oof_covered_rows"] = int(np.sum(covered))
    return weights, report, summary


def _mean_or_none(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not values:
        return None
    return float(np.mean(values))


def _std_or_none(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not values:
        return None
    return float(np.std(values))


def summarize_cv_metrics(fold_metrics):
    return {
        "mean_cv_accuracy": _mean_or_none([m.get("accuracy") for m in fold_metrics]),
        "std_cv_accuracy": _std_or_none([m.get("accuracy") for m in fold_metrics]),
        "mean_cv_fp_rate": _mean_or_none([m.get("fp_rate") for m in fold_metrics]),
        "mean_cv_precision": _mean_or_none([m.get("precision") for m in fold_metrics]),
        "mean_cv_recall": _mean_or_none([m.get("recall") for m in fold_metrics]),
        "cv_folds_completed": int(len(fold_metrics)),
    }


def choose_cv_model_search_record(records, accuracy_tolerance=0.0):
    eligible = [r for r in records if r.get("eligible")]
    if not eligible:
        return None
    tolerance = max(0.0, float(accuracy_tolerance))
    default_records = [r for r in eligible if r.get("is_default_params")]
    default_record = None
    if default_records:
        default_record = max(default_records, key=lambda r: float(r.get("mean_cv_accuracy") or 0.0))
        default_acc = float(default_record.get("mean_cv_accuracy") or 0.0)
        non_default = [r for r in eligible if not r.get("is_default_params")]
        best_non_default_acc = max(
            [float(r.get("mean_cv_accuracy") or 0.0) for r in non_default],
            default=float("-inf"),
        )
        if best_non_default_acc <= default_acc + tolerance:
            default_record["beats_default_params"] = False
            default_record["chosen_reason"] = "default_params_baseline_not_beaten"
            for r in records:
                if r is not default_record:
                    r["beats_default_params"] = bool(
                        float(r.get("mean_cv_accuracy") or 0.0) > default_acc + tolerance
                    )
            return default_record

    best_accuracy = max(float(r.get("mean_cv_accuracy") or 0.0) for r in eligible)
    candidates = [
        r for r in eligible
        if float(r.get("mean_cv_accuracy") or 0.0) >= best_accuracy - tolerance
    ]
    chosen = min(
        candidates,
        key=lambda r: (
            float(r.get("std_cv_accuracy") if r.get("std_cv_accuracy") is not None else 1.0),
            float(r.get("mean_cv_fp_rate") if r.get("mean_cv_fp_rate") is not None else 1.0),
            int(r.get("final_total_nodes", r.get("total_nodes", 0))),
            not bool(r.get("is_default_params", False)),
            -float(r.get("mean_cv_accuracy") or 0.0),
            int(r.get("rank_input_order", 0)),
        ),
    )
    default_acc = float(default_record.get("mean_cv_accuracy") or 0.0) if default_record else float("-inf")
    for r in records:
        r["beats_default_params"] = bool(float(r.get("mean_cv_accuracy") or 0.0) > default_acc + tolerance)
    chosen["chosen_reason"] = "best_cv_accuracy"
    return chosen


def build_model_search_result_rows(model_search_records):
    rows = []
    for r in model_search_records:
        row = {
            "rank_input_order": int(r["rank_input_order"]),
            "eligible": bool(r["eligible"]),
            "score": _json_safe_float(r["score"]),
            "fp_rate": _json_safe_float(r["fp_rate"]),
            "size_ratio": _json_safe_float(r["size_ratio"]),
            "total_nodes": int(r["total_nodes"]),
            "final_total_nodes": int(r.get("final_total_nodes", r.get("total_nodes", 0))),
            "avg_nodes_per_tree": _json_safe_float(r["avg_nodes_per_tree"]),
            "selection_threshold": _json_safe_float(r.get("selection_threshold", 0.5)),
            "selection_accuracy": _json_safe_float(r.get("selection_accuracy", 0.0)),
            "selection_fp_rate": _json_safe_float(r.get("selection_fp_rate", r.get("fp_rate", 0.0))),
            "mean_cv_accuracy": _json_safe_float(r.get("mean_cv_accuracy", 0.0)),
            "std_cv_accuracy": _json_safe_float(r.get("std_cv_accuracy", 0.0)),
            "mean_cv_fp_rate": _json_safe_float(r.get("mean_cv_fp_rate", 0.0)),
            "mean_cv_precision": _json_safe_float(r.get("mean_cv_precision", 0.0)),
            "mean_cv_recall": _json_safe_float(r.get("mean_cv_recall", 0.0)),
            "cv_folds_completed": int(r.get("cv_folds_completed", 0)),
            "is_default_params": bool(r.get("is_default_params", False)),
            "beats_default_params": bool(r.get("beats_default_params", False)),
            "chosen_reason": str(r.get("chosen_reason", "")),
            "feature_count": int(r.get("feature_count", r.get("n_features", 0))),
            "n_features": int(r.get("n_features", r.get("feature_count", 0))),
            "feature_set": str((r.get("deployment_feature_cost_summary") or {}).get("feature_set", "")),
            "deployment_fft_source_count": int((r.get("deployment_feature_cost_summary") or {}).get("fft_source_count", 0)),
            "deployment_forbidden_selected_count": int((r.get("deployment_feature_cost_summary") or {}).get("forbidden_selected_count", 0)),
        }
        for name, value in (r.get("metrics") or {}).items():
            if name == "confusion_matrix":
                cm = value or {}
                for cm_key in ["TN", "FP", "FN", "TP"]:
                    row[f"cm_{cm_key}"] = int(cm.get(cm_key, 0))
            else:
                row[f"metric_{name}"] = _json_safe_float(value) if value is not None else None
        for name, value in (r.get("selection_metrics") or {}).items():
            if name == "confusion_matrix":
                cm = value or {}
                for cm_key in ["TN", "FP", "FN", "TP"]:
                    row[f"selection_cm_{cm_key}"] = int(cm.get(cm_key, 0))
            elif name not in {"threshold", "accuracy", "fp_rate"}:
                row[f"selection_metric_{name}"] = _json_safe_float(value) if value is not None else None
        for name, value in (r.get("params") or {}).items():
            row[f"param_{name}"] = value
        rows.append(row)
    return rows


def summarize_model_search_stability(results_df, accuracy_close_margin=0.002):
    """Summarize whether top model-search candidates are clearly separated."""
    if results_df is None or len(results_df) == 0:
        return {"available": False, "reason": "empty_results"}
    df = results_df.copy()
    if "eligible" in df.columns:
        df = df[df["eligible"].astype(bool)]
    if df.empty or "mean_cv_accuracy" not in df.columns:
        return {"available": False, "reason": "missing_cv_accuracy"}
    df["mean_cv_accuracy"] = pd.to_numeric(df["mean_cv_accuracy"], errors="coerce")
    df = df[df["mean_cv_accuracy"].notna()].copy()
    if df.empty:
        return {"available": False, "reason": "no_finite_cv_accuracy"}

    sort_cols = ["mean_cv_accuracy"]
    ascending = [False]
    if "std_cv_accuracy" in df.columns:
        sort_cols.append("std_cv_accuracy")
        ascending.append(True)
    if "mean_cv_fp_rate" in df.columns:
        sort_cols.append("mean_cv_fp_rate")
        ascending.append(True)
    if "final_total_nodes" in df.columns:
        sort_cols.append("final_total_nodes")
        ascending.append(True)
    df = df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

    best_acc = float(df.loc[0, "mean_cv_accuracy"])
    second_acc = float(df.loc[1, "mean_cv_accuracy"]) if len(df) > 1 else None
    top_margin = None if second_acc is None else float(best_acc - second_acc)
    close_mask = (best_acc - df["mean_cv_accuracy"].astype(float)) <= float(accuracy_close_margin)
    close_df = df[close_mask]

    default_rank = None
    if "is_default_params" in df.columns:
        default_hits = df.index[df["is_default_params"].astype(bool)].tolist()
        if default_hits:
            default_rank = int(default_hits[0] + 1)

    feature_counts = []
    if "feature_count" in close_df.columns:
        feature_counts = sorted({int(v) for v in pd.to_numeric(close_df["feature_count"], errors="coerce").dropna()})

    is_unstable = bool(
        len(close_df) > 1
        or (top_margin is not None and top_margin <= float(accuracy_close_margin))
        or (default_rank is not None and default_rank <= 3)
    )
    return {
        "available": True,
        "best_mean_cv_accuracy": best_acc,
        "second_mean_cv_accuracy": second_acc,
        "top_accuracy_margin": top_margin,
        "accuracy_close_margin": float(accuracy_close_margin),
        "close_top_candidate_count": int(len(close_df)),
        "close_top_feature_counts": feature_counts,
        "default_params_rank": default_rank,
        "is_unstable": is_unstable,
    }


def _json_safe_float(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _json_safe_model_search_record(record):
    out = dict(record)
    out["score"] = _json_safe_float(out["score"])
    out["fp_rate"] = _json_safe_float(out["fp_rate"])
    out["size_ratio"] = _json_safe_float(out["size_ratio"])
    out["avg_nodes_per_tree"] = _json_safe_float(out["avg_nodes_per_tree"])
    out["final_total_nodes"] = int(out.get("final_total_nodes", out.get("total_nodes", 0)))
    out["selection_threshold"] = _json_safe_float(out.get("selection_threshold", 0.5))
    out["selection_accuracy"] = _json_safe_float(out.get("selection_accuracy", 0.0))
    out["selection_fp_rate"] = _json_safe_float(out.get("selection_fp_rate", out["fp_rate"]))
    for key in [
        "mean_cv_accuracy",
        "std_cv_accuracy",
        "mean_cv_fp_rate",
        "mean_cv_precision",
        "mean_cv_recall",
    ]:
        out[key] = _json_safe_float(out.get(key))
    out["cv_folds_completed"] = int(out.get("cv_folds_completed", 0))
    out["is_default_params"] = bool(out.get("is_default_params", False))
    out["beats_default_params"] = bool(out.get("beats_default_params", False))
    out["chosen_reason"] = str(out.get("chosen_reason", ""))
    return out


def train_xgb_with_params(params, X_train, y_train, sample_weight=None):
    fit_params = dict(params)
    fit_params["n_jobs"] = get_inner_n_jobs()
    model = xgb.XGBClassifier(**fit_params)
    fit_kwargs = {"verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def _finite_float_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def score_fixed_params_with_train_cv(args, X_train, y_train, groups, params,
                                     sample_weight=None, total_nodes=None):
    """Score one fixed feature set with train-only CV, without touching valid/test."""
    cv_splits, cv_meta = build_repeated_group_cv_splits(
        y_train,
        groups=groups,
        n_folds=args.model_search_cv_folds,
        n_repeats=args.model_search_cv_repeats,
        random_state=args.model_search_random_state,
    )
    fold_metrics = []
    for train_idx, valid_idx in cv_splits:
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        sw_fold = sample_weight[train_idx] if sample_weight is not None else None
        fold_model = train_xgb_with_params(
            params,
            X_train[train_idx],
            y_train[train_idx],
            sample_weight=sw_fold,
        )
        fold_metrics.append(evaluate_accuracy_first_threshold(
            fold_model,
            X_train[valid_idx],
            y_train[valid_idx],
        ))
    cv_summary = summarize_cv_metrics(fold_metrics)
    if total_nodes is None:
        final_model = train_xgb_with_params(params, X_train, y_train, sample_weight=sample_weight)
        total_nodes = count_xgb_nodes(final_model)
    total_nodes = int(total_nodes)
    mean_accuracy = float(cv_summary.get("mean_cv_accuracy") or 0.0)
    mean_fp_rate = float(cv_summary.get("mean_cv_fp_rate") or 0.0)
    max_model_nodes = int(args.max_model_nodes)
    size_ratio = float(total_nodes) / float(max(max_model_nodes, 1)) if max_model_nodes > 0 else 0.0
    eligible = not (max_model_nodes > 0 and total_nodes > max_model_nodes)
    score = (
        mean_accuracy
        - float(args.model_search_fp_cost) * mean_fp_rate
        - float(args.model_search_size_cost) * size_ratio
    ) if eligible else float("-inf")
    return {
        "score": float(score),
        "selection_metric": "train_cv_fixed_params_score",
        "cv_summary": cv_summary,
        "cv_split": cv_meta,
        "total_nodes": int(total_nodes),
        "eligible": bool(eligible),
        "fp_rate": float(mean_fp_rate),
        "size_ratio": float(size_ratio),
    }


def _search_xgb_hyperparameters_single_split(args, X_train, y_train, X_select, y_select,
                                             scale_pos_weight=1.0, sample_weight=None):
    grid = build_model_search_grid(args, scale_pos_weight=scale_pos_weight)
    n_workers = resolve_model_search_workers(
        getattr(args, "model_search_n_workers", 1), n_items=len(grid))
    logger.info(
        "model_search enabled: evaluating %d XGBoost candidates with %d outer workers",
        len(grid), n_workers,
    )

    def evaluate_candidate(item):
        idx, params = item
        candidate = train_xgb_with_params(params, X_train, y_train, sample_weight=sample_weight)
        metrics = eval_model(candidate, X_select, y_select, threshold=0.5)
        selection_metrics = evaluate_accuracy_first_threshold(candidate, X_select, y_select)
        total_nodes = count_xgb_nodes(candidate)
        score = score_model_search_candidate(
            selection_metrics,
            total_nodes=total_nodes,
            max_model_nodes=args.max_model_nodes,
            fp_cost=args.model_search_fp_cost,
            size_cost=args.model_search_size_cost,
        )
        return {
            "rank_input_order": int(idx),
            "eligible": bool(score["eligible"]),
            "score": float(score["score"]),
            "fp_rate": float(score["fp_rate"]),
            "size_ratio": float(score["size_ratio"]),
            "total_nodes": int(total_nodes),
            "avg_nodes_per_tree": float(total_nodes) / float(max(int(params["n_estimators"]), 1)),
            "selection_threshold": float(selection_metrics["threshold"]),
            "selection_accuracy": float(selection_metrics["accuracy"]),
            "selection_fp_rate": float(selection_metrics["fp_rate"]),
            "selection_metrics": selection_metrics,
            "is_default_params": bool(is_default_xgb_params(params)),
            "chosen_reason": "",
            "metrics": metrics,
            "params": params,
        }

    records = ordered_thread_map(
        evaluate_candidate,
        list(enumerate(grid, 1)),
        n_workers=n_workers,
    )

    best = choose_accuracy_first_model_search_record(
        records,
        accuracy_tolerance=args.model_search_accuracy_tolerance,
    )
    if best is None:
        raise RuntimeError(
            f"model_search found no candidate under max_model_nodes={args.max_model_nodes}. "
            "Relax --max_model_nodes or shrink the search grid."
        )
    best_model = train_xgb_with_params(
        best["params"], X_train, y_train, sample_weight=sample_weight)

    records.sort(key=lambda r: (
        not r["eligible"],
        -float(r.get("selection_accuracy", 0.0)),
        int(r.get("total_nodes", 0)),
        float(r.get("selection_fp_rate", 1.0)),
    ))
    return best_model, {
        "enabled": True,
        "strategy": "single_split",
        "selection_data": "valid_model_selection_split",
        "max_model_nodes": int(args.max_model_nodes),
        "fp_cost": float(args.model_search_fp_cost),
        "size_cost": float(args.model_search_size_cost),
        "accuracy_tolerance": float(args.model_search_accuracy_tolerance),
        "selection_policy": "accuracy_first_size_second",
        "grid_size": int(len(grid)),
        "outer_workers": int(n_workers),
        "best": _json_safe_model_search_record(best),
        "top_candidates": [_json_safe_model_search_record(r) for r in records[:20]],
    }, records


def _search_xgb_hyperparameters_staged_group_cv(args, X_train, y_train, groups=None,
                                                scale_pos_weight=1.0, sample_weight=None):
    axes = build_model_search_axes(args)
    total_combinations = _model_search_combo_count(axes)
    grid = build_model_search_grid(args, scale_pos_weight=scale_pos_weight)
    cv_splits, cv_meta = build_repeated_group_cv_splits(
        y_train,
        groups=groups,
        n_folds=args.model_search_cv_folds,
        n_repeats=args.model_search_cv_repeats,
        random_state=args.model_search_random_state,
    )
    if not cv_splits:
        idx = np.arange(len(y_train))
        cv_splits = [(idx, idx)]
        cv_meta = {"fallback": True, "reason": "empty_cv_splits", "n_splits": 1, "n_repeats": 1}

    stage_train_idx, stage_valid_idx = cv_splits[0]
    n_workers = resolve_model_search_workers(
        getattr(args, "model_search_n_workers", 1), n_items=len(grid))
    logger.info(
        "model_search staged_group_cv: stage A evaluating %d sampled candidates "
        "(total_combinations=%d, outer_workers=%d)",
        len(grid),
        total_combinations,
        n_workers,
    )

    def evaluate_stage_a_candidate(item):
        idx, params = item
        sw_stage = sample_weight[stage_train_idx] if sample_weight is not None else None
        candidate = train_xgb_with_params(
            params, X_train[stage_train_idx], y_train[stage_train_idx], sample_weight=sw_stage)
        selection_metrics = evaluate_accuracy_first_threshold(
            candidate,
            X_train[stage_valid_idx],
            y_train[stage_valid_idx],
        )
        total_nodes = count_xgb_nodes(candidate)
        score = score_model_search_candidate(
            selection_metrics,
            total_nodes=total_nodes,
            max_model_nodes=args.max_model_nodes,
            fp_cost=args.model_search_fp_cost,
            size_cost=args.model_search_size_cost,
        )
        return {
            "rank_input_order": int(idx),
            "eligible": bool(score["eligible"]),
            "score": float(score["score"]),
            "fp_rate": float(score["fp_rate"]),
            "size_ratio": float(score["size_ratio"]),
            "total_nodes": int(total_nodes),
            "avg_nodes_per_tree": float(total_nodes) / float(max(int(params["n_estimators"]), 1)),
            "selection_threshold": float(selection_metrics["threshold"]),
            "selection_accuracy": float(selection_metrics["accuracy"]),
            "selection_fp_rate": float(selection_metrics["fp_rate"]),
            "selection_metrics": selection_metrics,
            "is_default_params": bool(is_default_xgb_params(params)),
            "chosen_reason": "",
            "metrics": selection_metrics,
            "params": params,
        }

    stage_records = ordered_thread_map(
        evaluate_stage_a_candidate,
        list(enumerate(grid, 1)),
        n_workers=n_workers,
    )

    stage_records.sort(key=lambda r: (
        not r["eligible"],
        -float(r.get("selection_accuracy", 0.0)),
        float(r.get("selection_fp_rate", 1.0)),
        int(r.get("total_nodes", 0)),
    ))
    stage2_top_k = max(1, int(args.model_search_stage2_top_k))
    stage2_params = [r["params"] for r in stage_records if r["eligible"]][:stage2_top_k]
    stage2_params = _ensure_default_params_in_grid(
        stage2_params,
        scale_pos_weight=scale_pos_weight,
        max_candidates=0,
    )
    logger.info(
        "model_search staged_group_cv: stage B CV evaluating %d candidates across %d folds",
        len(stage2_params),
        len(cv_splits),
    )

    fold_count = len(cv_splits)

    def evaluate_stage_b_candidate(item):
        idx, params = item
        fold_metrics = []
        for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits, 1):
            sw_fold = sample_weight[train_idx] if sample_weight is not None else None
            fold_model = train_xgb_with_params(
                params, X_train[train_idx], y_train[train_idx], sample_weight=sw_fold)
            fold_metrics.append(evaluate_accuracy_first_threshold(
                fold_model,
                X_train[valid_idx],
                y_train[valid_idx],
            ))
        cv_summary = summarize_cv_metrics(fold_metrics)
        final_model = train_xgb_with_params(params, X_train, y_train, sample_weight=sample_weight)
        final_total_nodes = count_xgb_nodes(final_model)
        mean_accuracy = float(cv_summary.get("mean_cv_accuracy") or 0.0)
        mean_fp_rate = float(cv_summary.get("mean_cv_fp_rate") or 0.0)
        size_ratio = (
            float(final_total_nodes) / float(max(int(args.max_model_nodes), 1))
            if int(args.max_model_nodes) > 0 else 0.0
        )
        eligible = not (int(args.max_model_nodes) > 0 and final_total_nodes > int(args.max_model_nodes))
        score = (
            mean_accuracy
            - float(args.model_search_fp_cost) * mean_fp_rate
            - float(args.model_search_size_cost) * size_ratio
        ) if eligible else float("-inf")
        mean_threshold = _mean_or_none([m.get("threshold") for m in fold_metrics])
        record = {
            "rank_input_order": int(idx),
            "eligible": bool(eligible),
            "score": float(score),
            "fp_rate": float(mean_fp_rate),
            "size_ratio": float(size_ratio),
            "total_nodes": int(final_total_nodes),
            "final_total_nodes": int(final_total_nodes),
            "avg_nodes_per_tree": float(final_total_nodes) / float(max(int(params["n_estimators"]), 1)),
            "selection_threshold": float(mean_threshold if mean_threshold is not None else 0.5),
            "selection_accuracy": float(mean_accuracy),
            "selection_fp_rate": float(mean_fp_rate),
            "selection_metrics": {
                "threshold": float(mean_threshold if mean_threshold is not None else 0.5),
                "accuracy": mean_accuracy,
                "fp_rate": mean_fp_rate,
                "precision": cv_summary.get("mean_cv_precision"),
                "recall": cv_summary.get("mean_cv_recall"),
            },
            "metrics": {
                "accuracy": mean_accuracy,
                "fp_rate": mean_fp_rate,
                "precision": cv_summary.get("mean_cv_precision"),
                "recall": cv_summary.get("mean_cv_recall"),
            },
            "is_default_params": bool(is_default_xgb_params(params)),
            "beats_default_params": False,
            "chosen_reason": "",
            "params": params,
        }
        record.update(cv_summary)
        record["cv_folds_completed"] = int(min(record["cv_folds_completed"], fold_count))
        return record

    cv_records = ordered_thread_map(
        evaluate_stage_b_candidate,
        list(enumerate(stage2_params, 1)),
        n_workers=resolve_model_search_workers(
            getattr(args, "model_search_n_workers", 1), n_items=len(stage2_params)),
    )

    best = choose_cv_model_search_record(
        cv_records,
        accuracy_tolerance=args.model_search_accuracy_tolerance,
    )
    if best is None:
        raise RuntimeError(
            f"model_search found no candidate under max_model_nodes={args.max_model_nodes}. "
            "Relax --max_model_nodes or shrink the search grid."
        )
    best_model = train_xgb_with_params(
        best["params"], X_train, y_train, sample_weight=sample_weight)

    cv_records.sort(key=lambda r: (
        not r["eligible"],
        -float(r.get("mean_cv_accuracy") or 0.0),
        float(r.get("std_cv_accuracy") if r.get("std_cv_accuracy") is not None else 1.0),
        float(r.get("mean_cv_fp_rate") if r.get("mean_cv_fp_rate") is not None else 1.0),
        int(r.get("final_total_nodes", r.get("total_nodes", 0))),
    ))
    default_baseline = next((r for r in cv_records if r.get("is_default_params")), None)
    return best_model, {
        "enabled": True,
        "strategy": "staged_group_cv",
        "selection_data": "train_group_cv",
        "selection_policy": "mean_cv_accuracy_std_fp_nodes",
        "max_model_nodes": int(args.max_model_nodes),
        "fp_cost": float(args.model_search_fp_cost),
        "size_cost": float(args.model_search_size_cost),
        "accuracy_tolerance": float(args.model_search_accuracy_tolerance),
        "grid_size": int(len(grid)),
        "stage2_top_k": int(stage2_top_k),
        "stage2_candidate_count": int(len(stage2_params)),
        "cv_folds": int(args.model_search_cv_folds),
        "cv_repeats": int(args.model_search_cv_repeats),
        "cv_folds_completed": int(fold_count),
        "random_state": int(args.model_search_random_state),
        "max_candidates": int(args.model_search_max_candidates),
        "outer_workers": int(n_workers),
        "parameter_space": {
            "total_combinations": int(total_combinations),
            "sampled_grid_size": int(len(grid)),
            "axes": {k: list(v) for k, v in axes.items()},
        },
        "cv_split": cv_meta,
        "default_baseline": _json_safe_model_search_record(default_baseline) if default_baseline else None,
        "best": _json_safe_model_search_record(best),
        "top_candidates": [_json_safe_model_search_record(r) for r in cv_records[:20]],
    }, cv_records


def search_xgb_hyperparameters(args, X_train, y_train, X_select=None, y_select=None,
                               scale_pos_weight=1.0, groups=None, sample_weight=None):
    strategy = getattr(args, "model_search_strategy", "single_split")
    if strategy == "staged_group_cv":
        return _search_xgb_hyperparameters_staged_group_cv(
            args,
            X_train,
            y_train,
            groups=groups,
            scale_pos_weight=scale_pos_weight,
            sample_weight=sample_weight,
        )
    if X_select is None or y_select is None:
        raise ValueError("single_split model_search requires X_select and y_select")
    return _search_xgb_hyperparameters_single_split(
        args,
        X_train,
        y_train,
        X_select,
        y_select,
        scale_pos_weight=scale_pos_weight,
        sample_weight=sample_weight,
    )


def compute_threshold_curve(model, X, y, beta=0.5):
    probs = model.predict_proba(X)[:, 1]
    rows = []
    for th in np.linspace(0.05, 0.95, 181):
        pred = (probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        precision = float(precision_score(y, pred, zero_division=0))
        recall = float(recall_score(y, pred, zero_division=0))
        f1 = float(f1_score(y, pred, zero_division=0))
        rows.append({
            "threshold": float(th),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta": float(_fbeta(precision, recall, beta)),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tn": int(tn),
        })
    return rows


def export_roc_pr_curves(model, X_valid, y_valid, artifact_dir):
    """Export ROC and Precision-Recall curves for model evaluation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip ROC/PR plot: {e}")
        return None

    from sklearn.metrics import roc_curve, precision_recall_curve, auc as _auc_metric

    y_true = np.asarray(y_valid, dtype=int)
    probs = model.predict_proba(np.asarray(X_valid, dtype=float))[:, 1]
    probs = np.clip(probs, 0.0, 1.0)
    mask = np.isfinite(probs)
    y_true = y_true[mask]
    probs = probs[mask]

    out_dir = os.path.join(str(artifact_dir), "report_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s05_roc_pr_curves.png")

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        fig = plt.figure(figsize=(12, 10), facecolor="white")
        fig.suptitle("ROC / PR Curves", fontsize=16, weight="bold")
        fig.text(0.5, 0.5, "Single class in validation split — cannot compute ROC/PR curves",
                 ha="center", va="center", fontsize=13, color="#c44e52")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] s05 ROC/PR plot (single class notice) -> {out_path}")
        return out_path

    # ---- ROC ----
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = float(_auc_metric(fpr, tpr))

    # ---- PR ----
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, probs)
    pr_auc = float(_auc_metric(recall_vals, precision_vals))

    fig = plt.figure(figsize=(12, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)

    # (0,0) Full ROC
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_roc.plot(fpr, tpr, color="#2f6f73", linewidth=2.0, label=f"XGBoost (AUC={roc_auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="#9aa6ac", linewidth=1.2, linestyle="--", alpha=0.7)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.03)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.grid(alpha=0.18)
    ax_roc.legend(frameon=False, loc="lower right")

    # (0,1) Full PR
    ax_pr = fig.add_subplot(gs[0, 1])
    ax_pr.plot(recall_vals, precision_vals, color="#4c78a8", linewidth=2.0, label=f"XGBoost (AUC={pr_auc:.4f})")
    # iso-F1 contours
    f_scores = np.linspace(0.2, 0.9, 8)
    for f_score in f_scores:
        x_vals = np.linspace(0.01, 1, 100)
        y_vals = f_score * x_vals / (2 * x_vals - f_score + 1e-12)
        valid = (y_vals > 0) & (y_vals <= 1)
        ax_pr.plot(x_vals[valid], y_vals[valid], color="#d35f2d", alpha=0.15, linewidth=0.7)
    ax_pr.set_xlim(0, 1.03)
    ax_pr.set_ylim(0, 1.03)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.grid(alpha=0.18)
    ax_pr.legend(frameon=False, loc="lower left")

    # (1,0) Zoomed ROC
    ax_roc_z = fig.add_subplot(gs[1, 0])
    ax_roc_z.plot(fpr, tpr, color="#2f6f73", linewidth=2.0)
    ax_roc_z.set_xlim(0, min(0.2, max(fpr) * 1.1 + 0.01))
    ax_roc_z.set_ylim(0.5, 1.03)
    ax_roc_z.set_xlabel("False Positive Rate (zoomed)")
    ax_roc_z.set_ylabel("True Positive Rate")
    ax_roc_z.set_title(f"ROC (FPR ≤ {min(0.2, max(fpr) * 1.1 + 0.01):.1f})")
    ax_roc_z.grid(alpha=0.18)
    # mark operating region
    ax_roc_z.axhline(y=0.9, color="#9aa6ac", linewidth=0.8, linestyle=":", alpha=0.5)

    # (1,1) Zoomed PR
    ax_pr_z = fig.add_subplot(gs[1, 1])
    ax_pr_z.plot(recall_vals, precision_vals, color="#4c78a8", linewidth=2.0)
    ax_pr_z.set_ylim(0.5, 1.03)
    ax_pr_z.set_xlim(max(0, min(recall_vals) * 0.95), 1.03)
    ax_pr_z.set_xlabel("Recall (zoomed)")
    ax_pr_z.set_ylabel("Precision")
    ax_pr_z.set_title("PR (high-precision region)")
    ax_pr_z.grid(alpha=0.18)
    ax_pr_z.axhline(y=0.9, color="#9aa6ac", linewidth=0.8, linestyle=":", alpha=0.5)

    fig.suptitle("ROC / PR Curves — Validation Split", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] s05 ROC/PR plot -> {out_path}")
    return out_path


def export_training_report_plot(plot_data, artifact_dir):
    """Export threshold/calibration summary plot for training reports."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip s05 plot: {e}")
        return None

    out_dir = os.path.join(str(artifact_dir), "report_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s05_training_report.png")

    curve = plot_data.get("threshold_curve", []) or []
    th = np.asarray([r.get("threshold", 0.0) for r in curve], dtype=float)
    precision = np.asarray([r.get("precision", 0.0) for r in curve], dtype=float)
    recall = np.asarray([r.get("recall", 0.0) for r in curve], dtype=float)
    fbeta = np.asarray([r.get("fbeta", 0.0) for r in curve], dtype=float)
    f1_vals = np.asarray([r.get("f1", 0.0) for r in curve], dtype=float)
    best_th = float(plot_data.get("threshold_search", {}).get("threshold", 0.5))

    metric_blocks = [
        ("valid default", plot_data.get("valid_default_threshold_metrics", {})),
        ("valid best", plot_data.get("valid_best_threshold_metrics", {})),
        ("threshold best", plot_data.get("threshold_split_best_metrics", {})),
    ]

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.85])
    ax_curve = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    if len(th):
        ax_curve.plot(th, precision, label="precision", color="#2f6f73", linewidth=2)
        ax_curve.plot(th, recall, label="recall", color="#4c78a8", linewidth=2)
        ax_curve.plot(th, fbeta, label="F-beta", color="#d35f2d", linewidth=2)
        ax_curve.plot(th, f1_vals, label="F1", color="#8172b2", linewidth=1.8, alpha=0.85)
        ax_curve.axvline(best_th, color="#222222", linestyle="--", linewidth=1.5, label=f"selected={best_th:.3f}")
    ax_curve.set_ylim(0, 1.03)
    ax_curve.set_xlabel("window threshold")
    ax_curve.set_ylabel("metric")
    ax_curve.set_title("Threshold Selection Curve")
    ax_curve.grid(alpha=0.18)
    ax_curve.legend(ncol=5, frameon=False, loc="lower center")

    labels = [x[0] for x in metric_blocks]
    prec = [float(x[1].get("precision", 0.0)) for x in metric_blocks]
    rec = [float(x[1].get("recall", 0.0)) for x in metric_blocks]
    x = np.arange(len(labels))
    ax_bar.bar(x - 0.18, prec, width=0.36, color="#2f6f73", label="precision")
    ax_bar.bar(x + 0.18, rec, width=0.36, color="#4c78a8", label="recall")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=15, ha="right")
    ax_bar.set_ylim(0, 1.03)
    ax_bar.set_title("Validation Split Metrics")
    ax_bar.grid(axis="y", alpha=0.18)
    ax_bar.legend(frameon=False)

    calib = plot_data.get("threshold_policy", {}).get("calibration", {})
    ax_text.axis("off")
    lines = [
        "Calibration / Threshold Split",
        f"calibration applied: {calib.get('applied', False)}",
        f"method: {calib.get('method', 'none')}",
        f"selected threshold: {best_th:.3f}",
        "threshold data: valid_threshold_split",
    ]
    ax_text.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f7", edgecolor="#d9dee2"))

    fig.suptitle("Training and Threshold Report", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] s05 report plot -> {out_path}")
    return out_path


def export_threshold_tradeoff_plot(plot_data, artifact_dir):
    """Export threshold vs FP/recall tradeoff plot and source CSV."""
    curve = list(plot_data.get("threshold_curve", []) or [])
    if not curve:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip s05 threshold tradeoff plot: {e}")
        return None

    out_dir = os.path.join(str(artifact_dir), "report_plots")
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(curve).copy()
    if "false_positive_rate" not in df.columns:
        if "fpr" in df.columns:
            df["false_positive_rate"] = pd.to_numeric(df["fpr"], errors="coerce")
        elif {"fp", "tn"}.issubset(df.columns):
            fp = pd.to_numeric(df["fp"], errors="coerce").fillna(0.0)
            tn = pd.to_numeric(df["tn"], errors="coerce").fillna(0.0)
            denom = fp + tn
            df["false_positive_rate"] = np.where(denom > 0, fp / denom, 0.0)
        else:
            df["false_positive_rate"] = np.nan

    for col in ["threshold", "precision", "recall", "false_positive_rate"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    csv_path = os.path.join(out_dir, "s05_threshold_fp_recall_tradeoff.csv")
    df.to_csv(csv_path, index=False)

    th = df["threshold"].to_numpy(dtype=float)
    precision = df["precision"].to_numpy(dtype=float)
    recall = df["recall"].to_numpy(dtype=float)
    fpr = df["false_positive_rate"].to_numpy(dtype=float)
    best_th = float(plot_data.get("threshold_search", {}).get("threshold", 0.5))
    finite_fpr = fpr[np.isfinite(fpr)]
    fpr_ylim = min(1.0, max(0.05, float(np.max(finite_fpr)) * 1.15 if len(finite_fpr) else 0.05))

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_fp = ax.twinx()
    ax.plot(th, recall, color="#4c78a8", linewidth=2.0, label="recall")
    ax.plot(th, precision, color="#2f6f73", linewidth=2.0, label="precision")
    ax_fp.plot(th, fpr, color="#c44e52", linewidth=2.0, label="false positive rate")
    ax.axvline(best_th, color="#222222", linestyle="--", linewidth=1.3, label=f"selected={best_th:.3f}")
    ax.set_xlabel("window threshold")
    ax.set_ylabel("precision / recall")
    ax_fp.set_ylabel("false positive rate")
    ax.set_ylim(0, 1.03)
    ax_fp.set_ylim(0, fpr_ylim)
    ax.set_title("Threshold Tradeoff")
    ax.grid(alpha=0.18)
    lines, labels = ax.get_legend_handles_labels()
    lines_fp, labels_fp = ax_fp.get_legend_handles_labels()
    ax.legend(lines + lines_fp, labels + labels_fp, frameon=False, loc="best")

    ax_region = fig.add_subplot(gs[0, 1])
    ax_region.plot(recall, fpr, color="#8172b2", linewidth=2.0)
    ax_region.scatter(recall, fpr, s=16, color="#8172b2", alpha=0.35)
    if len(th):
        idx = int(np.nanargmin(np.abs(th - best_th)))
        ax_region.scatter([recall[idx]], [fpr[idx]], s=70, color="#222222", zorder=3)
    ax_region.set_xlabel("recall")
    ax_region.set_ylabel("false positive rate")
    ax_region.set_xlim(0, 1.03)
    ax_region.set_ylim(0, fpr_ylim)
    ax_region.set_title("Low-FP Operating Region")
    ax_region.grid(alpha=0.18)

    fig.suptitle("Window Threshold: False Positives vs Recall", fontsize=15, weight="bold")
    fig.subplots_adjust(top=0.86, left=0.08, right=0.92, bottom=0.12, wspace=0.28)
    fig_path = os.path.join(out_dir, "s05_threshold_fp_recall_tradeoff.png")
    scientific = save_scientific_figure(
        fig,
        fig_path,
        source_data=df,
        source_data_path=csv_path,
        inputs=[csv_path],
        core_conclusion=(
            "The selected validation threshold balances window accuracy, recall, "
            "precision, and false-positive risk."
        ),
        panel_map={
            "a": "Precision, recall, and false-positive rate across thresholds.",
            "b": "Recall versus false-positive rate in the operating region.",
        },
        split="valid",
        n_definition=f"{len(df)} threshold candidates on the valid threshold split",
        statistics={
            "metric": "window classification metrics",
            "selected_threshold": best_th,
            "interval": "none",
        },
        reviewer_risks=["Threshold is selected on valid and must remain frozen for test."],
        test_read_only=False,
    )
    plt.close(fig)
    print(f"[OK] s05 threshold tradeoff plot -> {fig_path}")
    return {
        "figure": str(scientific["png"]),
        "source_data": str(scientific["source_data"]),
        "manifest": str(scientific["manifest"]),
        "qa": str(scientific["qa"]),
    }


# =========================================================
# 质量阈值 / OOD 分位数 / fingerprint
# =========================================================

QUALITY_FEATURES_DEFAULT = ["Ambient_std", "G_mean_mean"]


def learn_quality_thresholds(df_train, features=None, q_high=0.99, q_low=0.01):
    """
    从 train 集学习 compute_quality 用的阈值。
    Ambient_std 用 q_high 分位（防过大）；G_mean_mean/IR_mean 用 q_low 分位（防过小）。
    返回 dict 可直接塞 bundle["quality_thresholds"]。
    """
    if features is None:
        features = QUALITY_FEATURES_DEFAULT
    out = {}
    for f in features:
        if f not in df_train.columns:
            continue
        x = df_train[f].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            continue
        if f == "Ambient_std":
            out[f] = {"type": "high", "thr": float(x.quantile(q_high))}
        else:
            out[f] = {"type": "low", "thr": float(np.abs(x).quantile(q_low))}
    out["_meta"] = {
        "learned_from": "train_only",
        "q_high": float(q_high),
        "q_low": float(q_low),
    }
    return out


def compute_feature_quantiles(df_train, features, q_low=0.05, q_high=0.95):
    """从 train 算每个特征的 [q_low, q_high]，给 s06 做 OOD 监控。"""
    out = {}
    for f in features:
        if f not in df_train.columns:
            continue
        x = df_train[f].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            continue
        out[f] = {
            "q_low": float(x.quantile(q_low)),
            "q_high": float(x.quantile(q_high)),
        }
    out["_meta"] = {
        "learned_from": "train_only",
        "q_low": float(q_low),
        "q_high": float(q_high),
    }
    return out


def build_fingerprint(artifact_dir, feature_pool_path, splits_path):
    """收集 provenance：版本号、数据 hash、git sha、训练时间。"""
    import hashlib
    import time
    import platform

    info = {
        "train_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import numpy as _np
        info["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import pandas as _pd
        info["pandas"] = _pd.__version__
    except Exception:
        pass
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import xgboost as _xgb
        info["xgboost"] = _xgb.__version__
    except Exception:
        pass

    def sha256_file(path):
        """Stream the complete file so provenance also detects tail changes."""
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    info["splits_sha256"] = sha256_file(splits_path)
    info["feature_pool_train_sha256"] = sha256_file(feature_pool_path)

    # git sha（如果可用）
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=2)
        if r.returncode == 0:
            info["git_sha"] = r.stdout.strip()
    except Exception:
        pass

    return info


def _compute_scale_pos_weight(args, neg_count, pos_count, p_train_pos):
    """计算 scale_pos_weight，返回 (sw, strategy_str)。"""
    if args.legacy_scale_pos_weight:
        sw = (neg_count / pos_count) if pos_count > 0 else 1.0
        return sw, "legacy_neg_over_pos"
    if args.target_deploy_ratio is not None:
        r = float(args.target_deploy_ratio)
        r = min(max(r, 1e-6), 1 - 1e-6)
        if 0.0 < p_train_pos < 1.0:
            sw = (r * (1 - p_train_pos)) / ((1 - r) * p_train_pos)
        else:
            sw = 1.0
        return sw, f"target_deploy_ratio={r}"
    return 1.0, "balanced_1.0"


def _train_for_k(args, k, features, df_train_raw, df_valid_raw, train_groups,
                 calibration_threshold_split, scale_pos_weight, clip_bounds_cache=None,
                 train_cv_feature_score=False):
    """用 top-k 特征训练并返回完整结果。

    返回 dict: model, search_summary, search_records, features, fill_values,
                clip_bounds, X_valid, y_valid, model_select_split, df_calib
    """
    # clip. In multi-k/local-swap search, reuse train-learned bounds for the
    # superset of ranked candidates to avoid recomputing quantiles per subset.
    if clip_bounds_cache is not None:
        clip_bounds = {
            f: clip_bounds_cache[f]
            for f in features
            if f in clip_bounds_cache
        }
        df_train = clip_outliers(df_train_raw, features, k=1.5, bounds=clip_bounds)
    else:
        df_train, clip_bounds = clip_outliers(df_train_raw, features, k=1.5, return_bounds=True)
    df_valid = clip_outliers(df_valid_raw, features, k=1.5, bounds=clip_bounds)

    # fill
    fill_values = prepare_fill_values(df_train, features)
    X_train, y_train, _ = prepare_xy(df_train, features, fill_values=fill_values)
    X_valid, y_valid, _ = prepare_xy(df_valid, features, fill_values=fill_values)

    # model selection split
    df_calib_pool = df_valid
    if args.model_search and args.model_search_strategy == "single_split":
        df_model_select, df_calib, model_select_split = split_calibration_for_model_search(
            df_calib_pool,
            search_fraction=args.model_search_valid_fraction,
            random_state=args.calibration_random_state,
        )
    elif args.model_search:
        df_model_select = df_calib_pool.copy()
        df_calib = df_calib_pool
        _calib_groups = (calibration_threshold_split or {}).get("calibration_groups")
        model_select_split = {
            "fallback": False, "reason": "train_group_cv",
            "model_selection_groups": None,
            "calibration_groups": _calib_groups,
        }
    else:
        df_model_select = df_calib_pool.copy()
        df_calib = df_calib_pool
        _calib_groups_disabled = (calibration_threshold_split or {}).get("calibration_groups")
        model_select_split = {
            "fallback": False, "reason": "model_search_disabled",
            "model_selection_groups": None,
            "calibration_groups": _calib_groups_disabled,
        }

    X_model_select, y_model_select, _ = prepare_xy(
        df_model_select, features, fill_values=fill_values)

    # train
    model_search_records = []
    model_search_summary = {
        "enabled": False, "selection_data": None,
        "max_model_nodes": int(args.max_model_nodes),
        "fp_cost": float(args.model_search_fp_cost),
        "size_cost": float(args.model_search_size_cost),
        "fixed_params": build_default_xgb_params(scale_pos_weight=scale_pos_weight),
    }
    if args.model_search:
        raw_model, model_search_summary, model_search_records = search_xgb_hyperparameters(
            args, X_train, y_train, X_model_select, y_model_select,
            scale_pos_weight=scale_pos_weight, groups=train_groups,
        )
        model_search_summary["split"] = model_select_split
    else:
        raw_model = train_xgb_with_params(
            build_default_xgb_params(scale_pos_weight=scale_pos_weight),
            X_train, y_train,
        )
        model_search_summary["split"] = model_select_split
    model_search_summary["feature_count"] = int(k)
    model_search_summary["selected_features"] = list(features)
    for _record in model_search_records:
        _record.setdefault("feature_count", int(k))
        _record.setdefault("n_features", len(features))
        _record.setdefault("deployment_feature_cost_summary", summarize_deployment_feature_costs(features))

    total_nodes = count_xgb_nodes(raw_model)
    best_score = _finite_float_or_none((model_search_summary.get("best") or {}).get("score"))
    if best_score is None:
        best_score = float("-inf")
    valid_default = eval_model(raw_model, X_valid, y_valid, threshold=0.5)
    valid_acc = float(valid_default.get("accuracy", 0.0))
    if args.model_search and getattr(args, "model_search_strategy", "single_split") == "staged_group_cv":
        selection_score = best_score
        selection_metric = "train_cv_model_search_score"
    else:
        selection_score = valid_acc
        selection_metric = "valid_accuracy"
    if train_cv_feature_score:
        fixed_cv = score_fixed_params_with_train_cv(
            args,
            X_train,
            y_train,
            train_groups,
            build_default_xgb_params(scale_pos_weight=scale_pos_weight),
            total_nodes=total_nodes,
        )
        selection_score = float(fixed_cv["score"])
        selection_metric = fixed_cv["selection_metric"]
        model_search_summary["feature_set_train_cv"] = fixed_cv
    model_search_summary["feature_set_selection_metric"] = selection_metric
    model_search_summary["feature_set_selection_score"] = float(selection_score)

    logger.info(f"  [k={k}] features={len(features)} nodes={total_nodes} "
                f"valid_acc={valid_acc:.4f} search_score={best_score:.4f} "
                f"selection_metric={selection_metric} selection_score={selection_score:.4f}")

    return {
        "k": k, "features": list(features), "model": raw_model,
        "fill_values": fill_values, "clip_bounds": clip_bounds,
        "X_valid": X_valid, "y_valid": y_valid, "df_calib": df_calib,
        "search_summary": model_search_summary, "search_records": model_search_records,
        "valid_acc": valid_acc, "search_score": best_score,
        "feature_set_selection_metric": selection_metric,
        "feature_set_selection_score": float(selection_score),
        "total_nodes": total_nodes,
    }


def build_local_swap_feature_sets(ranked_features, base_features, tail_size=3, pool_size=8, max_candidates=12):
    """Generate fixed-size local swaps from ranked candidates after the current top-k set."""
    base = [str(f) for f in base_features]
    base_set = set(base)
    ranked_names = []
    for item in ranked_features:
        name = item.get("feature") if isinstance(item, dict) else item
        if name is None:
            continue
        name = str(name)
        if name and name not in ranked_names:
            ranked_names.append(name)

    tail_size = max(0, min(int(tail_size or 0), len(base)))
    pool_size = max(0, int(pool_size or 0))
    max_candidates = max(0, int(max_candidates or 0))
    if tail_size <= 0 or pool_size <= 0 or max_candidates <= 0:
        return []

    replace_positions = list(range(len(base) - tail_size, len(base)))
    pool = [name for name in ranked_names if name not in base_set][:pool_size]
    candidates = []
    seen = {tuple(base)}
    for pos in replace_positions:
        for incoming in pool:
            candidate = list(base)
            candidate[pos] = incoming
            if len(set(candidate)) != len(candidate):
                continue
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def train_best_local_feature_set_for_k(args, k, ranked_features, base_features,
                                       df_train_raw, df_valid_raw, train_groups,
                                       calibration_threshold_split, scale_pos_weight,
                                       clip_bounds_cache=None):
    """Quick-score local swaps, then run full model search once on the best fixed-size set."""
    candidate_feature_sets = [(list(base_features), "ranked_top_k")]
    if args.feature_search_local_swap:
        swap_sets = build_local_swap_feature_sets(
            ranked_features,
            base_features,
            tail_size=args.feature_search_swap_tail_size,
            pool_size=args.feature_search_swap_pool_size,
            max_candidates=args.feature_search_swap_max_candidates,
        )
        candidate_feature_sets.extend(
            (swap_feats, f"local_swap_{idx}")
            for idx, swap_feats in enumerate(swap_sets, start=1)
        )

    original_model_search = bool(args.model_search)
    use_train_cv_quick_score = (
        original_model_search
        and getattr(args, "model_search_strategy", "single_split") == "staged_group_cv"
    )
    best_quick_result = None
    best_quick_score = float("-inf")
    def _feature_set_score(result):
        score = _finite_float_or_none(result.get("feature_set_selection_score"))
        if score is not None:
            return score
        if use_train_cv_quick_score:
            score = _finite_float_or_none(result.get("search_score"))
            if score is not None:
                result.setdefault("feature_set_selection_metric", "train_cv_model_search_score")
                result.setdefault("feature_set_selection_score", score)
                result["search_summary"].setdefault(
                    "feature_set_selection_metric",
                    result["feature_set_selection_metric"],
                )
                result["search_summary"].setdefault("feature_set_selection_score", score)
                return score
        return float(result["valid_acc"])

    for candidate_features, candidate_source in candidate_feature_sets:
        args.model_search = False
        try:
            result = _train_for_k(
                args, k, candidate_features, df_train_raw, df_valid_raw,
                train_groups, calibration_threshold_split, scale_pos_weight,
                clip_bounds_cache=clip_bounds_cache,
                train_cv_feature_score=use_train_cv_quick_score,
            )
        finally:
            args.model_search = original_model_search
        result["search_summary"]["scale_pos_weight"] = scale_pos_weight
        result["search_summary"]["feature_set_source"] = candidate_source
        result["_combined_score"] = _feature_set_score(result)
        if result["_combined_score"] > best_quick_score:
            best_quick_score = result["_combined_score"]
            best_quick_result = result

    if original_model_search:
        result = _train_for_k(
            args, k, best_quick_result["features"], df_train_raw, df_valid_raw,
            train_groups, calibration_threshold_split, scale_pos_weight,
            clip_bounds_cache=clip_bounds_cache,
        )
        result["search_summary"]["scale_pos_weight"] = scale_pos_weight
        result["search_summary"]["feature_set_source"] = (
            best_quick_result["search_summary"].get("feature_set_source")
        )
        result["_combined_score"] = _feature_set_score(result)
    else:
        result = best_quick_result

    result["search_summary"]["local_swap_search"] = {
        "enabled": bool(args.feature_search_local_swap),
        "candidates_tested": int(len(candidate_feature_sets)),
        "best_source": result["search_summary"].get("feature_set_source"),
        "tail_size": int(args.feature_search_swap_tail_size),
        "pool_size": int(args.feature_search_swap_pool_size),
        "max_candidates": int(args.feature_search_swap_max_candidates),
    }
    return result


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument(
        "--feature_selection_mode",
        choices=["manual", "auto"],
        default="manual",
    )
    parser.add_argument("--manual_feature_file", type=str, default=None)
    parser.add_argument("--max_features", type=int, default=None,
                        help="从 ranked_features.json 取 top-k 特征；默认 None 时回退到 selected_features.json")
    parser.add_argument("--model_search_feature_counts", type=str, default="8,10,12,15,18",
                        help="搜参时测试的特征数量，逗号分隔 (如 8,10,12,15,18)。留空则使用 --max_features 固定值")
    parser.add_argument("--feature_search_local_swap", action=argparse.BooleanOptionalAction, default=True,
                        help="try fixed-size local swaps around the ranked top-k feature set")
    parser.add_argument("--feature_search_swap_tail_size", type=int, default=3,
                        help="number of lowest-ranked selected features eligible for local swaps")
    parser.add_argument("--feature_search_swap_pool_size", type=int, default=8,
                        help="number of next-ranked candidate features considered for local swaps")
    parser.add_argument("--feature_search_swap_max_candidates", type=int, default=12,
                        help="maximum local-swap feature sets evaluated per k")
    parser.add_argument(
        "--threshold_objective", type=str, default="accuracy",
        choices=["f1", "precision", "recall", "fbeta", "precision_constrained", "accuracy"],
        help="阈值搜索目标。默认 accuracy，优先单窗口准确率。"
    )
    parser.add_argument("--threshold_beta", type=float, default=0.5,
                        help="F-beta 的 beta。<1 偏 precision，>1 偏 recall。")
    parser.add_argument("--threshold_min_precision", type=float, default=0.95,
                        help="precision_constrained 模式下的精度下限。")
    parser.add_argument("--calibration_method", type=str, default="isotonic",
                        choices=["none", "isotonic"],
                        help="Probability calibration method. Calibration uses a valid subgroup split.")
    parser.add_argument("--threshold_valid_fraction", type=float, default=0.5,
                        help="Fraction of valid sample groups reserved for threshold selection.")
    parser.add_argument("--calibration_random_state", type=int, default=42,
                        help="Random seed for splitting valid into calibration/threshold groups.")
    parser.add_argument("--window_sec", type=float, default=5.0,
                        help="Feature window seconds recorded into model metadata.")
    parser.add_argument("--step_sec", type=float, default=1.0,
                        help="Feature/evaluation stride seconds recorded into model metadata.")
    parser.add_argument("--use_stage2_ir", action=argparse.BooleanOptionalAction, default=False,
                        help="legacy compatibility flag; Stage2 model features are always ambient/green/ACC only")
    parser.add_argument("--model_search", action=argparse.BooleanOptionalAction, default=False,
                        help="search XGBoost params under a node-count budget before final calibration")
    parser.add_argument("--model_search_strategy", type=str, default="staged_group_cv",
                        choices=["staged_group_cv", "single_split"],
                        help="model search strategy; staged_group_cv uses train-group CV and preserves valid for calibration/threshold")
    parser.add_argument("--max_model_nodes", type=int, default=500,
                        help="maximum allowed total XGBoost tree nodes during --model_search; <=0 disables the cap")
    parser.add_argument("--model_search_fp_cost", type=float, default=2.0,
                        help="FP penalty in model-search score: accuracy - fp_cost*fp_rate - size_cost*size_ratio")
    parser.add_argument("--model_search_size_cost", type=float, default=0.1,
                        help="model-size penalty in model-search score")
    parser.add_argument("--model_search_accuracy_tolerance", type=float, default=0.0,
                        help="accuracy gap allowed when preferring a smaller model in --model_search; default 0.0 means accuracy strictly wins under the node budget")
    parser.add_argument("--model_search_valid_fraction", type=float, default=0.5,
                        help="fraction of the valid calibration pool reserved for single_split model-search selection")
    parser.add_argument("--model_search_max_candidates", type=int, default=360,
                        help="maximum sampled candidates from the search grid; <=0 disables sampling")
    parser.add_argument("--model_search_stage1_top_k", type=int, default=4,
                        help="number of stage-1 structure candidates advanced to stage-2 refine")
    parser.add_argument("--model_search_stage2_top_k", type=int, default=48,
                        help="number of stage-A candidates kept for staged_group_cv")
    parser.add_argument("--model_search_cv_folds", type=int, default=3,
                        help="group-CV folds for staged_group_cv")
    parser.add_argument("--model_search_cv_repeats", type=int, default=2,
                        help="group-CV repeats for staged_group_cv")
    parser.add_argument("--model_search_random_state", type=int, default=42,
                        help="random seed for deterministic candidate sampling and CV splits")
    parser.add_argument("--model_search_n_workers", type=int, default=1,
                        help="parallel outer model candidates; each XGBoost fit defaults to WL_INNER_N_JOBS=1")
    parser.add_argument("--model_search_n_estimators", type=str,
                        default=_model_search_default_csv("n_estimators"),
                        help="comma-separated n_estimators candidates for --model_search")
    parser.add_argument("--model_search_max_depth", type=str,
                        default=_model_search_default_csv("max_depth"),
                        help="comma-separated max_depth candidates for --model_search")
    parser.add_argument("--model_search_learning_rate", type=str,
                        default=_model_search_default_csv("learning_rate"),
                        help="comma-separated learning_rate candidates for --model_search")
    parser.add_argument("--model_search_min_child_weight", type=str,
                        default=_model_search_default_csv("min_child_weight"),
                        help="comma-separated min_child_weight candidates for --model_search")
    parser.add_argument("--model_search_reg_lambda", type=str,
                        default=_model_search_default_csv("reg_lambda"),
                        help="comma-separated reg_lambda candidates for --model_search")
    parser.add_argument("--model_search_reg_alpha", type=str,
                        default=_model_search_default_csv("reg_alpha"),
                        help="comma-separated reg_alpha candidates for --model_search")
    parser.add_argument("--model_search_subsample", type=str,
                        default=_model_search_default_csv("subsample"),
                        help="comma-separated subsample candidates for --model_search")
    parser.add_argument("--model_search_colsample_bytree", type=str,
                        default=_model_search_default_csv("colsample_bytree"),
                        help="comma-separated colsample_bytree candidates for --model_search")
    parser.add_argument("--mine_hard_negatives", action="store_true", default=False,
                        help="mine train-only OOF hard negatives and upweight them before final calibration")
    parser.add_argument("--hard_negative_weight", type=float, default=3.0,
                        help="sample weight assigned to mined train hard-negative windows")
    parser.add_argument("--hard_negative_top_percentile", type=float, default=0.10,
                        help="fraction of highest-probability train negatives selected as hard negatives")
    parser.add_argument("--hard_negative_min_probability", type=float, default=None,
                        help="minimum OOF probability for hard-negative mining; defaults to initial window threshold")
    parser.add_argument("--ood_q_low", type=float, default=0.05)
    parser.add_argument("--ood_q_high", type=float, default=0.95)
    parser.add_argument(
        "--target_deploy_ratio", type=float, default=None,
        help=("部署时 Stage2 输入的期望 P(target=1 | Stage1 pass)。"
              "给出后按公式 r*(1-p_train)/((1-r)*p_train) 计算 scale_pos_weight，"
              "让训练 effective 分布对齐部署条件分布。"
              "默认 None：scale_pos_weight=1.0（既不补偿不平衡也不强加权）。")
    )
    parser.add_argument(
        "--legacy_scale_pos_weight", action="store_true", default=False,
        help=("回退到旧行为 scale_pos_weight = neg/pos。仅用于对照实验，"
              "不建议生产使用（与 target_aware_stride 叠加会双倍偏置）。")
    )

    if args is None:
        args = parser.parse_args()

    feature_pool_train_path = os.path.join(args.artifact_dir, "feature_pool_train.csv")
    feature_pool_valid_path = os.path.join(args.artifact_dir, "feature_pool_valid.csv")
    splits_path = os.path.join(args.artifact_dir, "splits.json")

    df_train_raw = pd.read_csv(feature_pool_train_path)
    df_valid_raw = pd.read_csv(feature_pool_valid_path)
    validate_feature_pool_frames(df_train_raw, df_valid_raw)

    selected_features_path = os.path.join(args.artifact_dir, "selected_features.json")
    ranked_features_path = os.path.join(args.artifact_dir, "ranked_features.json")
    full_ranking_path = os.path.join(args.artifact_dir, "feature_ranking_full.json")
    selection_provenance = {
        "feature_selection_mode": str(args.feature_selection_mode),
        "feature_pool_version": FEATURE_POOL_VERSION,
    }
    if args.feature_selection_mode == "manual":
        manual_feature_file = (
            args.manual_feature_file
            or os.path.join(args.artifact_dir, "manual_feature_selection.csv")
        )
        selected_features, selection_provenance = load_manual_feature_selection(
            manual_feature_file,
            full_ranking_path,
            df_train_raw,
            df_valid_raw,
        )
        if args.feature_search_local_swap:
            raise ValueError(
                "manual feature selection conflicts with --feature_search_local_swap; "
                "use --no-feature_search_local_swap."
            )
        args.model_search_feature_counts = ""
    else:
        if os.path.exists(ranked_features_path):
            with open(ranked_features_path, "r", encoding="utf-8") as f:
                ranked = json.load(f)
            ranked = [
                r for r in ranked
                if r.get("feature") in set(filter_features_for_deployment([r.get("feature")]))
            ]
            _k = min(args.max_features if args.max_features is not None else 15, len(ranked))
            selected_features = [r["feature"] for r in ranked[:_k]]
            logger.info(f"从 ranked_features.json 取 top {_k} 特征（共 {len(ranked)} 个候选）")
        else:
            with open(selected_features_path, "r", encoding="utf-8") as f:
                fs = json.load(f)
            selected_features = fs["selected_features"]
        selected_features = enforce_no_stage2_ir_features(
            selected_features, "initial selected_features"
        )
        selected_features = filter_features_for_deployment(selected_features)
        if not selected_features:
            raise ValueError(
                "deployment-friendly feature filter left no selected features for s05; rerun s04."
            )

    # ── 特征数量搜参：解析 --model_search_feature_counts ──
    _fc_str = getattr(args, "model_search_feature_counts", "") or ""
    _feature_counts = []
    if _fc_str.strip():
        for _part in _fc_str.split(","):
            _part = _part.strip()
            if _part.isdigit():
                _feature_counts.append(int(_part))
        _feature_counts = sorted(set(_feature_counts))

    # ── 多 k/显式 k 特征数搜索分支 / 普通单 k 分支 ──
    _using_feature_count_search = bool(_feature_counts and os.path.exists(ranked_features_path))
    model_search_records = []
    model_search_summary = {}
    if _using_feature_count_search:
        # ============ 多 k 搜参 ============
        if not args.model_search:
            logger.warning("model_search_feature_counts 已指定但 model_search 未启用，"
                           "仅用默认 XGBoost 参数遍历不同 k 值。")

        with open(ranked_features_path, "r", encoding="utf-8") as _f:
            _ranked = json.load(_f)
        _ranked_before = len(_ranked)
        _ranked = [r for r in _ranked if not is_stage2_ir_feature(r.get("feature", ""))]
        _ranked = [
            r for r in _ranked
            if r.get("feature") in set(filter_features_for_deployment([r.get("feature")]))
        ]
        _ranked_dropped = _ranked_before - len(_ranked)
        if _ranked_dropped > 0:
            logger.warning(
                "ranked_features.json: removed %d IR-derived Stage2 candidates before feature-count search",
                _ranked_dropped,
            )
        if not _ranked:
            raise ValueError("ranked_features.json has no non-IR Stage2 candidates after filtering")

        _max_k = min(max(_feature_counts), len(_ranked))
        _ks = [k for k in _feature_counts if k <= _max_k]
        if not _ks:
            _ks = [min(args.max_features if args.max_features is not None else 15, len(_ranked))]

        logger.info(f"特征数搜参: 测试 k ∈ {_ks}（共 {len(_ks)} 个候选，"
                    f"ranked_features 共 {len(_ranked)} 个）")
        _clip_feature_limit = max(_ks) + max(0, int(args.feature_search_swap_pool_size))
        _clip_feature_candidates = [
            r["feature"] for r in _ranked[:_clip_feature_limit]
            if r.get("feature") in df_train_raw.columns
        ]
        _clip_bounds_cache = learn_clip_bounds(df_train_raw, _clip_feature_candidates, k=1.5)
        logger.info(
            "预计算 clip bounds: %d/%d candidate features (reuse across k/local-swap)",
            len(_clip_bounds_cache),
            len(_clip_feature_candidates),
        )

        # 前置：计算一次 scale_pos_weight（不随 k 变化）
        _all_features = [r["feature"] for r in _ranked]
        _feats_tmp = _all_features[:min(10, len(_all_features))]
        _tmp_bounds = {f: _clip_bounds_cache[f] for f in _feats_tmp if f in _clip_bounds_cache}
        _df_tmp = clip_outliers(df_train_raw, _feats_tmp, k=1.5, bounds=_tmp_bounds)
        _fv_tmp = prepare_fill_values(_df_tmp, _feats_tmp)
        _X_tmp, _y_tmp, _ = prepare_xy(_df_tmp, _feats_tmp, fill_values=_fv_tmp)
        _train_groups = (df_train_raw["sample_name"].astype("object").to_numpy()
                         if "sample_name" in df_train_raw.columns else None)
        _neg_count = int(np.sum(_y_tmp == 0))
        _pos_count = int(np.sum(_y_tmp == 1))
        _p_train_pos = _pos_count / float(max(_neg_count + _pos_count, 1))
        _sw, _sw_strategy = _compute_scale_pos_weight(args, _neg_count, _pos_count, _p_train_pos)

        _best_result = None
        _best_score = float("-inf")
        for _k in _ks:
            _feats = [r["feature"] for r in _ranked[:_k]]
            _result = train_best_local_feature_set_for_k(
                args,
                _k,
                _ranked,
                _feats,
                df_train_raw,
                df_valid_raw,
                _train_groups,
                None,
                _sw,
                clip_bounds_cache=_clip_bounds_cache,
            )
            _score = _result["_combined_score"]
            if _score > _best_score:
                _best_score = _score
                _best_result = _result

        logger.info(f"特征数搜参完成: best k={_best_result['k']} score={_best_score:.4f}")
        selected_features = _best_result["features"]
        raw_model = _best_result["model"]
        fill_values = _best_result["fill_values"]
        clip_bounds = _best_result["clip_bounds"]
        X_valid = _best_result["X_valid"]
        y_valid = _best_result["y_valid"]
        df_calib = _best_result["df_calib"]
        model_search_summary = _best_result["search_summary"]
        model_search_records = _best_result["search_records"]
        model_search_summary["feature_search"] = {
            "enabled": True,
            "candidates_tested": _ks,
            "best_k": int(_best_result["k"]),
            "selection_metric": _best_result.get("feature_set_selection_metric", "valid_accuracy"),
            "best_score": float(_best_score),
        }
        # 多 k 搜参只消耗 train 内部 group-CV；valid 仍重新拆成 calibration/threshold。
        scale_pos_weight = _sw
        scale_pos_weight_strategy = _sw_strategy
        calibration_threshold_split = None
        X_threshold = X_valid
        y_threshold = y_valid
        X_calib = X_valid
        y_calib = y_valid
        _df_valid_final = clip_outliers(
            df_valid_raw,
            selected_features,
            k=1.5,
            bounds=clip_bounds,
        )
        _valid_splits = prepare_valid_calibration_threshold_data(
            _df_valid_final,
            selected_features,
            fill_values,
            threshold_fraction=args.threshold_valid_fraction,
            random_state=args.calibration_random_state,
        )
        calibration_threshold_split = _valid_splits["split"]
        df_calib = _valid_splits["df_calib"]
        X_threshold = _valid_splits["X_threshold"]
        y_threshold = _valid_splits["y_threshold"]
        X_calib = _valid_splits["X_calib"]
        y_calib = _valid_splits["y_calib"]
        model_search_summary["calibration_threshold_split"] = calibration_threshold_split
        if isinstance(model_search_summary.get("split"), dict):
            model_search_summary["split"]["calibration_groups"] = calibration_threshold_split.get("calibration_groups")
            model_search_summary["split"]["threshold_groups"] = calibration_threshold_split.get("threshold_groups")
        train_groups = _train_groups
        neg_count, pos_count = _neg_count, _pos_count
        p_train_pos = _p_train_pos
        feature_quantiles = compute_feature_quantiles(
            clip_outliers(df_train_raw, selected_features, k=1.5, bounds=clip_bounds),
            selected_features, q_low=args.ood_q_low, q_high=args.ood_q_high,
        )

    else:
        # ============ 单 k 分支（原有逻辑）============
        logger.info("应用异常值裁剪 (train 学边界 → train/valid 同步应用)...")
        df_train, clip_bounds = clip_outliers(df_train_raw, selected_features, k=1.5,
                                              return_bounds=True)
        df_valid = clip_outliers(df_valid_raw, selected_features, k=1.5, bounds=clip_bounds)

        quality_thresholds = learn_quality_thresholds(df_train, QUALITY_FEATURES_DEFAULT)
        feature_quantiles = compute_feature_quantiles(
            df_train, selected_features,
            q_low=args.ood_q_low, q_high=args.ood_q_high
        )
        df_calib_pool, df_threshold, calibration_threshold_split = split_valid_for_calibration_threshold(
            df_valid,
            threshold_fraction=args.threshold_valid_fraction,
            random_state=args.calibration_random_state,
        )
        if args.model_search and args.model_search_strategy == "single_split":
            df_model_select, df_calib, model_selection_split = split_calibration_for_model_search(
                df_calib_pool,
                search_fraction=args.model_search_valid_fraction,
                random_state=args.calibration_random_state,
            )
        elif args.model_search:
            df_model_select = df_calib_pool.copy()
            df_calib = df_calib_pool
            model_selection_split = {
                "fallback": False,
                "reason": "train_group_cv",
                "model_selection_groups": None,
                "calibration_groups": calibration_threshold_split.get("calibration_groups"),
            }
        else:
            df_model_select = df_calib_pool.copy()
            df_calib = df_calib_pool
            model_selection_split = {
                "fallback": False,
                "reason": "model_search_disabled",
                "model_selection_groups": None,
                "calibration_groups": calibration_threshold_split.get("calibration_groups"),
            }

        fill_values = prepare_fill_values(df_train, selected_features)

        X_train, y_train, _ = prepare_xy(
            df_train, selected_features, fill_values=fill_values)
        train_groups = (
            df_train["sample_name"].astype("object").to_numpy()
            if "sample_name" in df_train.columns else None
        )

        X_valid, y_valid, _ = prepare_xy(
            df_valid, selected_features, fill_values=fill_values)
        X_calib, y_calib, _ = prepare_xy(
            df_calib, selected_features, fill_values=fill_values)
        X_model_select, y_model_select, _ = prepare_xy(
            df_model_select, selected_features, fill_values=fill_values)
        X_threshold, y_threshold, _ = prepare_xy(
            df_threshold, selected_features, fill_values=fill_values)

        # 样本权重策略
    #
    # 旧行为 (legacy)：scale_pos_weight = neg/pos
    #   问题：与 target_aware_stride（pos=3s, neg=1s）叠加，会双倍偏置模型预测正类，
    #   抬高 FP。本项目 FP 代价高于 FN，这是个严重 bias。
    #
    # 新行为（默认）：scale_pos_weight = 1.0
    #   不主动补偿；让模型按训练数据天然分布学。配合 F-beta(β=0.5) 阈值选择
    #   把 FP 控制在产品可接受范围。
    #
    # 显式期望（--target_deploy_ratio r）：
    #   按部署时 Stage2 输入的期望 P(target=1 | Stage1 pass) 重加权，
    #   使训练 effective 分布 ≈ 部署条件分布。
    #   公式: scale_pos_weight = r * (1 - p_train) / ((1 - r) * p_train)
    # 普通单 k 或空：走正常训练流程；显式 feature_counts 分支上方已经训练完成
    if not _using_feature_count_search:
        neg_count = int(np.sum(y_train == 0))
        pos_count = int(np.sum(y_train == 1))
        n_total = max(neg_count + pos_count, 1)
        p_train_pos = pos_count / float(n_total)
        scale_pos_weight_strategy = "balanced_1.0"
        if args.legacy_scale_pos_weight:
            scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
            scale_pos_weight_strategy = "legacy_neg_over_pos"
        elif args.target_deploy_ratio is not None:
            r = float(args.target_deploy_ratio)
            r = min(max(r, 1e-6), 1 - 1e-6)
            if 0.0 < p_train_pos < 1.0:
                scale_pos_weight = (r * (1 - p_train_pos)) / ((1 - r) * p_train_pos)
            else:
                scale_pos_weight = 1.0
            scale_pos_weight_strategy = f"target_deploy_ratio={r}"
        else:
            scale_pos_weight = 1.0

        logger.info("样本分布统计:")
        logger.info(f"  负样本(target=0): {neg_count}")
        logger.info(f"  正样本(target=1): {pos_count}")
        logger.info(f"  train 正类占比 p_train_pos: {p_train_pos:.4f}")
        logger.info(f"  scale_pos_weight 策略: {scale_pos_weight_strategy}")
        logger.info(f"  scale_pos_weight: {scale_pos_weight:.4f}")
        if args.target_deploy_ratio is None and not args.legacy_scale_pos_weight:
            logger.info("  提示：未指定 --target_deploy_ratio。FP 高代价场景建议先估部署条件分布再传入。")

        model_search_records = []
        model_search_summary = {
            "enabled": False,
            "selection_data": None,
            "max_model_nodes": int(args.max_model_nodes),
            "fp_cost": float(args.model_search_fp_cost),
            "size_cost": float(args.model_search_size_cost),
            "fixed_params": build_default_xgb_params(scale_pos_weight=scale_pos_weight),
        }
        if args.model_search:
            raw_model, model_search_summary, model_search_records = search_xgb_hyperparameters(
                args, X_train, y_train, X_model_select, y_model_select,
                scale_pos_weight=scale_pos_weight, groups=train_groups,
            )
            model_search_summary["split"] = model_selection_split
            logger.info("model_search best score=%.6f total_nodes=%d params=%s",
                        model_search_summary["best"]["score"],
                        model_search_summary["best"]["total_nodes"],
                        model_search_summary["best"]["params"])
        else:
            raw_model = train_xgb_with_params(
                build_default_xgb_params(scale_pos_weight=scale_pos_weight),
                X_train, y_train,
            )
            model_search_summary["split"] = model_selection_split

    hard_negative_summary = {
        "enabled": bool(args.mine_hard_negatives),
        "source_split": "train_only",
        "n_hard_negatives": 0,
        "hard_negative_weight": float(args.hard_negative_weight),
        "top_percentile": float(args.hard_negative_top_percentile),
        "min_probability": (
            None if args.hard_negative_min_probability is None
            else float(args.hard_negative_min_probability)
        ),
    }
    hard_negative_report_path = None
    hard_negative_weights_path = None
    hard_negative_decision = None
    model_candidate_decision = None
    if args.mine_hard_negatives:
        df_train_for_hn = clip_outliers(
            df_train_raw, selected_features, k=1.5, bounds=clip_bounds)
        X_train_hn, y_train_hn, _ = prepare_xy(
            df_train_for_hn, selected_features, fill_values=fill_values)
        groups_hn = (
            df_train_for_hn["sample_name"].astype("object").to_numpy()
            if "sample_name" in df_train_for_hn.columns else None
        )
        hn_min_probability = args.hard_negative_min_probability
        threshold_source = "cli"
        if hn_min_probability is None:
            threshold_source = "initial_valid_threshold_split"
            try:
                initial_threshold = search_threshold_by_valid(
                    raw_model,
                    X_threshold,
                    y_threshold,
                    objective=args.threshold_objective,
                    beta=args.threshold_beta,
                    min_precision=args.threshold_min_precision,
                )
                hn_min_probability = float(initial_threshold["threshold"])
            except Exception as e:
                logger.warning("hard-negative mining threshold fallback to 0.5: %s", e)
                threshold_source = "fallback_0.5"
                hn_min_probability = 0.5

        hn_params = dict(
            model_search_summary.get("best", {}).get("params")
            or build_default_xgb_params(scale_pos_weight=scale_pos_weight)
        )
        hn_weights, hn_report, hard_negative_summary = mine_hard_negative_training_weights(
            df_train_for_hn,
            X_train_hn,
            y_train_hn,
            groups_hn,
            hn_params,
            min_probability=hn_min_probability,
            top_percentile=args.hard_negative_top_percentile,
            hard_negative_weight=args.hard_negative_weight,
            n_folds=args.model_search_cv_folds,
            random_state=args.model_search_random_state,
        )
        hard_negative_summary["threshold_source"] = threshold_source
        hard_negative_summary["params_source"] = (
            "model_search_best" if model_search_summary.get("best") else "default_xgb_params"
        )
        hard_negative_report_path = os.path.join(args.artifact_dir, "hard_negative_mining_train.csv")
        hard_negative_weights_path = os.path.join(args.artifact_dir, "hard_negative_training_weights.csv")
        hn_report.to_csv(hard_negative_report_path, index=False)
        weight_cols = [
            c for c in ["sample_name", "h5_file", "window_index", "target", "mode"]
            if c in df_train_for_hn.columns
        ]
        weight_df = df_train_for_hn[weight_cols].copy() if weight_cols else pd.DataFrame(index=df_train_for_hn.index)
        weight_df["sample_weight"] = hn_weights
        weight_df["is_hard_negative"] = hn_weights > 1.0
        weight_df.to_csv(hard_negative_weights_path, index=False)
        hard_negative_summary["report_path"] = hard_negative_report_path
        hard_negative_summary["weights_path"] = hard_negative_weights_path
        logger.info(
            "hard-negative mining: selected %d/%d train rows; retraining final raw model with weights",
            int(hard_negative_summary.get("n_hard_negatives", 0)),
            int(hard_negative_summary.get("n_train_rows", len(y_train_hn))),
        )
        reference_raw_model = raw_model
        hard_negative_raw_model = train_xgb_with_params(
            hn_params,
            X_train_hn,
            y_train_hn,
            sample_weight=hn_weights,
        )
        reference_threshold = search_threshold_by_valid(
            reference_raw_model,
            X_threshold,
            y_threshold,
            objective=args.threshold_objective,
            beta=args.threshold_beta,
            min_precision=args.threshold_min_precision,
        )
        hard_negative_threshold = search_threshold_by_valid(
            hard_negative_raw_model,
            X_threshold,
            y_threshold,
            objective=args.threshold_objective,
            beta=args.threshold_beta,
            min_precision=args.threshold_min_precision,
        )
        reference_record = _candidate_record(
            "reference",
            reference_raw_model,
            eval_model(reference_raw_model, X_valid, y_valid, reference_threshold["threshold"]),
        )
        hard_negative_record = _candidate_record(
            "hard_negative",
            hard_negative_raw_model,
            eval_model(hard_negative_raw_model, X_valid, y_valid, hard_negative_threshold["threshold"]),
        )
        reference_record["threshold"] = float(reference_threshold["threshold"])
        hard_negative_record["threshold"] = float(hard_negative_threshold["threshold"])
        hard_negative_decision = accept_hard_negative_candidate(
            reference_record, hard_negative_record
        )
        raw_model = (
            hard_negative_raw_model
            if hard_negative_decision["accepted"]
            else reference_raw_model
        )
        model_candidate_decision = select_model_candidate(
            [reference_record, hard_negative_record],
            max_nodes=args.max_model_nodes,
            max_fpr=0.01,
        )
        _atomic_json_dump(
            hard_negative_decision,
            os.path.join(args.artifact_dir, "hard_negative_decision.json"),
        )
        _atomic_json_dump(
            model_candidate_decision,
            os.path.join(args.artifact_dir, "model_candidate_leaderboard.json"),
        )
        hard_negative_summary["decision"] = hard_negative_decision
        model_search_summary["hard_negative_mining"] = hard_negative_summary

    # 确保 quality_thresholds 已初始化（多 k 分支可能跳过）
    try:
        _ = quality_thresholds
    except NameError:
        _df_qc, _ = clip_outliers(df_train_raw, selected_features, k=1.5, return_bounds=True)
        quality_thresholds = learn_quality_thresholds(_df_qc, QUALITY_FEATURES_DEFAULT)

    # 确保 scale_pos_weight_strategy 已初始化
    try:
        _ = scale_pos_weight_strategy
    except NameError:
        scale_pos_weight_strategy = "balanced_1.0"

    # 确保 calibration_threshold_split / model_selection_split 已初始化（多 k 分支未设置）
    try:
        _ = calibration_threshold_split
    except NameError:
        calibration_threshold_split = {
            "fallback": True, "reason": "multi_k_full_valid",
            "calibration_groups": None, "threshold_groups": None,
        }
    try:
        _ = model_selection_split
    except NameError:
        model_selection_split = {
            "fallback": True, "reason": "multi_k_search",
            "model_selection_groups": None, "calibration_groups": None,
        }

    # Platt 概率校准 (校准器包裹原始模型)
    model = raw_model
    calibration_meta = {
        "method": args.calibration_method,
        "calibration_data": "valid_calibration_split",
        "threshold_selection_data": "valid_threshold_split",
        "split": calibration_threshold_split,
        "model_selection_split": model_selection_split,
        "applied": False,
    }
    if args.calibration_method == "isotonic":
        try:
            if len(np.unique(y_calib)) < 2:
                raise ValueError("calibration split has a single class")
            from sklearn.calibration import CalibratedClassifierCV
            try:
                from sklearn.frozen import FrozenEstimator
                calib = CalibratedClassifierCV(FrozenEstimator(raw_model), method="isotonic")
            except Exception:
                calib = CalibratedClassifierCV(raw_model, method="isotonic", cv="prefit")
            calib.fit(X_calib, y_calib)
            model = calib
            calibration_meta["applied"] = True
            logger.info("Isotonic calibration applied on valid calibration split")
        except Exception as e:
            model = raw_model
            calibration_meta["applied"] = False
            calibration_meta["reason"] = str(e)
            logger.warning(f"Calibration skipped: {e}")
    else:
        model = raw_model

    # 打印实际节点数，验证 ≤500
    total_nodes = count_xgb_nodes(raw_model)
    avg_nodes = total_nodes / max(raw_model.n_estimators, 1)
    logger.info(f"trained {raw_model.n_estimators} trees, total_nodes={total_nodes}, "
                f"avg_nodes/tree={avg_nodes:.1f} (部署目标 ≤500)")

    valid_default = eval_model(model, X_valid, y_valid, threshold=0.5)
    threshold_default = eval_model(model, X_threshold, y_threshold, threshold=0.5)

    best_threshold = search_threshold_by_valid(
        model,
        X_threshold,
        y_threshold,
        objective=args.threshold_objective,
        beta=args.threshold_beta,
        min_precision=args.threshold_min_precision,
    )
    best_threshold["selection_data"] = "valid_threshold_split"
    best_threshold["calibration_data"] = "valid_calibration_split"

    valid_best = eval_model(
        model,
        X_valid,
        y_valid,
        threshold=best_threshold["threshold"]
    )
    threshold_best = eval_model(
        model,
        X_threshold,
        y_threshold,
        threshold=best_threshold["threshold"]
    )
    threshold_curve = compute_threshold_curve(
        model, X_threshold, y_threshold, beta=args.threshold_beta
    )

    print("\nValid 默认阈值 0.5:")
    print(json.dumps(valid_default, indent=2, ensure_ascii=False))

    print("\nValid 选择出的窗口阈值:")
    print(json.dumps(best_threshold, indent=2, ensure_ascii=False))

    print("\nValid 最优阈值指标:")
    print(json.dumps(valid_best, indent=2, ensure_ascii=False))

    model_path = os.path.join(args.artifact_dir, "final_model.json")
    config_path = os.path.join(args.artifact_dir, "final_model_config.json")
    bundle_path = os.path.join(args.artifact_dir, "model_bundle.pkl")

    raw_model.save_model(model_path)

    fingerprint = build_fingerprint(
        args.artifact_dir, feature_pool_train_path, splits_path
    )

    model_search_results_path = None
    if model_search_records:
        model_search_results_path = os.path.join(args.artifact_dir, "model_search_results.csv")
        for _record in model_search_records:
            _record.setdefault("feature_count", len(selected_features))
            _record.setdefault("n_features", len(selected_features))
            _record.setdefault("deployment_feature_cost_summary", summarize_deployment_feature_costs(selected_features))
        rows = build_model_search_result_rows(model_search_records)
        results_df = pd.DataFrame(rows)
        if "cv_folds_completed" in results_df.columns and int(results_df["cv_folds_completed"].max()) > 0:
            results_df = results_df.sort_values(
                by=[
                    "eligible",
                    "mean_cv_accuracy",
                    "std_cv_accuracy",
                    "mean_cv_fp_rate",
                    "final_total_nodes",
                ],
                ascending=[False, False, True, True, True],
            )
        else:
            results_df = results_df.sort_values(
                by=["eligible", "selection_accuracy", "total_nodes", "selection_fp_rate"],
                ascending=[False, False, True, True],
            )
        model_search_summary["stability"] = summarize_model_search_stability(results_df)
        results_df.to_csv(model_search_results_path, index=False)
        model_search_summary["results_path"] = model_search_results_path

    model_bundle = {
        "version": "v3",
        "feature_pool_version": FEATURE_POOL_VERSION,
        "feature_selection": selection_provenance,
        "feature_names": selected_features,
        "fill_values": fill_values,
        "scaler": None,
        "model": model,  # calibrated model (wraps raw XGBoost)
        "raw_model": raw_model,  # raw XGBoost for tree export
        "threshold": best_threshold["threshold"],
        "threshold_policy": {
            "objective": args.threshold_objective,
            "beta": float(args.threshold_beta),
            "min_precision": float(args.threshold_min_precision),
            "selection_data": "valid_threshold_split",
            "calibration_data": "valid_calibration_split",
            "calibration": calibration_meta,
        },
        "clip_bounds": clip_bounds,            # train 学到的 IQR 边界
        "quality_thresholds": quality_thresholds,  # 给 s06.compute_quality 用
        "feature_quantiles": feature_quantiles,    # 给 s06 OOD 监控用
        "fingerprint": fingerprint,                # provenance
        "meta": {
            "feature_pool_version": FEATURE_POOL_VERSION,
            "feature_selection": selection_provenance,
            "fs_ppg": 25.0,
            "fs_acc": None,
            "win_sec": float(args.window_sec),
            "step_sec": float(args.step_sec),
            "use_stage2_ir": bool(args.use_stage2_ir),
            "model_search": model_search_summary,
            "hard_negative_mining": hard_negative_summary,
        },
    }
    joblib.dump(model_bundle, bundle_path)
    print(f"统一模型包已保存: {bundle_path}")
    print(f"bundle fingerprint: {fingerprint}")

    config = {
        "feature_pool_version": FEATURE_POOL_VERSION,
        "feature_selection": selection_provenance,
        "selected_features": selected_features,
        "selected_feature_count": len(selected_features),
        "fill_values": fill_values,

        "model_path": model_path,
        "window_model_threshold": best_threshold["threshold"],

        "anti_overfit_policy": {
            "model_train_data": "train_only",
            "calibration_data": "valid_calibration_split",
            "threshold_selection_data": "valid_threshold_split",
            "test_used": False,
            "feature_selection_data": (
                "manual_file"
                if args.feature_selection_mode == "manual"
                else (fs.get("selection_policy", {}).get("selection_data", "unknown")
                      if "fs" in dir() else "multi_k_ranked")
            ),
        },

        "class_balance": {
            "neg_count": neg_count,
            "pos_count": pos_count,
            "p_train_pos": float(p_train_pos),
            "scale_pos_weight": float(scale_pos_weight),
            "scale_pos_weight_strategy": scale_pos_weight_strategy,
            "target_deploy_ratio": args.target_deploy_ratio,
        },
        "hard_negative_mining": hard_negative_summary,

        "xgboost_params": raw_model.get_params(),
        "model_complexity": {
            "total_nodes": int(total_nodes),
            "avg_nodes_per_tree": float(avg_nodes),
            "max_model_nodes": int(args.max_model_nodes),
        },
        "model_search": model_search_summary,
        "deployment_feature_cost_summary": summarize_deployment_feature_costs(selected_features),
        "valid_default_threshold_metrics": valid_default,
        "valid_best_threshold_metrics": valid_best,
        "threshold_split_default_metrics": threshold_default,
        "threshold_split_best_metrics": threshold_best,
        "threshold_curve": threshold_curve,
        "threshold_search": best_threshold,
        "threshold_policy": {
            "objective": args.threshold_objective,
            "beta": float(args.threshold_beta),
            "min_precision": float(args.threshold_min_precision),
            "selection_data": "valid_threshold_split",
            "calibration_data": "valid_calibration_split",
            "calibration": calibration_meta,
        },
        "fingerprint": fingerprint,
        "window_sec": float(args.window_sec),
        "step_sec": float(args.step_sec),
        "use_stage2_ir": bool(args.use_stage2_ir),

        "postprocess": {
            "alpha": 0.4,
            "T_on": 0.75,
            "T_off": 0.35,
            "K_on": 5,
            "K_off": 3,
            "cooldown_sec": 5
        }
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n模型已保存: {model_path}")
    print(f"配置已保存: {config_path}")


    export_training_report_plot(config, args.artifact_dir)
    export_threshold_tradeoff_plot(config, args.artifact_dir)
    export_roc_pr_curves(model, X_threshold, y_threshold, args.artifact_dir)


if __name__ == "__main__":
    main()
