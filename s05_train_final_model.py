
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
import logging
import joblib
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb

# from sklearn.preprocessing import StandardScaler  # 去掉归一化
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold, train_test_split

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    "max_depth": [2, 3, 4],
    "learning_rate": [0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
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


def clip_outliers(df, columns, k=1.5, bounds=None, return_bounds=False):
    """
    基于 IQR 的异常值裁剪（向量化版）。

    参数:
        df: 输入 DataFrame
        columns: 需要裁剪的列名列表
        k: IQR 倍数，默认 1.5
        bounds: dict[col -> (lower, upper)] 预先计算好的裁剪边界。给定时跳过 IQR 估计直接应用。
        return_bounds: 是否同时返回 bounds 字典（仅 bounds=None 时有意义）。

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
        return (df, bounds) if return_bounds else df

    sub = df[cols]
    q1 = sub.quantile(0.25)
    q3 = sub.quantile(0.75)
    iqr = q3 - q1

    valid_mask = iqr.values > 1e-10
    if not valid_mask.any():
        return (df, {}) if return_bounds else df

    valid_cols = [c for c, ok in zip(cols, valid_mask) if ok]
    q1v = q1[valid_cols]
    q3v = q3[valid_cols]
    lower = q1v - k * (q3v - q1v)
    upper = q3v + k * (q3v - q1v)

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
            })

    if clipped_cols:
        logger.info(f"异常值裁剪统计 (k={k}):")
        for item in clipped_cols:
            logger.info(f"  {item['column']}: lower={item['lower']:.4f}, "
                        f"upper={item['upper']:.4f}, "
                        f"clipped_min={item['clipped_min']}, "
                        f"clipped_max={item['clipped_max']}")

    out_bounds = {c: (float(lower[c]), float(upper[c])) for c in valid_cols}
    return (df, out_bounds) if return_bounds else df


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


def search_threshold_by_valid(model, X_valid, y_valid, objective="f1",
                               beta=0.5, min_precision=None):
    """
    在 valid 上搜窗口阈值。

    objective:
      - "f1"                 : 默认 F1
      - "precision"          : 仅 precision
      - "recall"             : 仅 recall
      - "fbeta"              : F-beta (默认 beta=0.5 偏 precision)
      - "precision_constrained" : 在 precision >= min_precision 约束下最大化 recall

    本项目背景：FP（非佩戴误判为佩戴）代价更高，建议 fbeta(beta=0.5) 或 precision_constrained。
    """
    probs = model.predict_proba(X_valid)[:, 1]
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
        rng = np.random.RandomState(random_state)
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


def train_xgb_with_params(params, X_train, y_train):
    fit_params = dict(params)
    fit_params["n_jobs"] = get_inner_n_jobs()
    model = xgb.XGBClassifier(**fit_params)
    model.fit(X_train, y_train, verbose=False)
    return model


def _search_xgb_hyperparameters_single_split(args, X_train, y_train, X_select, y_select,
                                             scale_pos_weight=1.0):
    grid = build_model_search_grid(args, scale_pos_weight=scale_pos_weight)
    records = []
    models = []
    logger.info("model_search enabled: evaluating %d XGBoost candidates", len(grid))
    for idx, params in enumerate(grid, 1):
        candidate = train_xgb_with_params(params, X_train, y_train)
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
        record = {
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
        records.append(record)
        models.append(candidate)

    best = choose_accuracy_first_model_search_record(
        records,
        accuracy_tolerance=args.model_search_accuracy_tolerance,
    )
    if best is None:
        raise RuntimeError(
            f"model_search found no candidate under max_model_nodes={args.max_model_nodes}. "
            "Relax --max_model_nodes or shrink the search grid."
        )
    best_index = int(best["rank_input_order"]) - 1
    best_model = models[best_index]

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
        "best": _json_safe_model_search_record(best),
        "top_candidates": [_json_safe_model_search_record(r) for r in records[:20]],
    }, records


def _search_xgb_hyperparameters_staged_group_cv(args, X_train, y_train, groups=None,
                                                scale_pos_weight=1.0):
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
    stage_records = []
    logger.info(
        "model_search staged_group_cv: stage A evaluating %d sampled candidates "
        "(total_combinations=%d)",
        len(grid),
        total_combinations,
    )
    for idx, params in enumerate(grid, 1):
        candidate = train_xgb_with_params(params, X_train[stage_train_idx], y_train[stage_train_idx])
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
        stage_records.append({
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
        })

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

    cv_records = []
    final_models = []
    fold_count = len(cv_splits)
    for idx, params in enumerate(stage2_params, 1):
        fold_metrics = []
        for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits, 1):
            fold_model = train_xgb_with_params(params, X_train[train_idx], y_train[train_idx])
            fold_metrics.append(evaluate_accuracy_first_threshold(
                fold_model,
                X_train[valid_idx],
                y_train[valid_idx],
            ))
        cv_summary = summarize_cv_metrics(fold_metrics)
        final_model = train_xgb_with_params(params, X_train, y_train)
        final_total_nodes = count_xgb_nodes(final_model)
        final_models.append(final_model)
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
        cv_records.append(record)

    best = choose_cv_model_search_record(
        cv_records,
        accuracy_tolerance=args.model_search_accuracy_tolerance,
    )
    if best is None:
        raise RuntimeError(
            f"model_search found no candidate under max_model_nodes={args.max_model_nodes}. "
            "Relax --max_model_nodes or shrink the search grid."
        )
    best_model = final_models[int(best["rank_input_order"]) - 1]

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
                               scale_pos_weight=1.0, groups=None):
    strategy = getattr(args, "model_search_strategy", "single_split")
    if strategy == "staged_group_cv":
        return _search_xgb_hyperparameters_staged_group_cv(
            args,
            X_train,
            y_train,
            groups=groups,
            scale_pos_weight=scale_pos_weight,
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
    )


def compute_threshold_curve(model, X, y, beta=0.5):
    probs = model.predict_proba(X)[:, 1]
    rows = []
    for th in np.linspace(0.05, 0.95, 181):
        pred = (probs >= th).astype(int)
        precision = float(precision_score(y, pred, zero_division=0))
        recall = float(recall_score(y, pred, zero_division=0))
        f1 = float(f1_score(y, pred, zero_division=0))
        rows.append({
            "threshold": float(th),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fbeta": float(_fbeta(precision, recall, beta)),
        })
    return rows


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
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] s05 report plot -> {out_path}")
    return out_path


# =========================================================
# 质量阈值 / OOD 分位数 / fingerprint
# =========================================================

QUALITY_FEATURES_DEFAULT = ["Ambient_std", "G_mean_mean", "IR_mean"]


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

    def sha256_head(path, head_bytes=4 * 1024 * 1024):
        """对前 4MB 做 hash，避免大 CSV 全读。"""
        if not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(head_bytes))
        return h.hexdigest()

    info["splits_sha256_head"] = sha256_head(splits_path)
    info["feature_pool_train_sha256_head"] = sha256_head(feature_pool_path)

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
                 calibration_threshold_split, scale_pos_weight):
    """用 top-k 特征训练并返回完整结果。

    返回 dict: model, search_summary, search_records, features, fill_values,
                clip_bounds, X_valid, y_valid, model_select_split, df_calib
    """
    # clip
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

    total_nodes = count_xgb_nodes(raw_model)
    best_score = float(model_search_summary.get("best", {}).get("score", float("-inf")))
    valid_default = eval_model(raw_model, X_valid, y_valid, threshold=0.5)
    valid_acc = float(valid_default.get("accuracy", 0.0))

    logger.info(f"  [k={k}] features={len(features)} nodes={total_nodes} "
                f"valid_acc={valid_acc:.4f} search_score={best_score:.4f}")

    return {
        "k": k, "features": list(features), "model": raw_model,
        "fill_values": fill_values, "clip_bounds": clip_bounds,
        "X_valid": X_valid, "y_valid": y_valid, "df_calib": df_calib,
        "search_summary": model_search_summary, "search_records": model_search_records,
        "valid_acc": valid_acc, "search_score": best_score,
        "total_nodes": total_nodes,
    }


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--max_features", type=int, default=None,
                        help="从 ranked_features.json 取 top-k 特征；默认 None 时回退到 selected_features.json")
    parser.add_argument("--model_search_feature_counts", type=str, default="10,12,15,18,20",
                        help="搜参时测试的特征数量，逗号分隔 (如 10,12,15,18,20)。留空则使用 --max_features 固定值")
    parser.add_argument(
        "--threshold_objective", type=str, default="fbeta",
        choices=["f1", "precision", "recall", "fbeta", "precision_constrained"],
        help="阈值搜索目标。默认 fbeta（偏 precision，因为 FP 代价更高）。"
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
    parser.add_argument("--window_sec", type=float, default=3.0,
                        help="Feature window seconds recorded into model metadata.")
    parser.add_argument("--step_sec", type=float, default=1.0,
                        help="Feature/evaluation stride seconds recorded into model metadata.")
    parser.add_argument("--use_stage2_ir", action=argparse.BooleanOptionalAction, default=False,
                        help="whether Stage2 feature pools used IR channel values")
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
    parser.add_argument("--model_search_max_candidates", type=int, default=600,
                        help="maximum sampled candidates from the search grid; <=0 disables sampling")
    parser.add_argument("--model_search_stage1_top_k", type=int, default=4,
                        help="number of stage-1 structure candidates advanced to stage-2 refine")
    parser.add_argument("--model_search_stage2_top_k", type=int, default=80,
                        help="number of stage-A candidates kept for staged_group_cv")
    parser.add_argument("--model_search_cv_folds", type=int, default=3,
                        help="group-CV folds for staged_group_cv")
    parser.add_argument("--model_search_cv_repeats", type=int, default=2,
                        help="group-CV repeats for staged_group_cv")
    parser.add_argument("--model_search_random_state", type=int, default=42,
                        help="random seed for deterministic candidate sampling and CV splits")
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

    selected_features_path = os.path.join(args.artifact_dir, "selected_features.json")
    ranked_features_path = os.path.join(args.artifact_dir, "ranked_features.json")

    # 若存在 ranked_features.json，从中取 top-k（支持不同 max_features）；
    # 否则回退到 selected_features.json（向后兼容）。
    if os.path.exists(ranked_features_path):
        with open(ranked_features_path, "r", encoding="utf-8") as f:
            ranked = json.load(f)
        _k = min(args.max_features if args.max_features is not None else 15, len(ranked))
        selected_features = [r["feature"] for r in ranked[:_k]]
        logger.info(f"从 ranked_features.json 取 top {_k} 特征（共 {len(ranked)} 个候选）")
    else:
        with open(selected_features_path, "r", encoding="utf-8") as f:
            fs = json.load(f)
        selected_features = fs["selected_features"]

    feature_pool_train_path = os.path.join(args.artifact_dir, "feature_pool_train.csv")
    feature_pool_valid_path = os.path.join(args.artifact_dir, "feature_pool_valid.csv")
    splits_path = os.path.join(args.artifact_dir, "splits.json")

    df_train_raw = pd.read_csv(feature_pool_train_path)
    df_valid_raw = pd.read_csv(feature_pool_valid_path)

    # ── 特征数量搜参：解析 --model_search_feature_counts ──
    _fc_str = getattr(args, "model_search_feature_counts", "") or ""
    _feature_counts = []
    if _fc_str.strip():
        for _part in _fc_str.split(","):
            _part = _part.strip()
            if _part.isdigit():
                _feature_counts.append(int(_part))
        _feature_counts = sorted(set(_feature_counts))

    # ── 多 k 分支 / 单 k 分支 ──
    if _feature_counts and os.path.exists(ranked_features_path):
        # ============ 多 k 搜参 ============
        if not args.model_search:
            logger.warning("model_search_feature_counts 已指定但 model_search 未启用，"
                           "仅用默认 XGBoost 参数遍历不同 k 值。")

        with open(ranked_features_path, "r", encoding="utf-8") as _f:
            _ranked = json.load(_f)

        _max_k = min(max(_feature_counts), len(_ranked))
        _ks = [k for k in _feature_counts if k <= _max_k]
        if not _ks:
            _ks = [min(args.max_features if args.max_features is not None else 15, len(_ranked))]

        logger.info(f"特征数搜参: 测试 k ∈ {_ks}（共 {len(_ks)} 个候选，"
                    f"ranked_features 共 {len(_ranked)} 个）")

        # 前置：计算一次 scale_pos_weight（不随 k 变化）
        _all_features = [r["feature"] for r in _ranked]
        _feats_tmp = _all_features[:min(10, len(_all_features))]
        _df_tmp = clip_outliers(df_train_raw, _feats_tmp, k=1.5)
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
            _result = _train_for_k(
                args, _k, _feats, df_train_raw, df_valid_raw,
                _train_groups, None, _sw,
            )
            _result["search_summary"]["scale_pos_weight"] = _sw
            _score = _result["search_score"] if args.model_search else _result["valid_acc"]
            _result["_combined_score"] = _score
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
            "selection_metric": "search_score" if args.model_search else "valid_accuracy",
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
    if not _feature_counts:
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
        "version": "v2",
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
            "fs_ppg": 25.0,
            "fs_acc": None,
            "win_sec": float(args.window_sec),
            "step_sec": float(args.step_sec),
            "use_stage2_ir": bool(args.use_stage2_ir),
            "model_search": model_search_summary,
        },
    }
    joblib.dump(model_bundle, bundle_path)
    print(f"统一模型包已保存: {bundle_path}")
    print(f"bundle fingerprint: {fingerprint}")

    config = {
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
            "feature_selection_data": (fs.get("selection_policy", {}).get("selection_data", "unknown")
                                         if "fs" in dir() else "multi_k_ranked"),
        },

        "class_balance": {
            "neg_count": neg_count,
            "pos_count": pos_count,
            "p_train_pos": float(p_train_pos),
            "scale_pos_weight": float(scale_pos_weight),
            "scale_pos_weight_strategy": scale_pos_weight_strategy,
            "target_deploy_ratio": args.target_deploy_ratio,
        },

        "xgboost_params": raw_model.get_params(),
        "model_complexity": {
            "total_nodes": int(total_nodes),
            "avg_nodes_per_tree": float(avg_nodes),
            "max_model_nodes": int(args.max_model_nodes),
        },
        "model_search": model_search_summary,
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


if __name__ == "__main__":
    main()
