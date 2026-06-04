
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
from sklearn.model_selection import train_test_split

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


def get_inner_n_jobs(default=1):
    """Thread count for XGBoost training; keep conservative on shared servers."""
    try:
        return max(1, int(os.environ.get("WL_INNER_N_JOBS", default)))
    except (TypeError, ValueError):
        return max(1, int(default))


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
    stratify = group_df["target"] if group_df["target"].value_counts().min() >= 2 else None
    try:
        calib_groups, threshold_groups = train_test_split(
            group_df["sample_name"].values,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        calib_groups, threshold_groups = train_test_split(
            group_df["sample_name"].values,
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
    stratify = group_df["target"] if group_df["target"].value_counts().min() >= 2 else None
    try:
        calib_groups, search_groups = train_test_split(
            group_df["sample_name"].values,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        calib_groups, search_groups = train_test_split(
            group_df["sample_name"].values,
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


def build_model_search_grid(args, scale_pos_weight=1.0):
    axes = {
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
    grid = []
    keys = list(axes.keys())
    for values in product(*(axes[k] for k in keys)):
        params = build_default_xgb_params(scale_pos_weight=scale_pos_weight)
        params.update(dict(zip(keys, values)))
        grid.append(params)
    return grid


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


def _json_safe_float(value):
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
    return out


def train_xgb_with_params(params, X_train, y_train):
    fit_params = dict(params)
    fit_params["n_jobs"] = get_inner_n_jobs()
    model = xgb.XGBClassifier(**fit_params)
    model.fit(X_train, y_train, verbose=False)
    return model


def search_xgb_hyperparameters(args, X_train, y_train, X_select, y_select,
                               scale_pos_weight=1.0):
    grid = build_model_search_grid(args, scale_pos_weight=scale_pos_weight)
    records = []
    best = None
    best_model = None
    logger.info("model_search enabled: evaluating %d XGBoost candidates", len(grid))
    for idx, params in enumerate(grid, 1):
        candidate = train_xgb_with_params(params, X_train, y_train)
        metrics = eval_model(candidate, X_select, y_select, threshold=0.5)
        total_nodes = count_xgb_nodes(candidate)
        score = score_model_search_candidate(
            metrics,
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
            "metrics": metrics,
            "params": params,
        }
        records.append(record)
        if score["eligible"] and (best is None or score["score"] > best["score"]):
            best = record
            best_model = candidate

    if best_model is None:
        raise RuntimeError(
            f"model_search found no candidate under max_model_nodes={args.max_model_nodes}. "
            "Relax --max_model_nodes or shrink the search grid."
        )

    records.sort(key=lambda r: (not r["eligible"], -r["score"], r["total_nodes"]))
    return best_model, {
        "enabled": True,
        "selection_data": "valid_model_selection_split",
        "max_model_nodes": int(args.max_model_nodes),
        "fp_cost": float(args.model_search_fp_cost),
        "size_cost": float(args.model_search_size_cost),
        "grid_size": int(len(grid)),
        "best": _json_safe_model_search_record(best),
        "top_candidates": [_json_safe_model_search_record(r) for r in records[:20]],
    }, records


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


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
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
    parser.add_argument("--max_model_nodes", type=int, default=400,
                        help="maximum allowed total XGBoost tree nodes during --model_search; <=0 disables the cap")
    parser.add_argument("--model_search_fp_cost", type=float, default=2.0,
                        help="FP penalty in model-search score: accuracy - fp_cost*fp_rate - size_cost*size_ratio")
    parser.add_argument("--model_search_size_cost", type=float, default=0.1,
                        help="model-size penalty in model-search score")
    parser.add_argument("--model_search_valid_fraction", type=float, default=0.5,
                        help="fraction of the valid calibration pool reserved for model-search selection")
    parser.add_argument("--model_search_n_estimators", type=str, default="20,30,40",
                        help="comma-separated n_estimators candidates for --model_search")
    parser.add_argument("--model_search_max_depth", type=str, default="2,3",
                        help="comma-separated max_depth candidates for --model_search")
    parser.add_argument("--model_search_learning_rate", type=str, default="0.03,0.05,0.08",
                        help="comma-separated learning_rate candidates for --model_search")
    parser.add_argument("--model_search_min_child_weight", type=str, default="20,30,50",
                        help="comma-separated min_child_weight candidates for --model_search")
    parser.add_argument("--model_search_reg_lambda", type=str, default="10,20",
                        help="comma-separated reg_lambda candidates for --model_search")
    parser.add_argument("--model_search_reg_alpha", type=str, default="1,2",
                        help="comma-separated reg_alpha candidates for --model_search")
    parser.add_argument("--model_search_subsample", type=str, default="0.7,0.8,0.9",
                        help="comma-separated subsample candidates for --model_search")
    parser.add_argument("--model_search_colsample_bytree", type=str, default="0.7,0.8,0.9",
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

    with open(os.path.join(args.artifact_dir, "selected_features.json"), "r", encoding="utf-8") as f:
        fs = json.load(f)

    selected_features = fs["selected_features"]

    feature_pool_train_path = os.path.join(args.artifact_dir, "feature_pool_train.csv")
    feature_pool_valid_path = os.path.join(args.artifact_dir, "feature_pool_valid.csv")
    splits_path = os.path.join(args.artifact_dir, "splits.json")

    df_train_raw = pd.read_csv(feature_pool_train_path)
    df_valid_raw = pd.read_csv(feature_pool_valid_path)

    # 异常值裁剪：从 train 学边界，应用到 train 与 valid（保持分布一致，避免 valid 自带 IQR 泄漏）
    logger.info("应用异常值裁剪 (train 学边界 → train/valid 同步应用)...")
    df_train, clip_bounds = clip_outliers(df_train_raw, selected_features, k=1.5,
                                          return_bounds=True)
    df_valid = clip_outliers(df_valid_raw, selected_features, k=1.5, bounds=clip_bounds)

    # 质量阈值 / OOD 分位数都基于裁剪后的 train，特征池里所有可用的列都覆盖到（quality 看固定 3 个特征）
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
    if args.model_search:
        df_model_select, df_calib, model_selection_split = split_calibration_for_model_search(
            df_calib_pool,
            search_fraction=args.model_search_valid_fraction,
            random_state=args.calibration_random_state,
        )
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
        df_train,
        selected_features,
        fill_values=fill_values
    )

    X_valid, y_valid, _ = prepare_xy(
        df_valid,
        selected_features,
        fill_values=fill_values
    )
    X_calib, y_calib, _ = prepare_xy(
        df_calib,
        selected_features,
        fill_values=fill_values
    )
    X_model_select, y_model_select, _ = prepare_xy(
        df_model_select,
        selected_features,
        fill_values=fill_values
    )
    X_threshold, y_threshold, _ = prepare_xy(
        df_threshold,
        selected_features,
        fill_values=fill_values
    )

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

    # 部署约束: 总节点数 ≤500。深度3满树=15节点，强正则化下实际~8-10。/树
    # 40树 × 8-10 ≈ 320-400 设计节点数
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
            args,
            X_train,
            y_train,
            X_model_select,
            y_model_select,
            scale_pos_weight=scale_pos_weight,
        )
        model_search_summary["split"] = model_selection_split
        logger.info("model_search best score=%.6f total_nodes=%d params=%s",
                    model_search_summary["best"]["score"],
                    model_search_summary["best"]["total_nodes"],
                    model_search_summary["best"]["params"])
    else:
        raw_model = train_xgb_with_params(
            build_default_xgb_params(scale_pos_weight=scale_pos_weight),
            X_train,
            y_train,
        )
        model_search_summary["split"] = model_selection_split

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
        rows = []
        for r in model_search_records:
            row = {
                "rank_input_order": int(r["rank_input_order"]),
                "eligible": bool(r["eligible"]),
                "score": _json_safe_float(r["score"]),
                "fp_rate": _json_safe_float(r["fp_rate"]),
                "size_ratio": _json_safe_float(r["size_ratio"]),
                "total_nodes": int(r["total_nodes"]),
                "avg_nodes_per_tree": _json_safe_float(r["avg_nodes_per_tree"]),
            }
            for name, value in r["metrics"].items():
                if name == "confusion_matrix":
                    cm = value or {}
                    for cm_key in ["TN", "FP", "FN", "TP"]:
                        row[f"cm_{cm_key}"] = int(cm.get(cm_key, 0))
                else:
                    row[f"metric_{name}"] = _json_safe_float(value) if value is not None else None
            for name, value in r["params"].items():
                row[f"param_{name}"] = value
            rows.append(row)
        pd.DataFrame(rows).sort_values(
            by=["eligible", "score", "total_nodes"],
            ascending=[False, False, True],
        ).to_csv(model_search_results_path, index=False)
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
        "fill_values": fill_values,

        "model_path": model_path,
        "window_model_threshold": best_threshold["threshold"],

        "anti_overfit_policy": {
            "model_train_data": "train_only",
            "calibration_data": "valid_calibration_split",
            "threshold_selection_data": "valid_threshold_split",
            "test_used": False,
            "feature_selection_data": fs.get("selection_policy", {}).get("selection_data", "unknown"),
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
