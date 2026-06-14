# s10_generalization_audit.py
# -*- coding: utf-8 -*-
"""Commercial deployment generalization audit.

This script is intentionally read-only: it only consumes artifacts already
written by s05/s06/s07 and emits an audit package under
``artifacts/generalization_audit``. It does not train models, replay
postprocess, change thresholds, or mutate existing training outputs.

The audit joins three views that are easy to inspect separately but hard to
reason about together:
1. Stage2 window errors from ``window_error_analysis_*``.
2. Sample/state-machine metrics from ``end_to_end_eval_*``.
3. Hard-negative and model-search context from ``hard_negatives_*`` and
   ``model_search_results.csv``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


OPTIONAL_DIMENSIONS = ["subject_id", "device_id", "session_id"]
GREEN_RELIABILITY_FEATURES = [
    "G_2OF3_AC_SUPPORT",
    "G_TOP2_TO_ALL_AC_RATIO",
    "G_TOP2_CORR_MIN",
    "G_WEAK_CHANNEL_GAP",
    "G_SPATIAL_STABILITY_SCORE",
]
GREEN_RELIABILITY_DIMENSIONS = [
    "green_support_bin",
    "green_top2_ratio_bin",
    "green_top2_corr_bin",
    "green_weak_gap_bin",
    "green_stability_bin",
]
WINDOW_DIMENSIONS = [
    "mode",
    "h5_file",
    "sample_name",
    "record",
    "window_index",
    "time_bin",
    "quality_bin",
    "ood_bin",
] + GREEN_RELIABILITY_DIMENSIONS + OPTIONAL_DIMENSIONS
SAMPLE_DIMENSIONS = [
    "mode",
    "h5_file",
    "sample_name",
    "record",
] + OPTIONAL_DIMENSIONS
STRATA_COLUMNS = [
    "level",
    "dimension",
    "stratum",
    "n_windows",
    "n_samples",
    "accuracy",
    "precision",
    "recall",
    "fp_rate",
    "fn_rate",
    "fp",
    "fn",
    "tp",
    "tn",
    "low_support",
]


def _first_existing(patterns):
    for pattern in patterns:
        matches = sorted(glob.glob(os.fspath(pattern)))
        if matches:
            return matches[-1]
    return None


def _read_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_div(num, den):
    den = float(den)
    if den <= 0:
        return 0.0
    return float(num) / den


def _counts_from_target_pred(df, target_col="target", pred_col="pred_raw"):
    if df.empty or target_col not in df.columns or pred_col not in df.columns:
        return {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
    y = df[target_col].fillna(0).astype(int)
    p = df[pred_col].fillna(0).astype(int)
    return {
        "TN": int(((y == 0) & (p == 0)).sum()),
        "FP": int(((y == 0) & (p == 1)).sum()),
        "FN": int(((y == 1) & (p == 0)).sum()),
        "TP": int(((y == 1) & (p == 1)).sum()),
    }


def _metrics_from_counts(cm):
    tn = int(cm.get("TN", 0))
    fp = int(cm.get("FP", 0))
    fn = int(cm.get("FN", 0))
    tp = int(cm.get("TP", 0))
    total = tn + fp + fn + tp
    return {
        "n": int(total),
        "accuracy": _safe_div(tn + tp, total),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "fp_rate": _safe_div(fp, tn + fp),
        "fn_rate": _safe_div(fn, tp + fn),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }


def _as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        return _as_list(parsed)
    return []


def _finite_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _green_support_bin(value):
    v = _finite_float(value)
    if v is None:
        return "missing"
    if v < 2.0 / 3.0:
        return "<2of3"
    if v < 1.0:
        return "2of3"
    return "3of3"


def _green_corr_bin(value):
    v = _finite_float(value)
    if v is None:
        return "missing"
    if v < 0.50:
        return "low_corr"
    if v < 0.85:
        return "mid_corr"
    return "high_corr"


def _green_gap_bin(value):
    v = _finite_float(value)
    if v is None:
        return "missing"
    if v > 0.60:
        return "large_gap"
    if v > 0.25:
        return "mid_gap"
    return "low_gap"


def _green_stability_bin(value):
    v = _finite_float(value)
    if v is None:
        return "missing"
    if v < 0.35:
        return "low_stability"
    if v < 0.75:
        return "mid_stability"
    return "high_stability"


def add_green_reliability_bins(window_df):
    """Add deployment-readable bins for optional three-green reliability features."""
    if window_df.empty:
        return window_df
    df = window_df.copy()
    if "G_2OF3_AC_SUPPORT" in df.columns:
        df["green_support_bin"] = df["G_2OF3_AC_SUPPORT"].map(_green_support_bin)
    if "G_TOP2_TO_ALL_AC_RATIO" in df.columns:
        df["green_top2_ratio_bin"] = df["G_TOP2_TO_ALL_AC_RATIO"].map(
            lambda v: "single_dominant" if (_finite_float(v) or 0.0) > 0.90 else "balanced_top2"
        )
    if "G_TOP2_CORR_MIN" in df.columns:
        df["green_top2_corr_bin"] = df["G_TOP2_CORR_MIN"].map(_green_corr_bin)
    if "G_WEAK_CHANNEL_GAP" in df.columns:
        df["green_weak_gap_bin"] = df["G_WEAK_CHANNEL_GAP"].map(_green_gap_bin)
    if "G_SPATIAL_STABILITY_SCORE" in df.columns:
        df["green_stability_bin"] = df["G_SPATIAL_STABILITY_SCORE"].map(_green_stability_bin)
    return df


def _first_worn_latencies(sample_df):
    """Return positive-sample response latencies from explicit columns or states.

    s06 details normally keep ``window_states`` and ``window_end_sec`` rather
    than a precomputed latency column, so this derives first-worn latency from
    the first state-machine window whose state becomes 1.
    """
    if sample_df.empty or "target" not in sample_df.columns:
        return []
    positive = sample_df[sample_df["target"].fillna(0).astype(int) == 1]
    latencies = []
    for col in [
        "first_worn_output_sec",
        "first_worn_latency_sec",
        "first_worn_output_latency_sec",
        "first_worn_sec",
    ]:
        if col not in positive.columns:
            continue
        for value in positive[col].tolist():
            parsed = _finite_float(value)
            if parsed is not None:
                latencies.append(parsed)
    if latencies:
        return latencies

    for _, row in positive.iterrows():
        states = _as_list(row.get("window_states"))
        if not states:
            continue
        end_times = _as_list(row.get("window_end_sec"))
        start_times = _as_list(row.get("window_start_sec"))
        times = end_times if end_times else start_times
        for idx, state in enumerate(states):
            state_value = _finite_float(state)
            if state_value is None or int(state_value) != 1:
                continue
            if idx < len(times):
                latency = _finite_float(times[idx])
                if latency is not None:
                    latencies.append(latency)
            break
    return latencies


def summarize_window_metrics(window_df):
    return _metrics_from_counts(_counts_from_target_pred(window_df))


def _pred_col_for_sample_df(sample_df):
    for col in ["pred", "final_pred", "sample_pred", "prediction"]:
        if col in sample_df.columns:
            return col
    return None


def _sample_df_from_eval(eval_payload, window_df):
    details = eval_payload.get("details", []) if isinstance(eval_payload, dict) else []
    if details:
        return pd.DataFrame(details)
    if window_df.empty or "sample_name" not in window_df.columns:
        return pd.DataFrame()
    grouped = []
    for sample_name, sub in window_df.groupby("sample_name", dropna=False):
        row = {
            "sample_name": sample_name,
            "target": int(sub["target"].mode().iloc[0]) if "target" in sub.columns else 0,
            "pred": int((sub.get("pred_raw", pd.Series([0])) == 1).any()),
        }
        for col in SAMPLE_DIMENSIONS:
            if col in sub.columns and col not in row:
                row[col] = sub[col].iloc[0]
        grouped.append(row)
    return pd.DataFrame(grouped)


def summarize_sample_metrics(sample_df):
    if sample_df.empty or "target" not in sample_df.columns:
        return {
            "n": 0,
            "accuracy": 0.0,
            "false_worn_event_rate": 0.0,
            "first_worn_latency_p50_sec": None,
            "first_worn_latency_p95_sec": None,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
        }
    pred_col = _pred_col_for_sample_df(sample_df)
    if pred_col is None:
        pred = pd.Series(np.zeros(len(sample_df), dtype=int), index=sample_df.index)
    else:
        pred = sample_df[pred_col].fillna(0).astype(int)
    tmp = sample_df.copy()
    tmp["_pred"] = pred
    metrics = _metrics_from_counts(_counts_from_target_pred(tmp, pred_col="_pred"))
    latencies = _first_worn_latencies(sample_df)
    metrics.update({
        "false_worn_event_rate": metrics["fp_rate"],
        "first_worn_latency_p50_sec": float(np.percentile(latencies, 50)) if latencies else None,
        "first_worn_latency_p95_sec": float(np.percentile(latencies, 95)) if latencies else None,
    })
    return metrics


def _strata_rows(df, dimensions, min_support, level):
    rows = []
    if df.empty:
        return rows
    pred_col = "pred_raw" if level == "window" else _pred_col_for_sample_df(df)
    if pred_col is None:
        return rows
    for dim in dimensions:
        if dim not in df.columns:
            continue
        for value, sub in df.groupby(dim, dropna=False):
            metrics = _metrics_from_counts(_counts_from_target_pred(sub, pred_col=pred_col))
            n_samples = int(sub["sample_name"].nunique()) if "sample_name" in sub.columns else int(len(sub))
            support_n = n_samples if "sample_name" in sub.columns else int(len(sub))
            row = {
                "level": level,
                "dimension": dim,
                "stratum": str(value),
                "n_windows": int(len(sub)) if level == "window" else None,
                "n_samples": n_samples,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fp_rate": metrics["fp_rate"],
                "fn_rate": metrics["fn_rate"],
                "fp": metrics["confusion_matrix"]["FP"],
                "fn": metrics["confusion_matrix"]["FN"],
                "tp": metrics["confusion_matrix"]["TP"],
                "tn": metrics["confusion_matrix"]["TN"],
                "low_support": bool(support_n < int(min_support)),
            }
            rows.append(row)
    return rows


def build_strata(window_df, sample_df, min_support):
    window_df = add_green_reliability_bins(window_df)
    window_strata = pd.DataFrame(
        _strata_rows(window_df, WINDOW_DIMENSIONS, min_support, "window"),
        columns=STRATA_COLUMNS,
    )
    sample_strata = pd.DataFrame(
        _strata_rows(sample_df, SAMPLE_DIMENSIONS, min_support, "sample"),
        columns=STRATA_COLUMNS,
    )
    return window_strata, sample_strata


def _group_accuracy_variance(df, group_candidates, pred_col):
    if df.empty or "target" not in df.columns or pred_col not in df.columns:
        return {"available": False}
    for group_col in group_candidates:
        if group_col not in df.columns or df[group_col].nunique(dropna=False) <= 1:
            continue
        rows = []
        for value, sub in df.groupby(group_col, dropna=False):
            metrics = _metrics_from_counts(_counts_from_target_pred(sub, pred_col=pred_col))
            rows.append({
                "group": str(value),
                "n": int(len(sub)),
                "accuracy": metrics["accuracy"],
                "fp_rate": metrics["fp_rate"],
                "fn_rate": metrics["fn_rate"],
            })
        acc = np.asarray([r["accuracy"] for r in rows], dtype=float)
        return {
            "available": True,
            "group_col": group_col,
            "n_groups": int(len(rows)),
            "accuracy_mean": float(np.mean(acc)),
            "accuracy_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
            "accuracy_min": float(np.min(acc)),
            "accuracy_max": float(np.max(acc)),
        }
    return {"available": False}


def summarize_group_variance(window_df, sample_df):
    sample_pred_col = _pred_col_for_sample_df(sample_df)
    return {
        "window_by_group": _group_accuracy_variance(
            window_df,
            ["sample_name", "record", "h5_file", "mode"],
            "pred_raw",
        ),
        "sample_by_group": _group_accuracy_variance(
            sample_df,
            ["record", "h5_file", "mode", "sample_name"],
            sample_pred_col,
        ) if sample_pred_col else {"available": False},
    }


def _add_action(items, priority, issue_type, stratum, evidence_metric, n_samples, suggested_action):
    items.append({
        "priority": priority,
        "issue_type": issue_type,
        "stratum": str(stratum),
        "evidence_metric": str(evidence_metric),
        "n_samples": int(n_samples),
        "suggested_action": suggested_action,
    })


def _hard_negative_count(hard_payload):
    fps = hard_payload.get("false_positives", []) if isinstance(hard_payload, dict) else []
    return len(fps), fps


def _is_object_worn_fp(row):
    text = " ".join(
        str(row.get(col, ""))
        for col in ["negative_type", "scene_type", "subject_type", "sample_name", "h5_file", "record"]
        if isinstance(row, dict)
    ).lower()
    return any(token in text for token in [
        "object_worn", "object-worn", "non_human", "non-human", "reflective", "物体", "非人体"
    ])


def _top_rows(df, condition, limit=5):
    if df.empty:
        return []
    sub = df[condition(df)].copy()
    if sub.empty:
        return []
    return sub.head(limit).to_dict("records")


def _model_search_stability_summary(model_search_df, close_margin=0.002):
    if model_search_df is None or model_search_df.empty or "mean_cv_accuracy" not in model_search_df.columns:
        return {"available": False}
    df = model_search_df.copy()
    if "eligible" in df.columns:
        df = df[df["eligible"].astype(bool)]
    df["mean_cv_accuracy"] = pd.to_numeric(df["mean_cv_accuracy"], errors="coerce")
    df = df[df["mean_cv_accuracy"].notna()].copy()
    if df.empty:
        return {"available": False}
    sort_cols = ["mean_cv_accuracy"]
    ascending = [False]
    for col in ["std_cv_accuracy", "mean_cv_fp_rate", "final_total_nodes"]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(True)
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    best_acc = float(df.loc[0, "mean_cv_accuracy"])
    second_acc = float(df.loc[1, "mean_cv_accuracy"]) if len(df) > 1 else None
    margin = None if second_acc is None else float(best_acc - second_acc)
    close_df = df[(best_acc - df["mean_cv_accuracy"].astype(float)) <= float(close_margin)]
    default_rank = None
    if "is_default_params" in df.columns:
        hits = df.index[df["is_default_params"].astype(bool)].tolist()
        if hits:
            default_rank = int(hits[0] + 1)
    return {
        "available": True,
        "best_mean_cv_accuracy": best_acc,
        "second_mean_cv_accuracy": second_acc,
        "top_accuracy_margin": margin,
        "close_top_candidate_count": int(len(close_df)),
        "default_params_rank": default_rank,
        "is_unstable": bool(len(close_df) > 1 or (default_rank is not None and default_rank <= 3)),
    }


def _collect_feature_names(value):
    if isinstance(value, dict):
        out = []
        for key, nested in value.items():
            if key in {"selected_features", "feature_names", "features"}:
                out.extend(_collect_feature_names(nested))
            elif isinstance(nested, (dict, list, tuple)):
                out.extend(_collect_feature_names(nested))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, (dict, list, tuple)):
                out.extend(_collect_feature_names(item))
        return out
    return []


def green_reliability_feature_usage(final_config):
    features = list(dict.fromkeys(_collect_feature_names(final_config)))
    selected = [f for f in features if f in GREEN_RELIABILITY_FEATURES]
    return {
        "known_features": list(GREEN_RELIABILITY_FEATURES),
        "selected_features": selected,
        "selected_count": int(len(selected)),
        "selected_fraction": _safe_div(len(selected), len(features)) if features else 0.0,
    }


def build_action_items(window_strata, sample_strata, hard_payload, model_search_df,
                       window_metrics, min_support):
    """Translate recurring deployment-risk patterns into concrete next actions."""
    items = []
    hard_count, hard_fps = _hard_negative_count(hard_payload)
    if hard_count > 0:
        object_fps = [row for row in hard_fps if _is_object_worn_fp(row)]
        if object_fps:
            _add_action(
                items,
                "P0",
                "object_worn_false_positive_cluster",
                "object_worn/non_human hard negatives",
                f"object_worn_false_positives={len(object_fps)}",
                len(object_fps),
                "Prioritize object-worn/non-human negatives in data collection, hard-negative weighting, and acceptance checks before tuning state-machine latency.",
            )
        _add_action(
            items,
            "P0",
            "hard_negative_fp_cluster",
            "hard_negatives:false_positives",
            f"false_positive_samples={hard_count}",
            hard_count,
            "Prioritize collecting/labeling these negative scenes and add FP-proxy features.",
        )

    if not window_strata.empty:
        for row in _top_rows(
            window_strata,
            lambda df: (df["dimension"].isin(["quality_bin", "ood_bin"]))
            & (df["fn"] > 0)
            & (df["n_windows"] >= 1),
        ):
            _add_action(
                items,
                "P1",
                "fn_low_quality_or_ood",
                f"{row['dimension']}={row['stratum']}",
                f"fn={row['fn']}, fn_rate={row['fn_rate']:.4f}",
                row["n_samples"],
                "Check Stage1/quality gating and add matching positive low-quality/OOD data.",
            )
        for row in _top_rows(
            window_strata,
            lambda df: (df["dimension"].isin(["quality_bin", "ood_bin"]))
            & (df["fp"] > 0)
            & (df["n_windows"] >= 1),
        ):
            _add_action(
                items,
                "P1",
                "fp_low_quality_or_ood",
                f"{row['dimension']}={row['stratum']}",
                f"fp={row['fp']}, fp_rate={row['fp_rate']:.4f}",
                row["n_samples"],
                "Try quality-aware threshold or quality gating before changing the state machine.",
            )
        overall_acc = float(window_metrics.get("accuracy", 0.0))
        for row in _top_rows(
            window_strata,
            lambda df: (df["dimension"] == "mode")
            & (df["accuracy"] < overall_acc - 0.05)
            & (df["n_windows"] >= max(1, int(min_support))),
        ):
            _add_action(
                items,
                "P1",
                "mode_specific_drop",
                f"mode={row['stratum']}",
                f"accuracy={row['accuracy']:.4f}, overall={overall_acc:.4f}",
                row["n_samples"],
                "Inspect mode-specific feature selection or try a mode-specific threshold on valid only.",
            )
        for row in _top_rows(
            window_strata,
            lambda df: (df["dimension"] == "time_bin")
            & ((df["fp"] + df["fn"]) > 0)
            & (df["stratum"].astype(str).str.contains("0-10|early|<", regex=True)),
        ):
            _add_action(
                items,
                "P2",
                "early_window_errors",
                f"time_bin={row['stratum']}",
                f"errors={int(row['fp'] + row['fn'])}",
                row["n_samples"],
                "Review skip_initial_windows, warmup behavior, and window ordering.",
            )
        for row in _top_rows(
            window_strata,
            lambda df: (df["dimension"].isin([
                "green_support_bin",
                "green_top2_ratio_bin",
                "green_top2_corr_bin",
                "green_weak_gap_bin",
                "green_stability_bin",
            ]))
            & (df["fp"] > 0)
            & (
                df["stratum"].astype(str).isin([
                    "<2of3",
                    "single_dominant",
                    "low_corr",
                    "large_gap",
                    "low_stability",
                ])
            ),
        ):
            _add_action(
                items,
                "P1",
                "green_reliability_fp_cluster",
                f"{row['dimension']}={row['stratum']}",
                f"fp={row['fp']}, fp_rate={row['fp_rate']:.4f}",
                row["n_samples"],
                "Inspect hard negatives with poor three-green reliability; consider keeping these features if they reduce FP on valid/test.",
            )

    if not model_search_df.empty and "mean_cv_accuracy" in model_search_df.columns:
        cv_best = pd.to_numeric(model_search_df["mean_cv_accuracy"], errors="coerce").max()
        if np.isfinite(cv_best) and cv_best - float(window_metrics.get("accuracy", 0.0)) > 0.03:
            _add_action(
                items,
                "P1",
                "cv_test_generalization_gap",
                "model_search_vs_test",
                f"best_cv_accuracy={cv_best:.4f}, test_window_accuracy={window_metrics.get('accuracy', 0.0):.4f}",
                int(window_metrics.get("n", 0)),
                "Audit split leakage and record/person/device distribution shift.",
            )
        stability = _model_search_stability_summary(model_search_df)
        if stability.get("available") and stability.get("is_unstable"):
            _add_action(
                items,
                "P2",
                "model_search_unstable_top_candidates",
                "model_search_results",
                (
                    f"top_margin={stability.get('top_accuracy_margin')}, "
                    f"close_top={stability.get('close_top_candidate_count')}, "
                    f"default_rank={stability.get('default_params_rank')}"
                ),
                int(window_metrics.get("n", 0)),
                "Do not trust a single top candidate blindly; inspect top-k stability or increase group-CV repeats.",
            )

    if not items:
        _add_action(
            items,
            "P3",
            "no_major_cluster_detected",
            "overall",
            "no rule triggered",
            int(window_metrics.get("n", 0)),
            "Review summary.md and continue monitoring with larger deployment-like data.",
        )
    return pd.DataFrame(items)


def _path_map(artifact_dir, split, method):
    artifact_dir = Path(artifact_dir)
    return {
        "end_to_end": _first_existing([
            artifact_dir / f"end_to_end_eval_{split}_{method}.json",
            artifact_dir / f"end_to_end_eval_*_{method}.json",
            artifact_dir / "end_to_end_eval_*.json",
        ]),
        "window_error_csv": _first_existing([
            artifact_dir / f"window_error_analysis_{split}_{method}.csv",
            artifact_dir / f"window_error_analysis_*_{method}.csv",
            artifact_dir / "window_error_analysis_*.csv",
        ]),
        "hard_negatives": _first_existing([
            artifact_dir / f"hard_negatives_{split}_{method}.json",
            artifact_dir / f"hard_negatives_*_{method}.json",
            artifact_dir / "hard_negatives_*.json",
        ]),
        "model_search_results": _first_existing([artifact_dir / "model_search_results.csv"]),
        "final_model_config": _first_existing([artifact_dir / "final_model_config.json"]),
    }


def _missing_optional_dimensions(window_df, sample_df):
    missing = []
    for dim in OPTIONAL_DIMENSIONS:
        if dim not in window_df.columns and dim not in sample_df.columns:
            missing.append(dim)
    return missing


def _markdown_summary(summary, action_items):
    lines = [
        "# Generalization Audit",
        "",
        f"- Split: `{summary['split']}`",
        f"- Method: `{summary['method']}`",
        f"- Window accuracy: {summary['window_metrics']['accuracy']:.4f}",
        f"- Window FP rate: {summary['window_metrics']['fp_rate']:.4f}",
        f"- Window FN rate: {summary['window_metrics']['fn_rate']:.4f}",
        f"- Sample false-worn event rate: {summary['sample_metrics']['false_worn_event_rate']:.4f}",
        f"- Positive first-worn latency P95 sec: {summary['sample_metrics']['first_worn_latency_p95_sec']}",
        f"- Window group variance: `{summary['group_level_variance']['window_by_group']}`",
        "",
        "## Top Action Items",
        "",
    ]
    for row in action_items.head(10).to_dict("records"):
        lines.append(
            f"- {row['priority']} `{row['issue_type']}` {row['stratum']}: "
            f"{row['evidence_metric']} -> {row['suggested_action']}"
        )
    lines.append("")
    return "\n".join(lines)


def run_audit(artifact_dir, split="test", method="state_machine", min_support=10):
    """Build the complete read-only audit package for one split/method pair."""
    artifact_dir = Path(artifact_dir)
    out_dir = artifact_dir / "generalization_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _path_map(artifact_dir, split, method)
    eval_payload = _read_json(paths["end_to_end"])
    hard_payload = _read_json(paths["hard_negatives"])
    final_config = _read_json(paths["final_model_config"])
    window_df = _read_csv(paths["window_error_csv"])
    model_search_df = _read_csv(paths["model_search_results"])
    sample_df = _sample_df_from_eval(eval_payload, window_df)

    window_metrics = summarize_window_metrics(window_df)
    sample_metrics = summarize_sample_metrics(sample_df)
    group_level_variance = summarize_group_variance(window_df, sample_df)
    window_strata, sample_strata = build_strata(window_df, sample_df, min_support=min_support)
    action_items = build_action_items(
        window_strata,
        sample_strata,
        hard_payload,
        model_search_df,
        window_metrics,
        min_support=min_support,
    )

    summary = {
        "split": split,
        "method": method,
        "input_paths": {k: (str(v) if v else None) for k, v in paths.items()},
        "missing_optional_dimensions": _missing_optional_dimensions(window_df, sample_df),
        "min_support": int(min_support),
        "window_metrics": window_metrics,
        "sample_metrics": sample_metrics,
        "group_level_variance": group_level_variance,
        "model_search": final_config.get("model_search", {}),
        "green_reliability_feature_usage": green_reliability_feature_usage(final_config),
        "n_window_strata": int(len(window_strata)),
        "n_sample_strata": int(len(sample_strata)),
        "n_action_items": int(len(action_items)),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(
        _markdown_summary(summary, action_items),
        encoding="utf-8",
    )
    window_strata.to_csv(out_dir / "window_strata.csv", index=False)
    sample_strata.to_csv(out_dir / "sample_strata.csv", index=False)
    action_items.to_csv(out_dir / "action_items.csv", index=False)
    return {
        "out_dir": str(out_dir),
        "summary": summary,
        "paths": {
            "summary_json": str(out_dir / "summary.json"),
            "summary_md": str(out_dir / "summary.md"),
            "window_strata": str(out_dir / "window_strata.csv"),
            "sample_strata": str(out_dir / "sample_strata.csv"),
            "action_items": str(out_dir / "action_items.csv"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", default="artifacts")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--method", default="state_machine")
    parser.add_argument("--min_support", type=int, default=10)
    args = parser.parse_args()

    result = run_audit(
        args.artifact_dir,
        split=args.split,
        method=args.method,
        min_support=args.min_support,
    )
    print(f"[OK] generalization_audit -> {result['out_dir']}")
    print(json.dumps(result["paths"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
