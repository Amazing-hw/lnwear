# s07_postprocess_optimize.py
# -*- coding: utf-8 -*-
"""
Postprocess optimizer: reads NPZ window caches, searches richer state machine params
under FP-sensitive multi-objective scoring. No s03/s05/s06 re-run needed.
"""

import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from scientific_figures import save_scientific_figure
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from s06_deploy_eval import (
    WearStateMachine,
    default_t_on_candidates,
    sample_pred_from_states,
)

REQUIRED_NPZ_KEYS = [
    "sample_name", "target", "window_start_sec", "window_end_sec",
    "prob_raw", "pred_raw", "quality", "ood_rate",
    "mode", "fallback", "model_threshold", "window_sec", "stride_sec",
    "cache_schema_version", "model_fingerprint_json", "feature_names_json",
]

DEFAULT_WARMUP_FRAMES = 5
EXPECTED_CACHE_SCHEMA_VERSION = "xgboost_window_outputs_v1"
CACHE_CONTRACT_FIELDS = (
    "cache_schema_version",
    "model_fingerprint",
    "feature_names",
    "model_threshold",
    "window_sec",
    "stride_sec",
)


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_n_workers(n_workers=None, n_items=None, cap=4):
    """Resolve bounded postprocess-search workers with a serial escape hatch."""
    if _env_flag("WL_FORCE_SERIAL"):
        return 1
    if n_workers is None:
        n_workers = max(1, min(cap, (os.cpu_count() or cap) // 2))
    try:
        resolved = max(1, int(n_workers))
    except (TypeError, ValueError):
        resolved = 1
    if n_items is not None:
        resolved = min(resolved, max(1, int(n_items)))
    return resolved


def _scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr.tolist()


def load_window_cache_npz(path):
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_NPZ_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path} missing required keys: {missing}")
        out = {key: data[key].copy() for key in data.files}

    prob = np.asarray(out["prob_raw"], dtype=float)
    for key in ["window_start_sec", "window_end_sec", "pred_raw", "quality", "ood_rate"]:
        arr = np.asarray(out[key])
        if arr.shape != prob.shape:
            raise ValueError(
                f"{path} key {key} shape {arr.shape} != prob_raw shape {prob.shape}")
    for key in ["window_indices", "window_targets"]:
        if key in out:
            arr = np.asarray(out[key])
            if arr.shape != prob.shape:
                raise ValueError(
                    f"{path} key {key} shape {arr.shape} != prob_raw shape {prob.shape}")

    out["sample_name"] = str(_scalar(out["sample_name"]))
    out["target"] = int(_scalar(out["target"]))
    out["mode"] = int(_scalar(out["mode"]))
    out["fallback"] = int(_scalar(out["fallback"]))
    out["model_threshold"] = float(_scalar(out["model_threshold"]))
    out["window_sec"] = float(_scalar(out["window_sec"]))
    out["stride_sec"] = float(_scalar(out["stride_sec"]))
    out["cache_schema_version"] = str(_scalar(out["cache_schema_version"]))
    out["model_fingerprint"] = json.loads(str(_scalar(out["model_fingerprint_json"])))
    out["feature_names"] = json.loads(str(_scalar(out["feature_names_json"])))
    if not isinstance(out["model_fingerprint"], dict):
        raise ValueError(f"{path} model_fingerprint_json must encode an object")
    if (not isinstance(out["feature_names"], list)
            or not all(isinstance(name, str) for name in out["feature_names"])):
        raise ValueError(f"{path} feature_names_json must encode a string list")
    for key in ["model_threshold", "window_sec", "stride_sec"]:
        if not np.isfinite(out[key]):
            raise ValueError(f"{path} key {key} must be finite")
    if out["window_sec"] <= 0 or out["stride_sec"] <= 0:
        raise ValueError(f"{path} window_sec and stride_sec must be positive")
    return out


def cache_provenance(caches):
    if not caches:
        raise ValueError("cache provenance requires at least one cache")
    first = caches[0]
    return {field: first[field] for field in CACHE_CONTRACT_FIELDS}


def validate_cache_batch(caches, paths=None, model_bundle=None):
    """Validate all NPZ files as one immutable cache/model contract."""
    if not caches:
        raise ValueError("no usable window caches")
    paths = list(paths or [f"cache[{i}]" for i in range(len(caches))])
    if len(paths) != len(caches):
        raise ValueError("cache paths must align with caches")

    reference = cache_provenance(caches)
    if reference["cache_schema_version"] != EXPECTED_CACHE_SCHEMA_VERSION:
        raise ValueError(
            "cache_schema_version must equal "
            f"{EXPECTED_CACHE_SCHEMA_VERSION!r}, got "
            f"{reference['cache_schema_version']!r} in {paths[0]}"
        )
    for cache, path in zip(caches[1:], paths[1:]):
        for field in CACHE_CONTRACT_FIELDS:
            if cache[field] != reference[field]:
                raise ValueError(
                    f"mixed cache contract for {field}: {paths[0]} has "
                    f"{reference[field]!r}, {path} has {cache[field]!r}"
                )

    if model_bundle is not None:
        if ("fingerprint" in model_bundle
                and model_bundle.get("fingerprint") != reference["model_fingerprint"]):
            raise ValueError(
                "cache model_fingerprint does not match current model_bundle "
                f"fingerprint: {reference['model_fingerprint']!r} != "
                f"{model_bundle.get('fingerprint')!r}"
            )
        if ("feature_names" in model_bundle
                and list(model_bundle.get("feature_names") or [])
                != reference["feature_names"]):
            raise ValueError(
                "cache feature_names do not match current model_bundle "
                f"feature_names: {reference['feature_names']!r} != "
                f"{list(model_bundle.get('feature_names') or [])!r}"
            )
    return reference


# =========================================================
# Rich State Machine
# =========================================================

def causal_median_filter_1d(x, k):
    if k <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    k = int(k)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        hi = i + 1
        out[i] = float(np.median(x[lo:hi]))
    return out


def run_postprocess_on_cache(cache, params):
    probs = np.asarray(cache["prob_raw"], dtype=float)
    quality = np.asarray(cache["quality"], dtype=float)
    ends = np.asarray(cache["window_end_sec"], dtype=float)
    stride_sec = float(cache.get("stride_sec", 1.0))
    first_valid_probability_sec = (
        float(ends[0])
        if len(probs) and len(ends)
        else np.inf
    )
    probs = causal_median_filter_1d(probs, int(params.get("median_k", 1)))

    sm = WearStateMachine(
        alpha=float(params["ema_alpha"]),
        T_on=float(params["T_on"]),
        T_off=float(params["T_off"]),
        K_on=int(params["K_on"]),
        K_off=int(params["K_off"]),
        cooldown_sec=float(params.get("cooldown_sec", 0.0)),
    )
    states = []
    scores = []
    state = 0
    first_output_sec = None
    positive_output_sec = None
    time_to_correct_sec = None
    for i, p in enumerate(probs):
        decision_time = float(ends[i]) if i < len(ends) else float((i + 1) * stride_sec)
        q = float(np.clip(quality[i] if i < len(quality) else 1.0, 0.0, 1.0))
        state, score = sm.update(float(p), quality=q, stride_sec=stride_sec)
        output_state = int(state)
        scores.append(float(score))
        if output_state == 1:
            if first_output_sec is None:
                first_output_sec = decision_time
            if positive_output_sec is None:
                positive_output_sec = decision_time
        elif output_state == 0 and first_output_sec is None and sm.off_count >= int(params["K_off"]):
            first_output_sec = decision_time
        if (first_output_sec is not None and time_to_correct_sec is None
                and output_state == int(cache["target"])):
            time_to_correct_sec = decision_time
        states.append(output_state)
    pred = sample_pred_from_states(
        states,
        strategy=params.get("sample_pred_strategy", "final_state"),
        warmup_frames=params.get("sample_pred_warmup_frames", 0),
    )
    return {
        "sample_name": cache["sample_name"],
        "target": int(cache["target"]),
        "pred": int(pred),
        "states": states,
        "window_targets": np.asarray(
            cache.get("window_targets", np.full(len(states), int(cache["target"]))),
            dtype=int,
        )[:len(states)].tolist(),
        "state_times_sec": [float(x) for x in ends[:len(states)]],
        "scores": scores,
        "stride_sec": stride_sec,
        "first_output_sec": float(first_output_sec if first_output_sec is not None else np.inf),
        "positive_output_sec": float(positive_output_sec if positive_output_sec is not None else np.inf),
        "time_to_correct_sec": float(time_to_correct_sec if time_to_correct_sec is not None else np.inf),
        "first_valid_probability_sec": float(first_valid_probability_sec),
        "first_output_latency_sec": float(
            first_output_sec - first_valid_probability_sec
            if first_output_sec is not None and np.isfinite(first_valid_probability_sec)
            else np.inf
        ),
        "positive_output_latency_sec": float(
            positive_output_sec - first_valid_probability_sec
            if positive_output_sec is not None and np.isfinite(first_valid_probability_sec)
            else np.inf
        ),
    }


# =========================================================
# Multi-Objective Scoring
# =========================================================

def _state_at_deadline(detail, deadline_sec):
    first_output = float(detail.get("first_output_sec", np.inf))
    if not np.isfinite(first_output) or first_output > float(deadline_sec):
        return None
    states = list(detail.get("states", []))
    times = list(detail.get("state_times_sec", []))
    if not states or not times:
        return None
    pred = None
    for t, s in zip(times, states):
        if float(t) <= float(deadline_sec):
            pred = int(s)
        else:
            break
    return pred


def _false_worn_duration(detail):
    if int(detail.get("target", 0)) != 0:
        return 0.0
    stride_sec = float(detail.get("stride_sec", 1.0))
    return float(sum(1 for s in detail.get("states", []) if int(s) == 1) * stride_sec)


def _time_to_correct(detail):
    direct = float(detail.get("time_to_correct_sec", np.inf))
    if np.isfinite(direct):
        return direct
    first_output = float(detail.get("first_output_sec", np.inf))
    if not np.isfinite(first_output):
        return np.inf
    target = int(detail.get("target", 0))
    for t, s in zip(detail.get("state_times_sec", []), detail.get("states", [])):
        if float(t) >= first_output and int(s) == target:
            return float(t)
    return np.inf


def compute_dataset_metrics(details, params_label="", warmup_frames=0):
    y_true = np.array([d["target"] for d in details])
    y_pred = np.array([d["pred"] for d in details])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sample_accuracy = float(accuracy_score(y_true, y_pred))
    sample_recall = float(recall_score(y_true, y_pred, zero_division=0))
    sample_precision = float(precision_score(y_true, y_pred, zero_division=0))
    sample_f1 = float(f1_score(y_true, y_pred, zero_division=0))
    n_neg = max(int(tn + fp), 1)
    sample_fp_rate = float(fp) / float(n_neg)

    # Window-level metrics from states
    all_true, all_pred = [], []
    warmup_frames = max(0, int(warmup_frames))
    skipped_warmup_windows = 0
    for d in details:
        window_targets = list(d.get("window_targets", []))
        sample_target = int(d["target"])
        states = list(d.get("states", []))
        start = min(warmup_frames, len(states))
        skipped_warmup_windows += start
        for idx, s in enumerate(states[start:], start=start):
            t = int(window_targets[idx]) if idx < len(window_targets) else sample_target
            all_true.append(t)
            all_pred.append(int(s))
    if all_true:
        all_true = np.array(all_true); all_pred = np.array(all_pred)
        wtn, wfp, wfn, wtp = confusion_matrix(all_true, all_pred, labels=[0, 1]).ravel()
        window_accuracy = float(accuracy_score(all_true, all_pred))
        n_neg_win = max(int(wtn + wfp), 1)
        window_fp_rate = float(wfp) / float(n_neg_win)
    else:
        window_accuracy = 0.0
        window_fp_rate = 0.0

    first_output_secs = [
        float(d.get("first_output_latency_sec", np.inf)) for d in details
        if np.isfinite(float(d.get("first_output_latency_sec", np.inf)))
    ]
    first_output_p95 = float(np.percentile(first_output_secs, 95)) if first_output_secs else np.inf
    first_output_mean = float(np.mean(first_output_secs)) if first_output_secs else np.inf
    first_worn_output_secs = [
        float(d.get("positive_output_latency_sec", np.inf)) for d in details
        if d["target"] == 1 and d["pred"] == 1 and np.isfinite(float(d.get("positive_output_latency_sec", np.inf)))
    ]
    first_worn_output_p95 = (
        float(np.percentile(first_worn_output_secs, 95)) if first_worn_output_secs else np.inf
    )
    time_to_correct_secs = [_time_to_correct(d) for d in details]
    time_to_correct_secs = [x for x in time_to_correct_secs if np.isfinite(x)]
    time_to_correct_mean = (
        float(np.mean(time_to_correct_secs)) if time_to_correct_secs else np.inf
    )
    early_accuracy = {}
    for deadline in [3, 4, 5, 6, 7, 8]:
        correct = 0
        for d in details:
            pred_at_t = _state_at_deadline(d, deadline)
            if pred_at_t is not None and pred_at_t == int(d["target"]):
                correct += 1
        early_accuracy[f"accuracy_at_{deadline}s"] = float(correct / max(len(details), 1))

    neg_details = [d for d in details if int(d["target"]) == 0]
    false_worn_durations = [_false_worn_duration(d) for d in neg_details]
    false_worn_events = [1 if dur > 0 else 0 for dur in false_worn_durations]
    false_worn_event_rate = (
        float(np.mean(false_worn_events)) if false_worn_events else 0.0
    )
    false_worn_duration_mean = (
        float(np.mean(false_worn_durations)) if false_worn_durations else 0.0
    )
    state_flip_count = int(sum(
        sum(int(a) != int(b) for a, b in zip(
            list(d.get("states", []))[:-1], list(d.get("states", []))[1:]
        ))
        for d in details
    ))

    out = {
        "params": params_label,
        "sample_accuracy": sample_accuracy, "sample_precision": sample_precision,
        "sample_recall": sample_recall, "sample_f1": sample_f1,
        "sample_fp_rate": sample_fp_rate, "sample_tp": int(tp), "sample_fp": int(fp),
        "sample_fn": int(fn), "sample_tn": int(tn),
        "window_accuracy": window_accuracy, "window_fp_rate": window_fp_rate,
        "window_warmup_frames": int(warmup_frames),
        "skipped_warmup_windows": int(skipped_warmup_windows),
        "window_total_windows": int(len(all_true)),
        "first_output_mean_sec": first_output_mean,
        "first_output_p95_sec": first_output_p95,
        "first_worn_output_p95_sec": first_worn_output_p95,
        "time_to_correct_mean_sec": time_to_correct_mean,
        "false_worn_event_rate": false_worn_event_rate,
        "false_worn_duration_mean_sec": false_worn_duration_mean,
        "state_flip_count": state_flip_count,
    }
    out["evaluation_semantics"] = "xgboost_postprocess_only_v1"
    out.update(early_accuracy)
    return out


def score_metrics(metrics, fp_cost=1.5):
    return (
        1.00 * float(metrics["sample_accuracy"])
        + 0.60 * float(metrics["sample_recall"])
        + 0.75 * float(metrics.get("accuracy_at_8s", 0.0))
        + 0.25 * float(metrics["window_accuracy"])
        - float(fp_cost) * float(metrics["sample_fp_rate"])
        - float(fp_cost) * float(metrics.get("false_worn_event_rate", 0.0))
        - 0.10 * float(metrics.get("false_worn_duration_mean_sec", 0.0))
        - 0.30 * float(metrics["window_fp_rate"])
        - 0.10 * float(metrics["first_output_p95_sec"]) / 8.0
    )


def metrics_satisfy_constraints(metrics, constraints):
    return (
        float(metrics.get("sample_fp_rate", np.inf))
        <= float(constraints.get("max_sample_fp_rate", np.inf))
        and float(metrics.get("false_worn_event_rate", np.inf))
        <= float(constraints.get("max_false_worn_event_rate", np.inf))
        and float(metrics.get("first_worn_output_p95_sec", np.inf))
        <= float(constraints.get("max_first_worn_output_p95_sec", np.inf))
        and float(metrics.get("accuracy_at_8s", 0.0)) > 0.0
    )


# =========================================================
# Grid Search
# =========================================================

def iter_param_grid(model_threshold=0.5):
    for ema_alpha in [0.2, 0.4, 0.6]:
        for median_k in [1, 3]:
            for T_on in default_t_on_candidates(model_threshold):
                for T_off in [0.20, 0.35, 0.50]:
                    if T_on <= T_off:
                        continue
                    for K_on in [2, 3, 5]:
                        for K_off in [1, 2, 3]:
                            if K_on < K_off:
                                continue
                            for cooldown_sec in [0.0, 2.0, 5.0]:
                                for strategy in [
                                        "final_state",
                                        "majority_state_after_warmup",
                                        "any_worn_after_warmup"]:
                                    yield {
                                        "ema_alpha": ema_alpha,
                                        "median_k": median_k,
                                        "T_on": T_on,
                                        "T_off": T_off,
                                        "K_on": K_on,
                                        "K_off": K_off,
                                        "cooldown_sec": cooldown_sec,
                                        "sample_pred_strategy": strategy,
                                        "sample_pred_warmup_frames": 0,
                                    }


def _postprocess_grid_priority(params):
    """Prefer deploy-stable, FP-safe candidates when a runtime budget is used."""
    strategy_rank = {
        "majority_state_after_warmup": 0,
        "final_state": 1,
        "any_worn_after_warmup": 2,
    }
    return (
        strategy_rank.get(str(params.get("sample_pred_strategy")), 9),
        -float(params["T_on"]),
        -int(params["K_on"]),
        int(params["K_off"]),
        abs(float(params["ema_alpha"]) - 0.4),
        -int(params["median_k"]),
        float(params.get("cooldown_sec", 0.0)),
        -float(params["T_off"]),
    )


def select_postprocess_search_grid(grid, search_budget=240):
    """Return a deterministic, representative subset of the postprocess grid."""
    grid = list(grid)
    budget = int(search_budget or 0)
    if budget <= 0 or budget >= len(grid):
        return grid
    anchors = [grid[0], grid[len(grid) // 2], grid[-1]]
    for strategy in ["final_state", "majority_state_after_warmup", "any_worn_after_warmup"]:
        candidate = next(
            (item for item in grid if item.get("sample_pred_strategy") == strategy),
            None,
        )
        if candidate is not None:
            anchors.append(candidate)
    selected = []
    seen = set()

    def add(params):
        key = tuple(sorted(params.items()))
        if key in seen:
            return False
        seen.add(key)
        selected.append(params)
        return True

    for params in anchors:
        add(params)
    for params in sorted(grid, key=_postprocess_grid_priority):
        if len(selected) >= budget:
            break
        add(params)
    return selected[:budget]


def infer_model_threshold_from_caches(caches, default=0.5):
    values = []
    for cache in caches:
        try:
            values.append(float(cache.get("model_threshold", default)))
        except Exception:
            continue
    if not values:
        return float(default)
    return float(np.median(np.asarray(values, dtype=float)))


def _params_label(p):
    strategy = str(p.get("sample_pred_strategy", "final_state"))
    return (f"strategy={strategy}_a={p['ema_alpha']:.2f}_mk={p['median_k']}_"
            f"Ton={p['T_on']:.2f}_Toff={p['T_off']:.2f}_"
            f"Kon={p['K_on']}_Koff={p['K_off']}_cd={p.get('cooldown_sec', 0.0):.1f}")


def metrics_with_params(metrics, params):
    out = dict(metrics)
    out["parameter_complexity"] = int(
        int(params["median_k"])
        + int(params["K_on"])
        + int(params["K_off"])
        + (1 if float(params.get("cooldown_sec", 0.0)) > 0.0 else 0)
    )
    out.update({
        "param_ema_alpha": float(params["ema_alpha"]),
        "param_median_k": int(params["median_k"]),
        "param_T_on": float(params["T_on"]),
        "param_T_off": float(params["T_off"]),
        "param_K_on": int(params["K_on"]),
        "param_K_off": int(params["K_off"]),
        "param_cooldown_sec": float(params.get("cooldown_sec", 0.0)),
        "param_sample_pred_strategy": str(params.get("sample_pred_strategy", "final_state")),
        "param_sample_pred_warmup_frames": int(params.get("sample_pred_warmup_frames", 0)),
    })
    return out


def evaluate_postprocess_on_caches(caches, params, warmup_frames=0):
    details = [run_postprocess_on_cache(c, params) for c in caches]
    return details, compute_dataset_metrics(
        details, _params_label(params), warmup_frames=warmup_frames)


def build_replay_report(best_params, selection_split, selection_caches,
                        replay_split=None, replay_caches=None,
                        warmup_frames=0):
    selection_details, selection_metrics = evaluate_postprocess_on_caches(
        selection_caches, best_params, warmup_frames=warmup_frames)
    payload = {
        "source": "s07_postprocess_optimize",
        "best_params": {
            "ema_alpha": float(best_params["ema_alpha"]),
            "median_k": int(best_params["median_k"]),
            "T_on": float(best_params["T_on"]),
            "T_off": float(best_params["T_off"]),
            "K_on": int(best_params["K_on"]),
            "K_off": int(best_params["K_off"]),
            "cooldown_sec": float(best_params.get("cooldown_sec", 0.0)),
            "sample_pred_strategy": str(best_params.get("sample_pred_strategy", "final_state")),
            "sample_pred_warmup_frames": int(best_params.get("sample_pred_warmup_frames", 0)),
        },
        "selection": {
            "split": selection_split,
            "n_samples": int(len(selection_caches)),
            "metrics": selection_metrics,
        },
    }
    if replay_split and replay_caches is not None:
        replay_details, replay_metrics = evaluate_postprocess_on_caches(
            replay_caches, best_params, warmup_frames=warmup_frames)
        payload["replay"] = {
            "split": replay_split,
            "n_samples": int(len(replay_caches)),
            "metrics": replay_metrics,
        }
    else:
        payload["replay"] = None
    return payload


def _best_params_from_search_row(best):
    return {
        "ema_alpha": float(best["param_ema_alpha"]),
        "median_k": int(best["param_median_k"]),
        "T_on": float(best["param_T_on"]),
        "T_off": float(best["param_T_off"]),
        "K_on": int(best["param_K_on"]),
        "K_off": int(best["param_K_off"]),
        "cooldown_sec": float(best.get("param_cooldown_sec", 0.0)),
        "sample_pred_strategy": str(best.get("param_sample_pred_strategy", "final_state")),
        "sample_pred_warmup_frames": int(best.get("param_sample_pred_warmup_frames", 0)),
    }


def write_optimized_config(best, out_dir, split, constraints, cache_provenance=None):
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    postprocess = {
        "alpha": float(best["param_ema_alpha"]),
        "median_k": int(best["param_median_k"]),
        "T_on": float(best["param_T_on"]),
        "T_off": float(best["param_T_off"]),
        "K_on": int(best["param_K_on"]),
        "K_off": int(best["param_K_off"]),
        "cooldown_sec": float(best.get("param_cooldown_sec", 0.0)),
        "sample_pred_strategy": str(best.get("param_sample_pred_strategy", "final_state")),
        "sample_pred_warmup_frames": int(best.get("param_sample_pred_warmup_frames", 0)),
    }
    metric_keys = [
        "sample_accuracy", "sample_precision", "sample_recall", "sample_f1",
        "sample_fp_rate", "window_accuracy", "window_fp_rate",
        "window_warmup_frames", "skipped_warmup_windows", "window_total_windows",
        "first_output_p95_sec", "first_worn_output_p95_sec", "score",
        "accuracy_at_8s", "time_to_correct_mean_sec",
        "false_worn_event_rate", "false_worn_duration_mean_sec",
        "state_flip_count", "parameter_complexity",
    ]
    config_out = {
        "source": "s07_postprocess_optimize",
        "split": split,
        "constraints": constraints,
        "metrics": {k: float(best[k]) for k in metric_keys if k in best},
        "postprocess": postprocess,
    }
    if cache_provenance is not None:
        config_out["cache_provenance"] = dict(cache_provenance)
    out_path = os.path.join(out_dir, "postprocess_optimized.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2, ensure_ascii=False)
    return out_path


def update_final_model_config(artifact_dir, optimized_config):
    config_path = os.path.join(os.fspath(artifact_dir), "final_model_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}
    config["postprocess"] = dict(optimized_config["postprocess"])
    config["postprocess_optimization"] = {
        "source": optimized_config.get("source", "s07_postprocess_optimize"),
        "split": optimized_config.get("split"),
        "constraints": optimized_config.get("constraints", {}),
        "metrics": optimized_config.get("metrics", {}),
        "cache_provenance": optimized_config.get("cache_provenance", {}),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return config_path


def resolve_cache_dir(artifact_dir, cache_root, split):
    cache_dir = os.path.join(os.fspath(artifact_dir), cache_root, split)
    if not os.path.isdir(cache_dir) and cache_root == "window_outputs":
        legacy_dir = os.path.join(os.fspath(artifact_dir), "window_cache", split)
        if os.path.isdir(legacy_dir):
            cache_dir = legacy_dir
    return cache_dir


def load_split_caches(artifact_dir, cache_root, split):
    cache_dir = resolve_cache_dir(artifact_dir, cache_root, split)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(cache_dir)
    npz_files = sorted([f for f in os.listdir(cache_dir) if f.endswith(".npz")])
    if not npz_files:
        raise ValueError(f"no NPZ window caches in {cache_dir}")
    caches = []
    cache_paths = []
    for fn in npz_files:
        path = os.path.join(cache_dir, fn)
        caches.append(load_window_cache_npz(path))
        cache_paths.append(path)
    bundle = None
    bundle_path = os.path.join(os.fspath(artifact_dir), "model_bundle.pkl")
    if os.path.isfile(bundle_path):
        try:
            bundle = joblib.load(bundle_path)
        except Exception as exc:
            raise ValueError(
                f"failed to load current model_bundle for cache validation: "
                f"{bundle_path}: {exc}") from exc
    validate_cache_batch(caches, paths=cache_paths, model_bundle=bundle)
    return caches, cache_dir


# =========================================================
# Worker globals for parallel evaluation
_WORKER_CACHES = None
_WORKER_WARMUP_FRAMES = 0


def _init_worker_caches(caches, warmup_frames=0):
    """Initialize per-process cache state for ProcessPoolExecutor workers."""
    global _WORKER_CACHES, _WORKER_WARMUP_FRAMES
    _WORKER_CACHES = caches
    _WORKER_WARMUP_FRAMES = max(0, int(warmup_frames))


def _eval_one_param(params):
    """Evaluate a single parameter combination on all cached samples."""
    caches = _WORKER_CACHES
    if caches is None:
        raise RuntimeError("worker caches not initialized")
    details = [run_postprocess_on_cache(c, params) for c in caches]
    metrics = compute_dataset_metrics(
        details, _params_label(params), warmup_frames=_WORKER_WARMUP_FRAMES)
    return params, metrics


def select_postprocess_candidate(
        records, max_window_fp_rate=0.01, max_added_latency_sec=3.0):
    """Select a causal postprocess candidate by accuracy, FPR, and added latency."""
    records = [dict(record) for record in records]
    if not records:
        raise ValueError("postprocess candidate list is empty")
    for record in records:
        record["fpr_constraint_pass"] = (
            float(record.get("window_fp_rate", np.inf)) <= float(max_window_fp_rate)
        )
        record["latency_constraint_pass"] = (
            float(record.get("first_worn_output_p95_sec", np.inf))
            <= float(max_added_latency_sec)
        )
        record["deployment_constraint_pass"] = bool(
            record["fpr_constraint_pass"] and record["latency_constraint_pass"]
        )
    feasible = [record for record in records if record["deployment_constraint_pass"]]
    pool = feasible if feasible else records
    if feasible:
        key = lambda record: (
            -float(record.get("window_accuracy", 0.0)),
            float(record.get("window_fp_rate", np.inf)),
            float(record.get("first_worn_output_p95_sec", np.inf)),
            int(record.get("state_flip_count", 10**9)),
            int(record.get("parameter_complexity", 10**9)),
            str(record.get("params", "")),
        )
    else:
        key = lambda record: (
            float(record.get("window_fp_rate", np.inf)),
            -float(record.get("window_accuracy", 0.0)),
            float(record.get("first_worn_output_p95_sec", np.inf)),
            int(record.get("state_flip_count", 10**9)),
            int(record.get("parameter_complexity", 10**9)),
            str(record.get("params", "")),
        )
    selected = min(pool, key=key)
    return {
        "selected_params": selected.get("params"),
        "deployment_acceptance": bool(feasible),
        "status": "deployment_candidate" if feasible else "analysis_only",
        "selection_reason": (
            "max_streaming_accuracy_within_fpr_and_latency_constraints"
            if feasible else "no_candidate_met_fpr_and_latency_constraints"
        ),
        "max_window_fp_rate": float(max_window_fp_rate),
        "max_added_latency_sec": float(max_added_latency_sec),
        "leaderboard": sorted(
            records, key=lambda record: record.get("params") != selected.get("params")
        ),
    }


# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Postprocess optimizer on cached NPZ windows")
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--cache_root", type=str, default="window_outputs")
    parser.add_argument("--max_sample_fp_rate", type=float, default=0.02)
    parser.add_argument("--max_false_worn_event_rate", type=float, default=0.02)
    parser.add_argument("--max_window_fp_rate", type=float, default=0.01)
    parser.add_argument("--max_first_worn_output_p95_sec", type=float, default=3.0)
    parser.add_argument("--fp_cost", type=float, default=1.5)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--search_budget", type=int, default=240,
                        help="maximum postprocess candidates to evaluate; <=0 keeps the full grid")
    parser.add_argument("--warmup_frames", type=int, default=DEFAULT_WARMUP_FRAMES,
                        help="skip the first N state-machine windows when computing window-level postprocess metrics")
    parser.add_argument("--replay_split", type=str, default="",
                        help="optional split to replay the selected postprocess params on, e.g. test")
    args = parser.parse_args()

    if args.split != "valid":
        parser.error("selection split must be 'valid'; use --replay_split test for read-only evaluation")

    caches, cache_dir = load_split_caches(
        args.artifact_dir, args.cache_root, args.split)
    provenance = cache_provenance(caches)
    print(f"Loading {len(caches)} window caches from {cache_dir}...")
    print(f"  loaded {len(caches)} {args.split} caches")

    # Grid search (parallel)
    model_threshold = infer_model_threshold_from_caches(caches)
    full_grid = list(iter_param_grid(model_threshold=model_threshold))
    grid = select_postprocess_search_grid(full_grid, search_budget=args.search_budget)
    print(
        f"Searching {len(grid)} parameter combinations "
        f"(full_grid={len(full_grid)}, search_budget={args.search_budget}, "
        f"model_threshold={model_threshold:.4f})..."
    )
    n_workers = resolve_n_workers(args.n_workers, n_items=len(grid))
    t0 = time.time()

    _init_worker_caches(caches, args.warmup_frames)

    results = []
    n_done = 0
    constraints = {
        "max_sample_fp_rate": args.max_sample_fp_rate,
        "max_false_worn_event_rate": args.max_false_worn_event_rate,
        "max_first_worn_output_p95_sec": args.max_first_worn_output_p95_sec,
    }
    worker_errors = []
    if n_workers == 1:
        for params in grid:
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(grid)}...")
            try:
                params, metrics = _eval_one_param(params)
            except Exception as e:
                worker_errors.append(str(e))
                if len(worker_errors) <= 5:
                    print(f"  worker error: {e}")
                continue
            metrics["is_valid"] = metrics_satisfy_constraints(metrics, constraints)
            metrics["score"] = score_metrics(metrics, fp_cost=args.fp_cost)
            results.append(metrics_with_params(metrics, params))
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker_caches,
            initargs=(caches, args.warmup_frames),
        ) as ex:
            futures = {ex.submit(_eval_one_param, p): p for p in grid}
            for fut in as_completed(futures):
                n_done += 1
                if n_done % 200 == 0:
                    print(f"  {n_done}/{len(grid)}...")
                try:
                    params, metrics = fut.result()
                except Exception as e:
                    worker_errors.append(str(e))
                    if len(worker_errors) <= 5:
                        print(f"  worker error: {e}")
                    continue
                metrics["is_valid"] = metrics_satisfy_constraints(metrics, constraints)
                metrics["score"] = score_metrics(metrics, fp_cost=args.fp_cost)
                results.append(metrics_with_params(metrics, params))

    if not results:
        unique_errors = sorted(set(worker_errors))[:5]
        raise RuntimeError(
            "Postprocess grid search produced no successful candidates. "
            f"Loaded caches={len(caches)}, candidates={len(grid)}, "
            f"first_errors={unique_errors}"
        )

    dt = time.time() - t0
    print(f"  done in {dt:.1f}s")

    # Select the deployable candidate using window-level risk and added latency.
    decision = select_postprocess_candidate(
        results,
        max_window_fp_rate=args.max_window_fp_rate,
        max_added_latency_sec=args.max_first_worn_output_p95_sec,
    )
    best = next(
        record for record in decision["leaderboard"]
        if record.get("params") == decision["selected_params"]
    )
    print(f"\nBest config: {best['params']}")
    print(f"  sample_acc={best['sample_accuracy']:.4f}  "
          f"precision={best['sample_precision']:.4f}  "
          f"recall={best['sample_recall']:.4f}  "
          f"fp_rate={best['sample_fp_rate']:.4f}")
    print(f"  window_acc={best['window_accuracy']:.4f}  "
          f"first_output_p95={best['first_output_p95_sec']:.1f}s  "
          f"score={best['score']:.4f}")
    print(f"  deployment_status={decision['status']}  "
          f"window_fp_rate={best['window_fp_rate']:.4f}  "
          f"added_latency_p95={best['first_worn_output_p95_sec']:.3f}s")

    # Export
    out_dir = os.path.join(args.artifact_dir, "postprocess_opt")
    os.makedirs(out_dir, exist_ok=True)

    config_path = write_optimized_config(
        best,
        out_dir,
        split=args.split,
        constraints={
            "max_sample_fp_rate": args.max_sample_fp_rate,
            "max_false_worn_event_rate": args.max_false_worn_event_rate,
            "max_first_worn_output_p95_sec": args.max_first_worn_output_p95_sec,
            "max_window_fp_rate": args.max_window_fp_rate,
            "fp_cost": args.fp_cost,
            "warmup_frames": args.warmup_frames,
        },
        cache_provenance=provenance,
    )
    print(f"[OK] {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        optimized_config = json.load(f)
    optimized_config["selection_decision"] = {
        key: value for key, value in decision.items() if key != "leaderboard"
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(optimized_config, f, indent=2, ensure_ascii=False)
    if decision["deployment_acceptance"]:
        final_config_path = update_final_model_config(args.artifact_dir, optimized_config)
        print(f"[OK] updated final_model_config.json -> {final_config_path}")
    else:
        print("[WARN] analysis-only postprocess result; final_model_config.json was not updated")

    replay_caches = None
    replay_split = args.replay_split.strip()
    if replay_split:
        replay_caches, replay_cache_dir = load_split_caches(
            args.artifact_dir, args.cache_root, replay_split)
        validate_cache_batch(
            caches + replay_caches,
            paths=(
                [f"{args.split}:{cache['sample_name']}" for cache in caches]
                + [f"{replay_split}:{cache['sample_name']}" for cache in replay_caches]
            ),
        )
        print(f"  loaded {len(replay_caches)} replay caches from {replay_cache_dir}")
    replay_payload = build_replay_report(
        best_params=_best_params_from_search_row(best),
        selection_split=args.split,
        selection_caches=caches,
        replay_split=replay_split if replay_caches is not None else None,
        replay_caches=replay_caches,
        warmup_frames=args.warmup_frames,
    )
    replay_path = os.path.join(
        out_dir,
        f"postprocess_replay_{args.split}"
        + (f"_to_{replay_split}" if replay_caches is not None else "")
        + ".json",
    )
    with open(replay_path, "w", encoding="utf-8") as f:
        json.dump(replay_payload, f, indent=2, ensure_ascii=False)
    print(f"[OK] replay report -> {replay_path}")

    # Full search results
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    df.to_csv(os.path.join(out_dir, "postprocess_search_results.csv"), index=False)
    print(f"[OK] {out_dir}/postprocess_search_results.csv")

    # Export search visualisation
    try:
        export_postprocess_search_plots(df, out_dir)
    except Exception as e:
        print(f"[WARN] postprocess search plot failed: {e}")


def export_postprocess_search_plots(df_results, out_dir):
    """Export multi-panel postprocess parameter search visualisation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip s07 plot: {e}")
        return None

    df = df_results.copy() if hasattr(df_results, "copy") else pd.DataFrame(df_results)
    if len(df) < 2:
        print("[WARN] fewer than 2 search candidates, skip s07 viz")
        return None

    metric_cols = [
        "sample_accuracy", "sample_recall", "sample_fp_rate",
        "window_accuracy", "false_worn_event_rate",
        "first_worn_output_p95_sec", "score",
    ]
    available = [c for c in metric_cols if c in df.columns]
    df_plot = df[available].copy()
    for c in available:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    df_plot = df_plot.dropna(subset=["score"] if "score" in df_plot.columns else available[:1])
    if len(df_plot) < 2:
        print("[WARN] insufficient clean rows for s07 viz")
        return None

    best = df_plot.iloc[0]
    out_path = os.path.join(str(out_dir), "postprocess_search_summary.png")
    os.makedirs(str(out_dir), exist_ok=True)

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30)

    def _scatter(ax, x_col, y_col, xlabel, ylabel, title, best_x=None, best_y=None):
        x = df_plot[x_col].values
        y = df_plot[y_col].values
        c = df_plot["score"].values if "score" in df_plot.columns else np.arange(len(x))
        sc = ax.scatter(x, y, c=c, cmap="viridis_r", alpha=0.6, s=18, edgecolors="none")
        if best_x is not None and best_y is not None and np.isfinite(best_x) and np.isfinite(best_y):
            ax.scatter([best_x], [best_y], marker="*", s=280, color="#d35f2d",
                       edgecolors="#222222", linewidths=0.8, zorder=5, label="best")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.18)
        return sc

    fp_col = "sample_fp_rate"
    rec_col = "sample_recall"
    lat_col = "first_worn_output_p95_sec"

    # (0,0) FP Rate vs Recall
    ax1 = fig.add_subplot(gs[0, 0])
    if fp_col in df_plot.columns and rec_col in df_plot.columns:
        _scatter(ax1, rec_col, fp_col,
                 "Sample Recall", "Sample FP Rate", "FP Rate vs Recall",
                 best_x=best.get(rec_col), best_y=best.get(fp_col))
        ax1.invert_yaxis()
    else:
        ax1.text(0.5, 0.5, "FP/Recall columns missing", ha="center", va="center")
        ax1.set_axis_off()

    # (0,1) Latency vs FP Rate
    ax2 = fig.add_subplot(gs[0, 1])
    if lat_col in df_plot.columns and fp_col in df_plot.columns:
        _scatter(ax2, lat_col, fp_col,
                 "P95 First-Worn Latency (s)", "Sample FP Rate", "Latency vs FP Rate",
                 best_x=best.get(lat_col), best_y=best.get(fp_col))
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "Latency/FP columns missing", ha="center", va="center")
        ax2.set_axis_off()

    # (0,2) Score distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if "score" in df_plot.columns:
        scores = df_plot["score"].dropna().values
        ax3.hist(scores, bins=min(40, len(scores) // 4), color="#4c78a8", alpha=0.7, edgecolor="white")
        ax3.axvline(best["score"], color="#d35f2d", linewidth=2, linestyle="--",
                    label=f"best={best['score']:.4f}")
        ax3.set_xlabel("Composite Score")
        ax3.set_ylabel("Count")
        ax3.set_title("Score Distribution")
        ax3.grid(alpha=0.18)
        ax3.legend(frameon=False)
    else:
        ax3.text(0.5, 0.5, "No score column", ha="center", va="center")
        ax3.set_axis_off()

    # (1,0) T_on / T_off heatmap
    ax4 = fig.add_subplot(gs[1, 0])
    param_cols = [c for c in df.columns if c in {"T_on", "T_off", "K_on", "K_off",
                                                   "ema_alpha", "median_k", "cooldown_sec"}]
    t_on_vals = sorted(df["T_on"].dropna().unique()) if "T_on" in df.columns else []
    t_off_vals = sorted(df["T_off"].dropna().unique()) if "T_off" in df.columns else []
    if len(t_on_vals) >= 2 and len(t_off_vals) >= 2:
        heatmap = np.full((len(t_off_vals), len(t_on_vals)), np.nan)
        for i, toff in enumerate(t_off_vals):
            for j, ton in enumerate(t_on_vals):
                mask = (df["T_on"] == ton) & (df["T_off"] == toff)
                if mask.any() and "score" in df.columns:
                    heatmap[i, j] = df.loc[mask, "score"].mean()
        im = ax4.imshow(heatmap, aspect="auto", cmap="viridis_r", origin="lower")
        ax4.set_xticks(range(len(t_on_vals)))
        ax4.set_xticklabels([f"{v:.2f}" for v in t_on_vals])
        ax4.set_yticks(range(len(t_off_vals)))
        ax4.set_yticklabels([f"{v:.2f}" for v in t_off_vals])
        ax4.set_xlabel("T_on")
        ax4.set_ylabel("T_off")
        ax4.set_title("Score by T_on × T_off")
        plt.colorbar(im, ax=ax4, fraction=0.046)
    else:
        ax4.text(0.5, 0.5, "Insufficient T_on/T_off values for heatmap",
                 ha="center", va="center")
        ax4.set_axis_off()
        ax4.set_title("T_on × T_off")

    # (1,1) K_on / K_off heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    k_on_vals = sorted(df["K_on"].dropna().unique()) if "K_on" in df.columns else []
    k_off_vals = sorted(df["K_off"].dropna().unique()) if "K_off" in df.columns else []
    if len(k_on_vals) >= 2 and len(k_off_vals) >= 2:
        heatmap_k = np.full((len(k_off_vals), len(k_on_vals)), np.nan)
        for i, koff in enumerate(k_off_vals):
            for j, kon in enumerate(k_on_vals):
                mask = (df["K_on"] == kon) & (df["K_off"] == koff)
                if mask.any() and "score" in df.columns:
                    heatmap_k[i, j] = df.loc[mask, "score"].mean()
        im2 = ax5.imshow(heatmap_k, aspect="auto", cmap="viridis_r", origin="lower")
        ax5.set_xticks(range(len(k_on_vals)))
        ax5.set_xticklabels([str(v) for v in k_on_vals])
        ax5.set_yticks(range(len(k_off_vals)))
        ax5.set_yticklabels([str(v) for v in k_off_vals])
        ax5.set_xlabel("K_on")
        ax5.set_ylabel("K_off")
        ax5.set_title("Score by K_on × K_off")
        plt.colorbar(im2, ax=ax5, fraction=0.046)
    else:
        ax5.text(0.5, 0.5, "Insufficient K_on/K_off values for heatmap",
                 ha="center", va="center")
        ax5.set_axis_off()
        ax5.set_title("K_on × K_off")

    # (1,2) Constraint filtering breakdown
    ax6 = fig.add_subplot(gs[1, 2])
    constraint_cols = {
        "window_fp_rate": ("max_window_fp_rate", 0.01),
        "false_worn_event_rate": ("max_false_worn_event_rate", 0.02),
        "first_worn_output_p95_sec": ("max_first_worn_output_p95_sec", 3.0),
    }
    counts = {"total": len(df)}
    cum_pass = np.ones(len(df), dtype=bool)
    for col, (_, thresh) in constraint_cols.items():
        if col in df.columns:
            valid = df[col].notna()
            cum_pass = cum_pass & valid & (df[col].fillna(float("inf")) <= thresh)
        label = col.replace("sample_", "").replace("_", " ").title()
        counts[label] = int(cum_pass.sum())

    if len(counts) > 1:
        names = list(counts.keys())
        vals = list(counts.values())
        colors = ["#4c78a8"] + ["#c44e52"] * (len(counts) - 2) + ["#2f6f73"]
        ax6.bar(range(len(names)), vals, color=colors[:len(names)], width=0.6)
        ax6.set_xticks(range(len(names)))
        ax6.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax6.set_ylabel("Candidates Passed")
        ax6.set_title("Constraint Filtering Cascade")
        ax6.grid(axis="y", alpha=0.18)
        for i, v in enumerate(vals):
            ax6.text(i, v + max(1, max(vals) * 0.02), str(v), ha="center", fontsize=9)
    else:
        ax6.text(0.5, 0.5, "No constraint columns found",
                 ha="center", va="center")
        ax6.set_axis_off()

    fig.suptitle("Postprocess Parameter Search Summary", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_scientific_figure(
        fig,
        out_path,
        source_data=df,
        inputs=[os.path.splitext(out_path)[0] + "_source_data.csv"],
        core_conclusion=(
            "Causal postprocessing is selected for maximum window accuracy only among "
            "candidates meeting the window FPR and added-latency deployment constraints."
        ),
        panel_map={
            "a": "Recall versus false-positive rate.",
            "b": "Added response latency versus false-positive rate.",
            "c": "Candidate score distribution.",
            "d": "Window accuracy versus window false-positive rate.",
            "e": "Top candidate metrics.",
            "f": "Deployment constraint filtering cascade.",
        },
        split="valid",
        n_definition="one row per deterministic postprocess parameter candidate",
        statistics={"selection": "validation-only constrained deterministic ranking"},
        reviewer_risks=["Test data must not participate in parameter selection."],
    )
    plt.close(fig)
    print(f"[OK] s07 postprocess search plot -> {out_path}")
    return out_path


if __name__ == "__main__":
    main()
