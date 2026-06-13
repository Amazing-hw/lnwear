# s07_postprocess_optimize.py
# -*- coding: utf-8 -*-
"""
Postprocess optimizer: reads NPZ window caches, searches richer state machine params
under FP-sensitive multi-objective scoring. No s03/s05/s06 re-run needed.
"""

import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

REQUIRED_NPZ_KEYS = [
    "sample_name", "target", "window_start_sec", "window_end_sec",
    "stage1_enabled", "prob_raw", "pred_raw", "quality", "ood_rate",
    "mode", "fallback", "model_threshold", "window_sec", "stride_sec",
    "cache_schema_version", "model_fingerprint_json", "feature_names_json",
    "skip_initial_windows",
]


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
    for key in ["window_start_sec", "window_end_sec", "stage1_enabled",
                 "pred_raw", "quality", "ood_rate"]:
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
    out["skip_initial_windows"] = int(_scalar(out["skip_initial_windows"]))
    return out


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
    enabled = np.asarray(cache["stage1_enabled"], dtype=int)
    quality = np.asarray(cache["quality"], dtype=float)
    ends = np.asarray(cache["window_end_sec"], dtype=float)
    stride_sec = float(cache.get("stride_sec", 1.0))
    probs = np.where(enabled > 0, probs, 0.0)
    probs = causal_median_filter_1d(probs, int(params.get("median_k", 1)))

    ema = 0.0
    state = 0
    on_count = 0
    off_count = 0
    steps_since_flip = 999
    cooldown_steps = int(float(params.get("cooldown_sec", 0.0)) / stride_sec) if stride_sec > 0 else 0
    states = []
    scores = []
    first_output_sec = None
    positive_output_sec = None
    time_to_correct_sec = None
    for i, p in enumerate(probs):
        decision_time = float(ends[i]) if i < len(ends) else float((i + 1) * stride_sec)
        q = float(np.clip(quality[i] if i < len(quality) else 1.0, 0.0, 1.0))
        alpha = float(params["ema_alpha"]) * q
        ema = alpha * float(p) + (1.0 - alpha) * ema
        steps_since_flip += 1
        scores.append(float(ema))
        if enabled[i] <= 0 or ema < float(params["T_off"]):
            off_count += 1
            on_count = 0
        elif ema >= float(params["T_on"]):
            on_count += 1
            off_count = 0
        else:
            on_count = 0
            off_count = 0
        if state == 0 and on_count >= int(params["K_on"]) and steps_since_flip >= cooldown_steps:
            state = 1
            on_count = 0
            off_count = 0
            steps_since_flip = 0
            if first_output_sec is None:
                first_output_sec = decision_time
            if positive_output_sec is None:
                positive_output_sec = decision_time
        elif state == 1 and off_count >= int(params["K_off"]) and steps_since_flip >= cooldown_steps:
            state = 0
            on_count = 0
            off_count = 0
            steps_since_flip = 0
        elif state == 0 and first_output_sec is None and off_count >= int(params["K_off"]):
            first_output_sec = decision_time
        if first_output_sec is not None and time_to_correct_sec is None and state == int(cache["target"]):
            time_to_correct_sec = decision_time
        states.append(int(state))
    return {
        "sample_name": cache["sample_name"],
        "target": int(cache["target"]),
        "pred": int(state),
        "states": states,
        "state_times_sec": [float(x) for x in ends[:len(states)]],
        "scores": scores,
        "stride_sec": stride_sec,
        "first_output_sec": float(first_output_sec if first_output_sec is not None else np.inf),
        "positive_output_sec": float(positive_output_sec if positive_output_sec is not None else np.inf),
        "time_to_correct_sec": float(time_to_correct_sec if time_to_correct_sec is not None else np.inf),
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


def compute_dataset_metrics(details, params_label=""):
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
    for d in details:
        t = d["target"]
        for s in d.get("states", []):
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
        float(d.get("first_output_sec", np.inf)) for d in details
        if np.isfinite(float(d.get("first_output_sec", np.inf)))
    ]
    first_output_p95 = float(np.percentile(first_output_secs, 95)) if first_output_secs else np.inf
    first_output_mean = float(np.mean(first_output_secs)) if first_output_secs else np.inf
    first_worn_output_secs = [
        float(d.get("positive_output_sec", np.inf)) for d in details
        if d["target"] == 1 and d["pred"] == 1 and np.isfinite(float(d.get("positive_output_sec", np.inf)))
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

    out = {
        "params": params_label,
        "sample_accuracy": sample_accuracy, "sample_precision": sample_precision,
        "sample_recall": sample_recall, "sample_f1": sample_f1,
        "sample_fp_rate": sample_fp_rate, "sample_tp": int(tp), "sample_fp": int(fp),
        "sample_fn": int(fn), "sample_tn": int(tn),
        "window_accuracy": window_accuracy, "window_fp_rate": window_fp_rate,
        "first_output_mean_sec": first_output_mean,
        "first_output_p95_sec": first_output_p95,
        "first_worn_output_p95_sec": first_worn_output_p95,
        "time_to_correct_mean_sec": time_to_correct_mean,
        "false_worn_event_rate": false_worn_event_rate,
        "false_worn_duration_mean_sec": false_worn_duration_mean,
    }
    out.update(early_accuracy)
    return out


def score_metrics(metrics, fp_cost=4.0):
    return (
        1.00 * float(metrics["sample_accuracy"])
        + 0.50 * float(metrics["sample_recall"])
        + 0.75 * float(metrics.get("accuracy_at_8s", 0.0))
        + 0.25 * float(metrics["window_accuracy"])
        - float(fp_cost) * float(metrics["sample_fp_rate"])
        - float(fp_cost) * float(metrics.get("false_worn_event_rate", 0.0))
        - 0.20 * float(metrics.get("false_worn_duration_mean_sec", 0.0))
        - 0.50 * float(metrics["window_fp_rate"])
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

def iter_param_grid():
    for ema_alpha in [0.2, 0.4, 0.6]:
        for median_k in [1, 3]:
            for T_on in [0.55, 0.70, 0.85]:
                for T_off in [0.20, 0.35, 0.50]:
                    if T_on <= T_off:
                        continue
                    for K_on in [3, 5, 8]:
                        for K_off in [2, 3, 5]:
                            if K_on < K_off:
                                continue
                            for cooldown_sec in [0.0, 2.0, 5.0]:
                                yield {
                                    "ema_alpha": ema_alpha,
                                    "median_k": median_k,
                                    "T_on": T_on,
                                    "T_off": T_off,
                                    "K_on": K_on,
                                    "K_off": K_off,
                                    "cooldown_sec": cooldown_sec,
                                }


def _params_label(p):
    return (f"a={p['ema_alpha']:.2f}_mk={p['median_k']}_"
            f"Ton={p['T_on']:.2f}_Toff={p['T_off']:.2f}_"
            f"Kon={p['K_on']}_Koff={p['K_off']}_cd={p.get('cooldown_sec', 0.0):.1f}")


def metrics_with_params(metrics, params):
    out = dict(metrics)
    out.update({
        "param_ema_alpha": float(params["ema_alpha"]),
        "param_median_k": int(params["median_k"]),
        "param_T_on": float(params["T_on"]),
        "param_T_off": float(params["T_off"]),
        "param_K_on": int(params["K_on"]),
        "param_K_off": int(params["K_off"]),
        "param_cooldown_sec": float(params.get("cooldown_sec", 0.0)),
    })
    return out


def evaluate_postprocess_on_caches(caches, params):
    details = [run_postprocess_on_cache(c, params) for c in caches]
    return details, compute_dataset_metrics(details, _params_label(params))


def build_replay_report(best_params, selection_split, selection_caches,
                        replay_split=None, replay_caches=None):
    selection_details, selection_metrics = evaluate_postprocess_on_caches(
        selection_caches, best_params)
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
        },
        "selection": {
            "split": selection_split,
            "n_samples": int(len(selection_caches)),
            "metrics": selection_metrics,
        },
    }
    if replay_split and replay_caches is not None:
        replay_details, replay_metrics = evaluate_postprocess_on_caches(
            replay_caches, best_params)
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
    }


def write_optimized_config(best, out_dir, split, constraints):
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
    }
    metric_keys = [
        "sample_accuracy", "sample_precision", "sample_recall", "sample_f1",
        "sample_fp_rate", "window_accuracy", "window_fp_rate",
        "first_output_p95_sec", "first_worn_output_p95_sec", "score",
        "accuracy_at_8s", "time_to_correct_mean_sec",
        "false_worn_event_rate", "false_worn_duration_mean_sec",
    ]
    config_out = {
        "source": "s07_postprocess_optimize",
        "split": split,
        "constraints": constraints,
        "metrics": {k: float(best[k]) for k in metric_keys if k in best},
        "postprocess": postprocess,
    }
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
    caches = []
    for fn in npz_files:
        caches.append(load_window_cache_npz(os.path.join(cache_dir, fn)))
    return caches, cache_dir


# =========================================================
# Worker globals for parallel evaluation
_WORKER_CACHES = None


def _eval_one_param(params):
    """Evaluate a single parameter combination on all cached samples."""
    caches = _WORKER_CACHES
    if caches is None:
        raise RuntimeError("worker caches not initialized")
    details = [run_postprocess_on_cache(c, params) for c in caches]
    metrics = compute_dataset_metrics(details, _params_label(params))
    return params, metrics


# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Postprocess optimizer on cached NPZ windows")
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--cache_root", type=str, default="window_outputs")
    parser.add_argument("--max_sample_fp_rate", type=float, default=0.02)
    parser.add_argument("--max_false_worn_event_rate", type=float, default=0.02)
    parser.add_argument("--max_first_worn_output_p95_sec", type=float, default=6.0)
    parser.add_argument("--fp_cost", type=float, default=4.0)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--replay_split", type=str, default="",
                        help="optional split to replay the selected postprocess params on, e.g. test")
    args = parser.parse_args()

    cache_dir = resolve_cache_dir(args.artifact_dir, args.cache_root, args.split)
    if not os.path.isdir(cache_dir):
        print(f"[ERROR] cache dir not found: {cache_dir}")
        print("  Run s06_deploy_eval.py --export_window_cache first")
        return

    # Load all caches
    npz_files = sorted(
        [f for f in os.listdir(cache_dir) if f.endswith(".npz")])
    if not npz_files:
        print(f"[ERROR] no NPZ files in {cache_dir}")
        return

    print(f"Loading {len(npz_files)} window caches from {cache_dir}...")
    caches = []
    for fn in npz_files:
        try:
            caches.append(load_window_cache_npz(os.path.join(cache_dir, fn)))
        except Exception as e:
            print(f"  skip {fn}: {e}")
    print(f"  loaded {len(caches)} {args.split} caches")

    # Grid search (parallel)
    grid = list(iter_param_grid())
    print(f"Searching {len(grid)} parameter combinations...")
    n_workers = max(1, int(args.n_workers))
    t0 = time.time()

    # Module-level cache for worker access
    import s07_postprocess_optimize as _s07
    _s07._WORKER_CACHES = caches

    results = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_eval_one_param, p): p for p in grid}
        for fut in as_completed(futures):
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{len(grid)}...")
            try:
                params, metrics = fut.result()
            except Exception as e:
                print(f"  worker error: {e}")
                continue
            constraints = {
                "max_sample_fp_rate": args.max_sample_fp_rate,
                "max_false_worn_event_rate": args.max_false_worn_event_rate,
                "max_first_worn_output_p95_sec": args.max_first_worn_output_p95_sec,
            }
            metrics["is_valid"] = metrics_satisfy_constraints(metrics, constraints)
        metrics["score"] = score_metrics(metrics, fp_cost=args.fp_cost)
        results.append(metrics_with_params(metrics, params))

    dt = time.time() - t0
    print(f"  done in {dt:.1f}s")

    # Select best
    valid = [r for r in results if r["is_valid"]]
    if not valid:
        print(f"[WARN] no config meets max_fp_rate={args.max_sample_fp_rate}, "
              f"max_false_worn_event_rate={args.max_false_worn_event_rate}, "
              f"max_worn_latency={args.max_first_worn_output_p95_sec}s. Relaxing...")
        valid = results

    best = max(valid, key=lambda r: r["score"])
    print(f"\nBest config: {best['params']}")
    print(f"  sample_acc={best['sample_accuracy']:.4f}  "
          f"precision={best['sample_precision']:.4f}  "
          f"recall={best['sample_recall']:.4f}  "
          f"fp_rate={best['sample_fp_rate']:.4f}")
    print(f"  window_acc={best['window_accuracy']:.4f}  "
          f"first_output_p95={best['first_output_p95_sec']:.1f}s  "
          f"score={best['score']:.4f}")

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
            "fp_cost": args.fp_cost,
        },
    )
    print(f"[OK] {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        optimized_config = json.load(f)
    final_config_path = update_final_model_config(args.artifact_dir, optimized_config)
    print(f"[OK] updated final_model_config.json -> {final_config_path}")

    replay_caches = None
    replay_split = args.replay_split.strip()
    if replay_split:
        try:
            replay_caches, replay_cache_dir = load_split_caches(
                args.artifact_dir, args.cache_root, replay_split)
            print(f"  loaded {len(replay_caches)} replay caches from {replay_cache_dir}")
        except Exception as e:
            replay_caches = None
            print(f"[WARN] replay split {replay_split!r} skipped: {e}")
    replay_payload = build_replay_report(
        best_params=_best_params_from_search_row(best),
        selection_split=args.split,
        selection_caches=caches,
        replay_split=replay_split if replay_caches is not None else None,
        replay_caches=replay_caches,
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


if __name__ == "__main__":
    main()
