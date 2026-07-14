# -*- coding: utf-8 -*-
"""Compare the commercial baseline with the deployed project pipeline.

Commercial baseline:
  Stage1: IR DC gate only.
  Stage2: 5 s windows at 25 Hz, 8 engineered features, AdaBoost.

Project pipeline:
  Stage1: deployed IR DC + AC/DC gate.
  Stage2: the selected deploy feature extractor exported by s06 plus the
  deployed XGBoost model bundle.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from s03_extract_feature_pool import (
    _downsample_ppg,
    _is_25hz_sample,
    detect_green_mode,
    downsample_to_5hz,
    fft_peak_features,
    get_channels_from_window,
    is_prewindowed_signal,
    load_acc,
    load_grouped_window_metadata,
    load_ppg,
    moving_average_filter,
    normalized_autocorr,
    preprocess_signal,
    robust_mad,
    safe_corr,
    stage1_sample_pass,
    validate_h5_file,
)
from s06_deploy_eval import (
    DEFAULT_POSTPROCESS_CONFIG,
    STAGE1_FS,
    STAGE1_GATE_K,
    STAGE1_PRIMITIVE_SEC,
    Stage1StreamingGate,
    _advance_stage1_gate_to_step,
    apply_postprocess,
    compute_window_model_metrics,
    compute_window_stream_metrics,
    fuse_stage1_stage2_states,
    get_deploy_stage1_threshold,
    load_bundle,
    resolve_stage1_gate_flags,
    resolve_use_stage2_ir,
    sample_pred_from_states,
)
from scientific_figures import save_scientific_figure


EPS = 1e-12
FEATURE_FS = 25
COMMERCIAL_WIN_SEC = 5
COMMERCIAL_STRIDE_SEC = 1
DEFAULT_SKIP_INITIAL_WINDOWS = 3
COMMERCIAL_FEATURE_NAMES = [
    "GREEN_CORR",
    "GREEN_AC",
    "AMB_AC",
    "ACC_YSUM",
    "GREEN_DC",
    "AMB_DC",
    "GREEN_XCORR",
    "FFT_PEAK_MEDIAN_RATIO",
]

PLOT_COLORS = {
    "ours": "#2563eb",
    "commercial": "#f97316",
    "positive": "#16a34a",
    "negative": "#64748b",
    "danger": "#dc2626",
}


class CommercialStage1Gate:
    """Streaming DC-only Stage1 gate used by the commercial baseline."""

    def __init__(self, dc_threshold: float, K: int = 3):
        self.dc_threshold = float(dc_threshold)
        self.K = int(K)
        self.stage2_enabled = False
        self.pass_count = 0
        self.fail_count = 0

    def _check_one(self, ir_5hz_window) -> bool:
        x = np.asarray(ir_5hz_window, dtype=float)
        if len(x) >= 2:
            dc = float(np.min((x[:-1] + x[1:]) / 2.0))
        elif len(x) == 1:
            dc = float(x[0])
        else:
            dc = 0.0
        return dc > self.dc_threshold

    def update(self, ir_5hz_window) -> bool:
        ok = self._check_one(ir_5hz_window)
        if ok:
            self.pass_count += 1
            self.fail_count = 0
            if self.pass_count >= self.K:
                self.stage2_enabled = True
        else:
            self.fail_count += 1
            self.pass_count = 0
            if self.fail_count >= self.K:
                self.stage2_enabled = False
        return self.stage2_enabled


def _safe_float(value, default: float = 0.0) -> float:
    if value is None or not np.isfinite(value):
        return float(default)
    return float(value)


def _safe_confusion(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def _metrics(y_true, y_pred, count_key: str = "total"):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return {
            count_key: 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
        }
    return {
        count_key: int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": _safe_confusion(y_true, y_pred),
    }


def _adaboost_classifier(random_state: int = 42):
    tree = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    try:
        return AdaBoostClassifier(estimator=tree, n_estimators=16, random_state=random_state)
    except TypeError:
        model = AdaBoostClassifier(n_estimators=16, random_state=random_state)
        model.set_params(base_estimator=tree)
        return model


def _make_png_path(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def extract_8_commercial_features(ir, ambient, g1, g2, g3, acc_window=None, fs: int = FEATURE_FS):
    """Extract the 8 commercial features from one 25 Hz PPG window."""
    _, amb_bp, _ = preprocess_signal(ambient, fs)
    g1_raw, g1_bp, _ = preprocess_signal(g1, fs)
    g2_raw, g2_bp, _ = preprocess_signal(g2, fs)
    g3_raw, g3_bp, _ = preprocess_signal(g3, fs)

    g_mean_raw = (g1_raw + g2_raw + g3_raw) / 3.0
    g_mean_bp = (g1_bp + g2_bp + g3_bp) / 3.0

    ma_win = max(2, int(round(0.15 * fs)))
    g_smooth = moving_average_filter(g_mean_bp, window_size=ma_win)
    green_corr = safe_corr(g_mean_bp, g_smooth)

    green_ac = 0.5 * float(np.sqrt(np.mean(g_mean_bp**2))) + 0.5 * robust_mad(g_mean_bp) * 1.4826
    amb_ac = 0.5 * float(np.sqrt(np.mean(amb_bp**2))) + 0.5 * robust_mad(amb_bp) * 1.4826

    if acc_window is not None and len(acc_window) >= 4:
        acc_arr = np.asarray(acc_window, dtype=float)
        acc_mag = np.sqrt(np.sum(acc_arr**2, axis=1) + EPS)
        acc_ysum = float(np.mean(acc_mag))
    else:
        acc_ysum = 0.0

    ac = normalized_autocorr(g_mean_bp)
    lag_min = max(1, int(fs * 60.0 / 180.0))
    lag_max = min(len(ac) - 1, int(fs * 60.0 / 40.0))
    green_xcorr = float(np.max(ac[lag_min:lag_max + 1])) if lag_max > lag_min else 0.0

    peak_ratio, _ = fft_peak_features(g_mean_bp, fs, fmin=0.5, fmax=5.0)

    values = [
        green_corr,
        green_ac,
        amb_ac,
        acc_ysum,
        float(np.median(g_mean_raw)),
        float(np.median(np.asarray(ambient, dtype=float))),
        green_xcorr,
        peak_ratio,
    ]
    return [_safe_float(v) for v in values]


def _load_sample_arrays(sample):
    sample_name = sample.get("sample_name", "unknown")
    h5_file = sample.get("h5_file")
    if h5_file is None:
        raise ValueError("missing h5_file")
    ok, err = validate_h5_file(h5_file, sample_name)
    if not ok:
        raise ValueError(err)
    return load_ppg(sample), load_acc(sample)


def _prewindow_targets(sample, n_windows):
    """Return one label per pre-windowed record, preserving grouped metadata."""
    metadata = load_grouped_window_metadata(sample)
    labels = list(metadata.get("window_labels", [])) if metadata else []
    if labels and len(labels) != int(n_windows):
        raise ValueError(
            f"sample {sample.get('sample_name', 'unknown')} grouped-window label count "
            f"{len(labels)} does not match PPG window count {int(n_windows)}"
        )
    if not labels:
        labels = [int(sample.get("target", 0))] * int(n_windows)
    return [int(value) for value in labels]


def _to_25hz(sample, ppg, acc):
    native_25hz = _is_25hz_sample(sample)
    ppg_src_fs = 25 if native_25hz else 100
    if native_25hz:
        ppg_25 = np.asarray(ppg, dtype=np.float64)
        acc_25 = np.asarray(acc, dtype=np.float64) if acc is not None and len(acc) > 0 else None
    else:
        ppg_25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS)
        acc_25 = None
        if acc is not None and len(acc) > 0:
            acc_25 = resample_poly(np.asarray(acc, dtype=np.float32), FEATURE_FS, 100, axis=0).astype(np.float64)
    return ppg_25, acc_25, ppg_src_fs


def _slice_acc(acc_25, start, size):
    if acc_25 is None or start >= len(acc_25):
        return None
    end = min(start + size, len(acc_25))
    return acc_25[start:end]


def _prewindow_to_25hz(sample, window, window_sec):
    n_samples = int(window.shape[0])
    # Detect native 25 Hz by matching against common window lengths:
    #   3s @ 25Hz = 75,  5s @ 25Hz = 125,  3s @ 100Hz = 300,  5s @ 100Hz = 500
    # Also check the sample name hint and the caller-expected length.
    looks_25hz = (
        _is_25hz_sample(sample)
        or n_samples == int(round(float(window_sec) * FEATURE_FS))
        or (n_samples <= 200 and n_samples > 0 and n_samples % FEATURE_FS == 0)
    )
    looks_100hz = (
        n_samples == int(round(float(window_sec) * 100))
        or (n_samples > 200 and n_samples % 100 == 0)
    )
    if looks_25hz and not looks_100hz:
        return np.asarray(window, dtype=np.float64), 25
    if looks_100hz and not looks_25hz:
        return _downsample_ppg(np.asarray(window, dtype=np.float64), src_fs=100, tgt_fs=FEATURE_FS), 100
    # Ambiguous: fall back to the caller-expected length check
    if n_samples >= int(round(float(window_sec) * FEATURE_FS)):
        return np.asarray(window, dtype=np.float64), 25
    return _downsample_ppg(np.asarray(window, dtype=np.float64), src_fs=100, tgt_fs=FEATURE_FS), 100


def _commercial_stage1_window_pass(window, dc_threshold, ppg_src_fs):
    ir_5hz = downsample_to_5hz(window[:, 0], ppg_src_fs, STAGE1_FS)
    s1_win = int(round(STAGE1_PRIMITIVE_SEC * STAGE1_FS))
    if len(ir_5hz) < s1_win:
        return False
    gate = CommercialStage1Gate(dc_threshold, K=STAGE1_GATE_K)
    enabled = False
    for start in range(0, len(ir_5hz) - s1_win + 1, s1_win):
        enabled = bool(gate.update(ir_5hz[start:start + s1_win]))
    return enabled


def infer_one_sample_commercial(sample, dc_threshold: float):
    """Run both commercial stages in parallel for one sample.

    Stage1 records a fast gate only. Stage2 features are extracted for every
    valid evaluation window, including windows for which the gate is closed.
    """
    base = {
        "sample_name": sample.get("sample_name", "unknown"),
        "target": int(sample.get("target", 0)),
        "stage1_pass": False,
        "features": [],
        "window_targets": [],
        "stage1_gate_flags": [],
        "stage2_enabled_flags": [],
        "fallback": False,
        "fallback_reason": None,
    }
    try:
        ppg, acc = _load_sample_arrays(sample)
        if is_prewindowed_signal(ppg):
            base["window_targets"] = _prewindow_targets(sample, ppg.shape[0])
            mode = detect_green_mode(ppg)
            for idx in range(ppg.shape[0]):
                raw_window = ppg[idx]
                window_25, ppg_src_fs = _prewindow_to_25hz(sample, raw_window, window_sec=COMMERCIAL_WIN_SEC)
                enabled = _commercial_stage1_window_pass(raw_window, dc_threshold, ppg_src_fs)
                base["stage1_gate_flags"].append(int(enabled))
                base["stage2_enabled_flags"].append(int(enabled))
                try:
                    ir, ambient, g1, g2, g3 = get_channels_from_window(window_25, mode)
                    acc_seg = None
                    if acc is not None and is_prewindowed_signal(acc) and idx < acc.shape[0]:
                        acc_seg, _ = _prewindow_to_25hz(sample, acc[idx], window_sec=COMMERCIAL_WIN_SEC)
                    base["features"].append(extract_8_commercial_features(ir, ambient, g1, g2, g3, acc_seg))
                except Exception:
                    base["features"].append(None)
            base["stage1_pass"] = bool(any(base["stage1_gate_flags"]))
            return base
        ppg_25, acc_25, ppg_src_fs = _to_25hz(sample, ppg, acc)
    except Exception as exc:
        base["fallback"] = True
        base["fallback_reason"] = f"load: {exc}"
        return base

    ir_5hz = downsample_to_5hz(ppg[:, 0], ppg_src_fs, STAGE1_FS)
    s1_win = int(round(STAGE1_PRIMITIVE_SEC * STAGE1_FS))
    s1_stride = s1_win
    s2_win = COMMERCIAL_WIN_SEC * FEATURE_FS
    s2_stride = COMMERCIAL_STRIDE_SEC * FEATURE_FS
    n_s1 = (len(ir_5hz) - s1_win) // s1_stride + 1
    n_s2 = (len(ppg_25) - s2_win) // s2_stride + 1
    n_steps = max(0, n_s2)
    base["window_targets"] = [base["target"]] * n_steps

    mode = detect_green_mode(ppg)
    gate = CommercialStage1Gate(dc_threshold, K=STAGE1_GATE_K)
    last_s1_step = -1
    for step in range(n_steps):
        s2_start = step * s2_stride
        target_s1_step = int(np.floor(s2_start / FEATURE_FS + 1e-9))
        if target_s1_step >= n_s1:
            break
        enabled, last_s1_step = _advance_stage1_gate_to_step(
            gate, ir_5hz, s1_win, s1_stride, last_s1_step, target_s1_step
        )
        base["stage1_gate_flags"].append(int(enabled))
        base["stage2_enabled_flags"].append(int(enabled))
        try:
            window = ppg_25[s2_start:s2_start + s2_win, :]
            ir, ambient, g1, g2, g3 = get_channels_from_window(window, mode)
            acc_seg = _slice_acc(acc_25, s2_start, s2_win)
            base["features"].append(extract_8_commercial_features(ir, ambient, g1, g2, g3, acc_seg))
        except Exception:
            base["features"].append(None)
    base["stage1_pass"] = bool(any(base["stage1_gate_flags"]))
    return base


def collect_commercial_training_windows(samples, dc_threshold: float):
    X, y = [], []
    diag = {
        "total_samples": len(samples),
        "fallback_samples": 0,
        "stage1_pass_windows": 0,
        "stage1_fail_windows": 0,
        "extract_error_windows": 0,
        "invalid_feature_windows": 0,
        "valid_windows": 0,
    }
    for sample in samples:
        result = infer_one_sample_commercial(sample, dc_threshold)
        if result.get("fallback"):
            diag["fallback_samples"] += 1
            continue
        enabled_flags = resolve_stage1_gate_flags(result, len(result.get("features", [])))
        for idx, feature_vec in enumerate(result.get("features", [])):
            stage2_enabled = enabled_flags[idx] if idx < len(enabled_flags) else False
            if stage2_enabled:
                diag["stage1_pass_windows"] += 1
            else:
                diag["stage1_fail_windows"] += 1
            if feature_vec is None:
                diag["extract_error_windows"] += 1
                continue
            if len(feature_vec) == len(COMMERCIAL_FEATURE_NAMES) and np.isfinite(feature_vec).all():
                X.append(feature_vec)
                window_targets = result.get("window_targets", [])
                y.append(int(window_targets[idx]) if idx < len(window_targets) else int(result["target"]))
                diag["valid_windows"] += 1
            else:
                diag["invalid_feature_windows"] += 1
    if not X:
        msg_lines = [
            "commercial baseline has no valid training windows",
            f"  total samples: {diag['total_samples']}",
            f"  fallback/load-failure samples: {diag['fallback_samples']}",
            f"  Stage1-open windows (diagnostic only): {diag['stage1_pass_windows']}",
            f"  Stage1-closed windows (diagnostic only): {diag['stage1_fail_windows']}",
            f"  feature extraction errors: {diag['extract_error_windows']}",
            f"  invalid (shape/NaN) features: {diag['invalid_feature_windows']}",
            f"  valid windows collected: {diag['valid_windows']}",
            f"  dc_threshold used: {dc_threshold}",
            "",
            "常见原因与排查:",
            "  1. 训练集样本数不足或全部加载失败 — 检查 splits.json 中 train split 是否包含有效 H5",
            "  2. 窗口太短无法提取 8 个商业特征 — 确认 window_sec >= 5 且采样率元数据正确",
            "  3. 特征含 NaN/inf 或通道布局无法解析 — 检查 H5 通道、mode 和预切窗形状",
            "  注：Stage1 DC 阈值仅用于门控诊断，不会过滤商业 Stage2 的训练窗口。",
        ]
        raise RuntimeError("\n".join(msg_lines))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def train_commercial_model(train_samples, dc_threshold: float, random_state: int = 42):
    X_train, y_train = collect_commercial_training_windows(train_samples, dc_threshold)
    model = _adaboost_classifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model, {
        "n_windows": int(len(X_train)),
        "n_positive_windows": int(np.sum(y_train == 1)),
        "n_negative_windows": int(np.sum(y_train == 0)),
    }


def _attach_commercial_probs(raw_results, model):
    out = []
    for raw in raw_results:
        row = dict(raw)
        features = row.get("features", [])
        probs = [0.0] * len(features)
        valid_idx = [i for i, f in enumerate(features) if f is not None]
        if valid_idx:
            X = np.asarray([features[i] for i in valid_idx], dtype=float)
            pred_probs = model.predict_proba(X)[:, 1]
            for local_i, idx in enumerate(valid_idx):
                probs[idx] = float(pred_probs[local_i])
        row["window_probs"] = probs
        out.append(row)
    return out


def _finalize_commercial_results(raw_results, threshold: float):
    details = []
    for raw in raw_results:
        probs = [float(p) for p in raw.get("window_probs", [])]
        window_preds = [int(p >= threshold) for p in probs]
        gate_flags = resolve_stage1_gate_flags(raw, len(window_preds))
        output_states = fuse_stage1_stage2_states(window_preds, gate_flags)
        if raw.get("fallback") or not window_preds:
            stage2_pred = 0
            final_pred = 0
        else:
            stage2_pred = int(np.mean(window_preds) >= 0.5)
            final_pred = int(np.mean(output_states) >= 0.5)
        details.append({
            "sample_name": raw.get("sample_name"),
            "target": int(raw.get("target", 0)),
            "pred": int(final_pred),
            "stage2_pred": int(stage2_pred),
            "stage1_pred": int(bool(any(gate_flags))),
            "stage1_pass": bool(any(gate_flags)),
            "fallback": bool(raw.get("fallback", False)),
            "fallback_reason": raw.get("fallback_reason"),
            "window_probs": probs,
            "window_preds": window_preds,
            "window_targets": [int(value) for value in raw.get("window_targets", [])],
            "window_states": [int(s) for s in output_states],
            "stage2_states": [int(s) for s in window_preds],
            "output_states": [int(s) for s in output_states],
            "window_scores": [],
            "stage1_gate_flags": list(gate_flags),
            "stage2_enabled_flags": list(gate_flags),
            "n_windows": int(len(probs)),
        })
    return details


def metrics_from_details(details, count_key: str = "total_samples"):
    return _metrics([d["target"] for d in details], [d["pred"] for d in details], count_key=count_key)


def window_metrics_from_details(details, count_key: str = "total_windows"):
    y_true, y_pred = [], []
    for detail in details:
        target = int(detail.get("target", 0))
        targets = detail.get("window_targets", [])
        for idx, pred in enumerate(detail.get("window_preds", [])):
            y_true.append(int(targets[idx]) if idx < len(targets) else target)
            y_pred.append(int(pred))
    return _metrics(y_true, y_pred, count_key=count_key)


def _apply_state_machine_to_details(details, threshold: float, postprocess_cfg):
    cfg = dict(DEFAULT_POSTPROCESS_CONFIG)
    if postprocess_cfg:
        cfg.update(postprocess_cfg)
    for detail in details:
        probs = [float(p) for p in detail.get("window_probs", [])]
        if detail.get("fallback") or len(probs) == 0:
            detail["stream_pred"] = 0
            detail["stage2_stream_pred"] = 0
            detail["window_states"] = []
            detail["stage2_states"] = []
            detail["output_states"] = []
            detail["window_scores"] = []
            continue
        _pred, states, _window_preds, scores = apply_postprocess(
            probs,
            [{} for _ in probs],
            "state_machine",
            cfg,
            threshold,
        )
        gate_flags = resolve_stage1_gate_flags(detail, len(states))
        output_states = fuse_stage1_stage2_states(states, gate_flags)
        strategy = cfg.get("sample_pred_strategy", DEFAULT_POSTPROCESS_CONFIG["sample_pred_strategy"])
        warmup_frames = cfg.get(
            "sample_pred_warmup_frames",
            DEFAULT_POSTPROCESS_CONFIG["sample_pred_warmup_frames"],
        )
        detail["stage2_stream_pred"] = int(sample_pred_from_states(states, strategy, warmup_frames))
        detail["stream_pred"] = int(sample_pred_from_states(output_states, strategy, warmup_frames))
        detail["window_states"] = [int(s) for s in output_states]
        detail["stage2_states"] = [int(s) for s in states]
        detail["output_states"] = [int(s) for s in output_states]
        detail["window_scores"] = [float(s) for s in scores]
    return cfg


def _build_eval_payload(details, postprocess_cfg, model_threshold, warmup_frames=5):
    cfg = dict(DEFAULT_POSTPROCESS_CONFIG)
    if postprocess_cfg:
        cfg.update(postprocess_cfg)
    try:
        warmup_frames = max(0, int(warmup_frames))
    except (TypeError, ValueError):
        warmup_frames = 5
    summary = metrics_from_details(details)
    stage1_summary = _metrics(
        [int(d.get("target", 0)) for d in details],
        [int(d.get("stage1_pred", d.get("stage1_pass", False))) for d in details],
    )
    stage2_summary = _metrics(
        [int(d.get("target", 0)) for d in details],
        [int(d.get("stage2_pred", d.get("pred", 0))) for d in details],
    )
    summary.update({
        "parallel_semantics_version": "stage1_mask_stage2_continuous_v1",
        "stage1_only": stage1_summary,
        "stage2_independent": stage2_summary,
        "fused_output": dict(summary),
    })
    window_model_summary = compute_window_model_metrics(details)
    window_stream_summary = compute_window_stream_metrics(
        details,
        cfg,
        warmup_frames=warmup_frames,
        model_threshold=model_threshold,
    )
    return {
        "summary": summary,
        "window_model_summary": window_model_summary,
        "window_stream_summary": window_stream_summary,
        # Backward-compatible aliases for existing report consumers.
        "sample_metrics": summary,
        "window_metrics": window_model_summary,
        "details": details,
    }


def select_commercial_threshold(raw_valid_results, fp_cost: float = 1.5):
    best = {"threshold": 0.5, "score": -np.inf, "metrics": None}
    thresholds = np.linspace(0.05, 0.95, 91)
    for threshold in thresholds:
        details = _finalize_commercial_results(raw_valid_results, float(threshold))
        metrics = metrics_from_details(details)
        cm = metrics["confusion_matrix"]
        neg = max(1, cm["TN"] + cm["FP"])
        fp_rate = cm["FP"] / neg
        score = metrics["f1"] + 0.5 * metrics["precision"] - fp_cost * fp_rate
        if score > best["score"]:
            best = {"threshold": float(threshold), "score": float(score), "metrics": metrics}
    return best


def evaluate_commercial_model(model, threshold: float, dc_threshold: float, samples, warmup_frames: int = 5):
    raw = [infer_one_sample_commercial(sample, dc_threshold) for sample in samples]
    raw = _attach_commercial_probs(raw, model)
    details = _finalize_commercial_results(raw, threshold)
    postprocess_cfg = {
        "sample_pred_warmup_frames": DEFAULT_POSTPROCESS_CONFIG["K_on"],
        "sample_pred_strategy": "final_state",
    }
    _apply_state_machine_to_details(details, threshold, postprocess_cfg)
    return _build_eval_payload(details, postprocess_cfg, threshold, warmup_frames=warmup_frames)


def _load_deploy_extractor(artifact_dir: Path):
    candidates = [
        artifact_dir / "deploy_feature_extractor.py",
        artifact_dir / "deploy_package" / "deploy_feature_extractor.py",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "deploy_feature_extractor.py not found. Run s07 through the deploy export step first."
        )
    spec = importlib.util.spec_from_file_location("deploy_feature_extractor_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load deploy feature extractor from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Failed to execute deploy feature extractor from {path}: {exc}\n"
            "The generated script may be corrupted. Re-run s08 --export_deploy to regenerate it."
        ) from exc
    return module, path


def load_project_artifacts(artifact_dir: Path, method: str = "state_machine"):
    bundle_path = artifact_dir / "model_bundle.pkl"
    threshold_path = artifact_dir / "stage1_threshold.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing model bundle: {bundle_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"missing stage1 threshold: {threshold_path}")

    bundle = load_bundle(str(bundle_path))
    extractor, extractor_path = _load_deploy_extractor(artifact_dir)
    if list(getattr(extractor, "FEATURE_ORDER", [])) != list(bundle["feature_names"]):
        raise ValueError("deploy extractor FEATURE_ORDER does not match model_bundle feature_names")

    with open(threshold_path, "r", encoding="utf-8") as f:
        try:
            stage1_threshold = get_deploy_stage1_threshold(json.load(f))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Failed to parse stage1_threshold.json ({threshold_path}): {exc}"
            ) from exc

    postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)
    final_config_path = artifact_dir / "final_model_config.json"
    if final_config_path.exists():
        try:
            with open(final_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            postprocess_cfg.update(config.get("postprocess", {}))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Failed to read final_model_config.json, using defaults: {exc}")

    return {
        "bundle": bundle,
        "bundle_path": str(bundle_path),
        "extractor": extractor,
        "extractor_path": str(extractor_path),
        "stage1_threshold": stage1_threshold,
        "postprocess_cfg": postprocess_cfg,
        "method": method,
    }


def infer_one_sample_project(sample, artifacts, window_sec=None, stride_sec=None,
                             skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
                             use_stage2_ir=None):
    bundle = artifacts["bundle"]
    extractor = artifacts["extractor"]
    stage1 = artifacts["stage1_threshold"]
    method = artifacts["method"]
    postprocess_cfg = artifacts["postprocess_cfg"]
    model_threshold = float(bundle["threshold"])
    use_stage2_ir = resolve_use_stage2_ir(bundle, use_stage2_ir)
    window_sec = float(window_sec if window_sec is not None else bundle["meta"]["win_sec"])
    stride_sec = float(stride_sec if stride_sec is not None else 1.0)

    base = {
        "sample_name": sample.get("sample_name", "unknown"),
        "target": int(sample.get("target", 0)),
        "stage1_pass": False,
        "mode": 0,
        "window_probs": [],
        "window_preds": [],
        "window_targets": [],
        "stage1_gate_flags": [],
        "stage2_enabled_flags": [],
        "fallback": False,
        "fallback_reason": None,
    }
    try:
        ppg, acc = _load_sample_arrays(sample)
        if is_prewindowed_signal(ppg):
            mode = detect_green_mode(ppg)
            base["mode"] = int(mode)
            first_step = max(0, int(skip_initial_windows))
            all_targets = _prewindow_targets(sample, ppg.shape[0])
            base["window_targets"] = all_targets[first_step:]
            n_steps = max(0, ppg.shape[0] - first_step)
            probs = np.zeros(n_steps, dtype=float)
            window_preds = np.zeros(n_steps, dtype=int)
            flags = []
            valid_indices = []
            feature_vectors = []
            for local_i, idx in enumerate(range(first_step, ppg.shape[0])):
                raw_window = ppg[idx]
                window_25, ppg_src_fs = _prewindow_to_25hz(sample, raw_window, window_sec)
                enabled = stage1_sample_pass(
                    raw_window,
                    stage1["dc_threshold"],
                    stage1["ac_dc_threshold"],
                    ppg_fs=ppg_src_fs,
                )
                flags.append(int(enabled))
                try:
                    ir, ambient, g1, g2, g3 = get_channels_from_window(window_25, mode)
                    acc_seg = None
                    if acc is not None and is_prewindowed_signal(acc) and idx < acc.shape[0]:
                        acc_seg, _ = _prewindow_to_25hz(sample, acc[idx], window_sec)
                    feature_vec = extractor.extract_features(
                        ir, ambient, g1, g2, g3, acc=acc_seg, fs=FEATURE_FS, mode=mode
                    )
                    feature_vectors.append(feature_vec)
                    valid_indices.append(local_i)
                except Exception:
                    continue
            if valid_indices:
                X = np.asarray(feature_vectors, dtype=float)
                pred_probs = bundle["model"].predict_proba(X)[:, 1]
                for row_i, idx in enumerate(valid_indices):
                    probs[idx] = float(pred_probs[row_i])
                    window_preds[idx] = int(pred_probs[row_i] >= model_threshold)
            base["window_probs"] = probs.tolist()
            base["window_preds"] = window_preds.tolist()
            base["stage1_gate_flags"] = flags
            base["stage2_enabled_flags"] = flags
            base["stage1_pass"] = bool(any(flags))
            return _finalize_project_detail(base, method, postprocess_cfg, model_threshold)
        ppg_25, acc_25, ppg_src_fs = _to_25hz(sample, ppg, acc)
    except Exception as exc:
        base["fallback"] = True
        base["fallback_reason"] = f"load: {exc}"
        return _finalize_project_detail(base, method, postprocess_cfg, model_threshold)

    mode = detect_green_mode(ppg)
    base["mode"] = int(mode)
    ir_5hz = downsample_to_5hz(ppg[:, 0], ppg_src_fs, STAGE1_FS)
    s1_win = int(round(STAGE1_PRIMITIVE_SEC * STAGE1_FS))
    s1_stride = s1_win
    s2_win = int(round(window_sec * FEATURE_FS))
    s2_stride = max(1, int(round(stride_sec * FEATURE_FS)))
    n_s1 = (len(ir_5hz) - s1_win) // s1_stride + 1
    n_s2 = (len(ppg_25) - s2_win) // s2_stride + 1
    n_steps = max(0, n_s2)
    base["window_targets"] = [base["target"]] * n_steps

    gate = Stage1StreamingGate(
        stage1["dc_threshold"],
        stage1["ac_dc_threshold"],
        K=STAGE1_GATE_K,
    )
    probs = np.zeros(n_steps, dtype=float)
    window_preds = np.zeros(n_steps, dtype=int)
    flags = [0] * n_steps
    valid_indices = []
    feature_vectors = []

    last_s1_step = -1
    first_step = max(0, int(skip_initial_windows))
    for step in range(first_step, n_steps):
        s2_start = step * s2_stride
        target_s1_step = int(np.floor(s2_start / FEATURE_FS + 1e-9))
        if target_s1_step >= n_s1:
            break
        enabled, last_s1_step = _advance_stage1_gate_to_step(
            gate, ir_5hz, s1_win, s1_stride, last_s1_step, target_s1_step
        )
        flags[step] = int(enabled)
        try:
            window = ppg_25[s2_start:s2_start + s2_win, :]
            ir, ambient, g1, g2, g3 = get_channels_from_window(window, mode)
            acc_seg = _slice_acc(acc_25, s2_start, s2_win)
            feature_vec = extractor.extract_features(
                ir, ambient, g1, g2, g3, acc=acc_seg, fs=FEATURE_FS, mode=mode
            )
            feature_vectors.append(feature_vec)
            valid_indices.append(step)
        except Exception:
            continue

    if valid_indices:
        X = np.asarray(feature_vectors, dtype=float)
        pred_probs = bundle["model"].predict_proba(X)[:, 1]
        for local_i, idx in enumerate(valid_indices):
            probs[idx] = float(pred_probs[local_i])
            window_preds[idx] = int(pred_probs[local_i] >= model_threshold)

    base["window_probs"] = probs.tolist()
    base["window_preds"] = window_preds.tolist()
    base["stage1_gate_flags"] = flags
    base["stage2_enabled_flags"] = flags
    base["stage1_pass"] = bool(any(flags))
    return _finalize_project_detail(base, method, postprocess_cfg, model_threshold)


def _finalize_project_detail(base, method, postprocess_cfg, model_threshold):
    probs = base.get("window_probs", [])
    gate_flags = resolve_stage1_gate_flags(base, len(probs))
    if base.get("fallback") or len(probs) == 0:
        pred = 0
        stage2_pred = 0
        states = []
        output_states = []
        scores = []
        window_preds = list(base.get("window_preds", []))
    else:
        stage2_pred, states, window_preds, scores = apply_postprocess(
            probs,
            [{} for _ in probs],
            method,
            postprocess_cfg,
            model_threshold,
        )
        states = list(states) if len(states) == len(probs) else list(window_preds)
        output_states = fuse_stage1_stage2_states(states, gate_flags)
        cfg = dict(DEFAULT_POSTPROCESS_CONFIG)
        if postprocess_cfg:
            cfg.update(postprocess_cfg)
        pred = sample_pred_from_states(
            output_states,
            cfg.get("sample_pred_strategy", DEFAULT_POSTPROCESS_CONFIG["sample_pred_strategy"]),
            cfg.get(
                "sample_pred_warmup_frames",
                DEFAULT_POSTPROCESS_CONFIG["sample_pred_warmup_frames"],
            ),
        )
    return {
        "sample_name": base.get("sample_name"),
        "target": int(base.get("target", 0)),
        "pred": int(pred),
        "stage2_pred": int(stage2_pred),
        "stage1_pred": int(bool(any(gate_flags))),
        "stage1_pass": bool(any(gate_flags)),
        "fallback": bool(base.get("fallback", False)),
        "fallback_reason": base.get("fallback_reason"),
        "mode": int(base.get("mode", 0)),
        "window_probs": [float(p) for p in probs],
        "window_preds": [int(p) for p in window_preds],
        "window_targets": [int(value) for value in base.get("window_targets", [])],
        "window_states": [int(s) for s in output_states],
        "stage2_states": [int(s) for s in states],
        "output_states": [int(s) for s in output_states],
        "window_scores": [float(s) for s in scores],
        "stage1_gate_flags": list(gate_flags),
        "stage2_enabled_flags": list(gate_flags),
        "n_windows": int(len(probs)),
    }


def evaluate_project_pipeline(samples, artifacts, window_sec=None, stride_sec=None,
                              skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
                              use_stage2_ir=None, warmup_frames: int = 5):
    details = [
        infer_one_sample_project(
            sample,
            artifacts,
            window_sec=window_sec,
            stride_sec=stride_sec,
            skip_initial_windows=skip_initial_windows,
            use_stage2_ir=use_stage2_ir,
        )
        for sample in samples
    ]
    return _build_eval_payload(
        details,
        artifacts.get("postprocess_cfg", {}),
        float(artifacts["bundle"]["threshold"]),
        warmup_frames=warmup_frames,
    )


def _prob_stats(detail):
    probs = [float(p) for p in detail.get("window_probs", []) if p is not None and np.isfinite(p)]
    if not probs:
        return {"mean_prob": 0.0, "max_prob": 0.0}
    return {"mean_prob": float(np.mean(probs)), "max_prob": float(np.max(probs))}


def build_comparison_report(commercial_eval, project_eval, metadata):
    commercial_by_name = {d["sample_name"]: d for d in commercial_eval["details"]}
    project_by_name = {d["sample_name"]: d for d in project_eval["details"]}
    common_names = sorted(set(commercial_by_name) & set(project_by_name))

    disagreements = []
    categories = {
        "both_correct": 0,
        "both_wrong": 0,
        "ours_only_correct": 0,
        "commercial_only_correct": 0,
        "ours_fp_commercial_tn": 0,
        "commercial_fp_ours_tn": 0,
    }
    for name in common_names:
        commercial = commercial_by_name[name]
        ours = project_by_name[name]
        target = int(ours["target"])
        c_pred = int(commercial["pred"])
        o_pred = int(ours["pred"])
        c_ok = c_pred == target
        o_ok = o_pred == target
        if c_ok and o_ok:
            categories["both_correct"] += 1
        elif (not c_ok) and (not o_ok):
            categories["both_wrong"] += 1
        elif o_ok:
            categories["ours_only_correct"] += 1
        else:
            categories["commercial_only_correct"] += 1
        if target == 0 and o_pred == 1 and c_pred == 0:
            categories["ours_fp_commercial_tn"] += 1
        if target == 0 and c_pred == 1 and o_pred == 0:
            categories["commercial_fp_ours_tn"] += 1
        if c_pred != o_pred:
            c_stats = _prob_stats(commercial)
            o_stats = _prob_stats(ours)
            disagreements.append({
                "sample_name": name,
                "target": target,
                "commercial_pred": c_pred,
                "ours_pred": o_pred,
                "commercial_mean_prob": c_stats["mean_prob"],
                "ours_mean_prob": o_stats["mean_prob"],
                "commercial_n_windows": int(commercial.get("n_windows", 0)),
                "ours_n_windows": int(ours.get("n_windows", 0)),
            })

    metric_deltas = {}
    for scope in ("summary", "window_model_summary", "window_stream_summary", "sample_metrics", "window_metrics"):
        metric_deltas[scope] = {}
        for key in ("accuracy", "precision", "recall", "f1"):
            metric_deltas[scope][key] = float(project_eval[scope][key] - commercial_eval[scope][key])

    return {
        "metadata": metadata,
        "commercial": commercial_eval,
        "project": project_eval,
        "metric_deltas_project_minus_commercial": metric_deltas,
        "paired_comparison": {
            "n_common_samples": int(len(common_names)),
            "categories": categories,
            "disagreements": disagreements,
        },
    }


def build_window_metric_comparison_rows(report):
    commercial = report["commercial"]["window_metrics"]
    project = report["project"]["window_metrics"]
    deltas = report["metric_deltas_project_minus_commercial"]["window_metrics"]
    metrics = ["accuracy", "precision", "recall", "f1", "total_windows"]
    rows = []
    for metric in metrics:
        if metric == "total_windows":
            delta = float(project.get(metric, 0.0) - commercial.get(metric, 0.0))
        else:
            delta = float(deltas.get(metric, project.get(metric, 0.0) - commercial.get(metric, 0.0)))
        rows.append({
            "metric": metric,
            "commercial": float(commercial.get(metric, 0.0)),
            "project": float(project.get(metric, 0.0)),
            "delta_project_minus_commercial": delta,
        })
    return rows


def build_accuracy_scope_rows(report):
    scope_specs = [
        ("summary", "sample"),
        ("window_model_summary", "window_model"),
        ("window_stream_summary", "window_stream"),
    ]
    rows = []
    for scope, label in scope_specs:
        commercial = report["commercial"].get(scope, {})
        project = report["project"].get(scope, {})
        deltas = report["metric_deltas_project_minus_commercial"].get(scope, {})
        rows.append({
            "scope": scope,
            "label": label,
            "commercial_accuracy": float(commercial.get("accuracy", 0.0)),
            "project_accuracy": float(project.get("accuracy", 0.0)),
            "delta_project_minus_commercial": float(
                deltas.get("accuracy", project.get("accuracy", 0.0) - commercial.get("accuracy", 0.0))
            ),
        })
    return rows


def export_accuracy_scope_csv(report, out_dir: Path):
    out_path = Path(out_dir) / "accuracy_scope_compare.csv"
    rows = build_accuracy_scope_rows(report)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "label",
                "commercial_accuracy",
                "project_accuracy",
                "delta_project_minus_commercial",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return str(out_path)


def print_accuracy_scope_comparison(report):
    print("\n[Accuracy compare]")
    for row in build_accuracy_scope_rows(report):
        print(
            f"  {row['label']:<13} "
            f"commercial={row['commercial_accuracy']:.4f}  "
            f"project={row['project_accuracy']:.4f}  "
            f"delta={row['delta_project_minus_commercial']:.4f}"
        )


def export_window_metric_comparison_csv(report, out_dir: Path):
    out_path = Path(out_dir) / "window_level_compare.csv"
    rows = build_window_metric_comparison_rows(report)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "commercial", "project", "delta_project_minus_commercial"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return str(out_path)


def _draw_confusion(ax, cm, title):
    matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]], dtype=float)
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=13, fontweight="bold")
    ax.tick_params(length=0)


def _annotate_bars(ax, bars, fmt="{:.2f}"):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, fmt.format(h), ha="center", va="bottom", fontsize=8)


def plot_summary(report, out_dir: Path, inputs=()):
    out_path = Path(_make_png_path(out_dir / "commercial_compare_summary.png"))
    commercial = report["commercial"]
    project = report["project"]
    metrics = ["precision", "recall", "f1", "accuracy"]
    x = np.arange(len(metrics))
    width = 0.34

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=160)
    fig.suptitle("Commercial Baseline vs Deployed XGBoost", fontsize=16, fontweight="bold", x=0.03, ha="left")

    ax = axes[0, 0]
    b1 = ax.bar(x - width / 2, [commercial["summary"][m] for m in metrics], width, label="Commercial", color=PLOT_COLORS["commercial"])
    b2 = ax.bar(x + width / 2, [project["summary"][m] for m in metrics], width, label="Deployed", color=PLOT_COLORS["ours"])
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, [m.upper() for m in metrics])
    ax.set_title("Sample-level metrics", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    _annotate_bars(ax, b1)
    _annotate_bars(ax, b2)

    ax = axes[0, 1]
    b1 = ax.bar(x - width / 2, [commercial["window_stream_summary"][m] for m in metrics], width, label="Commercial", color=PLOT_COLORS["commercial"])
    b2 = ax.bar(x + width / 2, [project["window_stream_summary"][m] for m in metrics], width, label="Deployed", color=PLOT_COLORS["ours"])
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, [m.upper() for m in metrics])
    ax.set_title("Streaming window metrics", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    _annotate_bars(ax, b1)
    _annotate_bars(ax, b2)

    _draw_confusion(axes[1, 0], commercial["summary"]["confusion_matrix"], "Commercial sample confusion")
    _draw_confusion(axes[1, 1], project["summary"]["confusion_matrix"], "Deployed sample confusion")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    source_rows = []
    for scope in ("summary", "window_stream_summary"):
        for model_name, payload in (("commercial", commercial), ("project", project)):
            for metric in metrics:
                source_rows.append({
                    "panel": scope, "model": model_name, "metric": metric,
                    "value": payload[scope][metric],
                })
    split = str(report.get("metadata", {}).get("split", "comparison"))
    save_scientific_figure(
        fig, out_path, source_data=source_rows,
        core_conclusion="The deployed model is compared with the commercial baseline at sample and streaming-window levels.",
        panel_map={"a": "Sample metrics.", "b": "Streaming-window metrics.", "c": "Commercial confusion matrix.", "d": "Deployed confusion matrix."},
        inputs=inputs, split=split,
        n_definition="paired samples and their causal streaming windows",
        statistics={"comparison": "paired point estimates"},
        reviewer_risks=["Windows within one sample are correlated."],
        test_read_only=split.lower() == "test",
    )
    plt.close(fig)
    return str(out_path)


def plot_probabilities(report, out_dir: Path, inputs=()):
    out_path = Path(_make_png_path(out_dir / "commercial_compare_probabilities.png"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160, sharey=True)
    for ax, key, title, color in [
        (axes[0], "commercial", "Commercial AdaBoost probabilities", PLOT_COLORS["commercial"]),
        (axes[1], "project", "Deployed XGBoost probabilities", PLOT_COLORS["ours"]),
    ]:
        details = report[key]["details"]
        pos = [p for d in details if d["target"] == 1 for p in d.get("window_probs", [])]
        neg = [p for d in details if d["target"] == 0 for p in d.get("window_probs", [])]
        bins = np.linspace(0, 1, 21)
        ax.hist(neg, bins=bins, alpha=0.65, label="target=0", color=PLOT_COLORS["negative"], density=False)
        ax.hist(pos, bins=bins, alpha=0.58, label="target=1", color=color, density=False)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Window probability")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Window count")
    fig.tight_layout()
    source_rows = [
        {"model": key, "sample_name": detail.get("sample_name"), "target": detail["target"], "probability": float(probability)}
        for key in ("commercial", "project")
        for detail in report[key]["details"]
        for probability in detail.get("window_probs", [])
        if probability is not None and np.isfinite(probability)
    ]
    split = str(report.get("metadata", {}).get("split", "comparison"))
    save_scientific_figure(
        fig, out_path, source_data=source_rows,
        core_conclusion="Window-probability separation is compared for the commercial and deployed models.",
        panel_map={"a": "Commercial probability distributions.", "b": "Deployed probability distributions."},
        inputs=inputs, split=split,
        n_definition="one row per finite window probability",
        statistics={"display": "fixed-width probability histograms"},
        reviewer_risks=["Window probabilities are correlated within samples."],
        test_read_only=split.lower() == "test",
    )
    plt.close(fig)
    return str(out_path)


def plot_disagreements(report, out_dir: Path, inputs=()):
    out_path = Path(_make_png_path(out_dir / "commercial_compare_disagreements.png"))
    categories = report["paired_comparison"]["categories"]
    names = [
        "both_correct",
        "ours_only_correct",
        "commercial_only_correct",
        "both_wrong",
        "commercial_fp_ours_tn",
        "ours_fp_commercial_tn",
    ]
    values = [categories.get(name, 0) for name in names]
    labels = [
        "Both correct",
        "Only deployed correct",
        "Only commercial correct",
        "Both wrong",
        "Commercial FP only",
        "Deployed FP only",
    ]
    colors = ["#22c55e", PLOT_COLORS["ours"], PLOT_COLORS["commercial"], "#94a3b8", PLOT_COLORS["danger"], "#a855f7"]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    bars = ax.barh(np.arange(len(values)), values, color=colors)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Sample count")
    ax.set_title("Paired sample outcomes and false-positive ownership", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    for bar in bars:
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2, int(bar.get_width()), va="center", fontsize=9)
    fig.tight_layout()
    source_rows = [
        {"category": name, "label": label, "sample_count": value}
        for name, label, value in zip(names, labels, values)
    ]
    split = str(report.get("metadata", {}).get("split", "comparison"))
    save_scientific_figure(
        fig, out_path, source_data=source_rows,
        core_conclusion="Paired outcomes identify which model owns each sample-level error and false positive.",
        panel_map={"a": "Paired correctness and false-positive ownership counts."},
        inputs=inputs, split=split,
        n_definition="one paired outcome category per row",
        statistics={"comparison": "paired sample counts"},
        reviewer_risks=["Small disagreement strata may be unstable."],
        test_read_only=split.lower() == "test",
    )
    plt.close(fig)
    return str(out_path)


def export_comparison_plots(report, out_dir, inputs=()):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": plot_summary(report, out_dir, inputs=inputs),
        "probabilities": plot_probabilities(report, out_dir, inputs=inputs),
        "disagreements": plot_disagreements(report, out_dir, inputs=inputs),
    }
    return paths


def _json_ready_detail(detail, keep_probs: bool):
    out = dict(detail)
    if not keep_probs:
        out.pop("window_probs", None)
        out.pop("window_preds", None)
        out.pop("window_states", None)
        out.pop("window_scores", None)
        out.pop("stage2_enabled_flags", None)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare commercial AdaBoost baseline with deployed XGBoost artifacts.")
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--commercial_dc_threshold", type=float, default=0.1e6)
    parser.add_argument("--fp_cost", type=float, default=1.5)
    parser.add_argument("--method", type=str, default="state_machine", choices=["state_machine", "mean_vote", "prob_mean"])
    parser.add_argument("--window_sec", type=float, default=None, help="Override deployed XGBoost window seconds.")
    parser.add_argument("--stride_sec", type=float, default=1.0, help="Deployed XGBoost stride seconds.")
    parser.add_argument("--skip_initial_windows", type=int, default=DEFAULT_SKIP_INITIAL_WINDOWS,
                        help="drop this many leading project Stage2 windows per sample")
    parser.add_argument("--warmup_frames", type=int, default=5,
                        help="streaming window metrics skip this many leading state-machine windows per sample")
    parser.add_argument("--use_stage2_ir", action=argparse.BooleanOptionalAction, default=None,
                        help="whether project Stage2 uses IR; defaults to model bundle metadata")
    parser.add_argument("--keep_window_probs", action="store_true")
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    out_dir = artifact_dir / "commercial_compare"
    with open(artifact_dir / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_samples = splits["train"]
    valid_samples = splits["valid"]
    eval_samples = splits[args.split]

    print("=" * 72)
    print("s08 commercial comparison")
    print(f"  eval split: {args.split} ({len(eval_samples)} samples)")
    print(f"  commercial: DC-only Stage1, {COMMERCIAL_WIN_SEC}s/25Hz, 8 features, AdaBoost")
    print("  project: s07 deploy_feature_extractor.py + model_bundle.pkl")
    print("=" * 72)

    t0 = time.time()
    commercial_model, commercial_train = train_commercial_model(train_samples, args.commercial_dc_threshold)
    valid_raw = [infer_one_sample_commercial(sample, args.commercial_dc_threshold) for sample in valid_samples]
    valid_raw = _attach_commercial_probs(valid_raw, commercial_model)
    threshold_selection = select_commercial_threshold(valid_raw, fp_cost=args.fp_cost)
    commercial_threshold = threshold_selection["threshold"]
    commercial_eval = evaluate_commercial_model(
        commercial_model,
        commercial_threshold,
        args.commercial_dc_threshold,
        eval_samples,
        warmup_frames=args.warmup_frames,
    )

    artifacts = load_project_artifacts(artifact_dir, method=args.method)
    project_eval = evaluate_project_pipeline(
        eval_samples,
        artifacts,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        skip_initial_windows=args.skip_initial_windows,
        use_stage2_ir=args.use_stage2_ir,
        warmup_frames=args.warmup_frames,
    )

    metadata = {
        "split": args.split,
        "commercial": {
            "stage1_dc_threshold": float(args.commercial_dc_threshold),
            "stage2_window_sec": COMMERCIAL_WIN_SEC,
            "stage2_stride_sec": COMMERCIAL_STRIDE_SEC,
            "stage2_fs": FEATURE_FS,
            "features": COMMERCIAL_FEATURE_NAMES,
            "model": "AdaBoostClassifier(n_estimators=16, max_depth=5)",
            "threshold": float(commercial_threshold),
            "warmup_frames": int(args.warmup_frames),
            "threshold_selection": threshold_selection,
            "train_summary": commercial_train,
        },
        "project": {
            "bundle_path": artifacts["bundle_path"],
            "deploy_extractor_path": artifacts["extractor_path"],
            "stage1_threshold": artifacts["stage1_threshold"],
            "feature_order": list(artifacts["bundle"]["feature_names"]),
            "n_features": int(len(artifacts["bundle"]["feature_names"])),
            "model_threshold": float(artifacts["bundle"]["threshold"]),
            "window_sec": float(args.window_sec if args.window_sec is not None else artifacts["bundle"]["meta"]["win_sec"]),
            "stride_sec": float(args.stride_sec),
            "skip_initial_windows": int(args.skip_initial_windows),
            "warmup_frames": int(args.warmup_frames),
            "use_stage2_ir": bool(resolve_use_stage2_ir(artifacts["bundle"], args.use_stage2_ir)),
            "postprocess": artifacts["postprocess_cfg"],
            "method": args.method,
        },
        "elapsed_sec": None,
    }
    report = build_comparison_report(commercial_eval, project_eval, metadata)
    window_compare_path = export_window_metric_comparison_csv(report, out_dir)
    accuracy_scope_path = export_accuracy_scope_csv(report, out_dir)
    plot_paths = export_comparison_plots(
        report, out_dir, inputs=[window_compare_path, accuracy_scope_path]
    )
    report["metadata"]["plot_paths"] = plot_paths
    report["metadata"]["window_level_compare_csv"] = window_compare_path
    report["metadata"]["accuracy_scope_compare_csv"] = accuracy_scope_path
    report["metadata"]["elapsed_sec"] = float(time.time() - t0)

    if not args.keep_window_probs:
        report["commercial"]["details"] = [_json_ready_detail(d, False) for d in report["commercial"]["details"]]
        report["project"]["details"] = [_json_ready_detail(d, False) for d in report["project"]["details"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "commercial_compare.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n[Commercial sample]", commercial_eval["summary"])
    print("[Project sample]   ", project_eval["summary"])
    print("[Delta sample]     ", report["metric_deltas_project_minus_commercial"]["summary"])
    print_accuracy_scope_comparison(report)
    print(f"\n[OK] report: {report_path}")
    print(f"[OK] window_level_compare.csv: {window_compare_path}")
    print(f"[OK] accuracy_scope_compare.csv: {accuracy_scope_path}")
    for name, path in plot_paths.items():
        print(f"[OK] {name}: {path}")


if __name__ == "__main__":
    main()
