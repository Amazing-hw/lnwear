# s06_deploy_eval.py
# -*- coding: utf-8 -*-

"""
步骤6：部署/端到端推理评估（优化版）

主要改进：
1. 单趟推理 + ProcessPoolExecutor 并行；推理结果在主进程统一计算窗口与样本指标。
2. 删除每窗对绿光/IR 的重复 preprocess_signal 调用：
   直接走 extract_feature_pool_from_window(..., return_preprocessed=True)，
   从返回的 preprocessed dict 复用三光区聚合信号给交叉特征使用；5 秒且
   ACC 可用时，与训练/独立部署一致地应用精确商用字段覆盖。
3. 主结果仅使用 XGBoost 窗口概率和冻结阈值；后处理仅作可选独立分析。
4. 状态机网格搜索并行化（probs 缓存复用）。
5. 预切窗和 grouped-window H5 直接按窗口编号推理；窗口缓存保留
   window_indices/window_targets，供后处理按原始顺序组合。

公共接口包括 main / predict_sample / predict_sample_with_bundle /
predict_sample_safe / evaluate_streaming_window_accuracy /
optimize_state_machine_params / load_bundle。输出 JSON 使用显式语义版本字段。
"""

import os
import json
import argparse
import logging
import joblib
import glob
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scientific_figures import save_scientific_figure
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from s03_extract_feature_pool import (
    _downsample_ppg,
    load_ppg,
    load_acc,
    get_sample_frequency,
    get_sample_ppg_config,
    is_prewindowed_signal,
    get_channels_from_window,
    is_stage2_ir_feature,
    extract_feature_pool_from_window,
    extract_stage2_window,
    extract_commercial_feature_overrides,
    COMMERCIAL_STAGE2_FIELDS,
    align_acc_window,
    validate_h5_file,
    load_grouped_window_metadata,
)
from stage2_feature_catalog import (
    FEATURE_CATALOG,
    FEATURE_POOL_VERSION,
    build_selected_feature_contract,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


_BUNDLE = None

DEFAULT_POSTPROCESS_CONFIG = {
    "alpha": 0.4,
    "median_k": 1,
    "T_on": 0.75,
    "T_off": 0.35,
    "K_on": 5,
    "K_off": 3,
    "cooldown_sec": 5,
    "sample_pred_strategy": "final_state",
    "sample_pred_warmup_frames": 0,
}

DEFAULT_USE_STAGE2_IR = False


def _dedupe_sorted(values):
    out = []
    for value in sorted(float(v) for v in values):
        value = round(value, 6)
        if not out or abs(out[-1] - value) > 1e-9:
            out.append(value)
    return out


def default_t_on_candidates(model_threshold=0.5):
    threshold = float(model_threshold)
    raw = [
        threshold + 0.02,
        threshold + 0.05,
        threshold + 0.10,
        0.55,
        0.70,
        0.85,
    ]
    return _dedupe_sorted(min(0.95, max(0.05, v)) for v in raw)


def resolve_postprocess_thresholds(cfg, model_threshold=0.5):
    """Resolve state-machine thresholds, deriving defaults from model threshold."""
    cfg = cfg or {}
    model_threshold = float(model_threshold)
    t_on = float(cfg.get("T_on", DEFAULT_POSTPROCESS_CONFIG["T_on"]))
    t_off = float(cfg.get("T_off", DEFAULT_POSTPROCESS_CONFIG["T_off"]))

    if abs(t_on - float(DEFAULT_POSTPROCESS_CONFIG["T_on"])) < 1e-9 and model_threshold < t_on:
        t_on = float(min(0.95, max(0.50, model_threshold + 0.05)))
    if abs(t_off - float(DEFAULT_POSTPROCESS_CONFIG["T_off"])) < 1e-9:
        t_off = float(min(t_on - 0.05, max(0.10, model_threshold - 0.20)))

    return t_on, t_off


def resolve_use_stage2_ir(bundle, requested=None):
    # Legacy CLI compatibility only. The model uses ambient/green/ACC features;
    # IR-derived features remain excluded from Stage2 candidates.
    return DEFAULT_USE_STAGE2_IR


def assert_no_stage2_ir_features(feature_names, context):
    leaks = [str(name) for name in feature_names if is_stage2_ir_feature(str(name))]
    if leaks:
        raise ValueError(
            f"IR-derived Stage2 features found in {context}: {leaks[:10]}. "
            "Regenerate artifacts from s03; Stage2 uses only ambient/green/ACC."
        )
    return list(feature_names)


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_n_workers(n_workers=None, n_items=None, cap=4):
    """Resolve a conservative worker count for server-safe deploy evaluation."""
    if _env_flag("WL_FORCE_SERIAL"):
        return 1
    if n_workers is None:
        n_workers = max(1, min(cap, (os.cpu_count() or cap) // 2))
    try:
        resolved = max(1, int(n_workers))
    except (TypeError, ValueError):
        resolved = 1
    if n_items is not None and int(n_items) <= 2:
        return 1
    if n_items is not None:
        resolved = min(resolved, max(1, int(n_items)))
    return resolved


def multiprocessing_context_from_env():
    method = os.environ.get("WL_MP_START_METHOD", "").strip()
    if not method:
        return None
    import multiprocessing as mp
    return mp.get_context(method)


def load_bundle(path):
    """加载统一的模型包"""
    global _BUNDLE
    _BUNDLE = joblib.load(path)
    assert_bundle_ok(_BUNDLE)
    return _BUNDLE


def assert_bundle_ok(bundle):
    """校验模型包完整性"""
    needed = ["feature_names", "fill_values", "model", "threshold", "meta"]
    for k in needed:
        if k not in bundle:
            raise ValueError(f"model_bundle missing key: {k}")

    version = bundle.get("feature_pool_version")
    if version != FEATURE_POOL_VERSION:
        raise ValueError(
            f"model_bundle feature_pool_version={version!r} does not match "
            f"{FEATURE_POOL_VERSION}; rerun s03-s05 before deployment evaluation."
        )

    miss = [c for c in bundle["feature_names"] if c not in bundle["fill_values"]]
    if miss:
        raise ValueError(f"fill_values missing for: {miss[:5]} ...")
    assert_no_stage2_ir_features(bundle["feature_names"], "model_bundle.pkl feature_names")

    for k in ["fs_ppg", "win_sec", "step_sec"]:
        if k not in bundle["meta"]:
            raise ValueError(f"meta missing: {k}")


def bundle_meta_log_summary(bundle):
    """Return only concise deployment-relevant metadata for console logging."""
    meta = bundle.get("meta") or {}
    model_search = meta.get("model_search") or {}
    hard_negative = meta.get("hard_negative_mining") or {}
    return {
        "feature_pool_version": bundle.get(
            "feature_pool_version", meta.get("feature_pool_version")
        ),
        "feature_count": int(len(bundle.get("feature_names") or [])),
        "window_sec": meta.get("win_sec"),
        "stride_sec": meta.get("step_sec"),
        "model_search_enabled": bool(model_search.get("enabled", False)),
        "hard_negative_enabled": bool(hard_negative.get("enabled", False)),
    }


def validate_inference_window_contract(bundle, window_sec, stride_sec):
    """Reject inference windows that differ from the trained bundle contract."""
    meta = bundle.get("meta", {}) if isinstance(bundle, dict) else {}
    expected = {
        "window_sec": ("model_bundle.meta.win_sec", meta.get("win_sec")),
        "stride_sec": ("model_bundle.meta.step_sec", meta.get("step_sec")),
    }
    actual = {
        "window_sec": window_sec,
        "stride_sec": stride_sec,
    }
    for name, (bundle_name, bundle_value) in expected.items():
        if bundle_value is None:
            raise ValueError(f"{bundle_name} is required for inference")
        try:
            value = float(actual[name])
            trained_value = float(bundle_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} and {bundle_name} must be numeric") from exc
        if not np.isfinite(value) or not np.isfinite(trained_value):
            raise ValueError(f"{name} and {bundle_name} must be finite")
        if not np.isclose(value, trained_value, rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"{name}={value:g} does not match "
                f"{bundle_name}={trained_value:g}; rerun training or use the "
                "bundle window contract."
            )
    return float(window_sec), float(stride_sec)


def apply_preprocess(feat_dict_list, bundle=None):
    """列对齐 + inf处理 + 缺失填充 + clip（与训练侧 s05 一致）"""
    b = bundle if bundle is not None else _BUNDLE
    if b is None:
        raise RuntimeError("must call load_bundle() first")

    feature_names = b["feature_names"]
    fill_values = b["fill_values"]

    df = pd.DataFrame(feat_dict_list)

    for c in feature_names:
        if c not in df.columns:
            df[c] = np.nan
    df = df[feature_names]

    for c in feature_names:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(fill_values[c])
    return df.values.astype(np.float64)


def predict_proba_windows(feat_dict_list, bundle=None):
    """预测窗口概率"""
    b = bundle if bundle is not None else _BUNDLE
    X = apply_preprocess(feat_dict_list, bundle=b)
    proba = b["model"].predict_proba(X)[:, 1]
    return proba


def predict_label_windows(feat_dict_list, bundle=None):
    """预测窗口标签"""
    b = bundle if bundle is not None else _BUNDLE
    proba = predict_proba_windows(feat_dict_list, bundle=b)
    thr = b["threshold"]
    return (proba >= thr).astype(int), proba


# =========================================================
# 后处理：状态机
# =========================================================

class WearStateMachine:
    def __init__(self, alpha=0.4, T_on=0.75, T_off=0.35, K_on=5, K_off=3, cooldown_sec=5):
        self.alpha = alpha
        self.T_on = T_on
        self.T_off = T_off
        self.K_on = K_on
        self.K_off = K_off
        self.cooldown_sec = cooldown_sec  # 翻转后最小冷却秒数

        self.state = 0
        self.score = None  # 延迟初始化：首次 update() 用第一个窗的概率值初始化
        self.on_count = 0
        self.off_count = 0
        self._steps_since_flip = 999  # 距离上次翻转的步数（初始大值允许首次翻转）

    def update(self, p, quality=1.0, stride_sec=1.0):
        q = float(np.clip(quality, 0.0, 1.0))
        eff_alpha = self.alpha * q
        if self.score is None:
            # 首个窗口：用质量调制后的概率值初始化 EMA，消除从 0 起步的冷启动偏差
            # 佩戴样本 (p≈0.9) 首次 score 直接越过 T_on，开始计 K_on
            # 非佩戴样本 (p≈0.1) 首次 score 直接低于 T_off，开始计 K_off
            self.score = q * float(p)
        else:
            self.score = eff_alpha * p + (1 - eff_alpha) * self.score
        self._steps_since_flip += 1

        cooldown_steps = int(self.cooldown_sec / stride_sec) if stride_sec > 0 else self.cooldown_sec

        if self.state == 0:
            if self.score > self.T_on:
                self.on_count += 1
            else:
                self.on_count = max(0, self.on_count - 1)  # leakage decay

            if self.on_count >= self.K_on and self._steps_since_flip >= cooldown_steps:
                self.state = 1
                self.on_count = 0
                self.off_count = 0
                self._steps_since_flip = 0
        else:
            if self.score < self.T_off:
                self.off_count += 1
            else:
                self.off_count = max(0, self.off_count - 1)  # leakage decay

            if self.off_count >= self.K_off and self._steps_since_flip >= cooldown_steps:
                self.state = 0
                self.on_count = 0
                self.off_count = 0
                self._steps_since_flip = 0

        return self.state, self.score


def causal_median_filter_1d(x, k):
    if int(k) <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    k = int(k)
    out = np.zeros_like(x)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = float(np.median(x[lo:i + 1]))
    return out


def sample_pred_from_states(states, strategy="final_state", warmup_frames=0):
    """Convert streaming states into one sample-level prediction."""
    if states is None:
        states = []
    states = [int(s) for s in list(states)]
    if not states:
        return 0

    strategy = str(strategy or "final_state")
    start = min(max(0, int(warmup_frames or 0)), len(states))
    scored_states = states[start:]

    if strategy in {"final_state", "last_state"}:
        return int(states[-1])
    if strategy in {"any_worn", "any_worn_after_warmup"}:
        return int(any(s == 1 for s in scored_states))
    if strategy in {"majority_state", "majority_state_after_warmup"}:
        if not scored_states:
            return 0
        return int(float(np.mean(scored_states)) >= 0.5)
    raise ValueError(f"unknown sample_pred_strategy: {strategy}")


def _quality_soft(violation_ratio, floor=0.5):
    """连续衰减：0=未违规 → 1.0；1=刚刚违规 → floor。clip 到 [floor, 1]。"""
    v = max(0.0, min(1.0, float(violation_ratio)))
    return float(max(floor, 1.0 - (1.0 - floor) * v))


def compute_quality(feat_or_meta, thresholds=None):
    """
    质量分计算。

    - 若 thresholds 给出（bundle["quality_thresholds"]），用连续衰减、阈值来自 train 分位数。
    - 否则回退到旧版三个 magic numbers + 二值 0.5（向后兼容）。
    """
    if not hasattr(feat_or_meta, "get"):
        return 1.0

    if thresholds:
        q = 1.0
        for key, spec in thresholds.items():
            if key.startswith("_"):
                continue
            v = feat_or_meta.get(key, None)
            if v is None or not np.isfinite(v):
                continue
            thr = spec.get("thr", None)
            kind = spec.get("type", "high")
            if thr is None or thr == 0:
                continue
            if kind == "high":
                # 越大越差：violation = (v - thr) / thr （只在超过时累计）
                violation = (v - thr) / abs(thr)
            else:
                # 越小越差：violation = (thr - |v|) / thr
                violation = (thr - abs(v)) / abs(thr)
            q *= _quality_soft(violation, floor=0.5)
        return float(q)

    # 兼容老 bundle / 没有 quality_thresholds 的情况
    q = 1.0
    amb = feat_or_meta.get("Ambient_std", None)
    if amb is not None and amb > 1e7:
        q *= 0.5
    gmm = feat_or_meta.get("G_mean_mean", None)
    if gmm is not None and np.abs(gmm) < 1e-6:
        q *= 0.5
    return float(q)


def compute_ood_score(feat_dict, feature_quantiles, feature_names):
    """
    OOD 监控：返回该窗特征落到 train 训练 [q_low, q_high] 外的比例。
    feature_quantiles: dict[feat -> {"q_low", "q_high"}]，来自 bundle。
    """
    if not feature_quantiles or not feat_dict:
        return None
    out = 0
    total = 0
    for f in feature_names:
        spec = feature_quantiles.get(f)
        if not spec:
            continue
        v = feat_dict.get(f, None)
        if v is None or not np.isfinite(v):
            continue
        total += 1
        if v < spec["q_low"] or v > spec["q_high"]:
            out += 1
    if total == 0:
        return None
    return float(out) / float(total)


def apply_postprocess(window_probs, quality_metas, method, cfg, model_threshold,
                      stride_sec=1.0):
    """
    纯函数后处理。基于已缓存的 probs 直接产出 final_pred / states / window_preds / scores。

    返回:
        final_pred (int), states (list[int]), window_preds (list[int]), scores (list[float])
    """
    probs = np.asarray(window_probs, dtype=float)
    if probs.size == 0:
        return 0, [], [], []

    window_preds = (probs >= model_threshold).astype(int).tolist()
    probs_for_state = causal_median_filter_1d(probs, cfg.get("median_k", 1))

    if method == "state_machine":
        _t_on, _t_off = resolve_postprocess_thresholds(cfg, model_threshold)
        sm = WearStateMachine(
            alpha=cfg.get("alpha", 0.4),
            T_on=_t_on,
            T_off=_t_off,
            K_on=cfg.get("K_on", DEFAULT_POSTPROCESS_CONFIG["K_on"]),
            K_off=cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]),
            cooldown_sec=cfg.get("cooldown_sec", DEFAULT_POSTPROCESS_CONFIG["cooldown_sec"]),
        )
        qt = _BUNDLE.get("quality_thresholds") if _BUNDLE is not None else None
        states, scores = [], []
        for i, p in enumerate(probs_for_state):
            meta_i = quality_metas[i] if i < len(quality_metas) else None
            q = compute_quality(meta_i, thresholds=qt) if meta_i else 1.0
            state, score = sm.update(p, quality=q, stride_sec=stride_sec)
            states.append(int(state))
            scores.append(float(score))
        final_pred = sample_pred_from_states(
            states,
            strategy=cfg.get("sample_pred_strategy", DEFAULT_POSTPROCESS_CONFIG["sample_pred_strategy"]),
            warmup_frames=cfg.get("sample_pred_warmup_frames", DEFAULT_POSTPROCESS_CONFIG["sample_pred_warmup_frames"]),
        )
        return final_pred, states, window_preds, scores

    if method == "mean_vote":
        final_pred = int(np.mean(window_preds) >= 0.5)
        return final_pred, [], window_preds, []

    # prob_mean / 默认
    final_pred = int(np.mean(probs) >= model_threshold)
    return final_pred, [], window_preds, []


def _window_error_record(position, exc):
    return {
        "window_position": int(position),
        "error": f"{type(exc).__name__}: {exc}",
    }


def _apply_selected_commercial_overrides(
    feat, raw_ppg, raw_acc, frequency, ppg_config, selected_features
):
    """Apply the exact 5-second commercial port to selected mapped fields."""
    if raw_acc is None:
        return feat
    selected = None if selected_features is None else set(selected_features)
    if selected is not None and not selected.intersection(COMMERCIAL_STAGE2_FIELDS):
        return feat
    overrides = extract_commercial_feature_overrides(
        raw_ppg,
        raw_acc,
        frequency=int(frequency),
        ppg_config=int(ppg_config),
    )
    if selected is None:
        feat.update(overrides)
    else:
        for name in selected:
            if name in overrides:
                feat[name] = overrides[name]
    return feat


def _extract_model_window_features(
    ppg_window,
    acc_window,
    *,
    ppg_config,
    selected_features,
    use_stage2_ir=DEFAULT_USE_STAGE2_IR,
    source_ppg=None,
    source_acc=None,
    source_frequency=25,
):
    """Build one inference feature row through the training's canonical path.

    The governed optical and ACC candidates must be computed by
    ``extract_stage2_window`` on both sides.  ``source_*`` is used only for the
    exact commercial-field override, which intentionally retains its original
    25/100 Hz input contract.
    """
    features, _, _ = extract_stage2_window(
        ppg_window,
        mode=int(ppg_config),
        fs=25,
        acc_window=acc_window,
        use_stage2_ir=use_stage2_ir,
        selected_features=selected_features,
    )
    if source_ppg is not None:
        _apply_selected_commercial_overrides(
            features,
            source_ppg,
            source_acc,
            source_frequency,
            ppg_config,
            selected_features,
        )
    return features


def _finalize_window_inference(
    base,
    feats_list,
    quality_metas,
    window_start_sec,
    window_end_sec,
    bundle,
    window_errors,
    window_indices=None,
    window_targets=None,
):
    """Predict valid windows only and preserve aligned metadata."""
    valid_indices = [i for i, feat in enumerate(feats_list) if feat is not None]
    candidate_count = len(feats_list)
    failure_count = len(window_errors)
    base["candidate_window_count"] = int(candidate_count)
    base["window_feature_failure_count"] = int(failure_count)
    base["window_feature_failure_examples"] = list(window_errors[:5])

    if not valid_indices:
        base["fallback"] = True
        if candidate_count == 0:
            base["fallback_reason"] = "no_eligible_windows"
        else:
            first_error = window_errors[0]["error"] if window_errors else "unknown error"
            base["fallback_reason"] = (
                "all_window_feature_extraction_failed: "
                f"{failure_count}/{candidate_count}; first={first_error}"
            )
        base["window_probs"] = []
        base["window_preds"] = []
        base["quality_metas"] = []
        base["window_ood_scores"] = []
        base["window_start_sec"] = []
        base["window_end_sec"] = []
        if window_indices is not None:
            base["window_indices"] = []
        if window_targets is not None:
            base["window_targets"] = []
        return base

    valid_feats = [feats_list[i] for i in valid_indices]
    window_preds, window_probs = predict_label_windows(valid_feats, bundle=bundle)
    feature_quantiles = bundle.get("feature_quantiles")
    feature_names = bundle.get("feature_names", [])
    ood_scores = [
        None if not feature_quantiles else compute_ood_score(feat, feature_quantiles, feature_names)
        for feat in valid_feats
    ]

    base["window_probs"] = [float(value) for value in window_probs]
    base["window_preds"] = [int(value) for value in window_preds]
    base["quality_metas"] = [quality_metas[i] for i in valid_indices]
    base["window_ood_scores"] = ood_scores
    base["window_start_sec"] = [float(window_start_sec[i]) for i in valid_indices]
    base["window_end_sec"] = [float(window_end_sec[i]) for i in valid_indices]
    if window_indices is not None:
        base["window_indices"] = [int(window_indices[i]) for i in valid_indices]
    if window_targets is not None:
        base["window_targets"] = [int(window_targets[i]) for i in valid_indices]
    return base


def _infer_prewindowed_sample(base, ppg, acc,
                              window_sec, stride_sec, bundle, use_stage2_ir):
    """Run XGBoost inference on every stored window."""
    FEATURE_FS = 25
    window_meta = load_grouped_window_metadata(base)
    window_indices = list(window_meta.get("window_indices") or []) if window_meta else []
    window_labels = list(window_meta.get("window_labels") or []) if window_meta else []
    native_25hz = get_sample_frequency(base) == FEATURE_FS
    source_frequency = FEATURE_FS if native_25hz else 100
    expected_native_window_len = int(round(float(window_sec) * source_frequency))
    mode = get_sample_ppg_config(base)
    base["mode"] = int(mode)

    feats_list = []
    quality_metas = []
    window_start_sec = []
    window_end_sec = []
    emitted_window_indices = []
    emitted_window_targets = []
    window_errors = []

    for step in range(ppg.shape[0]):
        raw_window = ppg[step]
        window_number = int(window_indices[step]) if window_indices and step < len(window_indices) else int(step)
        window_target = int(window_labels[step]) if window_labels and step < len(window_labels) else int(base.get("target", 0))
        start_sec = float(window_number * stride_sec)
        window_start_sec.append(start_sec)
        window_end_sec.append(start_sec + float(window_sec))
        emitted_window_indices.append(window_number)
        emitted_window_targets.append(window_target)

        try:
            if raw_window.shape[0] != expected_native_window_len:
                raise ValueError(
                    f"stored PPG window has {raw_window.shape[0]} samples, which does not "
                    f"match requested window length {expected_native_window_len} samples "
                    f"({float(window_sec):g}s at {source_frequency} Hz)"
                )
            window = raw_window.astype(np.float64, copy=False) if native_25hz else _downsample_ppg(
                raw_window, src_fs=100, tgt_fs=FEATURE_FS
            )
            acc_seg = None
            raw_acc = None
            if acc is not None and is_prewindowed_signal(acc) and step < acc.shape[0]:
                try:
                    raw_acc = acc[step]
                    if raw_acc.shape[0] != expected_native_window_len:
                        raise ValueError(
                            f"stored ACC window has {raw_acc.shape[0]} samples, which does not "
                            f"match requested window length {expected_native_window_len} samples"
                        )
                    acc_seg = raw_acc.astype(np.float64, copy=False) if native_25hz else _downsample_ppg(
                        raw_acc, src_fs=100, tgt_fs=FEATURE_FS
                    )
                except Exception:
                    acc_seg = None
            commercial_input = (
                (raw_window, raw_acc, 25 if native_25hz else 100)
                if int(round(float(window_sec) * FEATURE_FS)) == 125 else (None, None, 25)
            )
            feat = _extract_model_window_features(
                window,
                acc_seg,
                ppg_config=mode,
                selected_features=bundle.get("feature_names"),
                use_stage2_ir=use_stage2_ir,
                source_ppg=commercial_input[0],
                source_acc=commercial_input[1],
                source_frequency=commercial_input[2],
            )

            feats_list.append(feat)
            quality_metas.append({
                "Ambient_std": feat.get("Ambient_std"),
                "G_mean_mean": feat.get("G_mean_mean"),
            })
        except Exception as exc:
            feats_list.append(None)
            quality_metas.append(None)
            window_errors.append(_window_error_record(step, exc))

    return _finalize_window_inference(
        base,
        feats_list,
        quality_metas,
        window_start_sec,
        window_end_sec,
        bundle,
        window_errors,
        window_indices=emitted_window_indices,
        window_targets=emitted_window_targets,
    )


# =========================================================
# 单样本 XGBoost 推理
# =========================================================

def _infer_one_sample(sample, window_sec, stride_sec, bundle,
                      use_stage2_ir=DEFAULT_USE_STAGE2_IR):
    """
    对全部合法窗口提取特征并执行 XGBoost 预测。
    """
    FEATURE_FS = 25

    sample_name = sample.get("sample_name", "unknown")
    target = int(sample.get("target", 0))

    base = {
        "sample_name": sample_name, "target": target,
        "frequency": sample.get("frequency"),
        "ppg_config": sample.get("ppg_config"),
        "mode": sample.get("ppg_config"),
        "window_probs": [], "window_preds": [], "quality_metas": [],
        "window_ood_scores": [],
        "window_start_sec": [], "window_end_sec": [],
        "window_layout": sample.get("window_layout"),
        "window_indices": list(sample.get("window_indices", [])),
        "window_labels": list(sample.get("window_labels", [])),
        "use_stage2_ir": False,
        "fallback": False, "fallback_reason": None,
    }

    # 1-2. 验证 + 加载
    try:
        h5_file = sample.get("h5_file")
        if h5_file is None or sample_name is None:
            base["fallback"] = True; base["fallback_reason"] = "incomplete"; return base
        frequency = get_sample_frequency(sample)
        ppg_config = get_sample_ppg_config(sample)
        base["frequency"] = int(frequency)
        base["ppg_config"] = int(ppg_config)
        base["mode"] = int(ppg_config)
        ok, err = validate_h5_file(h5_file, sample_name)
        if not ok:
            base["fallback"] = True; base["fallback_reason"] = f"h5: {err}"; return base
        ppg = load_ppg(sample)
        acc = load_acc(sample)
    except Exception as e:
        base["fallback"] = True; base["fallback_reason"] = f"load: {e}"; return base

    if is_prewindowed_signal(ppg):
        return _infer_prewindowed_sample(
            base, ppg, acc,
            window_sec, stride_sec, bundle, use_stage2_ir,
        )

    # 3. 100Hz 信号固定按 x[::4] 取为 25Hz（原生 25Hz 跳过）。
    try:
        mode = get_sample_ppg_config(sample)
        base["mode"] = int(mode)

        native_25hz = get_sample_frequency(sample) == FEATURE_FS

        if native_25hz:
            ppg_25 = ppg.astype(np.float64)
            acc_25 = acc.astype(np.float64) if (acc is not None and len(acc) > 0) else None
        else:
            ppg_25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS)
            acc_25 = None
            if acc is not None and len(acc) > 0:
                try:
                    acc_25 = _downsample_ppg(acc, src_fs=100, tgt_fs=FEATURE_FS)
                except Exception:
                    pass

        # Stage2 窗口参数
        win_25 = int(window_sec * FEATURE_FS)
        stride_25 = int(stride_sec * FEATURE_FS)

        n_steps_s2 = (len(ppg_25) - win_25) // stride_25 + 1
        quality_metas = []
        feats_list = []
        window_start_sec = []
        window_end_sec = []
        window_errors = []

        window_steps = list(range(max(0, n_steps_s2)))
        for step in window_steps:
            s2_start = step * stride_25
            window_start_sec.append(float(s2_start / FEATURE_FS))
            window_end_sec.append(float(s2_start / FEATURE_FS + window_sec))

            # Stage2: PPG @ 25Hz
            window = ppg_25[s2_start:s2_start + win_25, :]
            try:
                acc_seg = None
                if acc_25 is not None and len(acc_25) > 0:
                    acc_seg = align_acc_window(acc_25, len(ppg_25), s2_start, win_25,
                                               fs_ppg=FEATURE_FS, fs_acc=FEATURE_FS)
                raw_ppg = None
                raw_acc = None
                commercial_frequency = FEATURE_FS
                if win_25 == 125:
                    source_stride = int(frequency // FEATURE_FS)
                    source_start = int(s2_start * source_stride)
                    source_len = int(125 * source_stride)
                    raw_ppg = ppg[source_start:source_start + source_len]
                    raw_acc = None if acc is None else acc[
                        source_start:source_start + source_len
                    ]
                    commercial_frequency = frequency
                feat = _extract_model_window_features(
                    window,
                    acc_seg,
                    ppg_config=mode,
                    selected_features=bundle.get("feature_names"),
                    use_stage2_ir=use_stage2_ir,
                    source_ppg=raw_ppg,
                    source_acc=raw_acc,
                    source_frequency=commercial_frequency,
                )
                feats_list.append(feat)
                quality_metas.append({
                    "Ambient_std": feat.get("Ambient_std"),
                    "G_mean_mean": feat.get("G_mean_mean"),
                })
            except Exception as exc:
                feats_list.append(None)
                quality_metas.append(None)
                window_errors.append(_window_error_record(step, exc))

        return _finalize_window_inference(
            base,
            feats_list,
            quality_metas,
            window_start_sec,
            window_end_sec,
            bundle,
            window_errors,
        )

    except Exception as e:
        base["fallback"] = True
        base["fallback_reason"] = f"feature_or_predict_error: {e}"
        return base


# =========================================================
# 多进程推理
# =========================================================

_WORKER_BUNDLE = None


def _init_worker(bundle_path):
    """子进程初始化：加载 bundle 一次。"""
    global _WORKER_BUNDLE
    _WORKER_BUNDLE = joblib.load(bundle_path)
    assert_bundle_ok(_WORKER_BUNDLE)
    try:
        _WORKER_BUNDLE["model"].set_params(n_jobs=1)
    except Exception:
        pass


def _worker_infer(args_tuple):
    """子进程入口。"""
    (sample, window_sec, stride_sec, use_stage2_ir) = args_tuple
    return _infer_one_sample(
        sample, window_sec, stride_sec, _WORKER_BUNDLE,
        use_stage2_ir=use_stage2_ir,
    )


def run_inference_parallel(samples, window_sec, stride_sec, bundle_path, n_workers,
                           use_stage2_ir=None):
    """
    并行单趟推理，返回与 samples 同序的 results 列表。
    """
    n_workers = resolve_n_workers(n_workers, n_items=len(samples))
    if use_stage2_ir is None:
        bundle_for_meta = _BUNDLE if _BUNDLE is not None else joblib.load(bundle_path)
        use_stage2_ir = resolve_use_stage2_ir(bundle_for_meta)
    args_list = [
        (s, window_sec, stride_sec, bool(use_stage2_ir))
        for s in samples
    ]

    if n_workers == 1:
        # 单进程路径：复用 _BUNDLE
        bundle = _BUNDLE if _BUNDLE is not None else joblib.load(bundle_path)
        return [
            _infer_one_sample(
                s, window_sec, stride_sec, bundle,
                use_stage2_ir=use_stage2_ir,
            )
            for s in samples
        ]

    results = [None] * len(samples)
    pool_kwargs = {
        "max_workers": n_workers,
        "initializer": _init_worker,
        "initargs": (bundle_path,),
    }
    mp_ctx = multiprocessing_context_from_env()
    if mp_ctx is not None:
        pool_kwargs["mp_context"] = mp_ctx
    with ProcessPoolExecutor(**pool_kwargs) as ex:
        futures = {ex.submit(_worker_infer, a): i for i, a in enumerate(args_list)}
        total = len(futures)
        for done_count, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                logger.warning(f"sample {samples[i].get('sample_name')} inference crashed: {e}")
                results[i] = {
                    "sample_name": samples[i].get("sample_name", "unknown"),
                    "target": int(samples[i].get("target", 0)),
                    "mode": 0,
                    "window_probs": [],
                    "window_preds": [],
                    "quality_metas": [],
                    "fallback": True,
                    "fallback_reason": f"worker_crash: {e}",
                }
            if total >= 10 and (done_count % max(1, total // 10) == 0 or done_count == total):
                print(f"  s06 inference progress: {done_count}/{total} samples", flush=True)
    return results


# =========================================================
# 三套指标
# =========================================================

def _summarize_ood(results, alert_rate=0.3):
    """
    汇总 OOD：每条样本的 mean OOD score、超阈值窗占比。
    返回 dict + 每样本一条记录列表。bundle 未提供 feature_quantiles 时全为 None。
    """
    per_sample = []
    overall_total = 0
    overall_out = 0
    n_alert_samples = 0
    available = False
    for r in results:
        oods = r.get("window_ood_scores", []) or []
        valid = [v for v in oods if v is not None and np.isfinite(v)]
        if valid:
            available = True
            mean_o = float(np.mean(valid))
            high = float(np.mean([1.0 if v > alert_rate else 0.0 for v in valid]))
            overall_out += sum(valid)
            overall_total += len(valid)
            if mean_o > alert_rate:
                n_alert_samples += 1
        else:
            mean_o, high = None, None
        per_sample.append({
            "sample_name": r.get("sample_name"),
            "target": int(r.get("target", 0)),
            "ood_mean": mean_o,
            "ood_window_alert_rate": high,
        })

    return {
        "available": available,
        "alert_rate_threshold": float(alert_rate),
        "global_mean_ood": float(overall_out / overall_total) if overall_total else None,
        "n_alert_samples": int(n_alert_samples),
        "per_sample": per_sample,
    }


def _safe_confusion(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return int(tn), int(fp), int(fn), int(tp)
    except Exception:
        return 0, 0, 0, 0


def build_evaluation_contract(split):
    split = str(split)
    return {
        "split": split,
        "test_read_only": split.lower() == "test",
        "configuration_frozen": True,
        "selection_performed": False,
    }


def compute_sample_metrics(results, method, cfg, model_threshold, stride_sec=1.0):
    """Compute sample metrics from XGBoost probabilities only."""
    y_true, y_pred = [], []
    fallback_count = 0
    samples_with_window_feature_failures = 0
    total_window_feature_failures = 0
    details = []

    for r in results:
        target = int(r["target"])
        if r.get("fallback", False):
            fallback_count += 1
        window_feature_failure_count = max(
            0, int(r.get("window_feature_failure_count", 0) or 0)
        )
        total_window_feature_failures += window_feature_failure_count
        if window_feature_failure_count > 0:
            samples_with_window_feature_failures += 1
        probs = r.get("window_probs", [])
        scores = []
        if r.get("fallback", False) or len(probs) == 0:
            final_pred = 0
            window_preds = list(r.get("window_preds", []))
            states = []
        else:
            final_pred, states, window_preds, scores = apply_postprocess(
                probs, r.get("quality_metas", []), method, cfg, model_threshold,
                stride_sec=stride_sec,
            )

        y_true.append(target)
        y_pred.append(int(final_pred))

        fallback = r.get("fallback", False)
        details.append({
            "sample_name": r.get("sample_name"),
            "target": target,
            "pred": int(final_pred),
            "mode": int(r.get("mode", 0)),
            "fallback": bool(fallback),
            "fallback_reason": r.get("fallback_reason"),
            "window_feature_failure_count": window_feature_failure_count,
            "window_feature_failure_examples": list(
                r.get("window_feature_failure_examples", []) or []
            ),
            "candidate_window_count": max(
                0,
                int(r.get(
                    "candidate_window_count",
                    len(probs) + window_feature_failure_count,
                ) or 0),
            ),
            "window_probs": probs,
            "window_preds": list(window_preds),
            "window_targets": list(r.get("window_targets", []) or []),
            "window_states": list(states),
            "window_scores": list(scores) if not fallback else [],
            "window_start_sec": r.get("window_start_sec", []),
            "window_end_sec": r.get("window_end_sec", []),
            "quality_metas": r.get("quality_metas", []),
            "window_ood_scores": r.get("window_ood_scores", []),
            "n_windows": len(probs),
        })

    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    tn, fp, fn, tp = _safe_confusion(y_true_a, y_pred_a)

    summary = {
        "method": method,
        "total_samples": int(len(results)),
        "fallback_samples": int(fallback_count),
        "samples_with_window_feature_failures": int(samples_with_window_feature_failures),
        "total_window_feature_failures": int(total_window_feature_failures),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)) if len(y_true_a) > 0 else 0.0,
        "precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
        "postprocess": cfg,
        "evaluation_semantics": (
            "xgboost_postprocess_only_v1"
            if method == "state_machine"
            else "xgboost_only_v1"
        ),
    }
    return summary, details


PER_SAMPLE_INFERENCE_SUMMARY_COLUMNS = [
    "sample_name", "target", "pred", "sample_is_correct", "sample_error_type",
    "ppg_config", "is_fallback", "fallback_reason",
    "candidate_windows", "successful_windows", "feature_failed_windows",
    "window_correct", "window_wrong", "window_accuracy", "window_error_rate",
    "window_tn", "window_fp", "window_fn", "window_tp",
    "positive_windows", "negative_windows", "positive_window_rate",
    "window_probability_mean", "window_probability_min", "window_probability_max",
    "window_probability_std", "ood_mean", "ood_window_alert_rate",
]


def _binary_error_type(target, prediction):
    target = int(target)
    prediction = int(prediction)
    if target == 1:
        return "TP" if prediction == 1 else "FN"
    return "FP" if prediction == 1 else "TN"


def _finite_summary(values):
    valid = [float(value) for value in (values or []) if value is not None and np.isfinite(value)]
    if not valid:
        return None, None, None, None
    array = np.asarray(valid, dtype=float)
    return (
        float(np.mean(array)),
        float(np.min(array)),
        float(np.max(array)),
        float(np.std(array)),
    )


def build_per_sample_inference_summary_rows(details):
    """Return one all-sample inference-statistics record per evaluation detail."""
    rows = []
    for detail in details:
        target = int(detail.get("target", 0))
        pred = int(detail.get("pred", 0))
        preds = [int(value) for value in (detail.get("window_preds", []) or [])]
        raw_targets = list(detail.get("window_targets", []) or [])
        window_targets = [
            int(raw_targets[index]) if index < len(raw_targets) else target
            for index in range(len(preds))
        ]
        successful = len(preds)
        correct = sum(
            int(prediction == actual)
            for prediction, actual in zip(preds, window_targets)
        )
        wrong = successful - correct
        tn, fp, fn, tp = _safe_confusion(
            np.asarray(window_targets, dtype=int), np.asarray(preds, dtype=int)
        ) if successful else (0, 0, 0, 0)
        failures = max(0, int(detail.get("window_feature_failure_count", 0) or 0))
        candidates = max(
            0,
            int(detail.get("candidate_window_count", successful + failures) or 0),
        )
        prob_mean, prob_min, prob_max, prob_std = _finite_summary(
            detail.get("window_probs", [])
        )
        ood_mean, _, _, _ = _finite_summary(detail.get("window_ood_scores", []))
        rows.append({
            "sample_name": str(detail.get("sample_name", "")),
            "target": target,
            "pred": pred,
            "sample_is_correct": int(pred == target),
            "sample_error_type": _binary_error_type(target, pred),
            "ppg_config": int(detail.get("mode", 0) or 0),
            "is_fallback": int(bool(detail.get("fallback", False))),
            "fallback_reason": detail.get("fallback_reason"),
            "candidate_windows": candidates,
            "successful_windows": successful,
            "feature_failed_windows": failures,
            "window_correct": correct,
            "window_wrong": wrong,
            "window_accuracy": float(correct / successful) if successful else None,
            "window_error_rate": float(wrong / successful) if successful else None,
            "window_tn": tn,
            "window_fp": fp,
            "window_fn": fn,
            "window_tp": tp,
            "positive_windows": int(sum(preds)),
            "negative_windows": int(successful - sum(preds)),
            "positive_window_rate": float(sum(preds) / successful) if successful else None,
            "window_probability_mean": prob_mean,
            "window_probability_min": prob_min,
            "window_probability_max": prob_max,
            "window_probability_std": prob_std,
            "ood_mean": detail.get("ood_mean", ood_mean),
            "ood_window_alert_rate": detail.get("ood_window_alert_rate"),
        })
    return rows


def export_per_sample_inference_summary(details, artifact_dir, split, method):
    """Export all evaluated samples, including fallbacks and no-window records."""
    os.makedirs(artifact_dir, exist_ok=True)
    out_path = os.path.join(
        artifact_dir, f"per_sample_inference_summary_{split}_{method}.csv"
    )
    rows = build_per_sample_inference_summary_rows(details)
    pd.DataFrame(rows, columns=PER_SAMPLE_INFERENCE_SUMMARY_COLUMNS).to_csv(
        out_path, index=False, encoding="utf-8"
    )
    return out_path


def _detail_prob_stats(detail):
    probs = detail.get("window_probs", []) or []
    vals = [float(p) for p in probs if p is not None and np.isfinite(p)]
    if not vals:
        return {"mean_prob": 0.0, "max_prob": 0.0, "n_windows": int(detail.get("n_windows", 0))}
    return {
        "mean_prob": float(np.mean(vals)),
        "max_prob": float(np.max(vals)),
        "n_windows": int(detail.get("n_windows", len(vals))),
    }


def _metrics_for_details(rows):
    if not rows:
        return {
            "n_samples": 0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    y_true = np.asarray([int(d.get("target", 0)) for d in rows])
    y_pred = np.asarray([int(d.get("pred", 0)) for d in rows])
    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)
    return {
        "n_samples": int(len(rows)),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _bucket_n_windows(n):
    n = int(n or 0)
    if n == 0:
        return "nwin=0"
    if n <= 3:
        return "nwin=1-3"
    if n <= 10:
        return "nwin=4-10"
    return "nwin>10"


def _bucket_prob(p):
    p = float(p or 0.0)
    if p < 0.3:
        return "mean_prob<0.3"
    if p < 0.6:
        return "mean_prob=0.3-0.6"
    return "mean_prob>=0.6"


def _bucket_ood(v):
    if v is None or not np.isfinite(v):
        return "ood=missing"
    if v <= 0.0:
        return "ood=0"
    if v < 0.3:
        return "ood=0-0.3"
    return "ood>=0.3"


def compute_stratified_error_analysis(details):
    """Compute sample-level metrics by deployment-relevant strata."""
    strata = {
        "mode": defaultdict(list),
        "n_windows": defaultdict(list),
        "mean_prob": defaultdict(list),
        "ood_window_alert_rate": defaultdict(list),
    }
    for d in details:
        stats = _detail_prob_stats(d)
        strata["mode"][f"mode={int(d.get('mode', 0))}"].append(d)
        strata["n_windows"][_bucket_n_windows(stats["n_windows"])].append(d)
        strata["mean_prob"][_bucket_prob(stats["mean_prob"])].append(d)
        strata["ood_window_alert_rate"][_bucket_ood(d.get("ood_window_alert_rate"))].append(d)

    out = {}
    for name, buckets in strata.items():
        out[name] = {
            bucket: _metrics_for_details(rows)
            for bucket, rows in sorted(buckets.items(), key=lambda kv: kv[0])
        }
    out["overall"] = _metrics_for_details(details)
    return out


def mine_hard_negatives(details, top_k=50):
    """Return FP samples and high-risk negatives sorted by probability pressure."""
    rows = []
    for d in details:
        if int(d.get("target", 0)) != 0:
            continue
        stats = _detail_prob_stats(d)
        row = {
            "sample_name": d.get("sample_name"),
            "target": int(d.get("target", 0)),
            "pred": int(d.get("pred", 0)),
            "mode": int(d.get("mode", 0)),
            "n_windows": stats["n_windows"],
            "mean_prob": stats["mean_prob"],
            "max_prob": stats["max_prob"],
            "ood_window_alert_rate": d.get("ood_window_alert_rate"),
            "fallback": bool(d.get("fallback", False)),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: (r["pred"], r["max_prob"], r["mean_prob"]), reverse=True)
    false_positives = [r for r in rows if r["pred"] == 1][:top_k]
    high_risk = sorted(rows, key=lambda r: (r["max_prob"], r["mean_prob"]), reverse=True)[:top_k]
    return {
        "top_k": int(top_k),
        "false_positives": false_positives,
        "high_risk_negatives": high_risk,
    }


def _cm_array(cm):
    return np.asarray([
        [int(cm.get("TN", 0)), int(cm.get("FP", 0))],
        [int(cm.get("FN", 0)), int(cm.get("TP", 0))],
    ], dtype=int)


def export_deploy_report_plot(payload, artifact_dir, split="test", method="state_machine"):
    """Export a report-style PNG for deploy evaluation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip s06 plot: {e}")
        return None

    out_dir = os.path.join(str(artifact_dir), "report_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s06_deploy_report.png")

    sample = payload.get("summary", {}) or {}
    window_model = payload.get("window_model_summary", {}) or {}
    window_stream = payload.get("window_stream_summary", {}) or {}
    details = payload.get("details", []) or []
    hard = payload.get("hard_negative_report", {}) or {}
    strat = payload.get("stratified_errors", {}) or {}

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.15, 1.1])
    ax_funnel = fig.add_subplot(gs[0, 0])
    ax_cm = fig.add_subplot(gs[0, 1])
    ax_prob = fig.add_subplot(gs[0, 2])
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_hard = fig.add_subplot(gs[1, 1])
    ax_strata = fig.add_subplot(gs[1, 2])

    total = int(sample.get("total_samples", len(details)))
    pred_pos = int(sample.get("confusion_matrix", {}).get("FP", 0) + sample.get("confusion_matrix", {}).get("TP", 0))
    ax_funnel.bar(["input", "XGBoost positive"], [total, pred_pos],
                  color=["#4c78a8", "#72b7b2"])
    ax_funnel.set_title("XGBoost Sample Output")
    ax_funnel.set_ylabel("samples")
    ax_funnel.grid(axis="y", alpha=0.18)

    cm = _cm_array(sample.get("confusion_matrix", {}))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title("Sample Confusion Matrix")
    ax_cm.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax_cm.set_yticks([0, 1], ["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="#1f2933", weight="bold")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    neg_probs, pos_probs = [], []
    for d in details:
        probs = [float(p) for p in (d.get("window_probs", []) or []) if p is not None and np.isfinite(p)]
        if int(d.get("target", 0)) == 0:
            neg_probs.extend(probs)
        else:
            pos_probs.extend(probs)
    bins = np.linspace(0, 1, 21)
    if neg_probs:
        ax_prob.hist(neg_probs, bins=bins, alpha=0.65, color="#c44e52", label="target 0")
    if pos_probs:
        ax_prob.hist(pos_probs, bins=bins, alpha=0.65, color="#2f6f73", label="target 1")
    ax_prob.set_title("Window Probability Distribution")
    ax_prob.set_xlabel("probability")
    ax_prob.set_ylabel("windows")
    ax_prob.legend(frameon=False)
    ax_prob.grid(axis="y", alpha=0.18)

    metric_names = ["precision", "recall", "f1"]
    blocks = [("sample", sample), ("window model", window_model), ("state stream", window_stream)]
    x = np.arange(len(blocks))
    width = 0.24
    colors = ["#2f6f73", "#4c78a8", "#8172b2"]
    for i, m in enumerate(metric_names):
        ax_metrics.bar(x + (i - 1) * width, [float(b[1].get(m, 0.0)) for b in blocks],
                       width=width, label=m, color=colors[i])
    ax_metrics.set_xticks(x, [b[0] for b in blocks], rotation=15, ha="right")
    ax_metrics.set_ylim(0, 1.03)
    ax_metrics.set_title("Metric Comparison")
    ax_metrics.grid(axis="y", alpha=0.18)
    ax_metrics.legend(frameon=False)

    fps = hard.get("false_positives", [])[:10]
    if fps:
        names = [str(x.get("sample_name", ""))[-22:] for x in fps][::-1]
        vals = [float(x.get("max_prob", 0.0)) for x in fps][::-1]
        ax_hard.barh(np.arange(len(names)), vals, color="#c44e52")
        ax_hard.set_yticks(np.arange(len(names)), names, fontsize=8)
        ax_hard.set_xlim(0, 1)
        ax_hard.set_xlabel("max probability")
    else:
        ax_hard.text(0.5, 0.5, "No false positives", ha="center", va="center")
        ax_hard.set_axis_off()
    ax_hard.set_title("Top False Positives")
    ax_hard.grid(axis="x", alpha=0.18)

    mode_buckets = strat.get("mode", {}) or {}
    if mode_buckets:
        labels = list(mode_buckets.keys())
        fp_vals = [int(mode_buckets[k].get("confusion_matrix", {}).get("FP", 0)) for k in labels]
        fn_vals = [int(mode_buckets[k].get("confusion_matrix", {}).get("FN", 0)) for k in labels]
        xx = np.arange(len(labels))
        ax_strata.bar(xx - 0.18, fp_vals, width=0.36, color="#c44e52", label="FP")
        ax_strata.bar(xx + 0.18, fn_vals, width=0.36, color="#4c78a8", label="FN")
        ax_strata.set_xticks(xx, labels, rotation=20, ha="right")
        ax_strata.legend(frameon=False)
    else:
        ax_strata.text(0.5, 0.5, "No strata", ha="center", va="center")
        ax_strata.set_axis_off()
    ax_strata.set_title("Error by PPG Configuration")
    ax_strata.grid(axis="y", alpha=0.18)

    fig.suptitle(f"Deployment Evaluation Report ({split}, {method})", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    source_rows = []
    for level, block in blocks:
        for metric in ["accuracy", "precision", "recall", "f1", "fp_rate"]:
            if metric in block:
                source_rows.append({
                    "panel": "metrics", "level": level,
                    "metric": metric, "value": block.get(metric),
                })
    for detail in details:
        for index, prob in enumerate(detail.get("window_probs", []) or []):
            if prob is not None and np.isfinite(prob):
                source_rows.append({
                    "panel": "window_probability",
                    "sample_name": detail.get("sample_name"),
                    "target": detail.get("target"),
                    "window_index": index,
                    "value": float(prob),
                })
    if not source_rows:
        source_rows.append({"panel": "summary", "metric": "total_samples", "value": total})
    save_scientific_figure(
        fig,
        out_path,
        source_data=pd.DataFrame(source_rows),
        inputs=[os.path.splitext(out_path)[0] + "_source_data.csv"],
        core_conclusion=(
            "Frozen end-to-end evaluation reports sample and window behavior, error modes, "
            "and probability separation without changing the evaluated configuration."
        ),
        panel_map={
            "a": "Stage funnel.", "b": "Sample confusion matrix.",
            "c": "Window probability distributions.", "d": "Metric comparison.",
            "e": "Highest-confidence false positives.", "f": "Errors by PPG configuration.",
        },
        split=str(split),
        n_definition="samples in the frozen evaluation split; probability panel uses eligible windows",
        statistics={"metrics": "point estimates on the named frozen split"},
        reviewer_risks=["Window observations within a sample are correlated and are not independent replicates."],
        test_read_only=str(split).lower() == "test",
    )
    plt.close(fig)
    print(f"[OK] s06 report plot -> {out_path}")
    return out_path


def compute_window_model_metrics(results):
    """
    独立 Stage2 模型评估（全部合法窗口）：
    逐窗 XGBoost 预测 (window_preds) vs window_targets；旧数据缺少
    window_targets 时回退到整条样本的 target。

    模型无状态，不做 warmup 跳过。
    """
    y_true, y_pred = [], []
    samples_with_no_windows = 0
    total_input_samples = len(results)
    for r in results:
        wp = r.get("window_preds", [])
        if r.get("fallback", False) or len(wp) == 0:
            samples_with_no_windows += 1
            continue
        sample_target = int(r["target"])
        window_targets = r.get("window_targets", [])
        for idx, p in enumerate(wp):
            t = int(_safe_list_get(window_targets, idx, sample_target))
            y_true.append(t)
            y_pred.append(int(p))

    if len(y_true) == 0:
        return {
            "total_input_samples": total_input_samples,
            "samples_with_no_windows": samples_with_no_windows,
            "total_windows": 0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    tn, fp, fn, tp = _safe_confusion(y_true_a, y_pred_a)
    return {
        "total_input_samples": total_input_samples,
        "samples_with_no_windows": samples_with_no_windows,
        "total_windows": int(len(y_true_a)),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
    }


def _window_error_type(target, pred):
    target = int(target)
    pred = int(pred)
    if target == 0 and pred == 0:
        return "TN"
    if target == 0 and pred == 1:
        return "FP"
    if target == 1 and pred == 0:
        return "FN"
    return "TP"


def _bucket_window_prob(prob):
    prob = float(prob)
    if prob < 0.2:
        return "prob<0.2"
    if prob < 0.5:
        return "prob=0.2-0.5"
    if prob < 0.8:
        return "prob=0.5-0.8"
    return "prob>=0.8"


def _bucket_window_time(start_sec):
    start_sec = float(start_sec)
    if start_sec < 6.0:
        return "start<6s"
    if start_sec < 15.0:
        return "start=6-15s"
    return "start>=15s"


def _bucket_window_ood(value):
    if value is None or not np.isfinite(float(value)):
        return "ood=missing"
    value = float(value)
    if value <= 0.0:
        return "ood=0"
    if value < 0.3:
        return "ood=0-0.3"
    return "ood>=0.3"


def _quality_meta_score(meta):
    if not isinstance(meta, dict) or not meta:
        return None
    vals = []
    for value in meta.values():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            vals.append(abs(value))
    if not vals:
        return None
    return float(np.mean(vals))


def _bucket_quality_meta(meta):
    score = _quality_meta_score(meta)
    if score is None:
        return "quality=missing"
    if score < 1.0:
        return "quality_meta<1"
    if score < 10.0:
        return "quality_meta=1-10"
    return "quality_meta>=10"


def _safe_list_get(values, idx, default=None):
    if values is None:
        return default
    try:
        if idx < len(values):
            return values[idx]
    except TypeError:
        return default
    return default


def _window_rows_from_details(details):
    rows = []
    for detail in details:
        if detail.get("fallback", False):
            continue
        probs = detail.get("window_probs", []) or []
        preds = detail.get("window_preds", []) or []
        if not probs or not preds:
            continue
        sample_target = int(detail.get("target", 0))
        window_targets = detail.get("window_targets", [])
        window_indices = detail.get("window_indices", [])
        sample_name = detail.get("sample_name")
        mode = int(detail.get("mode", 0))
        starts = detail.get("window_start_sec", [])
        ends = detail.get("window_end_sec", [])
        q_metas = detail.get("quality_metas", [])
        oods = detail.get("window_ood_scores", [])
        for idx, prob in enumerate(probs):
            pred = int(_safe_list_get(preds, idx, 0))
            target = int(_safe_list_get(window_targets, idx, sample_target))
            window_index = int(_safe_list_get(window_indices, idx, idx))
            start = float(_safe_list_get(starts, idx, idx))
            end = float(_safe_list_get(ends, idx, start))
            q_meta = _safe_list_get(q_metas, idx, {})
            ood = _safe_list_get(oods, idx, None)
            error_type = _window_error_type(target, pred)
            rows.append({
                "sample_name": sample_name,
                "target": target,
                "pred_raw": pred,
                "error_type": error_type,
                "is_error": int(error_type in {"FP", "FN"}),
                "window_index": window_index,
                "window_start_sec": start,
                "window_end_sec": end,
                "prob_raw": float(prob),
                "prob_bin": _bucket_window_prob(prob),
                "time_bin": _bucket_window_time(start),
                "mode": mode,
                "ood_rate": None if ood is None else float(ood),
                "ood_bin": _bucket_window_ood(ood),
                "quality_bin": _bucket_quality_meta(q_meta),
            })
    return rows


def _summarize_window_rows(rows):
    if not rows:
        return {
            "total_windows": 0,
            "error_windows": 0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fp_rate": 0.0,
        }
    y_true = np.asarray([r["target"] for r in rows], dtype=int)
    y_pred = np.asarray([r["pred_raw"] for r in rows], dtype=int)
    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)
    n_neg = max(tn + fp, 1)
    return {
        "total_windows": int(len(rows)),
        "error_windows": int(fp + fn),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fp_rate": float(fp) / float(n_neg),
    }


def _summarize_strata(rows, key):
    out = {}
    buckets = sorted({str(r.get(key)) for r in rows})
    for bucket in buckets:
        sub = [r for r in rows if str(r.get(key)) == bucket]
        cm = _summarize_window_rows(sub)["confusion_matrix"]
        out[bucket] = {
            "n_windows": int(len(sub)),
            "errors": int(cm["FP"] + cm["FN"]),
            "fp": int(cm["FP"]),
            "fn": int(cm["FN"]),
            "tp": int(cm["TP"]),
            "tn": int(cm["TN"]),
            "accuracy": _summarize_window_rows(sub)["accuracy"],
        }
    return out


def compute_window_error_analysis(details):
    """Build a raw Stage2 window-level error report from s06 sample details."""
    rows = _window_rows_from_details(details)
    return {
        "summary": _summarize_window_rows(rows),
        "strata": {
            "error_type": _summarize_strata(rows, "error_type"),
            "prob_bin": _summarize_strata(rows, "prob_bin"),
            "time_bin": _summarize_strata(rows, "time_bin"),
            "mode": _summarize_strata(rows, "mode"),
            "ood_bin": _summarize_strata(rows, "ood_bin"),
            "quality_bin": _summarize_strata(rows, "quality_bin"),
        },
        "rows": rows,
    }


def export_window_error_analysis(report, artifact_dir, split, method):
    out_csv = os.path.join(
        os.fspath(artifact_dir), f"window_error_analysis_{split}_{method}.csv")
    out_json = os.path.join(
        os.fspath(artifact_dir), f"window_error_analysis_{split}_{method}.json")
    rows = report.get("rows", [])
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    payload = {
        "summary": report.get("summary", {}),
        "strata": report.get("strata", {}),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_csv, out_json


def compute_window_stream_metrics(results, cfg, warmup_frames=0, model_threshold=None,
                                  stride_sec=1.0):
    """Compute streaming metrics and attach the exact state trace to each row.

    ``stage2_states`` and ``stage2_scores`` are written to every mutable result
    row so downstream CSV/report generation uses the same trace as this summary.
    The sample-level ``pred`` remains governed by the explicitly selected method.
    """
    y_true, y_pred = [], []
    samples_with_no_windows = 0
    skipped_windows = 0

    # 对齐 T_on / T_off：当 cfg 中的值与默认值相同时，从模型阈值推导
    if model_threshold is None:
        model_threshold = float(_BUNDLE.get("threshold", 0.5)) if _BUNDLE is not None else 0.5
    _t_on, _t_off = resolve_postprocess_thresholds(cfg, model_threshold)

    for r in results:
        probs = r.get("window_probs", [])
        qm = r.get("quality_metas", [])
        if r.get("fallback", False) or len(probs) == 0:
            r["stage2_states"] = []
            r["stage2_scores"] = []
            samples_with_no_windows += 1
            continue

        sample_target = int(r["target"])
        window_targets = r.get("window_targets", [])
        qt = _BUNDLE.get("quality_thresholds") if _BUNDLE is not None else None

        # 全程运行状态机，使其积累 EMA 和 on/off count
        sample_states = []
        sample_scores = []
        sample_targets = []
        probs_for_state = causal_median_filter_1d(probs, cfg.get("median_k", DEFAULT_POSTPROCESS_CONFIG["median_k"]))
        sm = WearStateMachine(
            alpha=cfg.get("alpha", 0.4),
            T_on=_t_on,
            T_off=_t_off,
            K_on=cfg.get("K_on", DEFAULT_POSTPROCESS_CONFIG["K_on"]),
            K_off=cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]),
            cooldown_sec=cfg.get("cooldown_sec", DEFAULT_POSTPROCESS_CONFIG["cooldown_sec"]),
        )
        for i, p in enumerate(probs_for_state):
            meta_i = qm[i] if i < len(qm) else None
            q = compute_quality(meta_i, thresholds=qt) if meta_i else 1.0
            state, score = sm.update(p, quality=q, stride_sec=stride_sec)
            sample_states.append(int(state))
            sample_scores.append(float(score))
            sample_targets.append(int(_safe_list_get(window_targets, i, sample_target)))

        r["stage2_states"] = sample_states
        r["stage2_scores"] = sample_scores

        # Warmup windows are excluded from this state-machine metric.
        start = min(max(0, int(warmup_frames)), len(sample_states))
        skipped_windows += start
        for t, s in zip(sample_targets[start:], sample_states[start:]):
            y_true.append(t)
            y_pred.append(s)

    if len(y_true) == 0:
        return {
            "total_samples": len(results),
            "samples_with_no_windows": samples_with_no_windows,
            "warmup_frames": int(warmup_frames),
            "skipped_warmup_windows": int(skipped_windows),
            "total_windows": 0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    tn, fp, fn, tp = _safe_confusion(y_true_a, y_pred_a)
    return {
        "total_samples": len(results),
        "samples_with_no_windows": samples_with_no_windows,
        "warmup_frames": int(warmup_frames),
        "skipped_warmup_windows": int(skipped_windows),
        "total_windows": int(len(y_true_a)),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
    }


# =========================================================
# 状态机网格搜索（并行）
# =========================================================

def _score_grid_point(args_tuple):
    """子进程评分一个网格点，返回 (params, metrics_dict)。"""
    alpha, T_on, T_off, K_on, K_off, cache_pickle = args_tuple
    import pickle
    data = pickle.loads(cache_pickle)
    cache = data["samples"]
    quality_thresholds = data.get("quality_thresholds")
    stride_sec = float(data.get("stride_sec", 1.0))

    y_true, y_pred = [], []
    for s in cache:
        target = s["target"]
        probs = s.get("probs", [])
        qm = s.get("quality_metas", [])
        if len(probs) == 0:
            pred = 0
        else:
            sm = WearStateMachine(alpha=alpha, T_on=T_on, T_off=T_off, K_on=K_on, K_off=K_off)
            state = 0
            for i, p in enumerate(probs):
                q = compute_quality(qm[i], thresholds=quality_thresholds) if i < len(qm) and qm[i] else 1.0
                state, _ = sm.update(p, quality=q, stride_sec=stride_sec)
            pred = int(state)
        y_true.append(target)
        y_pred.append(pred)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    params = {"alpha": alpha, "T_on": T_on, "T_off": T_off, "K_on": K_on, "K_off": K_off}
    return params, {"recall": float(rec), "precision": float(prec), "f1": float(f1)}


def optimize_state_machine_params(samples, window_sec=5, stride_sec=1, min_recall=0.95,
                                   bundle_path=None, n_workers=None,
                                   use_stage2_ir=None):
    """
    网格搜索最优状态机参数。
    优化:
    - 推理一次性并行产出 probs 缓存
    - 网格点评分并行
    """
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    T_ons = [0.45, 0.5, 0.6, 0.7, 0.8]
    T_offs = [0.2, 0.3, 0.4]
    K_ons = [3, 5, 7, 10, 15]
    K_offs = [3, 5, 7, 10]   # K_off <= K_on (FP cost high, FN more tolerable)

    print("状态机参数优化配置:")
    print(f"  搜索空间 alpha={alphas} T_on={T_ons} T_off={T_offs}")
    print(f"    K_on={K_ons} K_off={K_offs}")
    print(f"  目标: recall>={min_recall*100}% 前提下最大化 F1")

    # 1. 单趟并行推理拿 probs
    if bundle_path is None:
        raise ValueError("optimize_state_machine_params 需要 bundle_path")
    n_workers = resolve_n_workers(n_workers, n_items=len(samples))

    print("预计算样本窗口概率（并行）...")
    results = run_inference_parallel(
        samples, window_sec, stride_sec, bundle_path, n_workers,
        use_stage2_ir=use_stage2_ir,
    )

    sample_cache = []
    for r in results:
        sample_cache.append({
            "sample_name": r["sample_name"],
            "target": int(r["target"]),
            "probs": r.get("window_probs", []),
            "quality_metas": r.get("quality_metas", []),
        })
    target_1 = [s for s in sample_cache if s["target"] == 1]
    print(f"有效样本: {len(sample_cache)}, target=1 样本: {len(target_1)}")

    # 2. 准备网格任务 (K_on >= K_off: FP cost high)
    grid = [(a, t_on, t_off, ko, kf)
            for a, t_on, t_off, ko, kf in product(alphas, T_ons, T_offs, K_ons, K_offs)
            if t_off < t_on and ko >= kf]
    print(f"待评分网格点: {len(grid)}")

    # 3. 并行评分（含 bundle 的 quality_thresholds）
    import pickle
    _bundle = joblib.load(bundle_path) if _BUNDLE is None else _BUNDLE
    score_data = {
        "samples": sample_cache,
        "quality_thresholds": _bundle.get("quality_thresholds"),
        "stride_sec": float(stride_sec),
    }
    cache_pickle = pickle.dumps(score_data, protocol=pickle.HIGHEST_PROTOCOL)
    task_args = [(a, t_on, t_off, ko, kf, cache_pickle) for (a, t_on, t_off, ko, kf) in grid]

    scored = []
    grid_workers = resolve_n_workers(n_workers, n_items=len(grid))
    if grid_workers == 1:
        for ta in task_args:
            scored.append(_score_grid_point(ta))
    else:
        pool_kwargs = {"max_workers": grid_workers}
        mp_ctx = multiprocessing_context_from_env()
        if mp_ctx is not None:
            pool_kwargs["mp_context"] = mp_ctx
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            futures = {ex.submit(_score_grid_point, ta): i for i, ta in enumerate(task_args)}
            total = len(futures)
            for done_count, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                scored.append(res)
                if done_count % max(1, total // 10) == 0 or done_count == total:
                    print(f"  s06 grid progress: {done_count}/{total}", flush=True)

    # 4. 选最优（recall>=min_recall 下最大 F1；若无则取 recall 最高）
    best_params, best_metrics, best_f1 = None, None, -1.0
    for params, metrics in scored:
        if metrics["recall"] >= min_recall and metrics["f1"] > best_f1:
            best_params, best_metrics, best_f1 = params, metrics, metrics["f1"]

    if best_params is None:
        print(f"警告: 未找到 recall>={min_recall} 的组合，放宽约束（取 recall 最高）")
        for params, metrics in scored:
            if best_metrics is None or metrics["recall"] > best_metrics["recall"] or \
               (metrics["recall"] == best_metrics["recall"] and metrics["f1"] > best_f1):
                best_params, best_metrics, best_f1 = params, metrics, metrics["f1"]

    print(f"最优参数: {best_params}")
    print(f"评估指标: {best_metrics}")
    return {"best_params": best_params, "best_metrics": best_metrics}


# =========================================================
# 向后兼容 API
# =========================================================

def predict_sample_with_bundle(sample, window_sec=5, stride_sec=1,
                                method="state_machine", postprocess_cfg=None,
                                use_stage2_ir=None):
    """
    单样本部署预测（向后兼容）。内部走优化后的推理路径（无重复 preprocess）。
    """
    assert _BUNDLE is not None, "must call load_bundle() first"

    if postprocess_cfg is None:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)

    r = _infer_one_sample(sample, window_sec, stride_sec, _BUNDLE,
                          use_stage2_ir=resolve_use_stage2_ir(_BUNDLE, use_stage2_ir))
    target = int(r["target"])
    if len(r["window_probs"]) == 0 or r["fallback"]:
        return {
            "sample_name": r["sample_name"],
            "target": target,
            "pred": 0,
            "window_probs": list(r.get("window_probs", [])),
            "window_preds": list(r.get("window_preds", [])),
        }

    final_pred, states, window_preds, _scores = apply_postprocess(
        r["window_probs"], r["quality_metas"], method, postprocess_cfg,
        _BUNDLE["threshold"], stride_sec=stride_sec,
    )
    return {
        "sample_name": r["sample_name"],
        "target": target,
        "pred": int(final_pred),
        "mode": int(r["mode"]),
        "window_probs": list(r["window_probs"]),
        "window_preds": list(window_preds),
        "window_states": [int(s) for s in states],
    }


def predict_sample_safe(sample, window_sec=5, stride_sec=1,
                        method="state_machine", postprocess_cfg=None,
                        use_stage2_ir=None):
    """带 fallback 的预测（向后兼容）。"""
    fallback_result = {
        "sample_name": sample.get("sample_name", "unknown"),
        "target": int(sample.get("target", 0)),
        "pred": 0,
        "mode": 0,
        "window_probs": [],
        "window_preds": [],
        "fallback": True,
        "fallback_reason": None,
    }
    h5_file = sample.get("h5_file")
    sample_name = sample.get("sample_name")
    if h5_file is None or sample_name is None:
        fallback_result["fallback_reason"] = "incomplete_sample_info"
        return fallback_result
    ok, err = validate_h5_file(h5_file, sample_name)
    if not ok:
        fallback_result["fallback_reason"] = f"h5_validation_failed: {err}"
        return fallback_result
    try:
        res = predict_sample_with_bundle(
            sample,
            window_sec=window_sec, stride_sec=stride_sec,
            method=method, postprocess_cfg=postprocess_cfg,
            use_stage2_ir=use_stage2_ir,
        )
        res["fallback"] = False
        res["fallback_reason"] = None
        return res
    except Exception as e:
        logger.warning(f"预测失败使用 fallback: {e}")
        fallback_result["fallback_reason"] = f"prediction_error: {e}"
        return fallback_result


def evaluate_streaming_window_accuracy(samples, window_sec=5, stride_sec=1, postprocess_cfg=None,
                                       bundle_path=None, n_workers=None,
                                       use_stage2_ir=None):
    """
    流式窗口级评估（向后兼容）。可选并行推理。
    返回结构与旧版一致：accuracy/precision/recall/f1/total_windows/total_samples/
    valid_samples/confusion_matrix/sample_details。
    """
    if postprocess_cfg is None:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)

    # 推理（如能并行则并行）
    if bundle_path is not None and (n_workers is None or n_workers > 1):
        if n_workers is None:
            n_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
        results = run_inference_parallel(
            samples, window_sec, stride_sec, bundle_path, n_workers,
            use_stage2_ir=use_stage2_ir,
        )
    else:
        # 单进程：要求 _BUNDLE 已加载
        assert _BUNDLE is not None, "must call load_bundle() first"
        results = [
            _infer_one_sample(
                s, window_sec, stride_sec, _BUNDLE,
                use_stage2_ir=resolve_use_stage2_ir(_BUNDLE, use_stage2_ir),
            )
            for s in samples
        ]

    all_true, all_pred, details = [], [], []
    for r in results:
        target = int(r["target"])
        probs = r.get("window_probs", [])
        qm = r.get("quality_metas", [])
        if r.get("fallback", False) or len(probs) == 0:
            details.append({
                "sample_name": r.get("sample_name"),
                "target": target,
                "n_windows": 0,
                "window_states": [],
            })
            continue
        model_threshold = float(_BUNDLE.get("threshold", 0.5)) if _BUNDLE is not None else 0.5
        _t_on, _t_off = resolve_postprocess_thresholds(postprocess_cfg, model_threshold)
        sm = WearStateMachine(
            alpha=postprocess_cfg.get("alpha", 0.4),
            T_on=_t_on,
            T_off=_t_off,
            K_on=postprocess_cfg.get("K_on", 5),
            K_off=postprocess_cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]),
        )
        qt = _BUNDLE.get("quality_thresholds") if _BUNDLE is not None else None
        states = []
        for i, p in enumerate(probs):
            q = compute_quality(qm[i], thresholds=qt) if i < len(qm) and qm[i] else 1.0
            state, _ = sm.update(p, quality=q, stride_sec=stride_sec)
            states.append(state)
            all_true.append(target)
            all_pred.append(state)
        details.append({
            "sample_name": r.get("sample_name"),
            "target": target,
            "n_windows": len(states),
            "window_states": states,
        })

    if len(all_true) == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "total_windows": 0, "total_samples": len(samples), "valid_samples": 0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
            "sample_details": details,
        }

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)
    valid = sum(1 for d in details if d["n_windows"] > 0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "total_windows": int(len(y_true)),
        "total_samples": int(len(samples)),
        "valid_samples": int(valid),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "sample_details": details,
    }


def predict_sample(sample, model, scaler, selected_features,
                    window_sec=5, stride_sec=1, fs=25,
                    method="state_machine",
                    model_threshold=0.5,
                    postprocess_cfg=None,
                    fill_values=None,
                    use_stage2_ir=DEFAULT_USE_STAGE2_IR):
    """
    [DEPRECATED] Legacy 路径（基于显式 scaler / 旧 fill_values 字典）。
    保留接口以防外部调用；新代码请使用 predict_sample_with_bundle。
    """
    import warnings
    warnings.warn(
        "predict_sample 已弃用，请改用 predict_sample_with_bundle（基于 model_bundle.pkl）。",
        DeprecationWarning, stacklevel=2
    )
    if fill_values is None:
        fill_values = {}
    ppg = load_ppg(sample)

    mode = get_sample_ppg_config(sample)
    win = window_sec * fs
    stride = stride_sec * fs

    if postprocess_cfg is None:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)

    _t_on, _t_off = resolve_postprocess_thresholds(postprocess_cfg, model_threshold)
    sm = WearStateMachine(
        alpha=postprocess_cfg.get("alpha", 0.4),
        T_on=_t_on,
        T_off=_t_off,
        K_on=postprocess_cfg.get("K_on", DEFAULT_POSTPROCESS_CONFIG["K_on"]),
        K_off=postprocess_cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]),
    )

    probs, window_preds, states = [], [], []

    for start in range(0, len(ppg) - win + 1, stride):
        window = ppg[start:start + win, :]
        try:
            ir, ambient, g1, g2, g3 = get_channels_from_window(window, mode)
            feat = extract_feature_pool_from_window(
                ir, ambient, g1, g2, g3, fs=fs,
                selected_features=selected_features,
            )
            feat["mode"] = float(mode)
            x = []
            for f in selected_features:
                v = feat.get(f, fill_values.get(f, 0.0))
                if not np.isfinite(v):
                    v = fill_values.get(f, 0.0)
                x.append(v)
            x = np.array(x, dtype=float).reshape(1, -1)
            if scaler is not None:
                x = scaler.transform(x)
            p = float(model.predict_proba(x)[:, 1][0])
            probs.append(p)
            wp = int(p >= model_threshold)
            window_preds.append(wp)
            if method == "state_machine":
                q = compute_quality(feat)
                state, _ = sm.update(p, quality=q, stride_sec=stride_sec)
                states.append(state)
        except Exception:
            continue

    if len(probs) == 0:
        final_pred = 0
    elif method == "mean_vote":
        final_pred = int(np.mean(window_preds) >= 0.5)
    elif method == "prob_mean":
        final_pred = int(np.mean(probs) >= model_threshold)
    elif method == "state_machine":
        final_pred = sample_pred_from_states(
            states,
            strategy=postprocess_cfg.get("sample_pred_strategy", DEFAULT_POSTPROCESS_CONFIG["sample_pred_strategy"]),
            warmup_frames=postprocess_cfg.get("sample_pred_warmup_frames", DEFAULT_POSTPROCESS_CONFIG["sample_pred_warmup_frames"]),
        )
    else:
        final_pred = int(np.mean(probs) >= model_threshold)

    return {
        "sample_name": sample["sample_name"],
        "target": int(sample["target"]),
        "pred": int(final_pred),
        "mode": int(mode),
        "window_probs": probs,
        "window_preds": window_preds,
    }


# =========================================================
# 部署产物导出
# =========================================================

def build_feature_formula_map(selected_features):
    """
    Build deployment formula metadata from the governed Stage2 catalog.

    返回: OrderedDict[feature_name -> formula_info]
    """
    from collections import OrderedDict
    selected_features = assert_no_stage2_ir_features(
        list(selected_features),
        "deploy feature formula selected_features",
    )
    unknown = [name for name in selected_features if name not in FEATURE_CATALOG]
    if unknown:
        raise ValueError(
            "unknown Stage2 model candidates in deployment formula export: "
            + ", ".join(unknown)
            + "; rerun s03-s05 with the current feature pool."
        )
    return OrderedDict(
        (
            name,
            OrderedDict([
                ("feature", name),
                ("category", str(FEATURE_CATALOG[name]["group"])),
                ("signals", [str(FEATURE_CATALOG[name]["signal_source"])]),
                ("formula", str(FEATURE_CATALOG[name]["formula"])),
                ("preprocessing", str(FEATURE_CATALOG[name]["preprocessing"])),
                ("unit", str(FEATURE_CATALOG[name]["unit"])),
                ("numerical_guard", str(FEATURE_CATALOG[name]["numerical_guard"])),
                ("c_operators", list(FEATURE_CATALOG[name]["c_operators"])),
                ("c_abs_tolerance", float(FEATURE_CATALOG[name]["c_abs_tolerance"])),
                ("c_rel_tolerance", float(FEATURE_CATALOG[name]["c_rel_tolerance"])),
            ]),
        )
        for name in selected_features
    )


def export_deploy_artifacts(artifact_dir):
    """
    导出所有部署所需产物到 artifacts/deploy_package/。

    从已有的 artifacts 中读取：
      - model_bundle.pkl
      - final_model_config.json (可选)

    产生：
      - deploy_config.json       总配置
      - feature_formulas.json    每个入选特征的计算公式
      - xgboost_trees.txt        所有树的文本 dump
      - xgboost_nodes.csv        所有节点的结构化 CSV
      - model_params.json        模型超参 / threshold / fill_values
      - postprocess_config.json  状态机参数 + quality 阈值
    """
    import os as _os
    import shutil

    # --- 读取已有 artifacts ---
    bundle_path = _os.path.join(artifact_dir, "model_bundle.pkl")
    features_path = _os.path.join(artifact_dir, "selected_features.json")
    config_path = _os.path.join(artifact_dir, "final_model_config.json")

    bundle = joblib.load(bundle_path)
    assert_bundle_ok(bundle)

    out_dir = _os.path.join(artifact_dir, "deploy_package")
    _os.makedirs(out_dir, exist_ok=True)
    print(f"\n导出部署产物到: {out_dir}")

    postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)
    if _os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            fcfg = json.load(f)
        if "postprocess" in fcfg:
            postprocess_cfg.update(fcfg["postprocess"])

    selected_features = assert_no_stage2_ir_features(
        list(bundle["feature_names"]),
        "model_bundle.pkl feature_names",
    )
    if _os.path.exists(features_path):
        try:
            with open(features_path, "r", encoding="utf-8") as f:
                features = json.load(f)
            stale_selected = list(features.get("selected_features", []))
            if stale_selected and stale_selected != selected_features:
                print("[WARN] selected_features.json differs from model_bundle.pkl; "
                      "deploy package uses bundle['feature_names']")
        except Exception as e:
            print(f"[WARN] selected_features.json could not be read: {e}")
    model = bundle["model"]
    raw = bundle.get("raw_model", model)
    booster = raw.get_booster()

    # =========================================================
    # 2. feature_formulas.json
    # =========================================================
    formula_map = build_feature_formula_map(selected_features)
    formulas_out = OrderedDict([
        ("pipeline_step", 2),
        ("description", f"Stage2 — {float(bundle['meta']['win_sec']):g}s sliding-window feature extraction + XGBoost"),
        ("window_config", {
            "duration_sec": float(bundle["meta"]["win_sec"]),
            "stride_sec": float(bundle["meta"]["step_sec"]),
            "use_stage2_ir": False,
            "fs_ppg": float(bundle["meta"]["fs_ppg"]),
        }),
        ("preprocessing_per_channel", [
            "finite_signal: NaN/Inf 用通道有限值中位数替换；全无有限值时用 0",
            "contact_raw = remove_burr(finite_signal, k=6.0): 仅修复同时偏离左右邻点的孤立毛刺",
            "baseline = rolling_median(contact_raw, nearest odd max(3, round(0.8*fs)))",
            "pulse = contact_raw - baseline",
            "若 round(0.04*fs)>=2，再对 pulse 做短窗 moving_average；25Hz 下该步骤不启用",
            "DC/contact 特征使用 contact_raw；脉搏/频域特征使用 pulse",
        ]),
        ("n_selected_features", len(selected_features)),
        ("features", formula_map),
    ])

    with open(_os.path.join(out_dir, "feature_formulas.json"), "w", encoding="utf-8") as f:
        json.dump(formulas_out, f, indent=2, ensure_ascii=False)
    print("[OK] feature_formulas.json")

    # =========================================================
    # 3. xgboost_trees.txt
    # =========================================================
    trees_txt = booster.get_dump(with_stats=True)
    with open(_os.path.join(out_dir, "xgboost_trees.txt"), "w", encoding="utf-8") as f:
        for i, tree in enumerate(trees_txt):
            f.write(f"booster[{i}]:\n")
            f.write(tree)
            f.write("\n\n")
    print(f"[OK] xgboost_trees.txt ({len(trees_txt)} trees)")

    # =========================================================
    # 4. xgboost_nodes.csv
    # =========================================================
    try:
        # 优先用 trees_to_data_frame (xgboost >= 1.6)
        nodes_df = booster.trees_to_data_frame()
        fmap = {i: name for i, name in enumerate(selected_features)}
        if "Feature" in nodes_df.columns:
            nodes_df["FeatureName"] = nodes_df["Feature"].apply(
                lambda idx: fmap.get(int(idx), str(idx)) if idx != "Leaf" else "Leaf"
            )
        nodes_df.to_csv(_os.path.join(out_dir, "xgboost_nodes.csv"), index=False)
        print(f"[OK] xgboost_nodes.csv ({len(nodes_df)} nodes)")
    except Exception:
        # 回退：从 get_dump 文本解析叶子值
        try:
            rows = []
            fmap = {i: name for i, name in enumerate(selected_features)}
            for tidx, tree in enumerate(trees_txt):
                for line in tree.split("\n"):
                    line = line.strip()
                    if not line or line.startswith("booster"):
                        continue
                    parts = line.split(":")
                    node_id = int(parts[0].strip())
                    node_str = parts[1].strip() if len(parts) > 1 else ""
                    if "leaf=" in node_str:
                        leaf_val = float(node_str.split("leaf=")[1].split(",")[0].split()[0])
                        rows.append({
                            "Tree": tidx, "Node": node_id,
                            "ID": f"{tidx}-{node_id}",
                            "Feature": "Leaf", "FeatureName": "Leaf",
                            "Split": "", "Yes": "", "No": "", "Missing": "",
                            "Gain": "", "Cover": "", "LeafValue": leaf_val,
                        })
                    elif "[" in node_str and "]" in node_str:
                        feat_part = node_str.split("[")[1].split("]")[0]
                        feat_idx = int(feat_part.replace("f", ""))
                        feat_name = fmap.get(feat_idx, f"f{feat_idx}")
                        condition = node_str.split("]")[1].strip().split(",")[0] if "]" in node_str else ""
                        yes_child = node_str.split("yes=")[1].split(",")[0] if "yes=" in node_str else ""
                        no_child = node_str.split("no=")[1].split(",")[0] if "no=" in node_str else ""
                        missing = node_str.split("missing=")[1].split(",")[0] if "missing=" in node_str else ""
                        gain = node_str.split("gain=")[1].split(",")[0] if "gain=" in node_str else ""
                        cover = node_str.split("cover=")[1].split(",")[0].split()[0] if "cover=" in node_str else ""
                        rows.append({
                            "Tree": tidx, "Node": node_id,
                            "ID": f"{tidx}-{node_id}",
                            "Feature": f"f{feat_idx}", "FeatureName": feat_name,
                            "Split": condition, "Yes": yes_child, "No": no_child, "Missing": missing,
                            "Gain": gain, "Cover": cover, "LeafValue": "",
                        })
            if rows:
                pd.DataFrame(rows).to_csv(_os.path.join(out_dir, "xgboost_nodes.csv"), index=False)
                print(f"[OK] xgboost_nodes.csv ({len(rows)} nodes, parsed from tree dump)")
            else:
                print("[WARN] xgboost_nodes.csv: parsed 0 nodes")
        except Exception as e2:
            print(f"[WARN] xgboost_nodes.csv 生成失败: {e2}")

    # =========================================================
    # 5. model_params.json
    # =========================================================
    model_params = OrderedDict([
        ("pipeline_step", 2),
        ("model_type", "XGBoost (XGBClassifier)"),
        ("n_estimators", int(raw.n_estimators)),
        ("hyperparameters", {
            k: v for k, v in raw.get_params().items()
            if k not in ("missing", "n_jobs", "random_state", "verbosity", "n_estimators")
        }),
        ("window_threshold", float(bundle["threshold"])),
        ("threshold_policy", bundle.get("threshold_policy", {})),
        ("n_selected_features", len(selected_features)),
        ("selected_features", selected_features),
        ("fill_values", bundle["fill_values"]),
        ("clip_bounds", {}),
        ("quality_thresholds", bundle.get("quality_thresholds", {})),
        ("feature_quantiles", bundle.get("feature_quantiles", {})),
        ("fingerprint", bundle.get("fingerprint", {})),
        ("meta", bundle["meta"]),
    ])

    with open(_os.path.join(out_dir, "model_params.json"), "w", encoding="utf-8") as f:
        json.dump(model_params, f, indent=2, ensure_ascii=False)
    print("[OK] model_params.json")

    # =========================================================
    # 6. postprocess_config.json
    # =========================================================
    postprocess_out = OrderedDict([
        ("pipeline_step", 4),
        ("description", "时序后处理 — 对窗口概率应用带滞回的状态机平滑"),
        ("state_machine", OrderedDict([
            ("algorithm", "EMA + hysteresis"),
            ("state", "0=not_worn, 1=worn"),
            ("parameters", OrderedDict([
                ("alpha", float(postprocess_cfg.get("alpha", 0.4))),
                ("T_on", float(postprocess_cfg.get("T_on", 0.75))),
                ("T_off", float(postprocess_cfg.get("T_off", 0.35))),
                ("K_on", int(postprocess_cfg.get("K_on", 5))),
                ("K_off", int(postprocess_cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]))),
                ("cooldown_sec", float(postprocess_cfg.get("cooldown_sec", DEFAULT_POSTPROCESS_CONFIG["cooldown_sec"]))),
                ("sample_pred_strategy", str(postprocess_cfg.get(
                    "sample_pred_strategy", DEFAULT_POSTPROCESS_CONFIG["sample_pred_strategy"]))),
                ("sample_pred_warmup_frames", int(postprocess_cfg.get(
                    "sample_pred_warmup_frames", DEFAULT_POSTPROCESS_CONFIG["sample_pred_warmup_frames"]))),
            ])),
            ("formula", OrderedDict([
                ("score_update", "first score = quality * p; later score = alpha * quality * p + (1 - alpha * quality) * score"),
                ("state_0→1", "IF score > T_on for K_on consecutive updates AND cooldown expired THEN state=1"),
                ("state_1→0", "IF score < T_off for K_off consecutive updates AND cooldown expired THEN state=0"),
                ("cooldown", "After flip, wait cooldown_sec before next flip allowed"),
                ("leakage_decay", "on_count/off_count decrement by 1 when threshold not met (soft reset)"),
                ("warmup_output_policy", "During warmup_frames, update score/counts normally but publish output_valid=false; do not substitute raw model predictions"),
                ("final_prediction", "sample prediction follows sample_pred_strategy"),
            ])),
        ])),
        ("quality_scoring", OrderedDict([
            ("description", "基于特征质量调整 EMA 平滑速度"),
            ("thresholds_source", "train — learned from bundle['quality_thresholds']"),
        ("features_used", ["Ambient_std", "G_mean_mean"]),
        ("thresholds", bundle.get("quality_thresholds", {})),
        ("fallback", "if thresholds missing: Ambient_std>1e7→×0.5, |G_mean_mean|<1e-6→×0.5"),
        ])),
        ("ood_monitoring", OrderedDict([
            ("description", "OOD 窗比例监控 — 窗特征超出 train 分位 [q_low, q_high] 的比例"),
            ("alert_rate", 0.3),
            ("quantiles", bundle.get("feature_quantiles", {})),
        ])),
    ])

    with open(_os.path.join(out_dir, "postprocess_config.json"), "w", encoding="utf-8") as f:
        json.dump(postprocess_out, f, indent=2, ensure_ascii=False)
    print("[OK] postprocess_config.json")

    # =========================================================
    # 7. deploy_config.json (总构型，聚合以上所有)
    # =========================================================
    deploy_config = OrderedDict([
        ("title", "手表佩戴活体检测 — 部署配置"),
        ("pipeline_overview", [
            f"XGBoost: {float(bundle['meta']['win_sec']):g}s feature window ({float(bundle['meta']['step_sec']):g}s stride) + window probability",
            "Optional postprocess: WearStateMachine (EMA + hysteresis)",
        ]),
        ("stage2_features", OrderedDict([
            ("window_config", OrderedDict([
                ("duration_sec", float(bundle["meta"]["win_sec"])),
                ("stride_sec", float(bundle["meta"]["step_sec"])),
                ("use_stage2_ir", False),
            ])),
            ("n_features", len(selected_features)),
            ("names", selected_features),
        ])),
        ("stage2_model", OrderedDict([
            ("type", "XGBoost"),
            ("n_trees", int(raw.n_estimators)),
            ("window_threshold", float(bundle["threshold"])),
            ("fill_strategy", "train median"),
        ])),
        ("stage4_postprocess", OrderedDict([
            ("alpha", float(postprocess_cfg.get("alpha", 0.4))),
            ("T_on", float(postprocess_cfg.get("T_on", 0.75))),
            ("T_off", float(postprocess_cfg.get("T_off", 0.35))),
            ("K_on", int(postprocess_cfg.get("K_on", 5))),
            ("K_off", int(postprocess_cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]))),
            ("cooldown_sec", float(postprocess_cfg.get("cooldown_sec", DEFAULT_POSTPROCESS_CONFIG["cooldown_sec"]))),
            ("sample_pred_strategy", str(postprocess_cfg.get(
                "sample_pred_strategy", DEFAULT_POSTPROCESS_CONFIG["sample_pred_strategy"]))),
            ("sample_pred_warmup_frames", int(postprocess_cfg.get(
                "sample_pred_warmup_frames", DEFAULT_POSTPROCESS_CONFIG["sample_pred_warmup_frames"]))),
        ])),
        ("bundle_fingerprint", bundle.get("fingerprint", {})),
    ])

    with open(_os.path.join(out_dir, "deploy_config.json"), "w", encoding="utf-8") as f:
        json.dump(deploy_config, f, indent=2, ensure_ascii=False)
    print("[OK] deploy_config.json")

    # The deployment package must carry the exact selected-feature contract.
    fs_ppg = float(bundle["meta"]["fs_ppg"])
    win_sec = float(bundle["meta"]["win_sec"])
    stage2_contract = build_selected_feature_contract(
        selected_features,
        fs=fs_ppg,
        window_samples=max(1, int(round(fs_ppg * win_sec))),
    )
    stage2_catalog = OrderedDict([
        ("feature_pool_version", stage2_contract["feature_pool_version"]),
        ("feature_order", list(stage2_contract["feature_order"])),
        ("features", stage2_contract["features"]),
    ])
    with open(_os.path.join(out_dir, "stage2_feature_catalog.json"),
              "w", encoding="utf-8") as f:
        json.dump(stage2_catalog, f, indent=2, ensure_ascii=False)
    with open(_os.path.join(out_dir, "stage2_c_contract.json"),
              "w", encoding="utf-8") as f:
        json.dump(stage2_contract, f, indent=2, ensure_ascii=False)
    print("[OK] stage2_feature_catalog.json")
    print("[OK] stage2_c_contract.json")

    # 复制已有的 xgboost JSON 模型文件
    model_json_path = _os.path.join(artifact_dir, "final_model.json")
    if _os.path.exists(model_json_path):
        shutil.copy2(model_json_path, _os.path.join(out_dir, "xgboost_model.json"))
        print("[OK] xgboost_model.json (copied from final_model.json)")

    print(f"\n部署产物导出完成: {out_dir}/")
    print(f"  共 {len(_os.listdir(out_dir))} 个文件")


# =========================================================
# main
# =========================================================

def _safe_cache_name(sample_name):
    text = str(sample_name).replace("\\", "_").replace("/", "_").replace(":", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "sample"


def write_window_cache_npz(result, out_dir, window_sec, stride_sec, model_threshold,
                           quality_thresholds=None, metadata=None):
    import os as _os
    _os.makedirs(out_dir, exist_ok=True)
    metadata = metadata or {}
    probs = np.asarray(result.get("window_probs", []), dtype=float)
    n = int(len(probs))
    starts = np.asarray(result.get("window_start_sec", []), dtype=float)
    if len(starts) != n:
        starts = np.arange(n, dtype=float) * float(stride_sec)
    ends = np.asarray(result.get("window_end_sec", []), dtype=float)
    if len(ends) != n:
        ends = starts + float(window_sec)
    preds = np.asarray(result.get("window_preds", []), dtype=int)
    window_indices = np.asarray(result.get("window_indices", np.arange(n, dtype=int)), dtype=int)
    window_targets = np.asarray(result.get("window_targets", np.full(n, int(result.get("target", 0)))), dtype=int)
    if len(preds) != n:
        preds = (probs >= float(model_threshold)).astype(int)
    if len(window_indices) != n:
        window_indices = np.arange(n, dtype=int)
    if len(window_targets) != n:
        window_targets = np.full(n, int(result.get("target", 0)), dtype=int)
    quality_metas = result.get("quality_metas", [])
    quality = np.ones(n, dtype=float)
    ood_rate = np.zeros(n, dtype=float)
    for i, meta in enumerate(quality_metas[:n]):
        if isinstance(meta, dict):
            if "quality" in meta:
                quality[i] = float(meta.get("quality", 1.0))
            else:
                quality[i] = float(compute_quality(meta, thresholds=quality_thresholds))
    ood_scores = result.get("window_ood_scores", [])
    for i, val in enumerate(ood_scores[:n]):
        if val is not None and np.isfinite(val):
            ood_rate[i] = float(val)
    sample_name = str(result.get("sample_name", "unknown"))
    fname = _safe_cache_name(sample_name) + ".npz"
    fpath = _os.path.join(out_dir, fname)
    np.savez_compressed(
        fpath,
        sample_name=np.array(sample_name),
        target=np.array(int(result.get("target", 0)), dtype=np.int64),
        window_start_sec=starts,
        window_end_sec=ends,
        window_indices=window_indices.astype(np.int64),
        window_targets=window_targets.astype(np.int64),
        prob_raw=probs,
        pred_raw=preds.astype(np.int64),
        quality=quality,
        ood_rate=ood_rate,
        mode=np.array(int(result.get("mode", 0)), dtype=np.int64),
        fallback=np.array(int(bool(result.get("fallback", False))), dtype=np.int64),
        model_threshold=np.array(float(model_threshold)),
        window_sec=np.array(float(window_sec)),
        stride_sec=np.array(float(stride_sec)),
        cache_schema_version=np.array("xgboost_window_outputs_v1"),
        model_fingerprint_json=np.array(json.dumps(metadata.get("model_fingerprint", {}), ensure_ascii=False)),
        feature_names_json=np.array(json.dumps(list(metadata.get("feature_names", [])), ensure_ascii=False)),
        use_stage2_ir=np.array(0, dtype=np.int64),
    )
    return fpath


def export_tree_feature_usage_plot(artifact_dir):
    """Export XGBoost tree-level feature usage visualisation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip tree viz: {e}")
        return None

    bundle_path = os.path.join(str(artifact_dir), "model_bundle.pkl")
    if not os.path.exists(bundle_path):
        print("[WARN] model_bundle.pkl not found, skip tree viz")
        return None

    bundle = joblib.load(bundle_path)
    assert_bundle_ok(bundle)
    raw_model = bundle.get("raw_model")
    if raw_model is None:
        print("[WARN] raw_model not in bundle, skip tree viz")
        return None

    try:
        booster = raw_model.get_booster()
    except Exception as e:
        print(f"[WARN] cannot access booster for tree viz: {e}")
        return None

    feature_names = list(bundle.get("feature_names", []))
    out_dir = os.path.join(str(artifact_dir), "report_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s06_tree_feature_usage.png")

    # Extract tree data
    try:
        nodes_df = booster.trees_to_data_frame()
    except Exception:
        try:
            dump = booster.get_dump(with_stats=True)
            rows = []
            for tidx, tree_str in enumerate(dump):
                for line in tree_str.splitlines():
                    stripped = line.strip()
                    if not stripped or "leaf" in stripped:
                        continue
                    import re
                    feat_match = re.search(r"f(\d+)", stripped)
                    feat_idx = int(feat_match.group(1)) if feat_match else -1
                    gain_match = re.search(r"gain=([\d.]+)", stripped)
                    cover_match = re.search(r"cover=([\d.]+)", stripped)
                    rows.append({
                        "Tree": tidx,
                        "Feature": f"f{feat_idx}" if feat_idx >= 0 else "",
                        "Gain": float(gain_match.group(1)) if gain_match else 0.0,
                        "Cover": float(cover_match.group(1)) if cover_match else 0.0,
                    })
            nodes_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception as e2:
            print(f"[WARN] tree data extraction failed: {e2}")
            return None

    if nodes_df.empty or "Feature" not in nodes_df.columns:
        print("[WARN] empty tree node data, skip tree viz")
        return None

    # Map feature indices to names
    if feature_names:
        def _feat_name(f):
            try:
                fi = int(str(f).replace("f", ""))
                return feature_names[fi] if 0 <= fi < len(feature_names) else str(f)
            except (ValueError, IndexError):
                return str(f)
        nodes_df["FeatureName"] = nodes_df["Feature"].astype(str).apply(_feat_name)
    else:
        nodes_df["FeatureName"] = nodes_df["Feature"].astype(str)

    split_nodes = nodes_df[nodes_df["FeatureName"].notna() & (nodes_df["FeatureName"] != "")
                           & (nodes_df["FeatureName"] != "f-1")]
    if split_nodes.empty:
        print("[WARN] no split nodes found, skip tree viz")
        return None

    # Compute per-feature aggregates
    feat_freq = split_nodes["FeatureName"].value_counts().head(15)
    feat_gain = split_nodes.groupby("FeatureName")["Gain"].mean().sort_values(ascending=False).head(15)
    feat_cover = split_nodes.groupby("FeatureName")["Cover"].mean().sort_values(ascending=False).head(15)
    tree_sizes = nodes_df.groupby("Tree").size() if "Tree" in nodes_df.columns else pd.Series(dtype=int)

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)

    # (0,0) Feature split frequency
    ax_freq = fig.add_subplot(gs[0, 0])
    names_f = list(feat_freq.index)[::-1]
    vals_f = [feat_freq[n] for n in names_f]
    ax_freq.barh(np.arange(len(names_f)), vals_f, color="#4c78a8", height=0.68)
    ax_freq.set_yticks(np.arange(len(names_f)))
    ax_freq.set_yticklabels(names_f, fontsize=8)
    ax_freq.set_xlabel("Split Count")
    ax_freq.set_title(f"Feature Split Frequency (Top {len(names_f)})")
    ax_freq.grid(axis="x", alpha=0.18)

    # (0,1) Feature mean gain
    ax_gain = fig.add_subplot(gs[0, 1])
    names_g = list(feat_gain.index)[::-1]
    vals_g = [feat_gain[n] for n in names_g]
    ax_gain.barh(np.arange(len(names_g)), vals_g, color="#2f6f73", height=0.68)
    ax_gain.set_yticks(np.arange(len(names_g)))
    ax_gain.set_yticklabels(names_g, fontsize=8)
    ax_gain.set_xlabel("Mean Gain")
    ax_gain.set_title(f"Feature Mean Gain (Top {len(names_g)})")
    ax_gain.grid(axis="x", alpha=0.18)

    # (1,0) Feature mean cover
    ax_cover = fig.add_subplot(gs[1, 0])
    names_c = list(feat_cover.index)[::-1]
    vals_c = [feat_cover[n] for n in names_c]
    ax_cover.barh(np.arange(len(names_c)), vals_c, color="#8172b2", height=0.68)
    ax_cover.set_yticks(np.arange(len(names_c)))
    ax_cover.set_yticklabels(names_c, fontsize=8)
    ax_cover.set_xlabel("Mean Cover")
    ax_cover.set_title(f"Feature Mean Cover (Top {len(names_c)})")
    ax_cover.grid(axis="x", alpha=0.18)

    # (1,1) Tree size distribution
    ax_tree = fig.add_subplot(gs[1, 1])
    if len(tree_sizes) > 0:
        sizes = tree_sizes.values
        ax_tree.hist(sizes, bins=min(20, len(tree_sizes)), color="#d35f2d", alpha=0.7, edgecolor="white")
        ax_tree.axvline(sizes.mean(), color="#222222", linewidth=1.5, linestyle="--",
                        label=f"mean={sizes.mean():.1f}")
        ax_tree.axvline(np.median(sizes), color="#4c78a8", linewidth=1.5, linestyle=":",
                        label=f"median={np.median(sizes):.0f}")
        ax_tree.set_xlabel("Nodes per Tree")
        ax_tree.set_ylabel("Tree Count")
        ax_tree.set_title(f"Tree Size Distribution ({len(tree_sizes)} trees, {int(sizes.sum())} total nodes)")
        ax_tree.grid(alpha=0.18)
        ax_tree.legend(frameon=False)
    else:
        ax_tree.text(0.5, 0.5, "No tree size data", ha="center", va="center")
        ax_tree.set_axis_off()
        ax_tree.set_title("Tree Size Distribution")

    fig.suptitle("XGBoost Tree Feature Usage Report", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] s06 tree feature usage plot -> {out_path}")
    return out_path


def export_window_cache(results, artifact_dir, split, window_sec, stride_sec, model_threshold,
                        quality_thresholds=None, metadata=None, cache_root="window_outputs"):
    import os as _os, pandas as _pd
    metadata = metadata or {}
    artifact_path = Path(artifact_dir).resolve()
    cache_path = (artifact_path / _os.fspath(cache_root)).resolve()
    try:
        cache_path.relative_to(artifact_path)
    except ValueError as exc:
        raise ValueError(
            f"cache_root={cache_root!r} must resolve under artifact_dir={artifact_path}") from exc
    out_path = (cache_path / _os.fspath(split)).resolve()
    try:
        out_path.relative_to(cache_path)
    except ValueError as exc:
        raise ValueError(
            f"split={split!r} must resolve under cache_root={cache_path}") from exc
    if out_path == cache_path:
        raise ValueError("split must name a directory below cache_root")
    out_path.mkdir(parents=True, exist_ok=True)
    out_dir = _os.fspath(out_path)
    manifest = []
    produced_paths = set()
    for r in results:
        try:
            npz_path = write_window_cache_npz(
                r, out_dir, window_sec, stride_sec, model_threshold,
                quality_thresholds=quality_thresholds,
                metadata=metadata,
            )
            resolved_npz = Path(npz_path).resolve()
            if resolved_npz.parent != out_path:
                raise ValueError(
                    f"cache output {resolved_npz} escaped target split directory {out_path}")
            produced_paths.add(resolved_npz)
            prob_arr = np.asarray(r.get("window_probs", []), dtype=float)
            n_windows = int(len(prob_arr))
            manifest.append({
                "sample_name": str(r.get("sample_name", "unknown")),
                "target": int(r.get("target", 0)),
                "npz_path": npz_path,
                "n_windows": n_windows,
                "cache_schema_version": "xgboost_window_outputs_v1",
                "model_fingerprint": metadata.get("model_fingerprint", {}),
                "feature_names": list(metadata.get("feature_names", [])),
                "use_stage2_ir": False,
                "fallback": int(bool(r.get("fallback", False))),
                "fallback_reason": str(r.get("fallback_reason", "")),
            })
        except Exception as exc:
            sample_name = str(r.get("sample_name", "unknown"))
            raise RuntimeError(
                f"failed to export window cache for sample {sample_name}: {exc}"
            ) from exc
    for existing in out_path.iterdir():
        resolved_existing = existing.resolve()
        if (existing.is_file() and existing.suffix.lower() == ".npz"
                and resolved_existing.parent == out_path
                and resolved_existing not in produced_paths):
            existing.unlink()
    if manifest:
        _pd.DataFrame(manifest).to_csv(_os.path.join(out_dir, "manifest.csv"), index=False)
        with open(_os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[OK] window cache: {len(manifest)} samples -> {out_dir}/")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--method", type=str, default="prob_mean",
                        choices=["mean_vote", "prob_mean", "state_machine"])
    parser.add_argument("--window_sec", type=int, default=5)
    parser.add_argument("--stride_sec", type=int, default=1)
    parser.add_argument("--use_stage2_ir", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="legacy compatibility flag; Stage2 IR is always disabled")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--optimize_split", type=str, default="valid",
                        choices=["train", "valid", "test"])
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")
    parser.add_argument("--warmup_frames", type=int, default=5,
                        help="窗口级指标跳过每条样本前 N 个窗（消除状态机冷启动偏差）")
    parser.add_argument("--optimize_thresholds", type=str, default="",
                        help="对一组候选窗口阈值在缓存 probs 上做窗口级指标扫描（如 '0.3,0.4,0.5,0.6'），输出 P/R/F0.5/F1 表。不修改 bundle.threshold；建议结合此扫描手选/重训。")
    parser.add_argument("--ood_alert_rate", type=float, default=0.3,
                        help="OOD 报警阈值：窗特征 OOD 比例 > 此值视为可疑窗")
    parser.add_argument("--export_window_cache", action="store_true",
                        help="导出逐窗模型输出 NPZ 到 artifacts/window_outputs/ 供 s07 后处理优化")
    parser.add_argument("--window_output_root", type=str, default="window_outputs",
                        help="directory under artifact_dir for per-sample model-output NPZ files")
    parser.add_argument("--export_deploy", action="store_true",
                        help="导出部署产物到 artifacts/deploy_package/")
    parser.add_argument("--generalization_audit", action="store_true",
                        help="只读取已有评估产物并导出 generalization_audit，不重新推理")
    parser.add_argument("--min_support", type=int, default=10,
                        help="generalization audit 最小分层样本数")

    if args is None:
        args = parser.parse_args()

    if args.generalization_audit:
        result = run_audit(
            args.artifact_dir,
            split=args.split,
            method=args.method,
            min_support=args.min_support,
        )
        print(f"[OK] generalization_audit -> {result['out_dir']}")
        print(json.dumps(result["paths"], indent=2, ensure_ascii=False))
        return

    with open(os.path.join(args.artifact_dir, "splits.json"), "r", encoding="utf-8") as f:
        split = json.load(f)
    bundle_path = os.path.join(args.artifact_dir, "model_bundle.pkl")

    print("=" * 80)
    print("加载统一模型包")
    print("=" * 80)
    bundle = load_bundle(bundle_path)
    validate_inference_window_contract(
        bundle,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
    )
    use_stage2_ir = resolve_use_stage2_ir(bundle, args.use_stage2_ir)
    print(f"feature_names: {len(bundle['feature_names'])} 个特征")
    print(f"threshold: {bundle['threshold']}")
    print("meta summary: " + json.dumps(
        bundle_meta_log_summary(bundle), ensure_ascii=False
    ))
    print(f"use_stage2_ir: {use_stage2_ir}")

    # 参数优化
    if args.optimize:
        print("\n" + "=" * 80)
        print(f"运行状态机参数优化 (split={args.optimize_split}, n_workers={args.n_workers})")
        print("=" * 80)
        opt = optimize_state_machine_params(
            samples=split[args.optimize_split],
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            bundle_path=bundle_path,
            n_workers=args.n_workers,
            use_stage2_ir=use_stage2_ir,
        )
        best_params = opt["best_params"]
        best_metrics = opt["best_metrics"]
        print("\n最优参数:")
        print(json.dumps(best_params, indent=2, ensure_ascii=False))
        print(f"\n评估指标: recall={best_metrics['recall']:.4f}, "
              f"precision={best_metrics['precision']:.4f}, F1={best_metrics['f1']:.4f}")

        config_path = os.path.join(args.artifact_dir, "final_model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_old = json.load(f)
        else:
            cfg_old = {}
        cfg_old["legacy_s06_postprocess"] = best_params
        cfg_old["legacy_s06_postprocess_optimization"] = {
            "optimized_on_split": args.optimize_split,
            "metrics": best_metrics,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_old, f, indent=2, ensure_ascii=False)
        print(f"\n最优参数已保存到: {config_path}")
        postprocess_cfg = best_params
    else:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)
        # 尝试读取之前 optimize 保存的最优参数
        config_path = os.path.join(args.artifact_dir, "final_model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
            if "postprocess" in saved_cfg:
                postprocess_cfg.update(saved_cfg["postprocess"])
                print(f"  (loaded optimized postprocess from {config_path})")

    # 单趟并行推理
    print("\n" + "=" * 80)
    print(f"并行推理 split={args.split} (n_workers={args.n_workers})")
    print("=" * 80)
    results = run_inference_parallel(
        samples=split[args.split],
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        bundle_path=bundle_path,
        n_workers=args.n_workers,
        use_stage2_ir=use_stage2_ir,
    )
    # XGBoost 主指标与可选后处理参考指标
    sample_summary, details = compute_sample_metrics(
        results, args.method, postprocess_cfg, bundle["threshold"],
        stride_sec=args.stride_sec,
    )
    window_model_summary = compute_window_model_metrics(results)
    window_stream_summary = compute_window_stream_metrics(
        details, postprocess_cfg, warmup_frames=args.warmup_frames,
        model_threshold=bundle["threshold"],
        stride_sec=args.stride_sec,
    )

    # OOD 汇总：每条样本计 OOD 窗占比和触发标志
    ood_summary = _summarize_ood(results, alert_rate=args.ood_alert_rate)

    sample_summary["split"] = args.split
    sample_summary["selected_features"] = bundle["feature_names"]
    sample_summary["model_threshold"] = float(bundle["threshold"])
    sample_summary["bundle_fingerprint"] = bundle.get("fingerprint")
    sample_summary["threshold_policy"] = bundle.get("threshold_policy")

    print("\n" + "=" * 80)
    print("主指标: XGBoost-only 样本评估")
    print("  窗口概率按所选 XGBoost-only 样本聚合方式生成预测")
    print("=" * 80)
    summary_for_print = {k: v for k, v in sample_summary.items() if k != "selected_features"}
    print(json.dumps(summary_for_print, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("窗口指标: 全部合法窗口的 XGBoost 评估")
    print("  XGBoost 逐窗预测 vs 样本真实标签 (无 warmup)")
    print("=" * 80)
    wm_print = {k: v for k, v in window_model_summary.items() if k != "confusion_matrix"}
    print(json.dumps(wm_print, indent=2, ensure_ascii=False))
    print(f"  confusion_matrix: {window_model_summary['confusion_matrix']}")

    print("\n" + "=" * 80)
    print("参考: Stage2+3 流式状态 (状态机逐窗状态)")
    print(f"  warmup_frames={args.warmup_frames} (跳过每条样本前N窗消除冷启动)")
    print("=" * 80)
    ws_print = {k: v for k, v in window_stream_summary.items() if k != "confusion_matrix"}
    print(json.dumps(ws_print, indent=2, ensure_ascii=False))
    print(f"  confusion_matrix: {window_stream_summary['confusion_matrix']}")

    print("\n" + "=" * 80)
    print("准确率对比")
    print("=" * 80)
    print(f"  XGBoost-only (样本): {sample_summary['accuracy']:.4f}")
    print(f"  XGBoost (逐窗):      {window_model_summary['accuracy']:.4f}")
    print(f"  可选状态机 (逐窗):  {window_stream_summary['accuracy']:.4f}")

    # Hard negative 输出: 列出 FP 和 FN 样本
    fp_samples = [d for d in details if d["pred"] == 1 and d["target"] == 0]
    fn_samples = [d for d in details if d["pred"] == 0 and d["target"] == 1]
    print("\n" + "=" * 80)
    print(f"Hard Negative 分析: FP={len(fp_samples)}, FN={len(fn_samples)}")
    print("=" * 80)
    if fp_samples:
        print("\n  False Positives (非佩戴判为佩戴):")
        for d in sorted(fp_samples, key=lambda x: np.mean(x.get("window_probs", []) or [0]), reverse=True)[:10]:
            avg_p = np.mean(d.get("window_probs", []) or [0])
            print(f"    {d['sample_name']:30s} target=0 pred=1  avg_prob={avg_p:.3f}  "
                  f"n_win={d.get('n_windows',0)}")
    if fn_samples:
        print("\n  False Negatives (佩戴判为非佩戴):")
        for d in sorted(fn_samples, key=lambda x: np.mean(x.get("window_probs", []) or [0]))[:10]:
            avg_p = np.mean(d.get("window_probs", []) or [0])
            print(f"    {d['sample_name']:30s} target=1 pred=0  avg_prob={avg_p:.3f}  "
                  f"n_win={d.get('n_windows',0)}")

    # 把 OOD 信息也合并进每条 detail
    ood_map = {s["sample_name"]: s for s in ood_summary["per_sample"]}
    for d in details:
        s = ood_map.get(d["sample_name"], {})
        d["ood_mean"] = s.get("ood_mean")
        d["ood_window_alert_rate"] = s.get("ood_window_alert_rate")

    per_sample_summary_csv = export_per_sample_inference_summary(
        details, args.artifact_dir, split=args.split, method=args.method
    )
    print(f"per-sample inference summary saved: {per_sample_summary_csv}")

    # 窗口阈值扫描（联合优化的轻量版）
    threshold_sweep = None
    stratified_errors = compute_stratified_error_analysis(details)
    hard_negative_report = mine_hard_negatives(details, top_k=50)
    window_error_report = compute_window_error_analysis(details)

    threshold_sweep = None
    if args.optimize_thresholds.strip():
        thr_list = []
        for t in args.optimize_thresholds.split(","):
            t = t.strip()
            if not t:
                continue
            try:
                thr_list.append(float(t))
            except ValueError:
                pass
        if thr_list:
            print("\n" + "=" * 80)
            print(f"窗口阈值扫描 (window-level, warmup={args.warmup_frames})")
            print("=" * 80)
            print(f"{'threshold':>10}  {'precision':>10}  {'recall':>10}  "
                  f"{'F0.5':>10}  {'F1':>10}  {'n_win':>8}")
            sweep_rows = []
            for thr in thr_list:
                y_t, y_p = [], []
                for r in results:
                    if (r.get("fallback", False)
                            or len(r.get("window_probs", [])) == 0):
                        continue
                    t_target = int(r["target"])
                    probs = np.asarray(r["window_probs"], dtype=float)
                    start = min(args.warmup_frames, len(probs))
                    for p in probs[start:]:
                        y_t.append(t_target)
                        y_p.append(int(p >= thr))
                if not y_t:
                    continue
                y_t = np.asarray(y_t)
                y_p = np.asarray(y_p)
                prec = float(precision_score(y_t, y_p, zero_division=0))
                rec = float(recall_score(y_t, y_p, zero_division=0))
                f1v = float(f1_score(y_t, y_p, zero_division=0))
                # F0.5
                denom = 0.25 * prec + rec
                f05 = 1.25 * prec * rec / denom if denom > 0 else 0.0
                row = {
                    "threshold": float(thr),
                    "precision": prec, "recall": rec,
                    "f0.5": float(f05), "f1": f1v,
                    "n_windows": int(len(y_t)),
                }
                sweep_rows.append(row)
                print(f"{thr:>10.3f}  {prec:>10.4f}  {rec:>10.4f}  "
                      f"{f05:>10.4f}  {f1v:>10.4f}  {len(y_t):>8d}")
            threshold_sweep = {
                "thresholds": thr_list,
                "rows": sweep_rows,
                "note": "本扫描不修改 bundle.threshold；若需控 FP，可在 s05 重训阶段显式使用 --threshold_objective precision_constrained 或 fbeta。",
            }

    out_path = os.path.join(args.artifact_dir, f"end_to_end_eval_{args.split}_{args.method}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_contract": build_evaluation_contract(args.split),
            "summary": sample_summary,
            "window_summary": window_stream_summary,         # 向后兼容旧字段
            "window_model_summary": window_model_summary,    # 新增
            "window_stream_summary": window_stream_summary,  # 新增（与 window_summary 等价）
            "ood_summary": ood_summary,                      # 新增 OOD 汇总
            "threshold_sweep": threshold_sweep,               # 新增 阈值扫描（可选）
            "stratified_errors": stratified_errors,
            "hard_negative_report": hard_negative_report,
            "window_error_report": {
                "summary": window_error_report.get("summary", {}),
                "strata": window_error_report.get("strata", {}),
            },
            "details": details,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {out_path}")

    # 部署产物导出
    export_deploy_report_plot({
        "summary": sample_summary,
        "window_summary": window_stream_summary,
        "window_model_summary": window_model_summary,
        "window_stream_summary": window_stream_summary,
        "ood_summary": ood_summary,
        "threshold_sweep": threshold_sweep,
        "stratified_errors": stratified_errors,
        "hard_negative_report": hard_negative_report,
        "details": details,
    }, args.artifact_dir, split=args.split, method=args.method)
    strat_path = os.path.join(args.artifact_dir, f"error_stratification_{args.split}_{args.method}.json")
    with open(strat_path, "w", encoding="utf-8") as f:
        json.dump(stratified_errors, f, indent=2, ensure_ascii=False)
    hard_path = os.path.join(args.artifact_dir, f"hard_negatives_{args.split}_{args.method}.json")
    with open(hard_path, "w", encoding="utf-8") as f:
        json.dump(hard_negative_report, f, indent=2, ensure_ascii=False)
    win_err_csv, win_err_json = export_window_error_analysis(
        window_error_report, args.artifact_dir, split=args.split, method=args.method)
    print(f"error stratification saved: {strat_path}")
    print(f"hard negatives saved: {hard_path}")
    print(f"window error analysis saved: {win_err_csv}")
    print(f"window error summary saved: {win_err_json}")

    if args.export_window_cache:
        export_window_cache(
            results=results,
            artifact_dir=args.artifact_dir,
            split=args.split,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            model_threshold=float(bundle["threshold"]),
            quality_thresholds=bundle.get("quality_thresholds"),
            metadata={
                "model_fingerprint": bundle.get("fingerprint", {}),
                "feature_names": bundle.get("feature_names", []),
                "use_stage2_ir": use_stage2_ir,
            },
            cache_root=args.window_output_root,
        )
    if args.export_deploy:
        export_deploy_artifacts(
            args.artifact_dir
        )


# =========================================================
# Generalization audit utilities (formerly s10).
# =========================================================
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
                "Review feature quality and add matching positive low-quality/OOD data.",
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
                "Review warmup behavior and window ordering.",
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

    # Export audit heatmap visualisation
    try:
        export_audit_heatmap(window_strata, out_dir, min_support=min_support)
    except Exception as e:
        print(f"[WARN] audit heatmap export failed: {e}")
    try:
        export_audit_ranked_error_bars(window_strata, sample_strata, out_dir)
    except Exception as e:
        print(f"[WARN] audit ranked error bars export failed: {e}")
    try:
        export_audit_latency_distribution(sample_df, out_dir)
    except Exception as e:
        print(f"[WARN] audit latency distribution export failed: {e}")

    return {
        "out_dir": str(out_dir),
        "summary": summary,
        "paths": {
            "summary_json": str(out_dir / "summary.json"),
            "summary_md": str(out_dir / "summary.md"),
            "window_strata": str(out_dir / "window_strata.csv"),
            "sample_strata": str(out_dir / "sample_strata.csv"),
            "action_items": str(out_dir / "action_items.csv"),
            "ranked_error_bars": str(out_dir / "audit_ranked_error_bars.png"),
            "latency_distribution": str(out_dir / "audit_latency_distribution.png"),
        },
    }


def export_audit_ranked_error_bars(window_strata, sample_strata, out_dir, top_k=15):
    """Export ranked FP/FN bars for deployment-relevant audit strata."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip audit ranked error bars: {e}")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for level, df in [("window", window_strata), ("sample", sample_strata)]:
        if df is None or len(df) == 0:
            continue
        work = df.copy()
        work["level"] = level
        frames.append(work)
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame(columns=["level", "dimension", "stratum", "fp", "fn"])

    for col in ["fp", "fn"]:
        if col not in combined.columns:
            combined[col] = 0
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0).astype(int)
    combined["total_errors"] = combined["fp"] + combined["fn"]
    ranked = combined[combined["total_errors"] > 0].copy()
    ranked = ranked.sort_values(["total_errors", "fp", "fn"], ascending=False).head(int(top_k))
    source_path = out_dir / "audit_ranked_error_bars.csv"
    ranked.to_csv(source_path, index=False)

    fig, ax = plt.subplots(figsize=(11, max(4.0, 0.42 * max(len(ranked), 1) + 1.2)), facecolor="white")
    if ranked.empty:
        ax.text(0.5, 0.5, "No FP/FN strata above support threshold", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [
            f"{row.level}:{row.dimension}={str(row.stratum)[:28]}"
            for row in ranked.itertuples(index=False)
        ][::-1]
        fp_vals = ranked["fp"].to_numpy(dtype=int)[::-1]
        fn_vals = ranked["fn"].to_numpy(dtype=int)[::-1]
        y = np.arange(len(labels))
        ax.barh(y, fp_vals, color="#c44e52", label="FP")
        ax.barh(y, fn_vals, left=fp_vals, color="#4c78a8", label="FN")
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlabel("error count")
        ax.set_title("Ranked FP/FN Strata")
        ax.grid(axis="x", alpha=0.18)
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "audit_ranked_error_bars.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] audit ranked error bars -> {out_path}")
    return str(out_path)


def export_audit_latency_distribution(sample_df, out_dir):
    """Export first-worn latency distribution and false-worn sample count."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip audit latency distribution: {e}")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latencies = _first_worn_latencies(sample_df)
    false_worn_count = 0
    true_negative_count = 0
    if sample_df is not None and not sample_df.empty and {"target", "pred"}.issubset(sample_df.columns):
        target = pd.to_numeric(sample_df["target"], errors="coerce").fillna(0).astype(int)
        pred = pd.to_numeric(sample_df["pred"], errors="coerce").fillna(0).astype(int)
        false_worn_count = int(((target == 0) & (pred == 1)).sum())
        true_negative_count = int((target == 0).sum())
    pd.DataFrame({"first_worn_latency_sec": latencies}).to_csv(
        out_dir / "audit_latency_distribution.csv",
        index=False,
    )

    fig = plt.figure(figsize=(10, 5), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.8], wspace=0.25)
    ax_hist = fig.add_subplot(gs[0, 0])
    if latencies:
        ax_hist.hist(latencies, bins=min(12, max(4, len(latencies))), color="#4c78a8", alpha=0.78)
        ax_hist.axvline(np.percentile(latencies, 95), color="#c44e52", linestyle="--", linewidth=1.4, label="P95")
        ax_hist.legend(frameon=False)
    else:
        ax_hist.text(0.5, 0.5, "No positive first-worn latency values", ha="center", va="center")
    ax_hist.set_xlabel("first-worn latency (sec)")
    ax_hist.set_ylabel("positive samples")
    ax_hist.set_title("Positive Output Latency")
    ax_hist.grid(axis="y", alpha=0.18)

    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.bar(["false worn", "target=0"], [false_worn_count, true_negative_count],
               color=["#c44e52", "#9aa6ac"])
    ax_bar.set_title("Negative Sample Risk")
    ax_bar.set_ylabel("samples")
    ax_bar.grid(axis="y", alpha=0.18)

    fig.suptitle("Latency and False-Worn Summary", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / "audit_latency_distribution.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] audit latency distribution -> {out_path}")
    return str(out_path)


def export_audit_heatmap(strata_df, out_dir, min_support=10):
    """Export heatmap of stratified evaluation metrics across dimensions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable, skip audit heatmap: {e}")
        return None

    if strata_df is None or len(strata_df) == 0:
        print("[WARN] empty strata, skip audit heatmap")
        return None

    df = strata_df.copy()
    out_path = os.path.join(str(out_dir), "audit_strata_heatmap.png")
    os.makedirs(str(out_dir), exist_ok=True)

    # Select columns for the heatmap
    metric_cols = ["accuracy", "precision", "recall", "fp_rate"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        print("[WARN] no metric columns in strata, skip audit heatmap")
        return None

    # Build "dimension:stratum" labels
    dim_col = "dimension" if "dimension" in df.columns else None
    strat_col = "stratum" if "stratum" in df.columns else None
    if dim_col and strat_col:
        df["label"] = df[dim_col].astype(str) + ":" + df[strat_col].astype(str)
    elif strat_col:
        df["label"] = df[strat_col].astype(str)
    else:
        df["label"] = df.index.astype(str)

    # Sort by accuracy descending
    if "accuracy" in df.columns:
        df = df.sort_values("accuracy", ascending=False)
    if "n_windows" in df.columns:
        low_mask = df["n_windows"] < min_support
    else:
        low_mask = pd.Series(False, index=df.index)

    # Truncate to top 50 for readability
    max_rows = 50
    if len(df) > max_rows:
        df = df.head(max_rows)
        low_mask = low_mask.loc[df.index]

    n_rows = len(df)
    n_cols = len(metric_cols)
    heatmap_data = df[metric_cols].values.astype(float)

    # Handle fp_rate: lower is better, so invert
    display_data = heatmap_data.copy()
    fp_idx = metric_cols.index("fp_rate") if "fp_rate" in metric_cols else None

    # Build a masked array for low-support strata
    mask = np.tile(low_mask.values.reshape(-1, 1), (1, n_cols))

    fig_height = max(6, 2.5 + 0.22 * n_rows)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, fig_height),
                              facecolor="white", squeeze=False)
    axes = axes[0]

    cmap = plt.get_cmap("RdYlGn")
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx]
        col_data = display_data[:, idx].reshape(-1, 1)
        col_mask = mask[:, idx].reshape(-1, 1)

        if idx == fp_idx:
            # Invert fp_rate: lower is better → higher score on heatmap
            col_data = 1.0 - np.clip(col_data, 0, 1)

        masked_data = np.ma.array(col_data, mask=col_mask)
        im = ax.imshow(masked_data, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                        interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(df["label"].values, fontsize=7)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        # Low-support hatching
        for r in range(n_rows):
            if low_mask.iloc[r]:
                ax.axhline(y=r, color="#9aa6ac", linewidth=1.5, alpha=0.4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Generalization Audit — Stratified Metrics (min_support={min_support})",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] audit heatmap -> {out_path}")
    return out_path


if __name__ == "__main__":
    main()
