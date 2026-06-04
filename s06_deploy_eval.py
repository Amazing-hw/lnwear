# s06_deploy_eval.py
# -*- coding: utf-8 -*-

"""
步骤6：部署/端到端推理评估（优化版）

主要改进：
1. 单趟推理 + ProcessPoolExecutor 并行；推理结果在主进程做三套指标计算。
2. 删除每窗对 g1 / ir 的重复 preprocess_signal 调用：
   直接走 extract_feature_pool_from_window(..., return_preprocessed=True)，
   从返回的 preprocessed dict 拿 g1_bp / ir_bp 给 cross-features 用。
3. 三套指标显式拆分：
   (a) sample-level（端到端）：final_pred vs target，
       fallback / Stage1-fail / 空 probs 一律 pred=0 参与统计。
   (b) window-level 模型+阈值：window_preds vs target，每窗一条。
   (c) window-level 流式状态机：state[i] vs target。
4. 状态机网格搜索并行化（probs 缓存复用）。

公共接口（main / predict_sample / predict_sample_with_bundle /
predict_sample_safe / evaluate_streaming_window_accuracy /
optimize_state_machine_params / get_deploy_stage1_threshold / load_bundle）
全部保留原签名。

CLI 参数与输出文件名、JSON 主要字段保持向后兼容。
"""

import os
import json
import argparse
import logging
import joblib
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from s03_extract_feature_pool import (
    _downsample_ppg,
    load_ppg,
    load_acc,
    stage1_sample_pass,
    _is_25hz_sample,
    is_prewindowed_signal,
    stage1_ambient_check,
    detect_green_mode,
    get_channels_from_window,
    apply_stage2_ir_policy,
    extract_feature_pool_from_window,
    align_acc_window,
    extract_acc_features,
    extract_acc_ppg_cross_features,
    validate_h5_file,
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
    "cooldown_sec": 2,
}

STAGE1_PRIMITIVE_SEC = 1.0
STAGE1_DECISION_SEC = 3.0
STAGE1_FS = 5
STAGE1_GATE_K = int(round(STAGE1_DECISION_SEC / STAGE1_PRIMITIVE_SEC))
DEFAULT_SKIP_INITIAL_WINDOWS = 3
DEFAULT_USE_STAGE2_IR = False


def resolve_use_stage2_ir(bundle, requested=None):
    if requested is not None:
        return bool(requested)
    if isinstance(bundle, dict):
        return bool(bundle.get("meta", {}).get("use_stage2_ir", DEFAULT_USE_STAGE2_IR))
    return DEFAULT_USE_STAGE2_IR


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
    needed = ["feature_names", "fill_values", "scaler", "model", "threshold", "meta"]
    for k in needed:
        assert k in bundle, f"model_bundle missing key: {k}"

    miss = [c for c in bundle["feature_names"] if c not in bundle["fill_values"]]
    assert not miss, f"fill_values missing for: {miss[:5]} ..."

    for k in ["fs_ppg", "win_sec", "step_sec"]:
        assert k in bundle["meta"], f"meta missing: {k}"


def apply_preprocess(feat_dict_list, bundle=None):
    """列对齐 + 缺失填充（无标准化）"""
    b = bundle if bundle is not None else _BUNDLE
    assert b is not None, "must call load_bundle() first"

    feature_names = b["feature_names"]
    fill_values = b["fill_values"]

    df = pd.DataFrame(feat_dict_list)

    for c in feature_names:
        if c not in df.columns:
            df[c] = np.nan
    df = df[feature_names]

    for c in feature_names:
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
    def __init__(self, alpha=0.4, T_on=0.75, T_off=0.35, K_on=5, K_off=3, cooldown_sec=2):
        self.alpha = alpha
        self.T_on = T_on
        self.T_off = T_off
        self.K_on = K_on
        self.K_off = K_off
        self.cooldown_sec = cooldown_sec  # 翻转后最小冷却秒数

        self.state = 0
        self.score = 0.0
        self.on_count = 0
        self.off_count = 0
        self._steps_since_flip = 999  # 距离上次翻转的步数（初始大值允许首次翻转）

    def update(self, p, quality=1.0, stride_sec=1.0):
        eff_alpha = self.alpha * quality
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
    irm = feat_or_meta.get("IR_mean", None)
    if irm is not None and np.abs(irm) < 1e-6:
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


def apply_postprocess(window_probs, quality_metas, method, cfg, model_threshold):
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
        sm = WearStateMachine(
            alpha=cfg.get("alpha", 0.4),
            T_on=cfg.get("T_on", 0.75),
            T_off=cfg.get("T_off", 0.35),
            K_on=cfg.get("K_on", DEFAULT_POSTPROCESS_CONFIG["K_on"]),
            K_off=cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]),
            cooldown_sec=cfg.get("cooldown_sec", DEFAULT_POSTPROCESS_CONFIG["cooldown_sec"]),
        )
        qt = _BUNDLE.get("quality_thresholds") if _BUNDLE is not None else None
        states, scores = [], []
        for i, p in enumerate(probs_for_state):
            meta_i = quality_metas[i] if i < len(quality_metas) else None
            q = compute_quality(meta_i, thresholds=qt) if meta_i else 1.0
            state, score = sm.update(p, quality=q)
            states.append(int(state))
            scores.append(float(score))
        final_pred = int(states[-1])
        return final_pred, states, window_preds, scores

    if method == "mean_vote":
        final_pred = int(np.mean(window_preds) >= 0.5)
        return final_pred, [], window_preds, []

    # prob_mean / 默认
    final_pred = int(np.mean(probs) >= model_threshold)
    return final_pred, [], window_preds, []


# =========================================================
# Stage1 流式门控 (1s stride, 3 consecutive same → toggle)
# =========================================================

class Stage1StreamingGate:
    """Stage1 流式滞回门控：连续 K 帧相同结果才翻转 Stage2 开关。"""

    def __init__(self, dc_threshold, ac_dc_threshold, K=STAGE1_GATE_K):
        self.dc_threshold = dc_threshold
        self.ac_dc_threshold = ac_dc_threshold
        self.K = K
        self.stage2_enabled = False   # 初始关闭
        self.pass_count = 0
        self.fail_count = 0

    def _check_one(self, ir_5hz_window):
        """对单个 3s/5Hz IR 窗做 DC/ACDC 判断。"""
        x = ir_5hz_window
        if len(x) >= 2:
            dc = float(np.min((x[:-1] + x[1:]) / 2.0))
        else:
            dc = float(np.mean(x))
        if len(x) >= 2:
            ac = float(np.median(np.abs(np.diff(x))))
        else:
            ac = 0.0
        return dc > self.dc_threshold and (ac / (np.abs(dc) + 1e-12)) < self.ac_dc_threshold

    def update(self, ir_5hz_window):
        """输入当前 1s stride 的 IR 5Hz 窗，返回当前 Stage2 是否启用。"""
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


def _advance_stage1_gate_to_step(gate, ir_5hz, s1_win_5hz, s1_stride_5hz,
                                 last_s1_step, target_s1_step):
    """Advance the 1s Stage1 gate through every primitive step up to target."""
    n_s1 = (len(ir_5hz) - s1_win_5hz) // s1_stride_5hz + 1
    if n_s1 <= 0:
        return False, last_s1_step
    target_s1_step = min(int(target_s1_step), n_s1 - 1)
    if target_s1_step <= last_s1_step:
        return bool(getattr(gate, "stage2_enabled", False)), last_s1_step

    enabled = bool(getattr(gate, "stage2_enabled", False))
    for s1_step in range(last_s1_step + 1, target_s1_step + 1):
        s1_start = s1_step * s1_stride_5hz
        enabled = bool(gate.update(ir_5hz[s1_start:s1_start + s1_win_5hz]))
        last_s1_step = s1_step
    return enabled, last_s1_step


def _infer_prewindowed_sample(base, ppg, acc, dc_threshold, ac_dc_threshold,
                              window_sec, stride_sec, bundle, use_stage2_ir,
                              skip_initial_windows):
    """Run deployed inference directly on stored 3s windows."""
    FEATURE_FS = 25
    native_25hz = _is_25hz_sample(base) or int(ppg.shape[1]) == int(round(float(window_sec) * FEATURE_FS))
    ppg_src_fs = 25 if native_25hz else 100
    mode = detect_green_mode(ppg)
    base["mode"] = int(mode)

    feats_list = []
    quality_metas = []
    stage2_enabled_flags = []
    window_start_sec = []
    window_end_sec = []

    first_step = max(0, int(skip_initial_windows))
    for step in range(first_step, ppg.shape[0]):
        raw_window = ppg[step]
        start_sec = float(step * stride_sec)
        window_start_sec.append(start_sec)
        window_end_sec.append(start_sec + float(window_sec))

        enabled = (
            stage1_sample_pass(raw_window, dc_threshold, ac_dc_threshold, ppg_fs=ppg_src_fs)
            and stage1_ambient_check(raw_window)
        )
        stage2_enabled_flags.append(int(enabled))
        if not enabled:
            feats_list.append(None)
            quality_metas.append(None)
            continue

        try:
            window = raw_window.astype(np.float64, copy=False) if native_25hz else _downsample_ppg(
                raw_window, src_fs=100, tgt_fs=FEATURE_FS
            )
            ir, ambient, g1, g2, g3 = get_channels_from_window(window, mode)
            ir = apply_stage2_ir_policy(ir, use_stage2_ir=use_stage2_ir)
            feat, preprocessed = extract_feature_pool_from_window(
                ir=ir, ambient=ambient, g1=g1, g2=g2, g3=g3,
                fs=FEATURE_FS, return_preprocessed=True
            )

            acc_seg = None
            if acc is not None and is_prewindowed_signal(acc) and step < acc.shape[0]:
                try:
                    raw_acc = acc[step]
                    acc_seg = raw_acc.astype(np.float64, copy=False) if native_25hz else _downsample_ppg(
                        raw_acc, src_fs=100, tgt_fs=FEATURE_FS
                    )
                except Exception:
                    acc_seg = None
            if acc_seg is not None and len(acc_seg) > 0:
                feat.update(extract_acc_features(acc_seg, fs=FEATURE_FS, prefix="ACC"))
                green_bp = preprocessed.get("g1_bp")
                ir_bp = preprocessed.get("ir_bp")
                if green_bp is not None and ir_bp is not None:
                    feat.update(extract_acc_ppg_cross_features(acc_seg, green_bp, ir_bp, fs=FEATURE_FS))

            feats_list.append(feat)
            quality_metas.append({
                "Ambient_std": feat.get("Ambient_std"),
                "G_mean_mean": feat.get("G_mean_mean"),
                "IR_mean": feat.get("IR_mean"),
            })
        except Exception:
            feats_list.append(None)
            quality_metas.append(None)

    valid_indices = [i for i, f in enumerate(feats_list) if f is not None]
    probs = np.zeros(len(feats_list), dtype=float)
    wpreds = np.zeros(len(feats_list), dtype=int)
    if valid_indices:
        valid_feats = [feats_list[i] for i in valid_indices]
        _wpreds, _probs = predict_label_windows(valid_feats, bundle=bundle)
        for vi, idx in enumerate(valid_indices):
            probs[idx] = float(_probs[vi])
            wpreds[idx] = int(_wpreds[vi])

    feature_quantiles = bundle.get("feature_quantiles")
    feature_names = bundle.get("feature_names", [])
    ood_scores = [
        None if f is None or not feature_quantiles else compute_ood_score(f, feature_quantiles, feature_names)
        for f in feats_list
    ]

    base["window_probs"] = probs.tolist()
    base["window_preds"] = wpreds.tolist()
    base["quality_metas"] = [qm if qm is not None else {} for qm in quality_metas]
    base["window_ood_scores"] = ood_scores
    base["stage2_enabled_flags"] = stage2_enabled_flags
    base["window_start_sec"] = window_start_sec
    base["window_end_sec"] = window_end_sec
    return base


# =========================================================
# 单样本推理（流式 Stage1 + Stage2）
# =========================================================

def _infer_one_sample(sample, dc_threshold, ac_dc_threshold, window_sec, stride_sec, bundle,
                      skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
                      use_stage2_ir=DEFAULT_USE_STAGE2_IR):
    """
    流式推理: Stage1(1s stride, 3s窗, 滞回K=3) → 按需调用 Stage2 → 每窗输出。

    Stage1 和 Stage2 都在 1s stride 上对齐。
    Stage1 连续 3 次通过 → 开启 Stage2
    Stage1 连续 3 次不通过 → 关闭 Stage2, 该窗 pred=0
    """
    from scipy.signal import resample_poly as _rp
    from s03_extract_feature_pool import downsample_to_5hz

    FEATURE_FS = 25

    sample_name = sample.get("sample_name", "unknown")
    target = int(sample.get("target", 0))

    base = {
        "sample_name": sample_name, "target": target,
        "stage1_pass": True,  # streaming mode: always "passed" in the old sense
        "mode": 0,
        "window_probs": [], "window_preds": [], "quality_metas": [],
        "window_ood_scores": [], "stage2_enabled_flags": [],
        "window_start_sec": [], "window_end_sec": [],
        "use_stage2_ir": bool(use_stage2_ir),
        "fallback": False, "fallback_reason": None,
    }

    # 1-2. 验证 + 加载
    try:
        h5_file = sample.get("h5_file")
        if h5_file is None or sample_name is None:
            base["fallback"] = True; base["fallback_reason"] = "incomplete"; return base
        ok, err = validate_h5_file(h5_file, sample_name)
        if not ok:
            base["fallback"] = True; base["fallback_reason"] = f"h5: {err}"; return base
        ppg = load_ppg(sample)
        acc = load_acc(sample)
    except Exception as e:
        base["fallback"] = True; base["fallback_reason"] = f"load: {e}"; return base

    if is_prewindowed_signal(ppg):
        return _infer_prewindowed_sample(
            base, ppg, acc, dc_threshold, ac_dc_threshold,
            window_sec, stride_sec, bundle, use_stage2_ir,
            skip_initial_windows,
        )

    # 2b. Stage1 环境光检查 (整条数据级别)
    if not stage1_ambient_check(ppg):
        base["stage1_pass"] = False
        return base  # 环境光异常 → 判为未佩戴

    # 3. 信号降采样 (25Hz 原生数据跳过)
    try:
        mode = detect_green_mode(ppg)
        base["mode"] = int(mode)

        native_25hz = _is_25hz_sample(sample)
        ppg_src_fs = 25 if native_25hz else 100

        if native_25hz:
            ppg_25 = ppg.astype(np.float64)
            acc_25 = acc.astype(np.float64) if (acc is not None and len(acc) > 0) else None
        else:
            ppg_25 = _rp(ppg.astype(np.float32, copy=False), 1, 4, axis=0).astype(np.float64)
            acc_25 = None
            if acc is not None and len(acc) > 0:
                try:
                    acc_25 = _rp(acc.astype(np.float32, copy=False), 1, 4, axis=0).astype(np.float64)
                except Exception:
                    pass

        # Stage1 用 IR 通道 → 5Hz
        ir_raw = ppg[:, 0]
        ir_5hz = downsample_to_5hz(ir_raw, ppg_src_fs, 5)
        s1_win_5hz = int(round(STAGE1_PRIMITIVE_SEC * STAGE1_FS))
        s1_stride_5hz = s1_win_5hz

        gate = Stage1StreamingGate(dc_threshold, ac_dc_threshold, K=STAGE1_GATE_K)

        # Stage2 窗口参数
        win_25 = int(window_sec * FEATURE_FS)
        stride_25 = int(stride_sec * FEATURE_FS)

        # 多少步可以对齐 Stage1 和 Stage2
        n_steps_s2 = (len(ppg_25) - win_25) // stride_25 + 1
        n_steps_s1 = (len(ir_5hz) - s1_win_5hz) // s1_stride_5hz + 1
        quality_metas = []
        feats_list = []
        stage2_enabled_flags = []
        window_start_sec = []
        window_end_sec = []
        last_s1_step = -1

        first_step = max(0, int(skip_initial_windows))
        for step in range(first_step, max(0, n_steps_s2)):
            # Stage1: IR @5Hz, 3s窗, 1s stride
            s2_start = step * stride_25
            target_s1_step = int(np.floor(s2_start / FEATURE_FS + 1e-9))
            if target_s1_step >= n_steps_s1:
                break
            s2_on, last_s1_step = _advance_stage1_gate_to_step(
                gate, ir_5hz, s1_win_5hz, s1_stride_5hz,
                last_s1_step, target_s1_step
            )
            stage2_enabled_flags.append(int(s2_on))
            window_start_sec.append(float(s2_start / FEATURE_FS))
            window_end_sec.append(float(s2_start / FEATURE_FS + window_sec))

            if not s2_on:
                # Stage2 关闭：跳过特征提取，pred=0, prob=0
                feats_list.append(None)
                quality_metas.append(None)
                continue

            # Stage2: PPG @ 25Hz
            window = ppg_25[s2_start:s2_start + win_25, :]
            try:
                ir, ambient, g1, g2, g3 = get_channels_from_window(window, mode)
                ir = apply_stage2_ir_policy(ir, use_stage2_ir=use_stage2_ir)
                feat, preprocessed = extract_feature_pool_from_window(
                    ir=ir, ambient=ambient, g1=g1, g2=g2, g3=g3,
                    fs=FEATURE_FS, return_preprocessed=True
                )
                if acc_25 is not None and len(acc_25) > 0:
                    acc_seg = align_acc_window(acc_25, len(ppg_25), s2_start, win_25,
                                               fs_ppg=FEATURE_FS, fs_acc=FEATURE_FS)
                    feat.update(extract_acc_features(acc_seg, fs=FEATURE_FS, prefix="ACC"))
                    green_bp = preprocessed.get("g1_bp")
                    ir_bp = preprocessed.get("ir_bp")
                    if green_bp is not None and ir_bp is not None:
                        feat.update(extract_acc_ppg_cross_features(
                            acc_seg, green_bp, ir_bp, fs=FEATURE_FS))
                        # ACC-PPG coherence
                        if acc_seg is not None and len(acc_seg) >= 16:
                            try:
                                from scipy.signal import coherence as _coh
                                _acc_mag = np.sqrt(np.sum(acc_seg.astype(float)**2, axis=1) + 1e-12)
                                _npseg = min(32, len(_acc_mag) // 2)
                                if _npseg >= 8:
                                    _f, _cxy = _coh(_acc_mag, green_bp, fs=FEATURE_FS, nperseg=_npseg)
                                    _cm = (_f >= 0.5) & (_f <= 3.0)
                                    if np.any(_cm):
                                        feat["ACC_PPG_coherence_mean"] = float(np.mean(_cxy[_cm]))
                                        feat["ACC_PPG_coherence_max"] = float(np.max(_cxy[_cm]))
                            except Exception:
                                pass
                feats_list.append(feat)
                quality_metas.append({
                    "Ambient_std": feat.get("Ambient_std"),
                    "G_mean_mean": feat.get("G_mean_mean"),
                    "IR_mean": feat.get("IR_mean"),
                })
            except Exception:
                feats_list.append(None)
                quality_metas.append(None)

        # 批量预测有效窗口
        valid_indices = [i for i, f in enumerate(feats_list) if f is not None]
        n_emitted_steps = len(feats_list)
        probs = np.zeros(n_emitted_steps, dtype=float)
        wpreds = np.zeros(n_emitted_steps, dtype=int)
        if valid_indices:
            valid_feats = [feats_list[i] for i in valid_indices]
            _wpreds, _probs = predict_label_windows(valid_feats, bundle=bundle)
            for vi, idx in enumerate(valid_indices):
                probs[idx] = float(_probs[vi])
                wpreds[idx] = int(_wpreds[vi])

        # OOD
        feature_quantiles = bundle.get("feature_quantiles")
        feature_names = bundle.get("feature_names", [])
        ood_scores = []
        for i, f in enumerate(feats_list):
            if f is None or not feature_quantiles:
                ood_scores.append(None)
            else:
                ood_scores.append(compute_ood_score(f, feature_quantiles, feature_names))

        base["window_probs"] = probs.tolist()
        base["window_preds"] = wpreds.tolist()
        base["quality_metas"] = [qm if qm is not None else {} for qm in quality_metas]
        base["window_ood_scores"] = ood_scores
        base["stage2_enabled_flags"] = stage2_enabled_flags
        base["window_start_sec"] = window_start_sec
        base["window_end_sec"] = window_end_sec
        return base

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
    (sample, dc_threshold, ac_dc_threshold, window_sec, stride_sec,
     skip_initial_windows, use_stage2_ir) = args_tuple
    return _infer_one_sample(
        sample, dc_threshold, ac_dc_threshold, window_sec, stride_sec, _WORKER_BUNDLE,
        skip_initial_windows=skip_initial_windows,
        use_stage2_ir=use_stage2_ir,
    )


def run_inference_parallel(samples, dc_threshold, ac_dc_threshold,
                           window_sec, stride_sec, bundle_path, n_workers,
                           skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
                           use_stage2_ir=None):
    """
    并行单趟推理，返回与 samples 同序的 results 列表。
    """
    n_workers = resolve_n_workers(n_workers, n_items=len(samples))
    if use_stage2_ir is None:
        bundle_for_meta = _BUNDLE if _BUNDLE is not None else joblib.load(bundle_path)
        use_stage2_ir = resolve_use_stage2_ir(bundle_for_meta)
    args_list = [
        (s, dc_threshold, ac_dc_threshold, window_sec, stride_sec,
         skip_initial_windows, bool(use_stage2_ir))
        for s in samples
    ]

    if n_workers == 1:
        # 单进程路径：复用 _BUNDLE
        bundle = _BUNDLE if _BUNDLE is not None else joblib.load(bundle_path)
        return [
            _infer_one_sample(
                s, dc_threshold, ac_dc_threshold, window_sec, stride_sec, bundle,
                skip_initial_windows=skip_initial_windows,
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
                    "stage1_pass": False,
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


def compute_sample_metrics(results, method, cfg, model_threshold):
    """
    Sample 级（端到端）：
    fallback / Stage1-fail / 空 probs 一律 pred=0 参与统计。
    """
    y_true, y_pred = [], []
    fallback_count = 0
    stage1_pass_count = 0
    details = []

    for r in results:
        target = int(r["target"])
        if r.get("fallback", False):
            fallback_count += 1
        if r.get("stage1_pass", False):
            stage1_pass_count += 1

        probs = r.get("window_probs", [])
        scores = []
        if r.get("fallback", False) or not r.get("stage1_pass", False) or len(probs) == 0:
            final_pred = 0
            states = []
            window_preds = list(r.get("window_preds", []))
        else:
            final_pred, states, window_preds, scores = apply_postprocess(
                probs, r.get("quality_metas", []), method, cfg, model_threshold
            )

        y_true.append(target)
        y_pred.append(int(final_pred))

        fallback = r.get("fallback", False)
        details.append({
            "sample_name": r.get("sample_name"),
            "target": target,
            "pred": int(final_pred),
            "stage1_pass": bool(r.get("stage1_pass", False)),
            "mode": int(r.get("mode", 0)),
            "fallback": bool(fallback),
            "fallback_reason": r.get("fallback_reason"),
            "window_probs": probs,
            "window_preds": list(window_preds),
            "window_states": list(states),
            "window_scores": list(scores) if not fallback else [],
            "stage2_enabled_flags": r.get("stage2_enabled_flags", []),
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
        "stage1_pass_samples": int(stage1_pass_count),
        "fallback_samples": int(fallback_count),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)) if len(y_true_a) > 0 else 0.0,
        "precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
        "postprocess": cfg,
    }
    return summary, details


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
        "stage1_pass": defaultdict(list),
        "mode": defaultdict(list),
        "n_windows": defaultdict(list),
        "mean_prob": defaultdict(list),
        "ood_window_alert_rate": defaultdict(list),
    }
    for d in details:
        stats = _detail_prob_stats(d)
        strata["stage1_pass"][f"stage1_pass={bool(d.get('stage1_pass', False))}"].append(d)
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
            "stage1_pass": bool(d.get("stage1_pass", False)),
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
    stage1 = int(sample.get("stage1_pass_samples", 0))
    pred_pos = int(sample.get("confusion_matrix", {}).get("FP", 0) + sample.get("confusion_matrix", {}).get("TP", 0))
    ax_funnel.bar(["input", "stage1 pass", "final positive"], [total, stage1, pred_pos],
                  color=["#4c78a8", "#72b7b2", "#d35f2d"])
    ax_funnel.set_title("Stage Funnel")
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

    stage1_buckets = strat.get("stage1_pass", {}) or {}
    if stage1_buckets:
        labels = list(stage1_buckets.keys())
        fp_vals = [int(stage1_buckets[k].get("confusion_matrix", {}).get("FP", 0)) for k in labels]
        fn_vals = [int(stage1_buckets[k].get("confusion_matrix", {}).get("FN", 0)) for k in labels]
        xx = np.arange(len(labels))
        ax_strata.bar(xx - 0.18, fp_vals, width=0.36, color="#c44e52", label="FP")
        ax_strata.bar(xx + 0.18, fn_vals, width=0.36, color="#4c78a8", label="FN")
        ax_strata.set_xticks(xx, labels, rotation=20, ha="right")
        ax_strata.legend(frameon=False)
    else:
        ax_strata.text(0.5, 0.5, "No strata", ha="center", va="center")
        ax_strata.set_axis_off()
    ax_strata.set_title("Error by Stage1 Stratum")
    ax_strata.grid(axis="y", alpha=0.18)

    fig.suptitle(f"Deployment Evaluation Report ({split}, {method})", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] s06 report plot -> {out_path}")
    return out_path


def compute_window_model_metrics(results):
    """
    Stage2 模型评估（仅通过 Stage1 的数据）：
    逐窗 XGBoost 预测 (window_preds) vs 整条样本的 target。

    模型无状态，不做 warmup 跳过——所有通过 Stage1 的窗口全部参与。
    未通过 Stage1 / fallback 的样本不参与（它们没有窗口数据）。
    """
    y_true, y_pred = [], []
    samples_with_no_windows = 0
    total_input_samples = len(results)
    stage1_pass_samples = 0
    for r in results:
        wp = r.get("window_preds", [])
        if r.get("fallback", False) or not r.get("stage1_pass", False) or len(wp) == 0:
            samples_with_no_windows += 1
            continue
        stage1_pass_samples += 1
        t = int(r["target"])
        for p in wp:
            y_true.append(t)
            y_pred.append(int(p))

    if len(y_true) == 0:
        return {
            "total_input_samples": total_input_samples,
            "stage1_pass_samples": stage1_pass_samples,
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
        "stage1_pass_samples": stage1_pass_samples,
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
        if detail.get("fallback", False) or not detail.get("stage1_pass", False):
            continue
        probs = detail.get("window_probs", []) or []
        preds = detail.get("window_preds", []) or []
        if not probs or not preds:
            continue
        target = int(detail.get("target", 0))
        sample_name = detail.get("sample_name")
        mode = int(detail.get("mode", 0))
        enabled = detail.get("stage2_enabled_flags", [])
        starts = detail.get("window_start_sec", [])
        ends = detail.get("window_end_sec", [])
        q_metas = detail.get("quality_metas", [])
        oods = detail.get("window_ood_scores", [])
        for idx, prob in enumerate(probs):
            pred = int(_safe_list_get(preds, idx, 0))
            start = float(_safe_list_get(starts, idx, idx))
            end = float(_safe_list_get(ends, idx, start))
            stage2_enabled = int(_safe_list_get(enabled, idx, 1))
            q_meta = _safe_list_get(q_metas, idx, {})
            ood = _safe_list_get(oods, idx, None)
            error_type = _window_error_type(target, pred)
            rows.append({
                "sample_name": sample_name,
                "target": target,
                "pred_raw": pred,
                "error_type": error_type,
                "is_error": int(error_type in {"FP", "FN"}),
                "window_index": int(idx),
                "window_start_sec": start,
                "window_end_sec": end,
                "prob_raw": float(prob),
                "prob_bin": _bucket_window_prob(prob),
                "time_bin": _bucket_window_time(start),
                "mode": mode,
                "stage2_enabled": stage2_enabled,
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
            "stage2_enabled": _summarize_strata(rows, "stage2_enabled"),
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


def compute_window_stream_metrics(results, cfg, warmup_frames=0):
    """
    Window 级（流式状态机）：state[i] vs target。
    warmup_frames: 跳过前 N 个窗，消除状态机冷启动 K_on 滞后的影响。
    """
    y_true, y_pred = [], []
    samples_with_no_windows = 0
    skipped_windows = 0
    for r in results:
        probs = r.get("window_probs", [])
        qm = r.get("quality_metas", [])
        if r.get("fallback", False) or not r.get("stage1_pass", False) or len(probs) == 0:
            samples_with_no_windows += 1
            continue

        sm = WearStateMachine(
            alpha=cfg.get("alpha", 0.4),
            T_on=cfg.get("T_on", 0.75),
            T_off=cfg.get("T_off", 0.35),
            K_on=cfg.get("K_on", DEFAULT_POSTPROCESS_CONFIG["K_on"]),
            K_off=cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]),
            cooldown_sec=cfg.get("cooldown_sec", DEFAULT_POSTPROCESS_CONFIG["cooldown_sec"]),
        )
        t = int(r["target"])
        qt = _BUNDLE.get("quality_thresholds") if _BUNDLE is not None else None
        # 先把所有窗的 state 跑出来，再按 warmup_frames 裁剪
        sample_states = []
        probs_for_state = causal_median_filter_1d(probs, cfg.get("median_k", DEFAULT_POSTPROCESS_CONFIG["median_k"]))
        for i, p in enumerate(probs_for_state):
            meta_i = qm[i] if i < len(qm) else None
            q = compute_quality(meta_i, thresholds=qt) if meta_i else 1.0
            state, _ = sm.update(p, quality=q)
            sample_states.append(int(state))
        start = min(warmup_frames, len(sample_states))
        skipped_windows += start
        for s in sample_states[start:]:
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

    y_true, y_pred = [], []
    for s in cache:
        target = s["target"]
        probs = s.get("probs", [])
        qm = s.get("quality_metas", [])
        if not s.get("stage1_pass", True) or len(probs) == 0:
            pred = 0
        else:
            sm = WearStateMachine(alpha=alpha, T_on=T_on, T_off=T_off, K_on=K_on, K_off=K_off)
            state = 0
            for i, p in enumerate(probs):
                q = compute_quality(qm[i], thresholds=quality_thresholds) if i < len(qm) and qm[i] else 1.0
                state, _ = sm.update(p, quality=q)
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


def optimize_state_machine_params(samples, dc_threshold, ac_dc_threshold,
                                   window_sec=3, stride_sec=1, min_recall=0.95,
                                   bundle_path=None, n_workers=None,
                                   skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
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
        samples, dc_threshold, ac_dc_threshold,
        window_sec, stride_sec, bundle_path, n_workers,
        skip_initial_windows=skip_initial_windows,
        use_stage2_ir=use_stage2_ir,
    )

    sample_cache = []
    for r in results:
        sample_cache.append({
            "sample_name": r["sample_name"],
            "target": int(r["target"]),
            "probs": r.get("window_probs", []),
            "quality_metas": r.get("quality_metas", []),
            "stage1_pass": bool(r.get("stage1_pass", False)) and not r.get("fallback", False),
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

def get_deploy_stage1_threshold(th):
    """读取部署阈值，兼容旧 schema。"""
    if "deploy_stage1_threshold" in th:
        return {
            "dc_threshold": float(th["deploy_stage1_threshold"]["dc_threshold"]),
            "ac_dc_threshold": float(th["deploy_stage1_threshold"]["ac_dc_threshold"]),
        }
    return {
        "dc_threshold": float(th["dc_threshold"]),
        "ac_dc_threshold": float(th["ac_dc_threshold"]),
    }


def predict_sample_with_bundle(sample, dc_threshold, ac_dc_threshold,
                                window_sec=3, stride_sec=1,
                                method="state_machine", postprocess_cfg=None,
                                skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
                                use_stage2_ir=None):
    """
    单样本部署预测（向后兼容）。内部走优化后的推理路径（无重复 preprocess）。
    """
    assert _BUNDLE is not None, "must call load_bundle() first"

    if postprocess_cfg is None:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)

    r = _infer_one_sample(sample, dc_threshold, ac_dc_threshold,
                          window_sec, stride_sec, _BUNDLE,
                          skip_initial_windows=skip_initial_windows,
                          use_stage2_ir=resolve_use_stage2_ir(_BUNDLE, use_stage2_ir))
    target = int(r["target"])
    if not r["stage1_pass"] or len(r["window_probs"]) == 0 or r["fallback"]:
        return {
            "sample_name": r["sample_name"],
            "target": target,
            "pred": 0,
            "stage1_pass": bool(r["stage1_pass"]),
            "window_probs": [],
            "window_preds": [],
        }

    final_pred, _states, window_preds, _scores = apply_postprocess(
        r["window_probs"], r["quality_metas"], method, postprocess_cfg, _BUNDLE["threshold"]
    )
    return {
        "sample_name": r["sample_name"],
        "target": target,
        "pred": int(final_pred),
        "stage1_pass": True,
        "mode": int(r["mode"]),
        "window_probs": list(r["window_probs"]),
        "window_preds": list(window_preds),
    }


def predict_sample_safe(sample, dc_threshold, ac_dc_threshold,
                        window_sec=3, stride_sec=1,
                        method="state_machine", postprocess_cfg=None,
                        skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
                        use_stage2_ir=None):
    """带 fallback 的预测（向后兼容）。"""
    fallback_result = {
        "sample_name": sample.get("sample_name", "unknown"),
        "target": int(sample.get("target", 0)),
        "pred": 0,
        "stage1_pass": False,
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
            sample, dc_threshold=dc_threshold, ac_dc_threshold=ac_dc_threshold,
            window_sec=window_sec, stride_sec=stride_sec,
            method=method, postprocess_cfg=postprocess_cfg,
            skip_initial_windows=skip_initial_windows,
            use_stage2_ir=use_stage2_ir,
        )
        res["fallback"] = False
        res["fallback_reason"] = None
        return res
    except Exception as e:
        logger.warning(f"预测失败使用 fallback: {e}")
        fallback_result["fallback_reason"] = f"prediction_error: {e}"
        return fallback_result


def evaluate_streaming_window_accuracy(samples, dc_threshold, ac_dc_threshold,
                                       window_sec=3, stride_sec=1, postprocess_cfg=None,
                                       bundle_path=None, n_workers=None,
                                       skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS,
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
            samples, dc_threshold, ac_dc_threshold,
            window_sec, stride_sec, bundle_path, n_workers,
            skip_initial_windows=skip_initial_windows,
            use_stage2_ir=use_stage2_ir,
        )
    else:
        # 单进程：要求 _BUNDLE 已加载
        assert _BUNDLE is not None, "must call load_bundle() first"
        results = [
            _infer_one_sample(
                s, dc_threshold, ac_dc_threshold, window_sec, stride_sec, _BUNDLE,
                skip_initial_windows=skip_initial_windows,
                use_stage2_ir=resolve_use_stage2_ir(_BUNDLE, use_stage2_ir),
            )
            for s in samples
        ]

    all_true, all_pred, details = [], [], []
    for r in results:
        target = int(r["target"])
        probs = r.get("window_probs", [])
        qm = r.get("quality_metas", [])
        if not r.get("stage1_pass", False) or r.get("fallback", False) or len(probs) == 0:
            details.append({
                "sample_name": r.get("sample_name"),
                "target": target,
                "n_windows": 0,
                "window_states": [],
            })
            continue
        sm = WearStateMachine(
            alpha=postprocess_cfg.get("alpha", 0.4),
            T_on=postprocess_cfg.get("T_on", 0.75),
            T_off=postprocess_cfg.get("T_off", 0.35),
            K_on=postprocess_cfg.get("K_on", 5),
            K_off=postprocess_cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]),
        )
        qt = _BUNDLE.get("quality_thresholds") if _BUNDLE is not None else None
        states = []
        for i, p in enumerate(probs):
            q = compute_quality(qm[i], thresholds=qt) if i < len(qm) and qm[i] else 1.0
            state, _ = sm.update(p, quality=q)
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
                    dc_threshold, ac_dc_threshold,
                    window_sec=3, stride_sec=1, fs=25,
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

    if not stage1_sample_pass(ppg, dc_threshold, ac_dc_threshold):
        return {
            "sample_name": sample["sample_name"],
            "target": int(sample["target"]),
            "pred": 0,
            "stage1_pass": False,
            "window_probs": [],
            "window_preds": [],
        }

    mode = detect_green_mode(ppg)
    win = window_sec * fs
    stride = stride_sec * fs

    if postprocess_cfg is None:
        postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)

    sm = WearStateMachine(
        alpha=postprocess_cfg.get("alpha", 0.4),
        T_on=postprocess_cfg.get("T_on", 0.75),
        T_off=postprocess_cfg.get("T_off", 0.35),
        K_on=postprocess_cfg.get("K_on", DEFAULT_POSTPROCESS_CONFIG["K_on"]),
        K_off=postprocess_cfg.get("K_off", DEFAULT_POSTPROCESS_CONFIG["K_off"]),
    )

    probs, window_preds, states = [], [], []

    for start in range(0, len(ppg) - win + 1, stride):
        window = ppg[start:start + win, :]
        try:
            ir, ambient, g1, g2, g3 = get_channels_from_window(window, mode)
            ir = apply_stage2_ir_policy(ir, use_stage2_ir=use_stage2_ir)
            feat = extract_feature_pool_from_window(ir, ambient, g1, g2, g3, fs=fs)
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
                state, _ = sm.update(p, quality=q)
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
        final_pred = int(states[-1]) if states else 0
    else:
        final_pred = int(np.mean(probs) >= model_threshold)

    return {
        "sample_name": sample["sample_name"],
        "target": int(sample["target"]),
        "pred": int(final_pred),
        "stage1_pass": True,
        "mode": int(mode),
        "window_probs": probs,
        "window_preds": window_preds,
    }


# =========================================================
# 部署产物导出
# =========================================================

def build_feature_formula_map(selected_features):
    """
    为每个入选特征生成计算公描述。
    按特征命名约定自动匹配公式模板。

    返回: OrderedDict[feature_name -> formula_info]
    """
    from collections import OrderedDict
    # ---- 公式片段模板 ----
    # 按通道替换
    CHANNEL_NAMES = {
        "IR":         {"raw": "ir_raw", "bp": "ir_bp", "dc": "ir_dc"},
        "GREEN":      {"raw": "g_mean_raw", "bp": "g_mean_bp", "dc": "g_mean_dc"},
        "IRX":        {"raw": "ir_raw", "bp": "ir_bp", "dc": "ir_dc"},
        "AMBX":       {"raw": "amb_raw", "bp": "amb_bp", "dc": "amb_dc"},
        "ACC":        {},  # special
    }

    # 单通道特征模板
    SINGLE_CH_TEMPLATES = OrderedDict([
        ("{pf}_DC_MEDIAN",      "{raw} = preprocess_signal(signal, fs)[0]; float(np.median({raw}))"),
        ("{pf}_DC_IQR",         "{raw} = preprocess_signal(signal, fs)[0]; robust_iqr({raw})"),
        ("{pf}_AC_RMS",         "{bp} = preprocess_signal(signal, fs)[1]; sqrt(mean({bp}²))"),
        ("{pf}_AC_MAD",         "{bp} = preprocess_signal(signal, fs)[1]; robust_mad({bp})"),
        ("{pf}_AC_DC_RATIO",    "{bp} = preprocess_signal(signal, fs)[1]; {dc} = preprocess_signal(signal, fs)[2]; AC_RMS / |DC_MEDIAN|"),
        ("{pf}_DERIV_MAD",      "{bp} = preprocess_signal(signal, fs)[1]; robust_mad(diff({bp}))"),
        ("{pf}_FFT_PEAK_MEDIAN_RATIO", "{bp} = preprocess_signal(signal, fs)[1]; fft_peak_features({bp}, fs, 0.5, 5.0)[0]"),
        ("{pf}_DOM_FREQ",       "{bp} = preprocess_signal(signal, fs)[1]; fft_peak_features({bp}, fs, 0.5, 5.0)[1]"),
        ("{pf}_AUTO_CORR_PEAK", "{bp} = preprocess_signal(signal, fs)[1]; autocorr_periodicity_features({bp}, fs, 40, 180)[0]"),
        ("{pf}_AUTO_CORR_LAG_SEC", "{bp} = preprocess_signal(signal, fs)[1]; autocorr_periodicity_features({bp}, fs, 40, 180)[1]"),
    ])

    # 核心低维特征
    CORE_TEMPLATES = OrderedDict([
        ("IR_mean",             "float(np.mean(ir_raw))"),
        ("IR_std",              "float(np.std(ir_raw))"),
        ("IR_p95",              "float(np.percentile(ir_raw, 95))"),
        ("IR_diff_std",         "float(np.std(np.diff(ir_raw)))"),
        ("IR_acdc",             "sqrt(mean(ir_bp²)) / |ir_dc|"),
        ("G_mean_mean",         "float(np.mean(g_mean_raw))"),
        ("G_mean_std",          "float(np.std(g_mean_raw))"),
        ("G_mean_diff_std",     "float(np.std(np.diff(g_mean_raw)))"),
        ("G_mean_acdc",         "sqrt(mean(g_mean_bp²)) / |g_mean_dc|"),
        ("log_IR_Gmean_mean",   "float(mean(log|ir_raw| - log|g_mean_raw|))"),
        ("IR_over_Gmean_mean",  "float(mean(ir_raw / g_mean_raw))"),
        ("IR_over_Gmean_std",   "float(std(ir_raw / g_mean_raw))"),
        ("corr_IR_Gmean",       "safe_corr(ir_raw, g_mean_raw)"),
        ("Ambient_mean",        "float(np.mean(amb_raw))"),
        ("Ambient_std",         "float(np.std(amb_raw))"),
        ("Ambient_p95",         "float(np.percentile(amb_raw, 95))"),
        ("corr_Ambient_IR",     "safe_corr(amb_raw, ir_raw)"),
        ("corr_Ambient_Gmean",  "safe_corr(amb_raw, g_mean_raw)"),
        ("SIG_LEN",             "float(len(window)) — 窗口采样点数"),
        ("SIG_SEC",             "float(len(window) / fs) — 窗口秒数"),
        ("mode",                "detect_green_mode(ppg) — 绿光模式 (1或2)"),
    ])

    # 绿光空间特征
    GREEN_SPATIAL_TEMPLATES = OrderedDict([
        ("G_imbalance_mean",    "g_stack=[g1_raw,g2_raw,g3_raw]; mean(std(g_stack,axis=0) / |mean(g_stack,axis=0)|)"),
        ("G_imbalance_p90",     "g_stack=[g1_raw,g2_raw,g3_raw]; percentile90(std(g_stack) / |mean(g_stack)|)"),
        ("G_imbalance_iqr",     "g_stack=[g1_raw,g2_raw,g3_raw]; robust_iqr(std(g_stack) / |mean(g_stack)|)"),
        ("G_rangeNorm_mean",    "mean((max(g_stack)-min(g_stack)) / (|g1_raw|+|g2_raw|+|g3_raw|))"),
        ("G_rangeNorm_p90",     "np.percentile(range_norm, 90)"),
        ("G_spatial_vmag_mean", "mean(|[vx,vy]| / (|g1_raw|+|g2_raw|+|g3_raw|)) where vx=g1-0.5g2-0.5g3, vy=√3/2*(g2-g3)"),
        ("G_spatial_vmag_p90",  "np.percentile(vmag, 90)"),
        ("G_spatial_vmag_iqr",  "robust_iqr(vmag)"),
        ("G_spatial_vmag_std",  "float(np.std(vmag))"),
        ("G_ch_dc_cv",          "std([median(g1),median(g2),median(g3)]) / |mean([...])|"),
        ("G_ch_dc_max_min_ratio","max(|dc_g1|,|dc_g2|,|dc_g3|) / min(|dc|)"),
        ("G_bp_corr_mean",      "mean([safe_corr(g1_bp,g2_bp), safe_corr(g2_bp,g3_bp), safe_corr(g3_bp,g1_bp)])"),
        ("G_bp_corr_min",       "min([safe_corr(g1_bp,g2_bp), safe_corr(g2_bp,g3_bp), safe_corr(g3_bp,g1_bp)])"),
        ("G_bp_corr_std",       "std([safe_corr(g1_bp,g2_bp), safe_corr(g2_bp,g3_bp), safe_corr(g3_bp,g1_bp)])"),
    ])

    # 跨通道特征
    CROSS_CH_TEMPLATES = OrderedDict([
        ("GREEN_IR_RAW_CORR",   "safe_corr(g_mean_raw, ir_raw)"),
        ("GREEN_IR_BP_CORR",    "safe_corr(g_mean_bp, ir_bp)"),
        ("GREEN_IR_ENV_CORR",   "safe_corr(smooth_envelope(g_mean_bp), smooth_envelope(ir_bp))"),
        ("GREEN_IR_MAX_XCORR",  "max_norm_xcorr(g_mean_bp, ir_bp, 0.3*fs)"),
        ("GREEN_IR_DOM_FREQ_DIFF", "|dom_freq(g_mean_bp) - dom_freq(ir_bp)|"),
        ("GREEN_AMB_BP_CORR",   "safe_corr(g_mean_bp, amb_bp)"),
        ("IR_AMB_BP_CORR",      "safe_corr(ir_bp, amb_bp)"),
        ("GREEN_AMB_ENV_CORR",  "safe_corr(smooth_envelope(g_mean_bp), smooth_envelope(amb_bp))"),
        ("IR_AMB_ENV_CORR",     "safe_corr(smooth_envelope(ir_bp), smooth_envelope(amb_bp))"),
        ("GREEN_AMB_LEAK",      "|GREEN_AMB_BP_CORR| * amb_rms / g_rms"),
        ("IR_AMB_LEAK",         "|IR_AMB_BP_CORR| * amb_rms / ir_rms"),
        ("GREEN_IR_AC_RATIO",   "g_rms / ir_rms"),
        ("GREEN_IR_DC_RATIO",   "|g_dc| / |ir_dc|"),
        ("GREEN_IR_ACDC_RATIO_RATIO", "(g_rms/|g_dc|) / (ir_rms/|ir_dc|)"),
    ])

    # 空间-光强耦合
    SPATIAL_COUPLING = OrderedDict([
        ("corr_Gmean_G_imbalance", "safe_corr(g_mean_raw, g_imbalance)"),
        ("corr_Gmean_vmag",        "safe_corr(g_mean_raw, vmag)"),
        ("corr_IR_G_imbalance",    "safe_corr(ir_raw, g_imbalance)"),
        ("corr_IR_vmag",           "safe_corr(ir_raw, vmag)"),
        ("corr_Ambient_vmag",      "safe_corr(amb_raw, vmag)"),
    ])

    # Hjorth 参数 (计算自 bp 信号)
    HJORTH_TEMPLATES = OrderedDict([
        ("Hjorth_Activity",     "float(np.var(bp)) — bp 信号方差"),
        ("Hjorth_Mobility",     "sqrt(var(diff(bp)) / var(bp)) — 一阶导数标准差比"),
        ("Hjorth_Complexity",   "sqrt(var(diff2(bp)) / var(diff(bp))) — 二阶与一阶导数标准差比"),
    ])

    # 熵特征 (计算自 bp 信号)
    ENTROPY_TEMPLATES = OrderedDict([
        ("Entropy_Shannon",     "Shannon entropy on 10-bin histogram of bp"),
        ("Entropy_ApEn",        "Approximate Entropy of bp (m=2, r=0.2*std)"),
        ("Entropy_SampEn",      "Sample Entropy of bp (m=2, r=0.2*std)"),
    ])

    # 导数特征 (计算自 bp 信号)
    DERIV_TEMPLATES = OrderedDict([
        ("Deriv_d1_mean",       "float(np.mean(np.diff(bp)))"),
        ("Deriv_d1_std",        "float(np.std(np.diff(bp)))"),
        ("Deriv_d1_max",        "float(np.max(np.diff(bp)))"),
        ("Deriv_d1_min",        "float(np.min(np.diff(bp)))"),
        ("Deriv_d1_zcr",        "zero-crossing rate of diff(bp) sign changes"),
        ("Deriv_d2_mean",       "float(np.mean(np.diff(np.diff(bp))))"),
        ("Deriv_d2_std",        "float(np.std(np.diff(np.diff(bp))))"),
        ("Deriv_d2_max",        "float(np.max(np.diff(np.diff(bp))))"),
        ("Deriv_d2_min",        "float(np.min(np.diff(np.diff(bp))))"),
        ("Deriv_d2_zcr",        "zero-crossing rate of 2nd-order diff sign changes"),
    ])

    # 时序动态特征 (计算自 bp 信号)
    TEMPORAL_TEMPLATES = OrderedDict([
        ("Temporal_slope_mean",      "linear regression slope of bp vs time"),
        ("Temporal_slope_std",       "std(residuals after linear detrend)"),
        ("Temporal_peak_prominence", "mean peak prominence from scipy.signal.find_peaks(bp)"),
        ("Temporal_peak_ratio",      "len(peaks) / len(bp)"),
        ("Temporal_valley_ratio",    "len(valleys) / len(bp)"),
    ])

    # ACC 特征
    ACC_TEMPLATES = OrderedDict([
        ("ACC_MAG_MEAN",        "float(np.mean(sqrt(acc_x²+acc_y²+acc_z²)))"),
        ("ACC_MAG_STD",         "float(np.std(mag)) — 加速度幅值标准差"),
        ("ACC_MAG_MAD",         "robust_mad(mag)"),
        ("ACC_AXIS_STD_SUM",    "sum(std(acc, axis=0)) — 三轴标准差之和"),
        ("ACC_GRAVITY_DOM_RATIO","max(|mean(acc_x)|,|mean(acc_y)|,|mean(acc_z)|) / sum(|mean|)"),
        ("ACC_BP_RMS",          "sqrt(mean(bandpass(mag, 0.5-5Hz)²))"),
        ("ACC_DIFF_MAD",        "robust_mad(diff(mag))"),
        ("ACC_STILL_SCORE",     "1/(1+50*mag_std/|mag_mean|) — 静止得分"),
    ])

    ACC_CROSS_TEMPLATES = OrderedDict([
        ("ACC_GREEN_BP_CORR",   "|safe_corr(bandpass(acc_mag), g_mean_bp)|"),
        ("ACC_IR_BP_CORR",      "|safe_corr(bandpass(acc_mag), ir_bp)|"),
        ("ACC_PPG_coherence_mean", "mean magnitude-squared coherence(acc_mag, g_mean_bp) over 0.5-3Hz"),
        ("ACC_PPG_coherence_max", "max magnitude-squared coherence(acc_mag, g_mean_bp) over 0.5-3Hz"),
    ])

    EXTRA_GREEN_TEMPLATES = OrderedDict([
        ("GREEN_FFT_harmonic_ratio", "2nd-harmonic FFT magnitude near 2*dom_freq divided by fundamental magnitude near dom_freq"),
        ("GREEN_FFT_harmonic_present", "1.0 if GREEN_FFT_harmonic_ratio > 0.2 else 0.0"),
    ])

    SHORT_WINDOW_TEMPLATES = OrderedDict([
        ("SQI_FLAT_RATIO", "mean(diff-flat-ratio over IR, Ambient, GreenMean before artifact removal)"),
        ("SQI_SPIKE_RATIO", "mean(diff > 6*MAD(diff) ratio over IR, Ambient, GreenMean before artifact removal)"),
        ("IR_ROBUST_RANGE_RATIO", "(p95(ir_raw)-p5(ir_raw)) / |median(ir_raw)|"),
        ("GREEN_ROBUST_RANGE_RATIO", "(p95(g_mean_raw)-p5(g_mean_raw)) / |median(g_mean_raw)|"),
        ("AMB_ROBUST_RANGE_RATIO", "(p95(amb_raw)-p5(amb_raw)) / |median(amb_raw)|"),
        ("IR_SEG_ACDC_CV", "std(AC/DC over three 1s IR segments) / mean(AC/DC)"),
        ("GREEN_SEG_ACDC_CV", "std(AC/DC over three 1s GreenMean segments) / mean(AC/DC)"),
        ("AMB_SEG_ACDC_CV", "std(AC/DC over three 1s Ambient segments) / mean(AC/DC)"),
        ("GREEN_BAND_ENERGY_RATIO", "sum(FFT(g_mean_bp)^2 over 0.7-3Hz) / sum(FFT(g_mean_bp)^2 over 0.5-5Hz)"),
        ("IR_BAND_ENERGY_RATIO", "sum(FFT(ir_bp)^2 over 0.7-3Hz) / sum(FFT(ir_bp)^2 over 0.5-5Hz)"),
        ("AMB_BAND_ENERGY_RATIO", "sum(FFT(amb_bp)^2 over 0.7-3Hz) / sum(FFT(amb_bp)^2 over 0.5-5Hz)"),
    ])

    # 合并所有模板
    ALL_TEMPLATES = OrderedDict()
    ALL_TEMPLATES.update(SHORT_WINDOW_TEMPLATES)
    ALL_TEMPLATES.update(CORE_TEMPLATES)
    ALL_TEMPLATES.update(CROSS_CH_TEMPLATES)
    ALL_TEMPLATES.update(SPATIAL_COUPLING)
    ALL_TEMPLATES.update(GREEN_SPATIAL_TEMPLATES)
    ALL_TEMPLATES.update(HJORTH_TEMPLATES)
    ALL_TEMPLATES.update(ENTROPY_TEMPLATES)
    ALL_TEMPLATES.update(DERIV_TEMPLATES)
    ALL_TEMPLATES.update(TEMPORAL_TEMPLATES)
    ALL_TEMPLATES.update(ACC_TEMPLATES)
    ALL_TEMPLATES.update(ACC_CROSS_TEMPLATES)
    ALL_TEMPLATES.update(EXTRA_GREEN_TEMPLATES)

    # 补充单通道模板（GREEN / IRX / AMBX）
    for pf in ["GREEN", "IRX", "AMBX"]:
        for tmpl, formula in SINGLE_CH_TEMPLATES.items():
            fname = tmpl.format(pf=pf)
            fml = formula
            if pf in CHANNEL_NAMES:
                for k, v in CHANNEL_NAMES[pf].items():
                    fml = fml.replace("{" + k + "}", v)
            fml = fml.replace("{pf}", pf)
            if fname not in ALL_TEMPLATES:
                ALL_TEMPLATES[fname] = fml

    # 为每个 selected feature 查找公式
    result = OrderedDict()
    for f in selected_features:
        info = OrderedDict()
        info["feature"] = f

        # 确定 category
        if f in SHORT_WINDOW_TEMPLATES:
            info["category"] = "short_window"
        elif f in CORE_TEMPLATES:
            info["category"] = "core"
        elif f in CROSS_CH_TEMPLATES:
            info["category"] = "cross_channel"
        elif f in SPATIAL_COUPLING:
            info["category"] = "spatial_coupling"
        elif f in GREEN_SPATIAL_TEMPLATES:
            info["category"] = "green_spatial"
        elif any(f.startswith(pf + "_") for pf in ["GREEN", "IRX", "AMBX"]):
            info["category"] = "single_channel"
        elif f in HJORTH_TEMPLATES:
            info["category"] = "hjorth"
        elif f in ENTROPY_TEMPLATES:
            info["category"] = "entropy"
        elif f in DERIV_TEMPLATES:
            info["category"] = "derivative"
        elif f in TEMPORAL_TEMPLATES:
            info["category"] = "temporal"
        elif f in ACC_TEMPLATES or f in ACC_CROSS_TEMPLATES:
            info["category"] = "acc"
        elif f in EXTRA_GREEN_TEMPLATES:
            info["category"] = "frequency"
        elif f == "mode":
            info["category"] = "mode"
        elif f in ("SIG_LEN", "SIG_SEC"):
            info["category"] = "meta"
        else:
            # 尝试去掉通道前缀匹配 (GREEN_/IRX_/AMBX_)
            base = f
            for cp in ("GREEN_", "IRX_", "AMBX_"):
                if f.startswith(cp):
                    base = f[len(cp):]
                    break
            if base in HJORTH_TEMPLATES:
                info["category"] = "hjorth"
            elif base in ENTROPY_TEMPLATES:
                info["category"] = "entropy"
            elif base in DERIV_TEMPLATES:
                info["category"] = "derivative"
            elif base in TEMPORAL_TEMPLATES:
                info["category"] = "temporal"
            else:
                info["category"] = "unknown"

        # 确定信号依赖
        sigs = set()
        if info["category"] in ("core", "cross_channel", "short_window"):
            sigs.update(["ir", "g_mean", "ambient"])
        elif info["category"] == "green_spatial":
            sigs.update(["g1", "g2", "g3"])
        elif info["category"] == "single_channel":
            ch = f.split("_")[0].lower()
            if ch == "green":
                sigs.add("g_mean")
            elif ch == "irx":
                sigs.add("ir")
            elif ch == "ambx":
                sigs.add("ambient")
        elif info["category"] in ("hjorth", "entropy", "derivative", "temporal"):
            sigs.add("bp")
        elif info["category"] == "acc":
            sigs.add("acc")
        elif info["category"] in ("spatial_coupling",):
            sigs.update(["ir", "g_mean", "ambient"])
        info["signals"] = sorted(sigs)

        # 查找公式：先直接匹配，再尝试去掉通道前缀匹配
        if f in ALL_TEMPLATES:
            info["formula"] = ALL_TEMPLATES[f]
        else:
            base = f
            for cp in ("GREEN_", "IRX_", "AMBX_"):
                if f.startswith(cp):
                    base = f[len(cp):]
                    break
            if base in ALL_TEMPLATES:
                channel = f[:len(f) - len(base)] if base != f else ""
                info["formula"] = f"[{channel.strip('_')}] " + ALL_TEMPLATES[base]
            else:
                info["formula"] = "[未匹配] — 请查看 s03_extract_feature_pool.py"

        result[f] = info

    return result


def export_deploy_artifacts(artifact_dir, skip_initial_windows=DEFAULT_SKIP_INITIAL_WINDOWS):
    """
    导出所有部署所需产物到 artifacts/deploy_package/。

    从已有的 artifacts 中读取：
      - stage1_threshold.json
      - model_bundle.pkl
      - selected_features.json
      - final_model_config.json (可选)

    产生：
      - deploy_config.json       总配置
      - stage1_config.json       Stage1 参数 + 公式
      - feature_formulas.json    每个入选特征的计算公式
      - xgboost_trees.txt        所有树的文本 dump
      - xgboost_nodes.csv        所有节点的结构化 CSV
      - model_params.json        模型超参 / threshold / fill_values
      - postprocess_config.json  状态机参数 + quality 阈值
    """
    import os as _os
    import shutil

    out_dir = _os.path.join(artifact_dir, "deploy_package")
    _os.makedirs(out_dir, exist_ok=True)
    print(f"\n导出部署产物到: {out_dir}")

    # --- 读取已有 artifacts ---
    stage1_path = _os.path.join(artifact_dir, "stage1_threshold.json")
    bundle_path = _os.path.join(artifact_dir, "model_bundle.pkl")
    features_path = _os.path.join(artifact_dir, "selected_features.json")
    config_path = _os.path.join(artifact_dir, "final_model_config.json")

    with open(stage1_path, "r", encoding="utf-8") as f:
        stage1 = json.load(f)
    bundle = joblib.load(bundle_path)
    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)
    postprocess_cfg = dict(DEFAULT_POSTPROCESS_CONFIG)
    if _os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            fcfg = json.load(f)
        if "postprocess" in fcfg:
            postprocess_cfg.update(fcfg["postprocess"])

    selected_features = features["selected_features"]
    model = bundle["model"]
    raw = bundle.get("raw_model", model)
    booster = raw.get_booster()

    # =========================================================
    # 1. stage1_config.json
    # =========================================================
    deploy_th = stage1.get("deploy_stage1_threshold", {})
    stage1_config = OrderedDict([
        ("pipeline_step", 1),
        ("description", "IR DC/ACDC 阈值粗筛 — 快速排除明显非佩戴样本"),
        ("input", "PPG 数据，IR 通道 (ch0)，100Hz 原始采样"),
        ("preprocessing", [
            "IR 100Hz → 5Hz 降采样 (resample_poly, gcd-based)",
        ]),
        ("window", {
            "duration_sec": 1.0,
            "points_5hz": 5,
            "stride_sec": 1.0,
        }),
        ("decision", OrderedDict([
            ("primitive_window_sec", 1.0),
            ("consecutive_pass_required", STAGE1_GATE_K),
            ("consecutive_fail_required", STAGE1_GATE_K),
            ("stage2_toggle_rule", "open Stage2 after 3 consecutive pass primitives; close after 3 consecutive fail primitives"),
        ])),
        ("features_per_window", OrderedDict([
            ("DC", {
                "formula": "min(neighbor_mean) where neighbor_mean[i] = (x[i] + x[i+1]) / 2.0",
                "purpose": "邻点均值的最小值，对单点毛刺鲁棒",
                "pseudocode": "x = ir_5hz[i:i+5]; dc = min((x[:-1] + x[1:]) / 2.0)",
            }),
            ("AC", {
                "formula": "median(|diff(x)|)",
                "purpose": "邻差的MAD，对单点抖动鲁棒",
                "pseudocode": "ac = float(np.median(np.abs(np.diff(x))))",
            }),
            ("AC_DC_RATIO", {
                "formula": "ac / (|dc| + 1e-8)",
                "purpose": "归一化交流分量",
            }),
        ])),
        ("window_pass_rule", "DC > dc_threshold AND AC_DC_RATIO < ac_dc_threshold"),
        ("streaming_gate_rule", "Stage2 opens after 3 consecutive 1s primitive windows pass and closes after 3 consecutive failures"),
        ("thresholds", OrderedDict([
            ("dc_threshold", float(deploy_th.get("dc_threshold", 0))),
            ("ac_dc_threshold", float(deploy_th.get("ac_dc_threshold", 0))),
        ])),
        ("search_source", deploy_th.get("search_source", "unknown")),
    ])

    with open(_os.path.join(out_dir, "stage1_config.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_config, f, indent=2, ensure_ascii=False)
    print("  [OK] stage1_config.json")

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
            "skip_initial_windows": int(skip_initial_windows),
            "use_stage2_ir": bool(bundle.get("meta", {}).get("use_stage2_ir", DEFAULT_USE_STAGE2_IR)),
            "fs_ppg": float(bundle["meta"]["fs_ppg"]),
        }),
        ("preprocessing_per_channel", [
            "remove_burr (k=6.0): 邻点超差异常→邻均值替换",
            "remove_step (k=10.0): 邻差过大→钳制为前一点",
            "medfilt (kernel=5): 中值滤波",
            "moving_average (win=3): 均值平滑",
            "bandpass_filter (0.4-6Hz, order=2): PPG 带通",
            "DC = median(raw_clean), BP = bandpass(raw_clean)",
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
        ("clip_bounds", bundle.get("clip_bounds", {})),
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
            ])),
            ("formula", OrderedDict([
                ("score_update", "score = alpha * quality * p + (1 - alpha * quality) * score"),
                ("state_0→1", "IF score > T_on for K_on consecutive updates AND cooldown expired THEN state=1"),
                ("state_1→0", "IF score < T_off for K_off consecutive updates AND cooldown expired THEN state=0"),
                ("cooldown", "After flip, wait cooldown_sec before next flip allowed"),
                ("leakage_decay", "on_count/off_count decrement by 1 when threshold not met (soft reset)"),
                ("final_prediction", "state machine final state"),
            ])),
        ])),
        ("quality_scoring", OrderedDict([
            ("description", "基于特征质量调整 EMA 平滑速度"),
            ("thresholds_source", "train — learned from bundle['quality_thresholds']"),
            ("features_used", ["Ambient_std", "G_mean_mean", "IR_mean"]),
            ("thresholds", bundle.get("quality_thresholds", {})),
            ("fallback", "if thresholds missing: Ambient_std>1e7→×0.5, |G_mean_mean|<1e-6→×0.5, |IR_mean|<1e-6→×0.5"),
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
            "Stage 1: IR DC/ACDC gate (1s primitive windows @5Hz, 3s consecutive decision)",
            f"Stage 2: {float(bundle['meta']['win_sec']):g}s feature window ({float(bundle['meta']['step_sec']):g}s stride) + XGBoost window probability",
            "Stage 3: [reserved for window-threshold optimization]",
            "Stage 4: WearStateMachine 时序后处理 (EMA + hysteresis)",
        ]),
        ("stage1", stage1_config),
        ("stage2_features", OrderedDict([
            ("window_config", OrderedDict([
                ("duration_sec", float(bundle["meta"]["win_sec"])),
                ("stride_sec", float(bundle["meta"]["step_sec"])),
                ("skip_initial_windows", int(skip_initial_windows)),
                ("use_stage2_ir", bool(bundle.get("meta", {}).get("use_stage2_ir", DEFAULT_USE_STAGE2_IR))),
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
        ])),
        ("bundle_fingerprint", bundle.get("fingerprint", {})),
    ])

    with open(_os.path.join(out_dir, "deploy_config.json"), "w", encoding="utf-8") as f:
        json.dump(deploy_config, f, indent=2, ensure_ascii=False)
    print("[OK] deploy_config.json")

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
    enabled = np.asarray(result.get("stage2_enabled_flags", np.ones(n, dtype=int)), dtype=int)
    if len(preds) != n:
        preds = (probs >= float(model_threshold)).astype(int)
    if len(enabled) != n:
        enabled = np.ones(n, dtype=int)
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
        stage1_enabled=enabled.astype(np.int64),
        prob_raw=probs,
        pred_raw=preds.astype(np.int64),
        quality=quality,
        ood_rate=ood_rate,
        mode=np.array(int(result.get("mode", 0)), dtype=np.int64),
        fallback=np.array(int(bool(result.get("fallback", False))), dtype=np.int64),
        model_threshold=np.array(float(model_threshold)),
        window_sec=np.array(float(window_sec)),
        stride_sec=np.array(float(stride_sec)),
        cache_schema_version=np.array("window_outputs_v2"),
        model_fingerprint_json=np.array(json.dumps(metadata.get("model_fingerprint", {}), ensure_ascii=False)),
        feature_names_json=np.array(json.dumps(list(metadata.get("feature_names", [])), ensure_ascii=False)),
        skip_initial_windows=np.array(int(metadata.get("skip_initial_windows", 0)), dtype=np.int64),
        use_stage2_ir=np.array(int(metadata.get("use_stage2_ir", result.get("use_stage2_ir", 0))), dtype=np.int64),
    )
    return fpath


def export_window_cache(results, artifact_dir, split, window_sec, stride_sec, model_threshold,
                        quality_thresholds=None, metadata=None, cache_root="window_outputs"):
    import os as _os, pandas as _pd
    metadata = metadata or {}
    out_dir = _os.path.join(artifact_dir, cache_root, split)
    _os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for r in results:
        try:
            npz_path = write_window_cache_npz(
                r, out_dir, window_sec, stride_sec, model_threshold,
                quality_thresholds=quality_thresholds,
                metadata=metadata,
            )
            prob_arr = np.asarray(r.get("window_probs", []), dtype=float)
            enabled_arr = np.asarray(r.get("stage2_enabled_flags", np.ones(len(prob_arr), dtype=int)), dtype=int)
            n_windows = int(len(prob_arr))
            n_enabled = int(np.sum(enabled_arr > 0))
            manifest.append({
                "sample_name": str(r.get("sample_name", "unknown")),
                "target": int(r.get("target", 0)),
                "npz_path": npz_path,
                "n_windows": n_windows,
                "n_stage1_enabled": n_enabled,
                "stage1_enabled_ratio": float(n_enabled / max(n_windows, 1)),
                "cache_schema_version": "window_outputs_v2",
                "model_fingerprint": metadata.get("model_fingerprint", {}),
                "feature_names": list(metadata.get("feature_names", [])),
                "skip_initial_windows": int(metadata.get("skip_initial_windows", 0)),
                "use_stage2_ir": bool(metadata.get("use_stage2_ir", r.get("use_stage2_ir", False))),
                "fallback": int(bool(r.get("fallback", False))),
                "fallback_reason": str(r.get("fallback_reason", "")),
            })
        except Exception:
            continue
    if manifest:
        _pd.DataFrame(manifest).to_csv(_os.path.join(out_dir, "manifest.csv"), index=False)
        with open(_os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[OK] window cache: {len(manifest)} samples -> {out_dir}/")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--method", type=str, default="state_machine",
                        choices=["mean_vote", "prob_mean", "state_machine"])
    parser.add_argument("--window_sec", type=int, default=3)
    parser.add_argument("--stride_sec", type=int, default=1)
    parser.add_argument("--skip_initial_windows", type=int, default=DEFAULT_SKIP_INITIAL_WINDOWS,
                        help="drop this many leading Stage2 windows per sample")
    parser.add_argument("--use_stage2_ir", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="whether Stage2 feature extraction uses IR channel values; defaults to model bundle metadata")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--optimize_split", type=str, default="valid",
                        choices=["train", "valid", "test"])
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")
    parser.add_argument("--warmup_frames", type=int, default=3,
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

    if args is None:
        args = parser.parse_args()

    with open(os.path.join(args.artifact_dir, "splits.json"), "r", encoding="utf-8") as f:
        split = json.load(f)
    with open(os.path.join(args.artifact_dir, "stage1_threshold.json"), "r", encoding="utf-8") as f:
        th = json.load(f)

    deploy_th = get_deploy_stage1_threshold(th)
    bundle_path = os.path.join(args.artifact_dir, "model_bundle.pkl")

    print("=" * 80)
    print("加载统一模型包")
    print("=" * 80)
    bundle = load_bundle(bundle_path)
    use_stage2_ir = resolve_use_stage2_ir(bundle, args.use_stage2_ir)
    print(f"feature_names: {len(bundle['feature_names'])} 个特征")
    print(f"threshold: {bundle['threshold']}")
    print(f"meta: {bundle['meta']}")
    print(f"use_stage2_ir: {use_stage2_ir}")

    print("\n" + "=" * 80)
    print("Deploy Stage1 threshold")
    print("=" * 80)
    print(f"dc_threshold   = {deploy_th['dc_threshold']}")
    print(f"acdc_threshold = {deploy_th['ac_dc_threshold']}")

    # 参数优化
    if args.optimize:
        print("\n" + "=" * 80)
        print(f"运行状态机参数优化 (split={args.optimize_split}, n_workers={args.n_workers})")
        print("=" * 80)
        opt = optimize_state_machine_params(
            samples=split[args.optimize_split],
            dc_threshold=deploy_th["dc_threshold"],
            ac_dc_threshold=deploy_th["ac_dc_threshold"],
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            bundle_path=bundle_path,
            n_workers=args.n_workers,
            skip_initial_windows=args.skip_initial_windows,
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
        dc_threshold=deploy_th["dc_threshold"],
        ac_dc_threshold=deploy_th["ac_dc_threshold"],
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        bundle_path=bundle_path,
        n_workers=args.n_workers,
        skip_initial_windows=args.skip_initial_windows,
        use_stage2_ir=use_stage2_ir,
    )

    # 三套指标
    sample_summary, details = compute_sample_metrics(
        results, args.method, postprocess_cfg, bundle["threshold"]
    )
    window_model_summary = compute_window_model_metrics(results)
    window_stream_summary = compute_window_stream_metrics(
        results, postprocess_cfg, warmup_frames=args.warmup_frames
    )

    # OOD 汇总：每条样本计 OOD 窗占比和触发标志
    ood_summary = _summarize_ood(results, alert_rate=args.ood_alert_rate)

    sample_summary["split"] = args.split
    sample_summary["selected_features"] = bundle["feature_names"]
    sample_summary["model_threshold"] = float(bundle["threshold"])
    sample_summary["bundle_fingerprint"] = bundle.get("fingerprint")
    sample_summary["threshold_policy"] = bundle.get("threshold_policy")

    print("\n" + "=" * 80)
    print("指标1: 端到端评估 (Stage1→Stage2→Stage3)")
    print("  最终预测 vs 样本真实标签")
    print("=" * 80)
    summary_for_print = {k: v for k, v in sample_summary.items() if k != "selected_features"}
    print(json.dumps(summary_for_print, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("指标2: Stage2 模型评估 (仅通过Stage1的数据)")
    print(f"  通过Stage1的样本: {window_model_summary.get('stage1_pass_samples', 0)}")
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
    print(f"  端到端 (Stage1→3):       {sample_summary['accuracy']:.4f}")
    print(f"  Stage2 模型 (逐窗):      {window_model_summary['accuracy']:.4f}")
    print(f"  Stage2+3 状态机 (逐窗):  {window_stream_summary['accuracy']:.4f}")

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
                  f"n_win={d.get('n_windows',0)}  s1_pass={d.get('stage1_pass',False)}")
    if fn_samples:
        print("\n  False Negatives (佩戴判为非佩戴):")
        for d in sorted(fn_samples, key=lambda x: np.mean(x.get("window_probs", []) or [0]))[:10]:
            avg_p = np.mean(d.get("window_probs", []) or [0])
            print(f"    {d['sample_name']:30s} target=1 pred=0  avg_prob={avg_p:.3f}  "
                  f"n_win={d.get('n_windows',0)}  s1_pass={d.get('stage1_pass',False)}")

    # 把 OOD 信息也合并进每条 detail
    ood_map = {s["sample_name"]: s for s in ood_summary["per_sample"]}
    for d in details:
        s = ood_map.get(d["sample_name"], {})
        d["ood_mean"] = s.get("ood_mean")
        d["ood_window_alert_rate"] = s.get("ood_window_alert_rate")

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
                    if (r.get("fallback", False) or not r.get("stage1_pass", False)
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
                "note": "本扫描不修改 bundle.threshold；用 F0.5（偏 precision）选合适操作点后，在 s05 重训阶段把 --threshold_objective fbeta --threshold_beta 0.5 固化。",
            }

    out_path = os.path.join(args.artifact_dir, f"end_to_end_eval_{args.split}_{args.method}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
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
                "skip_initial_windows": args.skip_initial_windows,
                "use_stage2_ir": use_stage2_ir,
            },
            cache_root=args.window_output_root,
        )
    if args.export_deploy:
        export_deploy_artifacts(
            args.artifact_dir,
            skip_initial_windows=args.skip_initial_windows,
        )


if __name__ == "__main__":
    main()
