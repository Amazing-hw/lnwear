# s03_extract_feature_pool.py
# -*- coding: utf-8 -*-

"""
步骤3：Stage2 特征池提取，增强鲁棒预处理版。

输入支持两种形态：
- 3D 预切窗 PPG：直接逐个使用 H5 中已有窗口，不再二次滑窗。
- 连续时序 PPG：按 H5 `frequency` 判断；100Hz 固定每 4 点取 1 点降到
  25Hz（保留索引 0,4,8,...），25Hz 直接使用，
  再按 5s/1s 滑窗（可显式切到 3s）。
- grouped-window H5：一个 record 下多个窗口 group，窗口名末尾为 *_w20_1；
  读取时按 w 后数字排序，label 来自最后一段。

功能：
1. 读取 artifacts/splits.json
2. 对全部合法样本/窗口提取 5s/25Hz XGBoost 特征
3. 读入并排序后保留每条数据的全部合法窗口。
4. 复用原始 H5 读取方式。
5. 只按 H5 `ppg_config` 构建三个固定物理光区，不做信号方差判断。
6. 模型候选使用与部署一致的短窗预处理：有限值替换、孤立毛刺修复、
   0.8s 滚动中位数去趋势；仅当 round(0.04*fs)>=2 时做短窗均值平滑。
   文件内保留的旧研究辅助函数不属于当前 126 项受治理候选的提取链路。
7. 输出特征池 CSV：
   - feature_pool_train.csv
   - feature_pool_valid.csv
   - feature_pool_test.csv
"""

import os
import json
import argparse
import re
import sys
import warnings as _commercial_warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd


import commercial_liveness_features
from direct_feature_selection import load_direct_feature_csv

from stage2_feature_catalog import (
    FEATURE_CATALOG,
    FEATURE_POOL_VERSION as STAGE2_FEATURE_POOL_VERSION,
    feature_record as stage2_feature_record,
    is_model_candidate as is_stage2_model_candidate,
    model_candidate_names as stage2_model_candidate_names,
    validate_candidate_names as validate_stage2_candidate_names,
)

_commercial_port_main = commercial_liveness_features.main

# =========================================================
# 基本配置
# =========================================================

EPS = 1e-12
MIN_ZONE_RELATIVE_AC_RMS = 1e-8
MIN_ZONE_ABSOLUTE_AC_RMS = 1e-9
DEFAULT_FS = 100.0
DEFAULT_USE_STAGE2_IR = False
COMMERCIAL_8_FEATURE_NAMES = [
    "GREEN_CORR",
    "GREEN_AC",
    "AMB_AC",
    "ACC_YSUM",
    "GREEN_DC",
    "AMB_DC",
    "GREEN_XCORR",
    "FFT_PEAK_MEDIAN_RATIO",
]

COMMERCIAL_STAGE2_FIELDS = (
    "GREEN_CORR",
    "COMM_GREEN_AC",
    "COMM_AMB_AC",
    "ACC_MAG_MEAN",
    "GREEN_DC_MEDIAN",
    "AMBX_DC_MEDIAN",
    "GREEN_AUTO_CORR_PEAK",
    "GREEN_FFT_PEAK_MEDIAN_RATIO",
)

WINDOW_NAME_RE = re.compile(r"(?:^|_)w(?P<index>\d+)_(?P<label>[01])$")


def apply_stage2_ir_policy(ir, use_stage2_ir=DEFAULT_USE_STAGE2_IR):
    """Return the IR signal used by Stage2 features according to the pipeline switch."""
    ir = np.asarray(ir, dtype=np.float64)
    if use_stage2_ir:
        return ir
    return np.zeros_like(ir, dtype=np.float64)


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_n_workers(n_workers=None, n_items=None, cap=4):
    """Resolve a conservative worker count for server-safe batch runs."""
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
    """Return an mp context when WL_MP_START_METHOD is set, otherwise default."""
    method = os.environ.get("WL_MP_START_METHOD", "").strip()
    if not method:
        return None
    import multiprocessing as mp
    return mp.get_context(method)


def export_feature_pool_analysis_plot(frames, artifact_dir):
    """Export split coverage, numerical quality, groups, and top separation as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scientific_figures import save_scientific_figure

    parts = [part for part in ("train", "valid", "test") if part in frames]
    candidate_names = stage2_model_candidate_names()
    source_rows = []
    window_counts = {part: [] for part in parts}
    finite_rates = []
    for part in parts:
        frame = frames[part]
        for target in (0, 1):
            count = int((frame.get("target", pd.Series(dtype=int)) == target).sum())
            window_counts[part].append(count)
            source_rows.append({"panel": "window_count", "split": part, "target": target, "value": count})
        available = [name for name in candidate_names if name in frame.columns]
        values = frame[available].to_numpy(dtype=float) if available and len(frame) else np.empty((0, 0))
        finite_rate = float(np.isfinite(values).mean()) if values.size else 0.0
        finite_rates.append(finite_rate)
        source_rows.append({"panel": "finite_rate", "split": part, "value": finite_rate})

    group_counts = OrderedDict()
    for name in candidate_names:
        group = str(stage2_feature_record(name)["group"])
        group_counts[group] = group_counts.get(group, 0) + 1
        source_rows.append({"panel": "feature_group", "group": group, "feature": name, "value": 1})

    separations = []
    train = frames.get("train", pd.DataFrame())
    if len(train) and "target" in train.columns:
        for name in candidate_names:
            if name not in train.columns:
                continue
            x0 = pd.to_numeric(train.loc[train["target"] == 0, name], errors="coerce")
            x1 = pd.to_numeric(train.loc[train["target"] == 1, name], errors="coerce")
            pooled = float(pd.concat([x0, x1]).std(ddof=0))
            score = abs(float(x1.mean()) - float(x0.mean())) / max(pooled, 1e-12)
            if np.isfinite(score):
                separations.append((name, score))
    separations = sorted(separations, key=lambda item: (-item[1], item[0]))[:12]
    for rank, (name, score) in enumerate(separations, start=1):
        source_rows.append({"panel": "train_separation", "rank": rank, "feature": name, "value": score})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="white")
    x = np.arange(len(parts))
    c0 = [window_counts[part][0] for part in parts]
    c1 = [window_counts[part][1] for part in parts]
    axes[0, 0].bar(x, c0, color="#4C78A8", label="not worn")
    axes[0, 0].bar(x, c1, bottom=c0, color="#E07B53", label="worn")
    axes[0, 0].set_xticks(x, parts)
    axes[0, 0].set_ylabel("windows")
    axes[0, 0].set_title("Window coverage", loc="left", weight="bold")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].bar(x, finite_rates, color="#2A9D8F")
    axes[0, 1].set_xticks(x, parts)
    axes[0, 1].set_ylim(0, 1.01)
    axes[0, 1].set_ylabel("finite fraction")
    axes[0, 1].set_title("Numerical completeness", loc="left", weight="bold")
    group_names = list(group_counts)
    axes[1, 0].barh(group_names, list(group_counts.values()), color="#6C8EBF")
    axes[1, 0].set_xlabel("features")
    axes[1, 0].set_title("Interpretable feature groups", loc="left", weight="bold")
    if separations:
        labels = [name for name, _score in separations][::-1]
        scores = [score for _name, score in separations][::-1]
        axes[1, 1].barh(labels, scores, color="#C89B3C")
        axes[1, 1].set_xlabel("standardized mean difference")
    else:
        axes[1, 1].text(0.5, 0.5, "No train separation data", ha="center", va="center")
        axes[1, 1].set_axis_off()
    axes[1, 1].set_title("Top train-only separation", loc="left", weight="bold")
    fig.suptitle("Stage2 feature-pool audit", fontsize=14, weight="bold", x=0.04, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    artifact_dir = os.fspath(artifact_dir)
    out_path = os.path.join(artifact_dir, "report_plots", "s03_feature_pool_analysis.png")
    inputs = [
        os.path.join(artifact_dir, f"feature_pool_{part}.csv")
        for part in parts
        if os.path.isfile(os.path.join(artifact_dir, f"feature_pool_{part}.csv"))
    ]
    outputs = save_scientific_figure(
        fig, out_path, source_data=source_rows,
        core_conclusion="The Stage2 pool is complete, numerically finite, physically grouped, and diagnostically diverse before manual selection.",
        panel_map={
            "a": "Window counts by split and target.",
            "b": "Finite feature-value rate by split.",
            "c": "Governed feature counts by interpretable group.",
            "d": "Top train-only standardized class separations.",
        },
        inputs=inputs,
        split="train_valid_test",
        n_definition="windows for coverage/finite panels and governed features for group/separation panels",
        statistics={"separation": "absolute train-only standardized mean difference", "interval": "none"},
        reviewer_risks=["Train-only separation is descriptive and must not replace grouped validation evidence."],
    )
    plt.close(fig)
    return outputs

# 已知高冗余特征（节省计算，尤其 Entropy/Derivative 类 O(N²) 操作）
# 这些特征在 s04 清洗阶段也会被 VIF/高相关移除，提前在 s03 跳过以加速提取
_REDUNDANT_FEATURES = {
    # -- ApEn (skip_apen=True, 不再计算) --
    # -- 二阶导数 (compute_d2=False, 不再计算) --
    # -- valley_ratio ≈ 1 - peak_ratio --
    "GREEN_Temporal_valley_ratio", "IRX_Temporal_valley_ratio", "AMBX_Temporal_valley_ratio",
    # -- AC_RMS ≈ 1.48 × AC_MAD (MAD 更鲁棒，保留 MAD) --
    "IRX_AC_RMS",
    "G1_AC_RMS", "G2_AC_RMS", "G3_AC_RMS",
    # -- AUTO_CORR_LAG_SEC ≈ 1/DOM_FREQ --
    "G1_AUTO_CORR_LAG_SEC", "G2_AUTO_CORR_LAG_SEC", "G3_AUTO_CORR_LAG_SEC",
    # -- DOM_FREQ 跨通道不变（同一心跳），consensus range/cv 始终为 0 --
    "G_consensus_DOM_FREQ_min", "G_consensus_DOM_FREQ_max",
    "G_consensus_DOM_FREQ_range", "G_consensus_DOM_FREQ_cv",
    "G_consensus_DOM_FREQ_top2_mean",
    # -- Hjorth_Activity = var(bp) ≈ (AC_RMS)²，仅 IRX/AMBX 砍 --
    "IRX_Hjorth_Activity", "AMBX_Hjorth_Activity",
    # -- Hjorth_Complexity 二阶导出，极不稳定 --
    "GREEN_Hjorth_Complexity", "IRX_Hjorth_Complexity", "AMBX_Hjorth_Complexity",
    # -- Entropy_Shannon 对连续信号区分度低，仅保留 GREEN --
    "IRX_Entropy_Shannon", "AMBX_Entropy_Shannon",
    # -- 与 IR_over_Gmean_mean 信息重叠 --
    "log_IR_Gmean_mean",
    # -- ≈ IRX_DERIV_MAD --
    "IR_diff_std",
    # -- Hjorth_Mobility ≈ Deriv_d1_std / AC_RMS，ratio 型 --
    "GREEN_Hjorth_Mobility", "IRX_Hjorth_Mobility", "AMBX_Hjorth_Mobility",
    # -- 手表佩戴姿态固定，区分度低 --
    "ACC_GRAVITY_DOM_RATIO",
    # Not in the deployment formula surface; keep them out of the final pool.
    "G_consensus_AC_MAD_range",
    "GREEN_FFT_harmonic_ratio", "GREEN_FFT_harmonic_present",
    # -- AC_DC_RATIO = AC_RMS / |DC|；当前单通道函数仍正确计算 AC_RMS 中间量，
    #    只是最终特征池中优先保留更鲁棒的 AC_MAD / |DC| 等价信息。--
}

# =========================================================
# H5 读取工具
# =========================================================

def normalize_ppg_array(arr):
    """Normalize H5 PPG arrays to (T, C) or (N_win, T_win, C) channel-last layout."""
    x = np.asarray(arr)
    if x.ndim == 2:
        return x.T
    if x.ndim == 3:
        return np.transpose(x, (0, 2, 1))
    raise ValueError(f"unsupported PPG ndim={x.ndim}, shape={x.shape}")


def normalize_acc_array(arr):
    """Normalize H5 ACC arrays to (T, C) or (N_win, T_win, C) channel-last layout."""
    x = np.asarray(arr)
    if x.ndim == 2:
        return x.T
    if x.ndim == 3:
        return np.transpose(x, (0, 2, 1))
    raise ValueError(f"unsupported ACC ndim={x.ndim}, shape={x.shape}")


def is_prewindowed_signal(arr):
    return np.asarray(arr).ndim == 3


def flatten_prewindowed_signal(arr):
    x = np.asarray(arr)
    if x.ndim != 3:
        return x
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2])


def parse_grouped_window_name(name):
    """Return (window_index, label) parsed from names ending in *_w20_1."""
    match = WINDOW_NAME_RE.search(str(name))
    if not match:
        return None
    return int(match.group("index")), int(match.group("label"))


def _raw_sorted_grouped_window_items(group):
    """Return all recognized grouped PPG windows in numeric window order."""
    items = []
    for child_name in group.keys():
        parsed = parse_grouped_window_name(child_name)
        if parsed is None:
            continue
        child = group[child_name]
        if not isinstance(child, h5py.Group) or "ppg" not in child:
            continue
        window_index, label = parsed
        items.append((window_index, label, child_name, child))
    return sorted(items, key=lambda item: item[0])


def load_grouped_window_metadata(sample):
    if sample.get("window_layout") != "grouped_windows":
        return None
    indices = sample.get("window_indices")
    labels = sample.get("window_labels")
    names = sample.get("window_names")
    if indices is not None and labels is not None:
        return {
            "window_indices": [int(value) for value in indices],
            "window_labels": [int(value) for value in labels],
            "window_names": [str(value) for value in names] if names is not None else None,
        }
    with h5py.File(sample["h5_file"], "r") as f:
        items = _raw_sorted_grouped_window_items(f[sample["sample_name"]])
    return {
        "window_indices": [int(item[0]) for item in items],
        "window_labels": [int(item[1]) for item in items],
        "window_names": [str(item[2]) for item in items],
    }


def load_ppg(sample):
    """
    Read PPG as continuous (T, C) or pre-windowed (N_win, T_win, C).
    Old H5 layout (C, T) remains supported.
    """
    with h5py.File(sample["h5_file"], "r") as f:
        grp = f[sample["sample_name"]]
        if sample.get("window_layout") == "grouped_windows" or "ppg" not in grp:
            grouped_items = _raw_sorted_grouped_window_items(grp)
            windows = []
            for _idx, _label, _name, child in grouped_items:
                windows.append(normalize_ppg_array(child["ppg"][:]))
            if not windows:
                raise KeyError(f"sample {sample['sample_name']} has no grouped PPG windows")
            ppg = np.stack(windows, axis=0)
        else:
            ppg = normalize_ppg_array(grp["ppg"][:])
    return ppg


def load_acc(sample):
    """
    读取ACC数据：
        f[sample_name]['acc'][:].T
    即原始为 (3, N)，转成 (N, 3)
    如果没有acc数据，返回None
    """
    with h5py.File(sample["h5_file"], "r") as f:
        grp = f[sample["sample_name"]]
        if sample.get("window_layout") == "grouped_windows" or "ppg" not in grp:
            acc_windows = []
            for _idx, _label, _name, child in _raw_sorted_grouped_window_items(grp):
                if "acc" not in child:
                    return None
                acc_windows.append(normalize_acc_array(child["acc"][:]))
            if not acc_windows:
                return None
            return np.stack(acc_windows, axis=0)
        if "acc" not in grp:
            return None
        acc = normalize_acc_array(grp["acc"][:])
    return acc

def get_sample_frequency(sample):
    """Return the validated H5 sampling frequency recorded by s01."""
    try:
        frequency = int(sample["frequency"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sample frequency metadata is missing or invalid") from exc
    if frequency not in {25, 100}:
        raise ValueError(f"sample frequency must be 25 or 100, got {frequency}")
    return frequency


def get_sample_ppg_config(sample):
    """Return the validated H5 green-channel configuration recorded by s01."""
    try:
        ppg_config = int(sample["ppg_config"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sample ppg_config metadata is missing or invalid") from exc
    if ppg_config not in {0, 1, 2}:
        raise ValueError(f"sample ppg_config must be 0, 1, or 2, got {ppg_config}")
    return ppg_config


def _is_25hz_sample(sample):
    """Compatibility helper backed only by explicit H5 frequency metadata."""
    return get_sample_frequency(sample) == 25
# =========================================================
# 绿光通道构建逻辑
# =========================================================

def get_channels_from_window(window, ppg_config):
    """
    根据 H5 ppg_config 统一输出三个固定物理光区：
    - 0: g1=ch3, g2=ch4, g3=ch5
    - 1: g1=(ch3+ch9)/2, g2=(ch4+ch10)/2, g3=(ch5+ch11)/2
    - 2: g1=(ch6+ch9+ch12)/3, g2=(ch7+ch10+ch13)/3,
         g3=(ch8+ch11+ch14)/3

    IR 固定为 ch0，Ambient 固定为 ch1。通道编号为零基索引。
    """
    ir = window[:, 0]

    if window.shape[1] > 1:
        ambient = window[:, 1]
    else:
        ambient = window[:, 0]

    try:
        ppg_config = int(ppg_config)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ppg_config must be 0, 1, or 2") from exc

    if ppg_config == 0 and window.shape[1] >= 6:
        g1 = window[:, 3]
        g2 = window[:, 4]
        g3 = window[:, 5]
    elif ppg_config == 1 and window.shape[1] >= 12:
        g1 = (window[:, 3] + window[:, 9]) / 2.0
        g2 = (window[:, 4] + window[:, 10]) / 2.0
        g3 = (window[:, 5] + window[:, 11]) / 2.0
    elif ppg_config == 2 and window.shape[1] >= 15:
        g1 = (window[:, 6] + window[:, 9] + window[:, 12]) / 3.0
        g2 = (window[:, 7] + window[:, 10] + window[:, 13]) / 3.0
        g3 = (window[:, 8] + window[:, 11] + window[:, 14]) / 3.0
    else:
        required = {0: 6, 1: 12, 2: 15}.get(ppg_config)
        if required is None:
            raise ValueError(f"ppg_config must be 0, 1, or 2, got {ppg_config}")
        raise ValueError(
            f"ppg_config={ppg_config} requires at least {required} PPG channels, "
            f"got {window.shape[1]}"
        )

    return ir, ambient, g1, g2, g3

# =========================================================
# 边界情况处理
# =========================================================

def validate_window(ppg_window, min_channels=6, min_length=100):
    """
    验证窗口有效性
    
    参数:
        ppg_window: PPG窗口数据，shape (N, C)
        min_channels: 最小通道数要求，默认6
        min_length: 最小长度要求，默认100
        
    返回:
        bool: 窗口是否有效
    """
    if ppg_window is None:
        return False
    
    ppg = np.asarray(ppg_window, dtype=np.float64)
    
    if ppg.ndim == 1:
        ppg = ppg.reshape(-1, 1)
    
    if len(ppg) < min_length:
        return False
    
    if ppg.shape[1] < min_channels:
        return False
    
    return True


def validate_h5_file(h5_file, sample_name):
    """
    验证H5文件是否可读
    
    参数:
        h5_file: H5文件路径
        sample_name: 样本名称
        
    返回:
        tuple: (是否有效, 错误信息)
    """
    try:
        with h5py.File(h5_file, "r") as f:
            if sample_name not in f:
                return False, f"样本 {sample_name} 不存在于H5文件中"

            grp = f[sample_name]
            if "ppg" not in grp:
                items = _raw_sorted_grouped_window_items(grp)
                if not items:
                    return False, f"样本 {sample_name} 缺少PPG数据"
                ppg = normalize_ppg_array(items[0][3]["ppg"][:])
                if ppg is None or len(ppg) == 0:
                    return False, f"样本 {sample_name} PPG数据为空"
                return True, None

            ppg = normalize_ppg_array(grp["ppg"][:])
            if ppg is None or len(ppg) == 0:
                return False, f"样本 {sample_name} PPG数据为空"
                
        return True, None
    except Exception as e:
        return False, f"H5文件读取失败: {str(e)}"


# =========================================================
# 鲁棒基础工具函数
# =========================================================

def safe_div(a, b, eps=EPS):
    return float(a) / (float(b) + eps)


STAGE2_IR_FEATURE_PREFIXES = (
    "IR_",
    "IRX_",
    "GREEN_IR_",
    "IR_AMB_",
    "IR_over_",
    "log_IR_",
    "corr_IR_",
    "ACC_IR_",
)

STAGE2_IR_FEATURE_NAMES = {
    "corr_Ambient_IR",
}


def is_stage2_ir_feature(name):
    """Return True for features that use the IR channel and must not enter Stage2."""
    n = str(name)
    return n in STAGE2_IR_FEATURE_NAMES or any(n.startswith(p) for p in STAGE2_IR_FEATURE_PREFIXES)


def filter_stage2_ir_features(features):
    """Remove all IR-derived Stage2 features while preserving input order."""
    if hasattr(features, "items"):
        return OrderedDict((k, v) for k, v in features.items() if not is_stage2_ir_feature(k))
    return [f for f in features if not is_stage2_ir_feature(f)]


DEPLOYMENT_ALLOWED_FFT_FEATURES = {
    name for name, record in FEATURE_CATALOG.items() if bool(record.get("fft"))
}
DEPLOYMENT_ALLOWED_NON_FFT_FEATURES = (
    set(FEATURE_CATALOG) - DEPLOYMENT_ALLOWED_FFT_FEATURES
)


def is_deployment_friendly_stage2_feature(name):
    """Return whether a feature belongs to the governed Stage2 model surface."""
    return is_stage2_model_candidate(name)


def filter_deployment_friendly_stage2_features(features):
    """Filter Stage2 features by the deployment-friendly allowlist."""
    filtered = filter_stage2_ir_features(features)
    if hasattr(filtered, "items"):
        return OrderedDict(
            (k, v) for k, v in filtered.items()
            if is_deployment_friendly_stage2_feature(k)
        )
    return [f for f in filtered if is_deployment_friendly_stage2_feature(f)]


def robust_mad(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0

    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def robust_iqr(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0

    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)

def safe_corr(x, y):
    """计算相关系数（带缓存优化）"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = min(len(x), len(y))
    if n < 8:
        return 0.0

    x = x[:n]
    y = y[:n]

    x = x - np.mean(x)
    y = y - np.mean(y)

    sx = np.std(x)
    sy = np.std(y)

    if sx < EPS or sy < EPS:
        return 0.0

    v = np.mean((x / sx) * (y / sy))

    if not np.isfinite(v):
        return 0.0

    return float(v)


def moving_average_filter(x, window_size=5):
    x = np.asarray(x, dtype=np.float64)

    if len(x) < window_size or window_size < 2:
        return x.copy()

    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(x, kernel, mode="same")

# =========================================================
# 鲁棒预处理
# =========================================================

def remove_burr(x, burr_k=6.0):
    """
    去毛刺：
    当前点与左右点都差异很大时，用邻点均值替换。

    向量化：判定/替换都基于原始邻点值，无序贯依赖。
    """
    x = np.asarray(x, dtype=np.float64).copy()

    if len(x) < 3:
        return x

    d = np.diff(x)
    mad_d = robust_mad(d)
    thr = max(burr_k * mad_d, EPS)

    left = x[:-2]
    mid = x[1:-1]
    right = x[2:]
    bad = (np.abs(mid - left) > thr) & (np.abs(mid - right) > thr)
    if bad.any():
        replaced = 0.5 * (left + right)
        x[1:-1] = np.where(bad, replaced, mid)

    return x

def remove_step(x, step_k=10.0):
    """
    去跳变：
    相邻点差异过大时，钳制为前一点。

    注意：
    这个规则偏保守，能抑制突发跳点；
    但如果真实戴摘瞬间进入窗口，也会被平滑掉一部分。
    对 3s 活体窗口通常是可以接受的。
    """
    x = np.asarray(x, dtype=np.float64).copy()

    if len(x) < 2:
        return x

    d = np.diff(x)
    mad_d = robust_mad(d)
    thr = max(step_k * mad_d, EPS)

    for i in range(1, len(x)):
        if abs(x[i] - x[i - 1]) > thr:
            x[i] = x[i - 1]

    return x

# 全局缓存 FIR bandpass 核，避免每窗重算。C 可直译：sinc + Hamming 窗 + 卷积。
_FIR_BANDPASS_CACHE: Dict[Tuple[float, float, float, int], np.ndarray] = {}


def _get_fir_bandpass_kernel(fs, lowcut, highcut, numtaps=65):
    """Windowed-sinc FIR bandpass kernel. Zero-phase via forward-backward convolve."""
    key = (float(fs), float(lowcut), float(highcut), int(numtaps))
    if key in _FIR_BANDPASS_CACHE:
        return _FIR_BANDPASS_CACHE[key]
    nyq = 0.5 * fs
    lo = lowcut / nyq
    hi = highcut / nyq
    # Ideal bandpass = lowpass(hi) - lowpass(lo)
    t = np.arange(numtaps, dtype=np.float64) - (numtaps - 1) / 2.0
    h = np.sinc(hi * t) * hi - np.sinc(lo * t) * lo
    h *= np.hamming(numtaps)
    h /= np.sum(np.abs(h)) + 1e-12
    _FIR_BANDPASS_CACHE[key] = h
    return h


def bandpass_filter(x, fs, lowcut=0.4, highcut=6.0, order=None, numtaps=65):
    """Windowed-sinc FIR bandpass (zero-phase via forward-backward convolve).

    Equivalent C pattern:
      - Design kernel once (sinc + Hamming window).
      - Convolve forward, then convolve reversed for zero-phase.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < numtaps:
        return x.copy()
    h = _get_fir_bandpass_kernel(fs, lowcut, highcut, numtaps)
    # Forward-backward convolve for zero-phase (like filtfilt)
    forward = np.convolve(x, h, mode='same')
    return np.convolve(forward[::-1], h, mode='same')[::-1]


def _median_filter_np(x, kernel_size):
    """Numpy median filter - directly portable to C (rolling window median)."""
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    half = k // 2
    n = len(x)
    out = np.empty(n, dtype=x.dtype)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out

def preprocess_signal(x, fs):
    """
    返回：
        raw_clean: 清理后的原始信号，用于 DC / IQR / raw corr
        bp: 带通信号，用于 AC / 频谱 / 相关 / 自相关
        dc: 直流中值

    滤波核为时间自适应（fs 变化时保持一致的时间尺度）：
        median_filter ≈ 50ms, moving_avg ≈ 30ms (all numpy, C-portable)
    """
    x = np.asarray(x, dtype=np.float64).copy()

    x = remove_burr(x, burr_k=6.0)
    x = remove_step(x, step_k=10.0)

    # 时间自适应: 50ms 中值滤波 (min 3)
    mf_kernel = max(3, int(round(0.05 * fs)))
    if mf_kernel % 2 == 0:
        mf_kernel += 1
    if len(x) >= mf_kernel:
        try:
            x = _median_filter_np(x, kernel_size=mf_kernel)
        except Exception:
            pass

    # 时间自适应: 30ms 滑动平均 (min 2)
    ma_win = max(2, int(round(0.03 * fs)))
    x = moving_average_filter(x, window_size=ma_win)

    dc = float(np.median(x))

    bp = bandpass_filter(x, fs, lowcut=0.4, highcut=6.0, order=4)
    bp = moving_average_filter(bp, window_size=ma_win)

    return x, bp, dc

# =========================================================
# 周期性 / 频域特征
# =========================================================

def fft_peak_features(x, fs, fmin=0.5, fmax=5.0):
    """
    返回：
        peak_median_ratio
        dom_freq
    """
    x = np.asarray(x, dtype=np.float64)

    if len(x) < 16:
        return 0.0, 0.0

    x = x - np.mean(x)
    xw = x * np.hamming(len(x))

    nfft = 1
    while nfft < len(x):
        nfft <<= 1

    nfft = max(256, nfft)

    spec = np.abs(np.fft.rfft(xw, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    mask = (freqs >= fmin) & (freqs <= fmax)

    if not np.any(mask):
        return 0.0, 0.0

    band_spec = spec[mask]
    band_freqs = freqs[mask]

    med = np.median(band_spec)
    peak = float(np.max(band_spec))

    if peak <= EPS:
        peak_ratio = 0.0
        dom_freq = 0.0
    elif med < EPS:
        peak_ratio = 0.0
        dom_freq = float(band_freqs[np.argmax(band_spec)])
    else:
        peak_ratio = float(peak / (med + EPS))
        dom_freq = float(band_freqs[np.argmax(band_spec)])

    return peak_ratio, dom_freq


def compute_fft_cache(x, fs, fmin=0.5, fmax=5.0):
    """
    一次性计算FFT，返回所有需要的信息（避免重复计算）
    
    返回：
        dict: 包含 peak_ratio, dom_freq, spec, freqs 等
    """
    x = finite_signal(x)
    
    result = {
        'peak_ratio': 0.0,
        'dom_freq': 0.0,
        'spec': None,
        'complex_spec': None,
        'freqs': None,
        'band_spec': None,
        'band_complex': None,
        'band_freqs': None
    }
    
    if len(x) < 16:
        return result
    
    x = x - np.mean(x)
    xw = x * np.hamming(len(x))
    
    nfft = 1
    while nfft < len(x):
        nfft <<= 1
    
    nfft = max(256, nfft)
    
    complex_spec = np.fft.rfft(xw, n=nfft)
    spec = np.abs(complex_spec)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    
    mask = (freqs >= fmin) & (freqs <= fmax)
    
    if not np.any(mask):
        result['spec'] = spec
        result['complex_spec'] = complex_spec
        result['freqs'] = freqs
        return result
    
    band_spec = spec[mask]
    band_freqs = freqs[mask]
    
    med = np.median(band_spec)
    peak = float(np.max(band_spec))

    if peak <= EPS:
        peak_ratio = 0.0
        dom_freq = 0.0
    elif med < EPS:
        peak_ratio = 0.0
        dom_freq = float(band_freqs[np.argmax(band_spec)])
    else:
        peak_ratio = float(peak / (med + EPS))
        dom_freq = float(band_freqs[np.argmax(band_spec)])
    
    result['peak_ratio'] = peak_ratio
    result['dom_freq'] = dom_freq
    result['spec'] = spec
    result['complex_spec'] = complex_spec
    result['freqs'] = freqs
    result['band_spec'] = band_spec
    result['band_complex'] = complex_spec[mask]
    result['band_freqs'] = band_freqs
    
    return result

def normalized_autocorr(x):
    x = np.asarray(x, dtype=np.float64)

    if len(x) < 4:
        return np.zeros(1, dtype=np.float64)

    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[len(x) - 1:]

    if corr[0] < EPS:
        return np.zeros_like(corr)

    return corr / (corr[0] + EPS)

def autocorr_periodicity_features(x, fs, bpm_min=40.0, bpm_max=180.0):
    """
    返回：
        ac_peak
        ac_lag_sec
    """
    x = np.asarray(x, dtype=np.float64)

    if len(x) < int(fs * 1.5):
        return 0.0, 0.0

    ac = normalized_autocorr(x)

    lag_min = int(fs * 60.0 / bpm_max)
    lag_max = int(fs * 60.0 / bpm_min)

    lag_min = max(1, lag_min)
    lag_max = min(len(ac) - 1, lag_max)

    if lag_max <= lag_min:
        return 0.0, 0.0

    seg = ac[lag_min:lag_max + 1]

    if len(seg) == 0:
        return 0.0, 0.0

    idx = int(np.argmax(seg))
    peak = float(seg[idx])
    lag = lag_min + idx
    lag_sec = float(lag / fs)

    return peak, lag_sec

def max_norm_xcorr(x, y, max_lag_samples):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = min(len(x), len(y))

    if n < 8:
        return 0.0

    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])

    sx = np.std(x)
    sy = np.std(y)

    if sx < EPS or sy < EPS:
        return 0.0

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-n + 1, n)

    mask = np.abs(lags) <= max_lag_samples

    corr = corr[mask]
    corr = corr / (n * sx * sy + EPS)

    if len(corr) == 0:
        return 0.0

    return float(np.max(np.abs(corr)))


def bounded_xcorr_peak_lag_samples(x, y, max_lag_samples):
    """Return absolute peak lag without assigning direction to symmetric zones."""
    x = finite_signal(x)
    y = finite_signal(y)
    n = min(len(x), len(y))
    if n < 8:
        return 0
    x = x[:n]
    y = y[:n]
    best_score = -1.0
    best_abs_lag = 0
    for lag in range(-int(max_lag_samples), int(max_lag_samples) + 1):
        if lag < 0:
            left, right = x[-lag:], y[:n + lag]
        elif lag > 0:
            left, right = x[:n - lag], y[lag:]
        else:
            left, right = x, y
        score = abs(guarded_corr(left, right))
        abs_lag = abs(lag)
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and abs_lag < best_abs_lag
        ):
            best_score = score
            best_abs_lag = abs_lag
    return int(best_abs_lag)

def smooth_envelope(x, fs, win_sec=0.25):
    x = np.abs(np.asarray(x, dtype=np.float64))

    win = max(3, int(round(win_sec * fs)))

    if win % 2 == 0:
        win += 1

    kernel = np.ones(win, dtype=np.float64) / win

    return np.convolve(x, kernel, mode="same")


def finite_signal(x, fill_value=0.0):
    """Replace NaN/Inf with the finite median so robust features stay finite."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if finite.any():
        fill = float(np.median(arr[finite]))
    else:
        fill = float(fill_value)
    return np.where(finite, arr, fill).astype(np.float64, copy=False)


# =========================================================
# Window-length-adaptive 25Hz robust features (3s or 5s)
# =========================================================

def robust_range_ratio(x):
    """Robust dynamic range normalized by median level."""
    x = finite_signal(x)
    if len(x) < 4:
        return 0.0
    p95, p5 = np.percentile(x, [95, 5])
    med = float(np.median(x))
    return guarded_ratio(float(p95 - p5), abs(med), scale=x)


def segment_acdc_cv(raw, n_segments=3):
    """CV of segment AC/DC values (window divided into n_segments equal parts)."""
    raw = finite_signal(raw)
    if len(raw) < n_segments * 4:
        return 0.0
    seg_len = max(1, len(raw) // n_segments)
    vals = []
    for i in range(n_segments):
        seg = raw[i * seg_len:(i + 1) * seg_len] if i < n_segments - 1 else raw[i * seg_len:]
        if len(seg) < 4:
            continue
        dc = float(np.median(seg))
        ac = robust_mad(np.diff(seg)) if len(seg) > 1 else 0.0
        vals.append(guarded_ratio(ac, abs(dc), scale=seg))
    if len(vals) < 2:
        return 0.0
    vals = np.asarray(vals, dtype=np.float64)
    return guarded_ratio(float(np.std(vals)), abs(float(np.mean(vals))), scale=vals)


def segment_acdc_values(raw, n_segments=3):
    """Return segment AC/DC values using the same primitive as segment_acdc_cv."""
    raw = finite_signal(raw)
    if len(raw) < n_segments * 4:
        return []
    seg_len = max(1, len(raw) // n_segments)
    vals = []
    for i in range(n_segments):
        seg = raw[i * seg_len:(i + 1) * seg_len] if i < n_segments - 1 else raw[i * seg_len:]
        if len(seg) < 4:
            continue
        dc = float(np.median(seg))
        ac = robust_mad(np.diff(seg)) if len(seg) > 1 else 0.0
        vals.append(guarded_ratio(ac, abs(dc), scale=seg))
    return vals


def zero_cross_rate(x):
    x = finite_signal(x)
    if len(x) < 2:
        return 0.0
    centered = x - np.median(x)
    signs = np.sign(centered)
    return float(np.mean(signs[1:] * signs[:-1] < 0))


def bp_shape_features(bp, prefix):
    bp = finite_signal(bp)
    feat = OrderedDict()
    if len(bp) < 3:
        feat[f"{prefix}_bp_skewness"] = 0.0
        feat[f"{prefix}_bp_kurtosis"] = 0.0
        feat[f"{prefix}_zero_cross_rate"] = 0.0
        feat[f"{prefix}_abs_diff_ratio"] = 0.0
        return feat
    mu = float(np.mean(bp))
    sigma = float(np.std(bp))
    if sigma > EPS:
        z = (bp - mu) / sigma
        feat[f"{prefix}_bp_skewness"] = float(np.mean(z ** 3))
        feat[f"{prefix}_bp_kurtosis"] = float(np.mean(z ** 4))
    else:
        feat[f"{prefix}_bp_skewness"] = 0.0
        feat[f"{prefix}_bp_kurtosis"] = 0.0
    feat[f"{prefix}_zero_cross_rate"] = zero_cross_rate(bp)
    feat[f"{prefix}_abs_diff_ratio"] = safe_div(
        float(np.mean(np.abs(np.diff(bp)))),
        float(np.mean(np.abs(bp - np.median(bp)))) + EPS,
    )
    return feat


def segment_stability_features(raw, prefix):
    feat = OrderedDict()
    half_vals = segment_acdc_values(raw, n_segments=2)
    third_vals = segment_acdc_values(raw, n_segments=3)
    if len(half_vals) == 2:
        feat[f"{prefix}_HALF_ACDC_DELTA"] = guarded_ratio(
            abs(float(half_vals[0]) - float(half_vals[1])),
            abs(float(np.mean(half_vals))),
            scale=half_vals,
        )
    else:
        feat[f"{prefix}_HALF_ACDC_DELTA"] = 0.0
    if len(third_vals) >= 2:
        feat[f"{prefix}_SEG_ACDC_RANGE"] = guarded_ratio(
            float(np.max(third_vals) - np.min(third_vals)),
            abs(float(np.mean(third_vals))),
            scale=third_vals,
        )
    else:
        feat[f"{prefix}_SEG_ACDC_RANGE"] = 0.0
    return feat


def ambient_green_leak_stability(amb_raw, green_raw):
    amb_vals = segment_acdc_values(amb_raw, n_segments=3)
    green_vals = segment_acdc_values(green_raw, n_segments=3)
    n = min(len(amb_vals), len(green_vals))
    if n < 2:
        return 0.0
    ratios = [
        guarded_ratio(amb_vals[i], green_vals[i], scale=green_vals)
        for i in range(n)
    ]
    return guarded_ratio(float(np.std(ratios)), abs(float(np.mean(ratios))), scale=ratios)


def segment_corr_range(x, y, n_segments=3):
    x = finite_signal(x)
    y = finite_signal(y)
    n = min(len(x), len(y))
    if n < n_segments * 4:
        return 0.0
    x = x[:n]
    y = y[:n]
    seg_len = max(1, n // n_segments)
    vals = []
    for i in range(n_segments):
        xs = x[i * seg_len:(i + 1) * seg_len] if i < n_segments - 1 else x[i * seg_len:]
        ys = y[i * seg_len:(i + 1) * seg_len] if i < n_segments - 1 else y[i * seg_len:]
        if len(xs) >= 4 and len(ys) >= 4:
            vals.append(safe_corr(xs, ys))
    if len(vals) < 2:
        return 0.0
    return float(np.max(vals) - np.min(vals))


def band_energy_ratio_from_fft_cache(fft_cache, low=0.7, high=3.0):
    """Physiological-band energy ratio using the cached short-window FFT."""
    spec = fft_cache.get("spec")
    freqs = fft_cache.get("freqs")
    if spec is None or freqs is None or len(spec) == 0:
        return 0.0
    spec = finite_signal(spec)
    freqs = finite_signal(freqs)
    total_mask = (freqs >= 0.5) & (freqs <= 5.0)
    band_mask = (freqs >= low) & (freqs <= high)
    total = float(np.sum(spec[total_mask] ** 2))
    band = float(np.sum(spec[band_mask] ** 2))
    return guarded_ratio(band, total, scale=spec)


def _diff_flat_ratio(x):
    x = finite_signal(x)
    if len(x) < 2:
        return 1.0
    d = np.abs(np.diff(x))
    scale = max(abs(float(np.median(x))), 1.0)
    tol = max(scale * 1e-7, 1e-6)
    return float(np.mean(d <= tol))


def _diff_spike_ratio(x, k=6.0):
    x = finite_signal(x)
    if len(x) < 3:
        return 0.0
    d = np.abs(np.diff(x))
    mad = robust_mad(d)
    if mad <= EPS:
        # A flat or constant-slope trace has zero derivative MAD.  Retain the
        # no-spike result for a constant slope, but still expose isolated
        # excursions around that slope (for example, a single ADC glitch).
        baseline = float(np.median(d))
        return float(np.mean(np.abs(d - baseline) > _scale_floor(x)))
    return float(np.mean(d > k * mad))


def short_window_sqi_features(ir_raw_in, amb_raw_in, g_raw_in):
    """Fast Stage2 SQI features from ambient and green only.

    IR is intentionally ignored here so XGBoost does not
    carry hidden IR information through generic SQI feature names.
    """
    channels = [
        finite_signal(amb_raw_in),
        finite_signal(g_raw_in),
    ]
    return OrderedDict([
        ("SQI_FLAT_RATIO", float(np.mean([_diff_flat_ratio(x) for x in channels]))),
        ("SQI_SPIKE_RATIO", float(np.mean([_diff_spike_ratio(x) for x in channels]))),
    ])

# =========================================================
# 单通道特征
# =========================================================

def extract_single_channel_features(raw, bp, dc, fs, prefix, fft_cache=None):
    """
    单通道特征提取
    
    参数：
        fft_cache: 可选的FFT缓存字典，避免重复计算
    """
    feat = OrderedDict()

    raw = np.asarray(raw, dtype=np.float64)
    bp = np.asarray(bp, dtype=np.float64)

    if len(raw) == 0 or len(bp) == 0:
        return feat

    ac_rms = float(np.sqrt(np.mean(bp ** 2)))
    ac_mad = robust_mad(bp)
    dc_iqr = robust_iqr(raw)

    deriv = np.diff(bp) if len(bp) > 1 else np.array([0.0])
    deriv_mad = robust_mad(deriv)

    # 使用缓存的FFT结果，避免重复计算
    if fft_cache is not None:
        fft_peak_ratio = fft_cache.get('peak_ratio', 0.0)
        dom_freq = fft_cache.get('dom_freq', 0.0)
    else:
        fft_peak_ratio, dom_freq = fft_peak_features(bp, fs, fmin=0.5, fmax=5.0)
    
    ac_peak, ac_lag_sec = autocorr_periodicity_features(bp, fs, bpm_min=40.0, bpm_max=180.0)

    feat[f"{prefix}_DC_MEDIAN"] = float(dc)
    feat[f"{prefix}_DC_IQR"] = dc_iqr
    feat[f"{prefix}_AC_RMS"] = ac_rms
    feat[f"{prefix}_AC_MAD"] = ac_mad
    feat[f"{prefix}_AC_DC_RATIO"] = safe_div(ac_rms, abs(dc) + EPS)
    feat[f"{prefix}_DERIV_MAD"] = deriv_mad
    feat[f"{prefix}_FFT_PEAK_MEDIAN_RATIO"] = fft_peak_ratio
    feat[f"{prefix}_DOM_FREQ"] = dom_freq
    feat[f"{prefix}_AUTO_CORR_PEAK"] = ac_peak
    feat[f"{prefix}_AUTO_CORR_LAG_SEC"] = ac_lag_sec

    return feat

# =========================================================
# 三通道绿光空间特征
# =========================================================

def extract_green_spatial_features(g1_raw, g2_raw, g3_raw, g1_bp, g2_bp, g3_bp,
                                   g1_input=None, g2_input=None, g3_input=None):
    feat = OrderedDict()
    eps = 1e-8

    g_stack = np.vstack([g1_raw, g2_raw, g3_raw])
    # ---------- 空间不均衡 ----------
    g_spatial_std = np.std(g_stack, axis=0)
    g_spatial_mean = np.mean(g_stack, axis=0)

    g_imbalance = g_spatial_std / (np.abs(g_spatial_mean) + eps)

    feat["G_imbalance_mean"] = float(np.mean(g_imbalance))
    feat["G_imbalance_p90"] = float(np.percentile(g_imbalance, 90))
    feat["G_imbalance_iqr"] = robust_iqr(g_imbalance)

    # ---------- 归一化极差 ----------
    g_max = np.max(g_stack, axis=0)
    g_min = np.min(g_stack, axis=0)

    g_range_norm = (g_max - g_min) / (
        np.abs(g1_raw) + np.abs(g2_raw) + np.abs(g3_raw) + eps
    )

    feat["G_rangeNorm_mean"] = float(np.mean(g_range_norm))
    feat["G_rangeNorm_p90"] = float(np.percentile(g_range_norm, 90))

    # ---------- 中心对称空间向量 ----------
    vx = g1_raw - 0.5 * g2_raw - 0.5 * g3_raw
    vy = (np.sqrt(3) / 2.0) * (g2_raw - g3_raw)

    vmag = np.sqrt(vx ** 2 + vy ** 2) / (
        np.abs(g1_raw) + np.abs(g2_raw) + np.abs(g3_raw) + eps
    )

    feat["G_spatial_vmag_mean"] = float(np.mean(vmag))
    feat["G_spatial_vmag_p90"] = float(np.percentile(vmag, 90))
    feat["G_spatial_vmag_iqr"] = robust_iqr(vmag)
    feat["G_spatial_vmag_std"] = float(np.std(vmag))
    feat["G_SPATIAL_VMAG_RANGE"] = float(np.percentile(vmag, 90) - np.percentile(vmag, 10))

    # ---------- 三通道 DC 相对强弱 ----------
    ch_dc = np.array([
        np.median(g1_raw),
        np.median(g2_raw),
        np.median(g3_raw)
    ], dtype=np.float64)

    feat["G_ch_dc_cv"] = float(np.std(ch_dc) / (np.abs(np.mean(ch_dc)) + eps))
    feat["G_ch_dc_max_min_ratio"] = float(
        np.max(np.abs(ch_dc)) / (np.min(np.abs(ch_dc)) + eps)
    )

    # ---------- 三通道 BP 一致性 ----------
    c12 = safe_corr(g1_bp, g2_bp)
    c23 = safe_corr(g2_bp, g3_bp)
    c31 = safe_corr(g3_bp, g1_bp)

    feat["G_bp_corr_mean"] = float(np.mean([c12, c23, c31]))
    feat["G_bp_corr_min"] = float(np.min([c12, c23, c31]))
    feat["G_bp_corr_std"] = float(np.std([c12, c23, c31]))

    # ---------- 三绿光可靠性 ----------
    # These scalar features keep the 120-degree green-channel geometry deployable:
    # no voting state, just compact reliability cues for XGBoost.
    ch_bp = [g1_bp, g2_bp, g3_bp]
    ch_raw = [
        g1_raw if g1_input is None else np.asarray(g1_input, dtype=np.float64),
        g2_raw if g2_input is None else np.asarray(g2_input, dtype=np.float64),
        g3_raw if g3_input is None else np.asarray(g3_input, dtype=np.float64),
    ]
    ch_ac = np.array([
        float(np.sqrt(np.mean((x - np.median(x)) ** 2))) for x in ch_raw
    ], dtype=np.float64)
    max_ac = float(np.max(ch_ac)) if len(ch_ac) else 0.0
    total_ac = float(np.sum(ch_ac))
    if max_ac <= eps:
        feat["G_2OF3_AC_SUPPORT"] = 0.0
        feat["G_TOP2_TO_ALL_AC_RATIO"] = 0.0
        feat["G_TOP2_CORR_MIN"] = 0.0
        feat["G_WEAK_CHANNEL_GAP"] = 0.0
        feat["G_SPATIAL_STABILITY_SCORE"] = 0.0
        feat["G_TOP1_TO_TOP2_AC_RATIO"] = 0.0
        feat["G_TOP2_RANK_STABILITY"] = 0.0
        feat["G_TOP2_SWITCH_RATE"] = 0.0
    else:
        support_count = int(np.sum(ch_ac >= 0.5 * max_ac))
        top2_idx = np.argsort(ch_ac)[-2:]
        top2_ac = ch_ac[top2_idx]
        top2_corr = safe_corr(ch_bp[int(top2_idx[0])], ch_bp[int(top2_idx[1])])
        top2_mean_ac = float(np.mean(top2_ac))
        weak_ac = float(np.min(ch_ac))
        weak_gap = safe_div(top2_mean_ac - weak_ac, top2_mean_ac + eps)
        corr_quality = max(0.0, float(top2_corr))
        spatial_penalty = 1.0 / (1.0 + float(np.mean(vmag)))

        feat["G_2OF3_AC_SUPPORT"] = float(support_count / 3.0)
        feat["G_TOP2_TO_ALL_AC_RATIO"] = safe_div(float(np.sum(top2_ac)), total_ac + eps)
        feat["G_TOP2_CORR_MIN"] = float(top2_corr)
        feat["G_WEAK_CHANNEL_GAP"] = float(max(0.0, weak_gap))
        feat["G_SPATIAL_STABILITY_SCORE"] = float(
            feat["G_2OF3_AC_SUPPORT"] * corr_quality * spatial_penalty
        )
        feat["G_TOP1_TO_TOP2_AC_RATIO"] = safe_div(max_ac, top2_mean_ac + eps)
        global_top2 = set(int(i) for i in top2_idx)
        n_seg = 3
        n = min(len(ch_raw[0]), len(ch_raw[1]), len(ch_raw[2]))
        seg_len = max(1, n // n_seg)
        switches = []
        for i in range(n_seg):
            lo = i * seg_len
            hi = (i + 1) * seg_len if i < n_seg - 1 else n
            if hi - lo < 4:
                continue
            seg_ac = np.array([
                float(np.sqrt(np.mean((np.asarray(ch_raw[j][lo:hi]) - np.median(ch_raw[j][lo:hi])) ** 2)))
                for j in range(3)
            ], dtype=np.float64)
            seg_top2 = set(int(j) for j in np.argsort(seg_ac)[-2:])
            switches.append(0.0 if seg_top2 == global_top2 else 1.0)
        switch_rate = float(np.mean(switches)) if switches else 0.0
        feat["G_TOP2_SWITCH_RATE"] = switch_rate
        feat["G_TOP2_RANK_STABILITY"] = float(1.0 - switch_rate)

    return feat, g_imbalance, vmag

# =========================================================
# 通道间特征
# =========================================================

def extract_cross_channel_features(g_raw, g_bp, g_dc,
                                   ir_raw, ir_bp, ir_dc,
                                   amb_raw, amb_bp,
                                   fs,
                                   fft_cache_green=None,
                                   fft_cache_ir=None):
    """
    通道间特征提取
    
    参数：
        fft_cache_green: 绿光FFT缓存
        fft_cache_ir: IR FFT缓存
    """
    feat = OrderedDict()

    g_env = smooth_envelope(g_bp, fs)
    amb_env = smooth_envelope(amb_bp, fs)

    # 使用缓存的FFT结果，避免重复计算
    if fft_cache_green is not None:
        g_dom = fft_cache_green.get('dom_freq', 0.0)
    else:
        _, g_dom = fft_peak_features(g_bp, fs, fmin=0.5, fmax=5.0)
    
    feat["GREEN_AMB_BP_CORR"] = safe_corr(g_bp, amb_bp)
    feat["GREEN_AMB_ENV_CORR"] = safe_corr(g_env, amb_env)

    g_rms = np.sqrt(np.mean(g_bp ** 2)) + EPS
    amb_rms = np.sqrt(np.mean(amb_bp ** 2)) + EPS

    feat["GREEN_AMB_LEAK"] = abs(feat["GREEN_AMB_BP_CORR"]) * safe_div(amb_rms, g_rms)

    return feat

# =========================================================
# ACC 特征（轻量版）
# =========================================================

def _acc_magnitude(acc_window):
    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    return np.sqrt(np.sum(acc * acc, axis=1) + 1e-12)


def extract_acc_features(acc_window, fs=100.0, prefix="ACC"):
    feats = OrderedDict()

    _all_keys = ["MAG_MEAN", "MAG_STD", "MAG_MAD", "AXIS_STD_SUM",
                 "GRAVITY_DOM_RATIO", "BP_RMS", "DIFF_MAD", "STILL_SCORE",
                 "MAG_P50", "MAG_P90", "YSUM",
                 "X_MEAN", "X_STD", "X_ENERGY",
                 "Y_MEAN", "Y_STD", "Y_ENERGY",
                 "Z_MEAN", "Z_STD", "Z_ENERGY",
                 "AXIS_MEAN_SUM", "MAG_ENERGY", "MAG_P2P",
                 "TILT_ANGLE", "DOM_AXIS", "GRAVITY_RATIO"]

    if acc_window is None or len(acc_window) < 4:
        for k in _all_keys:
            feats[f"{prefix}_{k}"] = 0.0
        return feats

    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)

    mag = _acc_magnitude(acc)
    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    mag_mad = robust_mad(mag)
    mag_energy = float(np.sum(mag ** 2))
    mag_p2p = float(np.max(mag) - np.min(mag))

    axis_std = np.std(acc, axis=0)
    axis_std_sum = float(np.sum(axis_std))

    axis_mean_abs = np.abs(np.mean(acc, axis=0))
    dom_axis_ratio = float(np.max(axis_mean_abs) / (np.sum(axis_mean_abs) + 1e-8))

    # Per-axis features (Tier 1): mean, std, energy
    n_axes = acc.shape[1]
    axis_labels = ["X", "Y", "Z"]
    axis_means = np.mean(acc, axis=0)
    axis_energies = np.sum(acc ** 2, axis=0)
    for i in range(min(n_axes, 3)):
        lbl = axis_labels[i]
        feats[f"{prefix}_{lbl}_MEAN"] = float(axis_means[i])
        feats[f"{prefix}_{lbl}_STD"] = float(axis_std[i])
        feats[f"{prefix}_{lbl}_ENERGY"] = float(axis_energies[i])
    for i in range(n_axes, 3):
        lbl = axis_labels[i]
        feats[f"{prefix}_{lbl}_MEAN"] = 0.0
        feats[f"{prefix}_{lbl}_STD"] = 0.0
        feats[f"{prefix}_{lbl}_ENERGY"] = 0.0
    feats[f"{prefix}_AXIS_MEAN_SUM"] = float(np.sum(np.abs(axis_means)))

    # ACC orientation (Tier 2): tilt angle from gravity vector
    # When worn, one axis typically points down (gravity); off-wrist orientation is random.
    _grav = axis_means.copy()
    _grav_norm = float(np.sqrt(np.sum(_grav ** 2)))
    if _grav_norm > 1e-8:
        _grav_unit = _grav / _grav_norm
        # Tilt angle: angle between gravity vector and vertical (Z-up assumed)
        # cos(theta) = |gz| / |g|, theta=0 means device flat, theta=90 means vertical
        feats[f"{prefix}_TILT_ANGLE"] = float(np.degrees(np.arccos(np.clip(np.abs(_grav_unit[2]) if n_axes >= 3 else 0.0, 0.0, 1.0))))
        # Dominant axis index (0=X, 1=Y, 2=Z)
        feats[f"{prefix}_DOM_AXIS"] = float(np.argmax(np.abs(_grav_unit)))
        # Gravity concentration: how much of total ACC energy is in the gravity component
        feats[f"{prefix}_GRAVITY_RATIO"] = float(_grav_norm / (mag_mean + 1e-8))
    else:
        feats[f"{prefix}_TILT_ANGLE"] = 0.0
        feats[f"{prefix}_DOM_AXIS"] = 0.0
        feats[f"{prefix}_GRAVITY_RATIO"] = 0.0

    mag_centered = mag - np.mean(mag)
    try:
        mag_bp = bandpass_filter(mag_centered, fs, lowcut=0.5, highcut=5.0, order=2)
    except Exception:
        mag_bp = mag_centered
    bp_rms = float(np.sqrt(np.mean(mag_bp ** 2)))

    diff_mad = robust_mad(np.diff(mag)) if len(mag) > 1 else 0.0

    rel_std = mag_std / (abs(mag_mean) + 1e-6)
    still_score = float(1.0 / (1.0 + 50.0 * rel_std))

    feats[f"{prefix}_MAG_MEAN"] = mag_mean
    feats[f"{prefix}_YSUM"] = mag_mean
    feats[f"{prefix}_MAG_STD"] = mag_std
    feats[f"{prefix}_MAG_MAD"] = mag_mad
    feats[f"{prefix}_MAG_ENERGY"] = mag_energy
    feats[f"{prefix}_MAG_P2P"] = mag_p2p
    feats[f"{prefix}_AXIS_STD_SUM"] = axis_std_sum
    feats[f"{prefix}_GRAVITY_DOM_RATIO"] = dom_axis_ratio
    feats[f"{prefix}_BP_RMS"] = bp_rms
    feats[f"{prefix}_DIFF_MAD"] = diff_mad
    feats[f"{prefix}_STILL_SCORE"] = still_score
    feats[f"{prefix}_MAG_P50"] = float(np.percentile(mag, 50))
    feats[f"{prefix}_MAG_P90"] = float(np.percentile(mag, 90))

    return feats


def extract_acc_ppg_cross_features(acc_window, green_bp, ir_bp=None, fs=100.0):
    feats = OrderedDict()

    if acc_window is None or len(acc_window) < 4:
        feats["ACC_GREEN_BP_CORR"] = 0.0
        return feats

    mag = _acc_magnitude(acc_window)
    mag_centered = mag - np.mean(mag)

    try:
        mag_bp = bandpass_filter(mag_centered, fs, lowcut=0.5, highcut=5.0, order=2)
    except Exception:
        mag_bp = mag_centered

    n = min(len(mag_bp), len(green_bp))
    if n < 8:
        feats["ACC_GREEN_BP_CORR"] = 0.0
        return feats

    feats["ACC_GREEN_BP_CORR"] = abs(safe_corr(mag_bp[:n], green_bp[:n]))

    return feats


def extract_acc_green_coupling_features(acc_window, green_raw, green_bp):
    feats = OrderedDict()
    keys = [
        "ACC_TO_GTOP2_AC_RATIO",
        "ACC_STILL_X_GREEN_STABILITY",
        "ACC_DIFF_TO_GTOP2_DIFF_RATIO",
        "ACC_STILL_GREEN_MISMATCH",
    ]
    if acc_window is None or green_raw is None or green_bp is None or len(acc_window) < 4:
        for k in keys:
            feats[k] = 0.0
        return feats
    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    mag = _acc_magnitude(acc)
    green_raw = finite_signal(green_raw)
    green_bp = finite_signal(green_bp)
    n = min(len(mag), len(green_raw), len(green_bp))
    if n < 4:
        for k in keys:
            feats[k] = 0.0
        return feats
    mag = mag[:n]
    green_raw = green_raw[:n]
    green_bp = green_bp[:n]
    acc_diff_mad = robust_mad(np.diff(mag)) if len(mag) > 1 else 0.0
    green_diff_mad = robust_mad(np.diff(green_raw)) if len(green_raw) > 1 else 0.0
    green_ac = float(np.sqrt(np.mean(green_bp ** 2)))
    green_stability = 1.0 / (1.0 + segment_acdc_cv(green_raw))
    still_score = 1.0 / (1.0 + float(np.std(mag)) + acc_diff_mad)
    feats["ACC_TO_GTOP2_AC_RATIO"] = safe_div(acc_diff_mad, green_ac + EPS)
    feats["ACC_STILL_X_GREEN_STABILITY"] = float(still_score * green_stability)
    feats["ACC_DIFF_TO_GTOP2_DIFF_RATIO"] = safe_div(acc_diff_mad, green_diff_mad + EPS)
    feats["ACC_STILL_GREEN_MISMATCH"] = float(still_score * safe_div(green_ac, acc_diff_mad + EPS))
    return feats


# =========================================================
# Hjorth 参数特征
# =========================================================

def extract_hjorth_parameters(x, prefix=""):
    """
    计算 Hjorth 三参数: Activity, Mobility, Complexity

    Activity: 信号方差（功率）
    Mobility: 一阶导数标准差与信号标准差的比值
    Complexity: 二阶导数 mobility 与 一阶导数 mobility 的比值

    参数:
        x: 输入信号 (numpy array)
        prefix: 通道前缀 (如 "GREEN", "IRX", "AMBX")

    返回:
        OrderedDict: {'{prefix}_Hjorth_Activity': ..., ...}
    """
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""

    if len(x) < 4:
        feat[f"{pf}Hjorth_Activity"] = 0.0
        feat[f"{pf}Hjorth_Mobility"] = 0.0
        feat[f"{pf}Hjorth_Complexity"] = 0.0
        return feat

    # Activity = 方差
    activity = float(np.var(x))
    feat[f"{pf}Hjorth_Activity"] = activity

    # 一阶导数
    d1 = np.diff(x)
    if len(d1) < 2:
        feat[f"{pf}Hjorth_Mobility"] = 0.0
        feat[f"{pf}Hjorth_Complexity"] = 0.0
        return feat

    # Mobility = sqrt(var(d1) / var(x))
    var_d1 = np.var(d1)
    if activity > EPS:
        mobility = float(np.sqrt(var_d1 / activity))
    else:
        mobility = 0.0
    feat[f"{pf}Hjorth_Mobility"] = mobility
    
    # 二阶导数 -> Complexity
    d2 = np.diff(d1)
    if len(d2) < 2:
        feat[f"{pf}Hjorth_Complexity"] = 0.0
        return feat
    
    var_d2 = np.var(d2)
    if var_d1 > EPS:
        complexity = float(np.sqrt(var_d2 / var_d1))
    else:
        complexity = 0.0
    feat[f"{pf}Hjorth_Complexity"] = complexity

    return feat


# =========================================================
# 熵特征
# =========================================================

def extract_entropy_features(x, r_std_ratio=0.2, m=2, prefix="", skip_apen=True):
    """
    计算熵特征: Shannon熵, 近似熵(ApEn), 样本熵(SampEn)

    参数:
        x: 输入信号 (numpy array)
        r_std_ratio: ApEn/SampEn 的阈值参数（std的倍数），默认0.2
        m: 嵌入维度，默认2
        prefix: 通道前缀 (如 "GREEN", "IRX", "AMBX")

    返回:
        OrderedDict: {'{prefix}_Entropy_Shannon': ..., ...}
    """
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""

    if len(x) < 10:
        feat[f"{pf}Entropy_Shannon"] = 0.0
        feat[f"{pf}Entropy_ApEn"] = 0.0
        feat[f"{pf}Entropy_SampEn"] = 0.0
        return feat
    
    # ---------- Shannon 熵 ----------
    try:
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        shannon_entropy = float(-np.sum(hist * np.log(hist + EPS)))
    except Exception:
        shannon_entropy = 0.0
    feat[f"{pf}Entropy_Shannon"] = shannon_entropy
    # ---------- ApEn (O(N^2)) — skip_apen=True 跳过 ----------
    if not skip_apen:
        try:
            N = len(x)
            r = r_std_ratio * np.std(x)
            if r < EPS:
                apen = 0.0
            else:
                def phi(m_val):
                    patterns = np.array([x[i:i + m_val] for i in range(N - m_val)])
                    if len(patterns) == 0:
                        return 0.0
                    distances = np.max(np.abs(patterns[:, np.newaxis, :] - patterns[np.newaxis, :, :]), axis=2)
                    count = np.sum(distances <= r, axis=1)
                    count = count / (N - m_val)
                    count = count[count > 0]
                    if len(count) == 0:
                        return 0.0
                    return np.mean(np.log(count + EPS))
                phi_m = phi(m)
                phi_m1 = phi(m + 1)
                apen = float(phi_m - phi_m1)
                if not np.isfinite(apen):
                    apen = 0.0
        except Exception:
            apen = 0.0
        feat[f"{pf}Entropy_ApEn"] = apen

    
    # ---------- 样本熵 (SampEn) ----------
    # SampEn(m, r) = -ln( A / B )
    #   B = #{ pairs (i,j), i!=j, max|x[i:i+m]-x[j:j+m]| <= r }
    #   A = #{ pairs (i,j), i!=j, max|x[i:i+m+1]-x[j:j+m+1]| <= r }
    # 部署端 s07 _sample_entropy 与此实现保持一致。
    try:
        N = len(x)
        r = r_std_ratio * np.std(x)
        if r < EPS:
            sampen = 0.0
        else:
            def sampen_count(m_val):
                patterns = np.array([x[i:i + m_val] for i in range(N - m_val)])
                if len(patterns) < 2:
                    return 0.0
                distances = np.max(np.abs(patterns[:, np.newaxis, :] - patterns[np.newaxis, :, :]), axis=2)
                np.fill_diagonal(distances, np.inf)
                return float(np.sum(distances <= r))

            B_m = sampen_count(m)         # 长度 m 的匹配对数
            B_m1 = sampen_count(m + 1)    # 长度 m+1 的匹配对数

            if B_m > 0 and B_m1 > 0:
                sampen = float(-np.log(B_m1 / (B_m + EPS)))
            else:
                sampen = 0.0
            if not np.isfinite(sampen):
                sampen = 0.0
    except Exception:
        sampen = 0.0
    feat[f"{pf}Entropy_SampEn"] = sampen
    
    return feat


# =========================================================
# 导数特征
# =========================================================

def extract_derivative_features(x, fs=100.0, prefix="", compute_d2=False):
    """
    计算导数特征: 一阶/二阶导数统计, 零交叉率

    参数:
        x: 输入信号 (numpy array)
        fs: 采样率，默认100Hz
        prefix: 通道前缀 (如 "GREEN", "IRX", "AMBX")

    返回:
        OrderedDict: 包含一阶/二阶导数的均值、标准差、零交叉率等
    """
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""
    
    if len(x) < 4:
        feat[f"{pf}Deriv_d1_mean"] = 0.0
        feat[f"{pf}Deriv_d1_std"] = 0.0
        feat[f"{pf}Deriv_d1_max"] = 0.0
        feat[f"{pf}Deriv_d1_min"] = 0.0
        feat[f"{pf}Deriv_d1_zcr"] = 0.0
        feat[f"{pf}Deriv_d2_mean"] = 0.0
        feat[f"{pf}Deriv_d2_std"] = 0.0
        feat[f"{pf}Deriv_d2_max"] = 0.0
        feat[f"{pf}Deriv_d2_min"] = 0.0
        feat[f"{pf}Deriv_d2_zcr"] = 0.0
        return feat
    
    # 一阶导数
    d1 = np.diff(x)
    if len(d1) > 0:
        feat[f"{pf}Deriv_d1_mean"] = float(np.mean(d1))
        feat[f"{pf}Deriv_d1_std"] = float(np.std(d1))
        feat[f"{pf}Deriv_d1_max"] = float(np.max(d1))
        feat[f"{pf}Deriv_d1_min"] = float(np.min(d1))
        zcr_d1 = np.sum(np.abs(np.diff(np.sign(d1)))) / (2.0 * len(d1))
        feat[f"{pf}Deriv_d1_zcr"] = float(zcr_d1)
    else:
        feat[f"{pf}Deriv_d1_mean"] = 0.0
        feat[f"{pf}Deriv_d1_std"] = 0.0
        feat[f"{pf}Deriv_d1_max"] = 0.0
        feat[f"{pf}Deriv_d1_min"] = 0.0
        feat[f"{pf}Deriv_d1_zcr"] = 0.0
    
    # 二阶导数 (compute_d2=False 跳过)
    if compute_d2:
        d2 = np.diff(d1) if len(d1) > 1 else np.array([])
        if len(d2) > 0:
            feat[f"{pf}Deriv_d2_mean"] = float(np.mean(d2))
            feat[f"{pf}Deriv_d2_std"] = float(np.std(d2))
            feat[f"{pf}Deriv_d2_max"] = float(np.max(d2))
            feat[f"{pf}Deriv_d2_min"] = float(np.min(d2))
            zcr_d2 = np.sum(np.abs(np.diff(np.sign(d2)))) / (2.0 * len(d2))
            feat[f"{pf}Deriv_d2_zcr"] = float(zcr_d2)
        else:
            feat[f"{pf}Deriv_d2_mean"] = 0.0
            feat[f"{pf}Deriv_d2_std"] = 0.0
            feat[f"{pf}Deriv_d2_max"] = 0.0
            feat[f"{pf}Deriv_d2_min"] = 0.0
            feat[f"{pf}Deriv_d2_zcr"] = 0.0
    
    return feat


# =========================================================
# 时序动态特征
# =========================================================

def extract_temporal_dynamic_features(x, fs=100.0, prefix=""):
    """
    计算时序动态特征: 信号斜率, 峰值突出度
    
    参数:
        x: 输入信号 (numpy array)
        fs: 采样率，默认100Hz
    
    返回:
        OrderedDict: 包含斜率、峰值突出度等特征
    """
    feat = OrderedDict()
    x = np.asarray(x, dtype=np.float64)
    pf = f"{prefix}_" if prefix else ""
    
    if len(x) < 4:
        feat[f"{pf}Temporal_slope_mean"] = 0.0
        feat[f"{pf}Temporal_slope_std"] = 0.0
        feat[f"{pf}Temporal_peak_prominence"] = 0.0
        feat[f"{pf}Temporal_peak_ratio"] = 0.0
        feat[f"{pf}Temporal_valley_ratio"] = 0.0
        return feat
    
    # ---------- 信号斜率 (基于线性回归) ----------
    t = np.arange(len(x))
    t_mean = np.mean(t)
    x_mean = np.mean(x)
    
    slope_numerator = np.sum((t - t_mean) * (x - x_mean))
    slope_denominator = np.sum((t - t_mean) ** 2)
    
    if slope_denominator > EPS:
        slope = slope_numerator / slope_denominator
    else:
        slope = 0.0
    
    fitted = x_mean + slope * (t - t_mean)
    residuals = x - fitted
    slope_std = float(np.std(residuals))
    
    feat[f"{pf}Temporal_slope_mean"] = float(slope)
    feat[f"{pf}Temporal_slope_std"] = slope_std
    
    # Keep this legacy helper numpy-only; the deployment-friendly pool no longer
    # calls temporal peak features, but standalone use should not require scipy.
    if len(x) >= 3:
        peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
        valleys = np.where((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))[0] + 1
        feat[f"{pf}Temporal_peak_ratio"] = float(len(peaks) / len(x))
        feat[f"{pf}Temporal_valley_ratio"] = float(len(valleys) / len(x))
        if len(peaks) > 0:
            local_base = np.maximum(x[peaks - 1], x[peaks + 1])
            feat[f"{pf}Temporal_peak_prominence"] = float(np.mean(np.maximum(0.0, x[peaks] - local_base)))
        else:
            feat[f"{pf}Temporal_peak_prominence"] = 0.0
    else:
        feat[f"{pf}Temporal_peak_prominence"] = 0.0
        feat[f"{pf}Temporal_peak_ratio"] = 0.0
        feat[f"{pf}Temporal_valley_ratio"] = 0.0
    
    return feat


def align_acc_window(acc, ppg_len, start_ppg, win_ppg, fs_ppg=100.0, fs_acc=None):
    """
    按时间比例对齐ACC窗口。
    ACC采样率默认与PPG相同（100Hz）。
    """
    if acc is None or len(acc) == 0:
        return None

    if fs_acc is None:
        fs_acc = fs_ppg

    acc_per_ppg = fs_acc / fs_ppg
    start_acc = int(start_ppg * acc_per_ppg)
    end_acc = start_acc + int(win_ppg * acc_per_ppg)

    if start_acc >= len(acc):
        return None
    return acc[start_acc:min(end_acc, len(acc))]


# =========================================================
# 单窗口特征提取主函数（保留原有接口，内部被extract_window_features调用）
# =========================================================

def _extract_legacy_feature_pool_from_window(ir, ambient, g1, g2, g3, fs=25, return_preprocessed=False):
    """
    输入：
        ir, ambient, g1, g2, g3: 5秒窗口信号 (默认已降采样至 25Hz)
        fs: 采样率 (默认 25)
        return_preprocessed: 是否返回预处理结果（供复用）

    输出：
        如果 return_preprocessed=False: OrderedDict 特征
        如果 return_preprocessed=True: (OrderedDict 特征, dict 预处理结果)
        
    优化：
        - 使用FFT缓存避免重复计算
        - 一次性计算所有需要的FFT结果
        - 可返回预处理结果供ACC交叉特征使用，避免重复计算
    """
    feat = OrderedDict()

    ir = np.asarray(ir, dtype=np.float64).reshape(-1)
    ambient = np.asarray(ambient, dtype=np.float64).reshape(-1)
    g1 = np.asarray(g1, dtype=np.float64).reshape(-1)
    g2 = np.asarray(g2, dtype=np.float64).reshape(-1)
    g3 = np.asarray(g3, dtype=np.float64).reshape(-1)

    n = min(len(ir), len(ambient), len(g1), len(g2), len(g3))

    if n < int(1.0 * fs):
        raise ValueError(f"窗口太短: n={n}, 需要 >= {int(fs)}")

    ir = ir[:n]
    ambient = ambient[:n]
    g1 = g1[:n]
    g2 = g2[:n]
    g3 = g3[:n]
    g1_input = g1.copy()
    g2_input = g2.copy()
    g3_input = g3.copy()
    g_mean_input = (g1 + g2 + g3) / 3.0

    # Short-window SQI is computed on the raw window before artifact removal.
    feat.update(short_window_sqi_features(ir, ambient, g_mean_input))

    # ---------- 鲁棒预处理 ----------
    ir_raw, ir_bp, ir_dc = preprocess_signal(ir, fs)
    amb_raw, amb_bp, amb_dc = preprocess_signal(ambient, fs)

    g1_raw, g1_bp, g1_dc = preprocess_signal(g1, fs)
    g2_raw, g2_bp, g2_dc = preprocess_signal(g2, fs)
    g3_raw, g3_bp, g3_dc = preprocess_signal(g3, fs)

    g_mean_raw = (g1_raw + g2_raw + g3_raw) / 3.0
    g_mean_bp = (g1_bp + g2_bp + g3_bp) / 3.0
    g_mean_dc = float(np.median(g_mean_raw))

    # =====================================================
    # ★ 优化：一次性计算所有FFT，避免重复
    # =====================================================
    fft_cache = {
        'green': compute_fft_cache(g_mean_bp, fs, fmin=0.5, fmax=5.0),
        'amb': compute_fft_cache(amb_bp, fs, fmin=0.5, fmax=5.0),
        'g1': compute_fft_cache(g1_bp, fs, fmin=0.5, fmax=5.0),
        'g2': compute_fft_cache(g2_bp, fs, fmin=0.5, fmax=5.0),
        'g3': compute_fft_cache(g3_bp, fs, fmin=0.5, fmax=5.0),
    }

    feat["GREEN_ROBUST_RANGE_RATIO"] = robust_range_ratio(g_mean_raw)
    feat["AMB_ROBUST_RANGE_RATIO"] = robust_range_ratio(amb_raw)
    feat["GREEN_SEG_ACDC_CV"] = segment_acdc_cv(g_mean_raw)
    feat["AMB_SEG_ACDC_CV"] = segment_acdc_cv(amb_raw)
    feat["GREEN_BAND_ENERGY_RATIO"] = band_energy_ratio_from_fft_cache(fft_cache['green'])
    feat["AMB_BAND_ENERGY_RATIO"] = band_energy_ratio_from_fft_cache(fft_cache['amb'])

    # =====================================================
    # 1. 基础长度
    # =====================================================
    feat["SIG_LEN"] = float(n)
    feat["SIG_SEC"] = float(n / fs)

    # =====================================================
    # 2. 与之前流程兼容的核心低维特征名
    # =====================================================

    # IR 类
    # G_mean 类
    feat["G_mean_mean"] = float(np.mean(g_mean_raw))
    feat["G_mean_std"] = float(np.std(g_mean_raw))
    feat["G_mean_diff_std"] = float(np.std(np.diff(g_mean_raw)))
    feat["G_mean_acdc"] = safe_div(np.sqrt(np.mean(g_mean_bp ** 2)), abs(g_mean_dc) + EPS)

    # Ambient statistics
    feat["Ambient_mean"] = float(np.mean(amb_raw))
    feat["Ambient_std"] = float(np.std(amb_raw))
    feat["Ambient_p95"] = float(np.percentile(amb_raw, 95))
    feat["corr_Ambient_Gmean"] = safe_corr(amb_raw, g_mean_raw)

    # =====================================================
    # 3. 单通道增强特征（使用FFT缓存）
    # =====================================================
    feat.update(extract_single_channel_features(
        g_mean_raw, g_mean_bp, g_mean_dc, fs, "GREEN", fft_cache=fft_cache['green']
    ))

    feat.update(extract_single_channel_features(
        amb_raw, amb_bp, amb_dc, fs, "AMBX", fft_cache=fft_cache['amb']
    ))

    # =====================================================
    # 3b. 波形形态: 偏度/峰度 — 真实 PPG 有特征性不对称（收缩峰尖锐、舒张缓慢）
    # =====================================================
    for _pf, _bp in [("GREEN", g_mean_bp)]:
        _m = float(np.mean(_bp))
        _s = float(np.std(_bp))
        if _s > EPS:
            feat[f"{_pf}_bp_skewness"] = float(np.mean((_bp - _m) ** 3) / (_s ** 3))
            feat[f"{_pf}_bp_kurtosis"] = float(np.mean((_bp - _m) ** 4) / (_s ** 4))
        else:
            feat[f"{_pf}_bp_skewness"] = 0.0
            feat[f"{_pf}_bp_kurtosis"] = 0.0

    # =====================================================
    # 4. 绿光三通道空间特征
    # =====================================================
    spatial_feat, g_imbalance, g_vmag = extract_green_spatial_features(
        g1_raw, g2_raw, g3_raw,
        g1_bp, g2_bp, g3_bp,
        g1_input=g1_input,
        g2_input=g2_input,
        g3_input=g3_input,
    )

    feat.update(spatial_feat)

    # =====================================================
    # 4a2. GREEN_TOP2 aggregation signal (Tier 3)
    #       Per-channel quality → select best 2 of 3 channels.
    #       TOP2 is robust to single-channel dropout (watch tilt):
    #       the worst channel is excluded, preserving signal quality.
    # =====================================================
    _ch_ac_rms = np.array([
        float(np.sqrt(np.mean(g1_bp ** 2))),
        float(np.sqrt(np.mean(g2_bp ** 2))),
        float(np.sqrt(np.mean(g3_bp ** 2))),
    ])
    _top2_idx = np.argsort(_ch_ac_rms)[-2:]  # indices of 2 highest AC channels
    _ch_bp = [g1_bp, g2_bp, g3_bp]
    _ch_raw = [g1_raw, g2_raw, g3_raw]
    g_top2_bp = np.mean([_ch_bp[i] for i in _top2_idx], axis=0)
    g_top2_raw = np.mean([_ch_raw[i] for i in _top2_idx], axis=0)
    feat["G_TOP2_CHANNEL_COUNT"] = float(len(_top2_idx))  # always 2, for consistency
    feat["G_TOP2_WORST_IDX"] = float(np.argmin(_ch_ac_rms))  # which channel was excluded

    # =====================================================
    # 4b. Per-channel G1/G2/G3 individual features (Tier 3)
    #     Extract the same lightweight features from each
    #     green channel independently. Combined with consensus
    #     statistics below, this captures directional tilt
    #     patterns without needing 3 separate models.
    # =====================================================
    _ch_data = [
        ("G1", g1_raw, g1_bp, g1_dc, fft_cache['g1']),
        ("G2", g2_raw, g2_bp, g2_dc, fft_cache['g2']),
        ("G3", g3_raw, g3_bp, g3_dc, fft_cache['g3']),
    ]
    for _pf, _raw, _bp, _dc, _fc in _ch_data:
        feat.update(extract_single_channel_features(
            _raw, _bp, _dc, fs, _pf, fft_cache=_fc
        ))

    # =====================================================
    # 4c. Cross-channel consensus statistics (Tier 3)
    #     For each base feature, compute min/max/range/cv
    #     across G1/G2/G3. The 120-degree symmetric LED layout
    #     means the *pattern* of channel deviations encodes
    #     tilt direction — e.g. G1 low + G2/G3 normal = tilted
    #     toward G1 direction (still on skin), while all 3 low
    #     = genuinely off-wrist.
    # =====================================================
    _consensus_base = [
        "DC_MEDIAN", "AC_RMS", "AC_MAD", "AC_DC_RATIO",
        "DERIV_MAD", "FFT_PEAK_MEDIAN_RATIO",
        "AUTO_CORR_PEAK",
    ]
    for _base in _consensus_base:
        _vals = []
        for _pf, _, _, _, _ in _ch_data:
            _v = feat.get(f"{_pf}_{_base}", 0.0)
            _vals.append(float(_v))
        _arr = np.array(_vals, dtype=np.float64)
        # _mean skipped: numerically ≈ GREEN_* / G_mean_* (already extracted)
        feat[f"G_consensus_{_base}_min"] = float(np.min(_arr))
        feat[f"G_consensus_{_base}_max"] = float(np.max(_arr))
        feat[f"G_consensus_{_base}_range"] = float(np.max(_arr) - np.min(_arr))
        _mean_abs = np.mean(np.abs(_arr))
        feat[f"G_consensus_{_base}_cv"] = float(np.std(_arr) / (_mean_abs + EPS))
        # top2_mean: mean of best 2 channels, robust to single-channel dropout
        _sorted = np.sort(_arr)
        feat[f"G_consensus_{_base}_top2_mean"] = float(np.mean(_sorted[-2:]))

    # =====================================================
    # 4d. Channel dropout indicators (Tier 3)
    #     Count how many channels appear "dark" and which
    #     direction the dropout vector points.
    # =====================================================
    _ch_ac = np.array([feat.get(f"{p}_AC_RMS", 0.0) for p in ["G1", "G2", "G3"]])
    _ch_ac_thr = 0.05 * np.max(_ch_ac) if np.max(_ch_ac) > EPS else EPS
    _dropout_mask = _ch_ac < _ch_ac_thr
    feat["G_DROPOUT_COUNT"] = float(np.sum(_dropout_mask))
    feat["G_MIN_CHANNEL_ID"] = float(np.argmin(_ch_ac))  # always report weakest channel
    # Dropout direction: angle of the vector from G1/G2/G3 AC values
    # in the 120-degree coordinate system (same transform as spatial vmag)
    _vx_drop = _ch_ac[0] - 0.5 * _ch_ac[1] - 0.5 * _ch_ac[2]
    _vy_drop = (np.sqrt(3.0) / 2.0) * (_ch_ac[1] - _ch_ac[2])
    feat["G_DROPOUT_ANGLE"] = float(np.degrees(np.arctan2(_vy_drop, _vx_drop + EPS)))

    # =====================================================
    # 5. 光学通道交叉特征（使用FFT缓存）
    # =====================================================
    feat.update(extract_cross_channel_features(
        g_mean_raw, g_mean_bp, g_mean_dc,
        ir_raw, ir_bp, ir_dc,
        amb_raw, amb_bp,
        fs,
        fft_cache_green=fft_cache['green'],
        fft_cache_ir=None
    ))


    # =====================================================
    # 5b. GREEN_CORR — bp 与自身平滑版本的相关性 (商用8特征之一)
    # =====================================================
    _ma_win_gc = max(2, int(round(0.15 * fs)))
    feat["GREEN_CORR"] = safe_corr(g_mean_bp,
                                   moving_average_filter(g_mean_bp, window_size=_ma_win_gc))
    feat["GREEN_AC"] = float(
        0.5 * np.sqrt(np.mean(g_mean_bp ** 2)) + 0.5 * robust_mad(g_mean_bp) * 1.4826
    )
    feat["AMB_AC"] = float(
        0.5 * np.sqrt(np.mean(amb_bp ** 2)) + 0.5 * robust_mad(amb_bp) * 1.4826
    )
    feat["GREEN_DC"] = float(np.median(g_mean_raw))
    feat["AMB_DC"] = float(np.median(ambient))
    _ac = normalized_autocorr(g_mean_bp)
    _lag_min = max(1, int(fs * 60.0 / 180.0))
    _lag_max = min(len(_ac) - 1, int(fs * 60.0 / 40.0))
    feat["GREEN_XCORR"] = float(np.max(_ac[_lag_min:_lag_max + 1])) if _lag_max > _lag_min else 0.0
    feat["FFT_PEAK_MEDIAN_RATIO"] = float(fft_cache["green"].get("peak_ratio", 0.0))

    # =====================================================
    # 6b. FFT 峰值宽度 + 绿光三通道相位一致性
    # =====================================================
    _fc = fft_cache['green']
    if _fc.get('band_spec') is not None and len(_fc['band_spec']) > 0:
        _bs = np.asarray(_fc['band_spec'], dtype=float)
        _bf = np.asarray(_fc['band_freqs'], dtype=float)
        _peak_val = np.max(_bs)
        _above_half = _bs > _peak_val * 0.5
        feat["GREEN_FFT_peak_width_Hz"] = float(_bf[_above_half][-1] - _bf[_above_half][0]) if np.any(_above_half) else 0.0
        # SNR: 带内功率 / 带外功率
        _in_band = float(np.sum(_bs ** 2))
        _out_band = float(np.sum(_fc['spec'] ** 2)) - _in_band
        feat["GREEN_FFT_SNR"] = float(_in_band / (_out_band + EPS))
    else:
        feat["GREEN_FFT_peak_width_Hz"] = 0.0
        feat["GREEN_FFT_SNR"] = 0.0

    # 谐波比: 2倍频功率 / 基频功率 (真实 PPG 有谐波结构，噪声没有)
    _fc = fft_cache['green']
    if _fc.get('band_spec') is not None and len(_fc['band_spec']) > 0:
        _bs = np.asarray(_fc['band_spec'], dtype=float)
        _bf = np.asarray(_fc['band_freqs'], dtype=float)
        _f0_idx = int(np.argmax(_bs))
        _f0 = _bf[_f0_idx]
        _f0_power = float(_bs[_f0_idx] ** 2)
        # 搜索 2*f0 附近 (±0.3Hz)
        _h2_mask = (_bf >= _f0 * 2 - 0.3) & (_bf <= _f0 * 2 + 0.3)
        if np.any(_h2_mask):
            _h2_power = float(np.max(_bs[_h2_mask] ** 2))
            feat["GREEN_FFT_harmonic_ratio"] = float(_h2_power / (_f0_power + EPS))
            feat["GREEN_FFT_harmonic_present"] = 1.0 if _h2_power > _f0_power * 0.1 else 0.0
        else:
            feat["GREEN_FFT_harmonic_ratio"] = 0.0
            feat["GREEN_FFT_harmonic_present"] = 0.0
    else:
        feat["GREEN_FFT_harmonic_ratio"] = 0.0
        feat["GREEN_FFT_harmonic_present"] = 0.0

    # 绿光三通道互相关峰值位置一致性（同一生理源 → lag 一致）
    _c12 = np.correlate(g1_bp - np.mean(g1_bp), g2_bp - np.mean(g2_bp), mode='same')
    _c23 = np.correlate(g2_bp - np.mean(g2_bp), g3_bp - np.mean(g3_bp), mode='same')
    _lag12 = int(np.argmax(np.abs(_c12)) - len(_c12) // 2) if len(_c12) > 0 else 0
    _lag23 = int(np.argmax(np.abs(_c23)) - len(_c23) // 2) if len(_c23) > 0 else 0
    feat["G_bp_lag_std"] = float(np.std([_lag12, _lag23]))

    # =====================================================
    # 7. 空间-光强耦合特征
    # =====================================================
    feat["corr_Gmean_G_imbalance"] = safe_corr(g_mean_raw, g_imbalance)
    feat["corr_Gmean_vmag"] = safe_corr(g_mean_raw, g_vmag)
    feat["corr_Ambient_vmag"] = safe_corr(amb_raw, g_vmag)

    # =====================================================
    # 7a2. 坏接触/离腕质量比值特征 (Tier 3)
    #      Ratio 型特征，跨设备/肤色鲁棒，区分：
    #      贴腕静止 vs 离腕环境光干扰 vs 松戴漏光
    # =====================================================
    _g_ac_rms = float(np.sqrt(np.mean(g_mean_bp ** 2)))
    _amb_ac_rms = float(np.sqrt(np.mean(amb_bp ** 2)))
    _g_dc = float(np.median(g_mean_raw))
    _amb_dc = float(np.median(amb_raw))
    _ch_dc_vals = np.array([float(np.median(g1_raw)), float(np.median(g2_raw)), float(np.median(g3_raw))])
    _ch_ac_vals = np.array([float(np.sqrt(np.mean(g1_bp ** 2))), float(np.sqrt(np.mean(g2_bp ** 2))), float(np.sqrt(np.mean(g3_bp ** 2)))])
    feat["AMB_AC_TO_GREEN_AC"] = safe_div(_amb_ac_rms, _g_ac_rms)
    feat["AMB_DC_TO_GREEN_DC"] = safe_div(_amb_dc, abs(_g_dc))
    feat["GCH_DC_RANGE_RATIO"] = safe_div(float(np.max(_ch_dc_vals) - np.min(_ch_dc_vals)), abs(float(np.mean(_ch_dc_vals))))
    feat["GCH_AC_RANGE_RATIO"] = safe_div(float(np.max(_ch_ac_vals) - np.min(_ch_ac_vals)), abs(float(np.mean(_ch_ac_vals))))

    # =====================================================
    # 7c. AMB spectral features (Tier 2)
    #     Ambient light spectrum helps distinguish skin-contact (light blocked)
    #     from off-wrist (ambient fluctuations visible).
    # =====================================================
    _fc_amb = fft_cache['amb']
    if _fc_amb.get('band_spec') is not None and len(_fc_amb['band_spec']) > 0:
        _bs = np.asarray(_fc_amb['band_spec'], dtype=float)
        _bf = np.asarray(_fc_amb['band_freqs'], dtype=float)
        feat["AMB_DOM_FREQ"] = float(_bf[int(np.argmax(_bs))])
        feat["AMB_FFT_PEAK_MEDIAN_RATIO"] = float(_fc_amb.get('peak_ratio', 0.0))
    else:
        feat["AMB_DOM_FREQ"] = 0.0
        feat["AMB_FFT_PEAK_MEDIAN_RATIO"] = 0.0

    # =====================================================
    # 7d. Per-channel skewness/kurtosis for AMB (Tier 2)
    #     GREEN and IRX already have these; AMB waveform shape
    #     indicates ambient light modulation patterns.
    # =====================================================
    _m_amb = float(np.mean(amb_bp))
    _s_amb = float(np.std(amb_bp))
    if _s_amb > EPS:
        feat["AMBX_bp_skewness"] = float(np.mean((amb_bp - _m_amb) ** 3) / (_s_amb ** 3))
        feat["AMBX_bp_kurtosis"] = float(np.mean((amb_bp - _m_amb) ** 4) / (_s_amb ** 4))
    else:
        feat["AMBX_bp_skewness"] = 0.0
        feat["AMBX_bp_kurtosis"] = 0.0

    # =====================================================
    # 7e. Signal quality indicators (Tier 2)
    #     Detect sensor saturation, clipping, and poor contact.
    # =====================================================
    for _ch_label, _ch_raw in [("GREEN", g_mean_raw)]:
        _sat_thr = 0.98 * np.max(_ch_raw)
        feat[f"{_ch_label}_SAT_FRAC"] = float(np.mean(_ch_raw >= _sat_thr))
        _d = np.diff(_ch_raw)
        _clip_eps = 1e-10
        feat[f"{_ch_label}_CLIP_RATE"] = float(np.mean(np.abs(_d) < _clip_eps))

    # GTOP2 基础特征：端侧只需要这一组 top-2 绿光统计和 1 个 FFT 来源。
    _gtop2_dc = float(np.median(g_top2_raw))
    _gtop2_cache = compute_fft_cache(g_top2_bp, fs, fmin=0.5, fmax=5.0)
    feat["GTOP2_ROBUST_RANGE_RATIO"] = robust_range_ratio(g_top2_raw)
    feat["GTOP2_SEG_ACDC_CV"] = segment_acdc_cv(g_top2_raw)
    feat["GTOP2_BAND_ENERGY_RATIO"] = band_energy_ratio_from_fft_cache(_gtop2_cache)
    feat.update(extract_single_channel_features(
        g_top2_raw, g_top2_bp, _gtop2_dc, fs, "GTOP2", fft_cache=_gtop2_cache
    ))
    feat.update(bp_shape_features(g_top2_bp, "GTOP2"))
    feat.update(segment_stability_features(g_top2_raw, "GTOP2"))
    feat["GREEN_AMB_LEAK_STABILITY"] = ambient_green_leak_stability(amb_raw, g_top2_raw)
    feat["GREEN_AMB_SEG_CORR_RANGE"] = segment_corr_range(amb_raw, g_top2_raw)

    # =====================================================
    # 12. 清理异常数值 + 移除已知冗余特征
    # =====================================================
    # 注意: 此处将 None / NaN / inf 统一替换为 0.0，这是 s03 层面的"兜底"处理。
    # 这意味着下游 s04/s05 从 CSV 读取特征时通常不会看到 NaN/inf（已被 0.0 替代）。
    # 0.0 不一定是该特征的合理默认值，但后续会被 s05 clip_outliers 裁剪到 IQR 下界，
    # 且 s05/s06 的 fill+clip 链会在此基础上做额外的防御性处理。
    # 训练与推理使用相同的 s03 代码，因此这一行为在两端一致，不构成训练-部署 gap。

    # ---- Invalid feature count (before NaN→0.0 fill) ----
    # XGBoost runs on every legal window and excludes IR-derived candidates.
    # IR-derived keys must not enter the Stage2 model-facing pool; failing here
    # catches new cross-stage feature leaks at the source.
    _stage2_ir_leaks = [k for k in feat.keys() if is_stage2_ir_feature(k)]
    if _stage2_ir_leaks:
        raise ValueError(
            "IR-derived Stage2 features generated in s03; remove them at source: "
            + ", ".join(_stage2_ir_leaks[:10])
        )

    # Count features that could not be computed. This lets XGBoost
    # distinguish "genuinely low value" from "computation failed".
    # Computed BEFORE the 0.0 fill below so we capture the true invalid count.
    _invalid_total = 0
    _invalid_ppg = 0
    _invalid_green = 0
    for k, v in feat.items():
        if v is None or not np.isfinite(v):
            _invalid_total += 1
            if any(k.startswith(p) for p in ["GREEN", "G_", "G1", "G2", "G3", "GTOP2", "AMBX", "AMB_", "Ambient", "corr_"]):
                if any(k.startswith(p) for p in ["GREEN", "G_", "G1", "G2", "G3", "GTOP2"]):
                    _invalid_green += 1
                _invalid_ppg += 1
    feat["TOTAL_INVALID_COUNT"] = float(_invalid_total)
    feat["PPG_INVALID_COUNT"] = float(_invalid_ppg)
    feat["GREEN_INVALID_COUNT"] = float(_invalid_green)

    for k in list(feat.keys()):
        if k in _REDUNDANT_FEATURES:
            del feat[k]
            continue
        v = feat[k]
        if v is None or not np.isfinite(v):
            feat[k] = 0.0
        else:
            feat[k] = float(v)

    # 如果需要返回预处理结果（供ACC交叉特征复用）
    if return_preprocessed:
        preprocessed = {
            'g1_bp': g1_bp, 'g2_bp': g2_bp, 'g3_bp': g3_bp,
            'g_top2_bp': g_top2_bp, 'g_top2_raw': g_top2_raw,
            'amb_bp': amb_bp,
            'g_mean_bp': g_mean_bp
        }
        return feat, preprocessed
    
    return feat


def _scale_floor(x, relative=1e-6, absolute=1e-9):
    x = finite_signal(x)
    if len(x) == 0:
        return float(absolute)
    scale = float(np.median(np.abs(x)))
    return float(max(scale * float(relative), float(absolute)))


def guarded_ratio(numerator, denominator, scale=None, relative=1e-6, absolute=1e-9):
    """Finite ratio with a signal-scale denominator floor suitable for C."""
    num = float(numerator)
    den = float(denominator)
    if not np.isfinite(num):
        num = 0.0
    if not np.isfinite(den):
        den = 0.0
    if scale is None:
        scale_value = abs(den)
    elif np.isscalar(scale):
        scale_value = abs(float(scale))
    else:
        scale_value = _scale_floor(scale, relative=1.0, absolute=absolute)
    floor = max(scale_value * float(relative), float(absolute))
    signed_den = den if abs(den) >= floor else (floor if den >= 0 else -floor)
    return float(num / signed_den)


def guarded_corr(x, y, relative_std_floor=1e-6):
    """Correlation with finite replacement and scale-aware variance guards."""
    x = finite_signal(x)
    y = finite_signal(y)
    n = min(len(x), len(y))
    if n < 8:
        return 0.0
    x = x[:n]
    y = y[:n]
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    sx = float(np.std(x_centered))
    sy = float(np.std(y_centered))
    x_floor = max(float(np.sqrt(np.mean(x * x))) * relative_std_floor, 1e-9)
    y_floor = max(float(np.sqrt(np.mean(y * y))) * relative_std_floor, 1e-9)
    if sx <= x_floor or sy <= y_floor:
        return 0.0
    value = float(np.mean((x_centered / sx) * (y_centered / sy)))
    return value if np.isfinite(value) else 0.0


def _top2_candidate_pairs(scores, relative_tolerance=1e-9):
    """Return every maximal two-zone pair, including exact/near ties."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != 3:
        raise ValueError("three-zone top2 aggregation requires exactly 3 scores")
    values = np.where(np.isfinite(values), values, 0.0)
    pairs = ((0, 1), (1, 2), (2, 0))
    pair_scores = np.asarray(
        [values[left] + values[right] for left, right in pairs],
        dtype=np.float64,
    )
    best = float(np.max(pair_scores))
    tolerance = max(abs(best) * float(relative_tolerance), 1e-12)
    return tuple(
        pair for pair, score in zip(pairs, pair_scores)
        if best - float(score) <= tolerance
    )


def _tie_aware_top2_composite(raw_stack, pulse_stack, scores):
    """Average all equally optimal top2 pair views to remove index tie bias."""
    raw = np.asarray(raw_stack, dtype=np.float64)
    pulse = np.asarray(pulse_stack, dtype=np.float64)
    pairs = _top2_candidate_pairs(scores)
    raw_pair_means = [0.5 * (raw[left] + raw[right]) for left, right in pairs]
    pulse_pair_means = [
        0.5 * (pulse[left] + pulse[right]) for left, right in pairs
    ]
    return (
        np.mean(np.vstack(raw_pair_means), axis=0),
        np.mean(np.vstack(pulse_pair_means), axis=0),
        pairs,
    )


def _frequency_evidence_valid(raw, pulse, fft_cache, fs):
    """Require amplitude, spectral-peak, and autocorrelation evidence."""
    raw = finite_signal(raw)
    pulse = finite_signal(pulse)
    pulse_rms = float(np.sqrt(np.mean(pulse ** 2))) if len(pulse) else 0.0
    raw_level = float(np.median(np.abs(raw))) if len(raw) else 0.0
    amplitude_floor = max(
        raw_level * MIN_ZONE_RELATIVE_AC_RMS,
        MIN_ZONE_ABSOLUTE_AC_RMS,
    )
    if pulse_rms <= amplitude_floor:
        return False
    band_spec = fft_cache.get("band_spec")
    if band_spec is None or len(band_spec) == 0:
        return False
    if float(fft_cache.get("dom_freq", 0.0)) <= 0.0:
        return False
    if float(fft_cache.get("peak_ratio", 0.0)) < 3.0:
        return False
    periodicity = autocorr_periodicity_features(pulse, fs)[0]
    return bool(np.isfinite(periodicity) and periodicity >= 0.20)


def _ambient_projection_residual(zone_pulse, ambient_pulse):
    """Remove the guarded linear ambient component while preserving zone mean."""
    zone = finite_signal(zone_pulse)
    ambient = finite_signal(ambient_pulse)
    n = min(len(zone), len(ambient))
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    zone = zone[:n]
    ambient = ambient[:n]
    zone_centered = zone - np.mean(zone)
    ambient_centered = ambient - np.mean(ambient)
    ambient_var = float(np.mean(ambient_centered ** 2))
    variance_floor = max(float(np.mean(ambient ** 2)) * 1e-12, 1e-18)
    if ambient_var <= variance_floor:
        return finite_signal(zone.copy())
    beta = float(np.mean(zone_centered * ambient_centered) / ambient_var)
    return finite_signal(zone - beta * ambient_centered)


def _phase_concentration(fft_caches, reference_hz):
    """Permutation-invariant phase concentration at a shared frequency bin."""
    if not np.isfinite(reference_hz) or float(reference_hz) <= 0.0:
        return 0.0
    phasors = []
    for cache in fft_caches:
        freqs = cache.get("band_freqs")
        values = cache.get("band_complex")
        if freqs is None or values is None or len(freqs) == 0 or len(values) == 0:
            return 0.0
        idx = int(np.argmin(np.abs(np.asarray(freqs, dtype=float) - float(reference_hz))))
        value = complex(values[idx])
        magnitude = abs(value)
        band_scale = float(np.max(np.abs(values))) if len(values) else 0.0
        if magnitude <= max(band_scale * 1e-9, 1e-12):
            return 0.0
        phasors.append(value / magnitude)
    if not phasors:
        return 0.0
    return float(np.clip(abs(np.mean(np.asarray(phasors, dtype=np.complex128))), 0.0, 1.0))


def _spectral_power_cosine_from_cache(left_cache, right_cache):
    """Cosine similarity between aligned cached 0.5-5 Hz power spectra."""
    left = left_cache.get("band_spec")
    right = right_cache.get("band_spec")
    if left is None or right is None:
        return 0.0
    count = min(len(left), len(right))
    if count < 2:
        return 0.0
    left_power = np.asarray(left[:count], dtype=np.float64) ** 2
    right_power = np.asarray(right[:count], dtype=np.float64) ** 2
    denominator = float(np.linalg.norm(left_power) * np.linalg.norm(right_power))
    floor = max(float(np.max(np.concatenate([left_power, right_power]))) * 1e-12, 1e-18)
    if denominator <= floor:
        return 0.0
    return float(np.clip(np.dot(left_power, right_power) / denominator, 0.0, 1.0))


def _contact_raw_signal(x):
    """Minimal raw path: finite replacement plus isolated-spike repair only."""
    return remove_burr(finite_signal(x), burr_k=6.0)


def _pulse_signal(contact_raw, fs):
    """Short-window pulse path using rolling-median detrending.

    This avoids the long forward-backward FIR edge response on 3-5 second
    windows and maps directly to a bounded rolling-median C implementation.
    """
    x = finite_signal(contact_raw)
    if len(x) < 4:
        return np.zeros_like(x)
    detrend_window = max(3, int(round(0.8 * float(fs))))
    if detrend_window % 2 == 0:
        detrend_window += 1
    baseline = _median_filter_np(x, detrend_window)
    pulse = x - baseline
    smooth_window = max(1, int(round(0.04 * float(fs))))
    if smooth_window >= 2:
        pulse = moving_average_filter(pulse, smooth_window)
    return finite_signal(pulse)


def _quality_features(ambient_input, green_input):
    ambient = finite_signal(ambient_input)
    green = finite_signal(green_input)
    green_flat = _diff_flat_ratio(green)
    green_spike = _diff_spike_ratio(green)
    ambient_flat = _diff_flat_ratio(ambient)
    ambient_spike = _diff_spike_ratio(ambient)
    return OrderedDict([
        ("GREEN_FLAT_RATIO", float(green_flat)),
        ("GREEN_SPIKE_RATIO", float(green_spike)),
        ("AMB_FLAT_RATIO", float(ambient_flat)),
        ("AMB_SPIKE_RATIO", float(ambient_spike)),
    ])


def _channel_candidate_features(raw, pulse, prefix, fs, fft_cache):
    dc = float(np.median(raw)) if len(raw) else 0.0
    ac_rms = float(np.sqrt(np.mean(pulse ** 2))) if len(pulse) else 0.0
    ac_peak, _ = autocorr_periodicity_features(pulse, fs, bpm_min=40.0, bpm_max=180.0)
    return OrderedDict([
        (f"{prefix}_DC_MEDIAN", dc),
        (f"{prefix}_DC_IQR", robust_iqr(raw)),
        (f"{prefix}_AC_RMS", ac_rms),
        (f"{prefix}_AC_MAD", robust_mad(pulse)),
        (f"{prefix}_AC_DC_RATIO", guarded_ratio(ac_rms, abs(dc), scale=raw)),
        (f"{prefix}_DERIV_MAD", robust_mad(np.diff(pulse)) if len(pulse) > 1 else 0.0),
        (f"{prefix}_FFT_PEAK_MEDIAN_RATIO", float(fft_cache.get("peak_ratio", 0.0))),
        (f"{prefix}_DOM_FREQ", float(fft_cache.get("dom_freq", 0.0))),
        (f"{prefix}_AUTO_CORR_PEAK", float(ac_peak)),
    ])


def _fixed_position_zone_candidates(raw_zones, pulse_zones, ambient_pulse, fs):
    """Return normalized candidates for the three stable physical green zones."""
    if len(raw_zones) != 3 or len(pulse_zones) != 3:
        raise ValueError("fixed-position extraction requires exactly three green zones")

    raw_zones = [finite_signal(zone) for zone in raw_zones]
    pulse_zones = [finite_signal(zone) for zone in pulse_zones]
    ambient_pulse = finite_signal(ambient_pulse)
    zone_dc = np.asarray(
        [float(np.median(zone)) if len(zone) else 0.0 for zone in raw_zones],
        dtype=np.float64,
    )
    zone_ac = np.asarray(
        [float(np.sqrt(np.mean(zone ** 2))) if len(zone) else 0.0 for zone in pulse_zones],
        dtype=np.float64,
    )
    median_dc = float(np.median(zone_dc))
    median_ac = float(np.median(zone_ac))

    features = OrderedDict()
    for index in range(3):
        features[f"GZONE{index + 1}_DC_CONTRAST"] = guarded_ratio(
            zone_dc[index] - median_dc,
            abs(zone_dc[index]) + abs(median_dc),
            scale=zone_dc,
        )
    for index in range(3):
        features[f"GZONE{index + 1}_AC_CONTRAST"] = guarded_ratio(
            zone_ac[index] - median_ac,
            zone_ac[index] + median_ac,
            scale=zone_ac,
        )
    for index in range(3):
        features[f"GZONE{index + 1}_AC_DC_RATIO"] = guarded_ratio(
            zone_ac[index], abs(zone_dc[index]), scale=raw_zones[index]
        )
    for index in range(3):
        features[f"GZONE{index + 1}_PERIODICITY"] = autocorr_periodicity_features(
            pulse_zones[index], fs
        )[0]
    for index in range(3):
        features[f"GZONE{index + 1}_AMB_ABS_CORR"] = abs(
            guarded_corr(pulse_zones[index], ambient_pulse)
        )
    return features


def normalized_spectral_entropy(fft_cache):
    """Normalized entropy of in-band spectral power, finite and bounded [0, 1]."""
    band_spec = fft_cache.get("band_spec")
    if band_spec is None or len(band_spec) < 2:
        return 0.0
    power = np.asarray(band_spec, dtype=np.float64) ** 2
    total = float(np.sum(power))
    if total <= _scale_floor(power):
        return 0.0
    probabilities = power / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + EPS)))
    return float(np.clip(entropy / np.log(len(probabilities)), 0.0, 1.0))


def robust_quantile_skewness(x):
    """Outlier-resistant Bowley-like skewness using P10/P50/P90."""
    values = finite_signal(x)
    if len(values) < 4:
        return 0.0
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    value = guarded_ratio(p90 + p10 - 2.0 * p50, p90 - p10, scale=values)
    return float(np.clip(value, -1.0, 1.0))


def spectral_power_cosine(x, y, fs):
    """Cosine similarity of two aligned 0.5-5 Hz power spectra."""
    n = min(len(x), len(y))
    if n < 16:
        return 0.0
    x_cache = compute_fft_cache(np.asarray(x)[:n], fs, fmin=0.5, fmax=5.0)
    y_cache = compute_fft_cache(np.asarray(y)[:n], fs, fmin=0.5, fmax=5.0)
    x_spec = x_cache.get("band_spec")
    y_spec = y_cache.get("band_spec")
    if x_spec is None or y_spec is None:
        return 0.0
    count = min(len(x_spec), len(y_spec))
    x_power = np.asarray(x_spec[:count], dtype=np.float64) ** 2
    y_power = np.asarray(y_spec[:count], dtype=np.float64) ** 2
    denominator = float(np.linalg.norm(x_power) * np.linalg.norm(y_power))
    if denominator <= _scale_floor(np.concatenate([x_power, y_power])):
        return 0.0
    return float(np.clip(np.dot(x_power, y_power) / denominator, 0.0, 1.0))


def _fft_shape_features(fft_cache):
    spec = fft_cache.get("spec")
    freqs = fft_cache.get("freqs")
    band_spec = fft_cache.get("band_spec")
    band_freqs = fft_cache.get("band_freqs")
    if spec is None or freqs is None or band_spec is None or len(band_spec) == 0:
        return 0.0, 0.0
    peak = float(np.max(band_spec))
    above_half = band_spec >= 0.5 * peak
    width = (
        float(band_freqs[above_half][-1] - band_freqs[above_half][0])
        if peak > 0.0 and np.any(above_half) else 0.0
    )
    non_dc = freqs > 0.0
    total = float(np.sum(np.asarray(spec)[non_dc] ** 2))
    in_band = float(np.sum(np.asarray(band_spec) ** 2))
    snr = guarded_ratio(in_band, max(total - in_band, 0.0), scale=total)
    return width, snr


def _green_spatial_candidates(g_raw, g_pulse, fs):
    raw_stack = np.vstack(g_raw)
    pulse_stack = np.vstack(g_pulse)
    spatial_mean = np.mean(raw_stack, axis=0)
    imbalance = np.std(raw_stack, axis=0) / (
        np.abs(spatial_mean) + _scale_floor(spatial_mean)
    )
    range_norm = (np.max(raw_stack, axis=0) - np.min(raw_stack, axis=0)) / (
        np.sum(np.abs(raw_stack), axis=0) + _scale_floor(raw_stack.reshape(-1))
    )
    channel_dc = np.median(raw_stack, axis=1)
    channel_ac = np.sqrt(np.mean(pulse_stack ** 2, axis=1))
    correlations = np.asarray([
        guarded_corr(pulse_stack[0], pulse_stack[1]),
        guarded_corr(pulse_stack[1], pulse_stack[2]),
        guarded_corr(pulse_stack[2], pulse_stack[0]),
    ], dtype=np.float64)
    periodicity = np.asarray([
        autocorr_periodicity_features(channel, fs, bpm_min=40.0, bpm_max=180.0)[0]
        for channel in pulse_stack
    ], dtype=np.float64)
    max_lag = max(1, int(round(0.4 * float(fs))))
    pair_lags = np.asarray([
        bounded_xcorr_peak_lag_samples(pulse_stack[0], pulse_stack[1], max_lag),
        bounded_xcorr_peak_lag_samples(pulse_stack[1], pulse_stack[2], max_lag),
        bounded_xcorr_peak_lag_samples(pulse_stack[2], pulse_stack[0], max_lag),
    ], dtype=np.float64)
    max_ac = float(np.max(channel_ac))
    total_ac = float(np.sum(channel_ac))
    if max_ac <= _scale_floor(channel_ac):
        support = 0.0
        top2_ratio = 0.0
        top2_corr = 0.0
        weak_gap = 0.0
        top1_top2 = 0.0
        rank_stability = 0.0
    else:
        top2_pairs = _top2_candidate_pairs(channel_ac)
        top2_pair_sums = [
            float(channel_ac[left] + channel_ac[right])
            for left, right in top2_pairs
        ]
        top2_sum = float(np.max(top2_pair_sums))
        top2_mean = 0.5 * top2_sum
        support = float(np.mean(channel_ac >= 0.5 * max_ac))
        top2_ratio = guarded_ratio(top2_sum, total_ac, scale=channel_ac)
        top2_corr = float(np.median([
            guarded_corr(pulse_stack[left], pulse_stack[right])
            for left, right in top2_pairs
        ]))
        weak_gap = max(
            0.0,
            guarded_ratio(
                top2_mean - float(np.min(channel_ac)),
                top2_mean,
                scale=channel_ac,
            ),
        )
        top1_top2 = guarded_ratio(max_ac, top2_mean, scale=channel_ac)
        global_top2 = set(top2_pairs)
        switches = []
        n = pulse_stack.shape[1]
        for indices in np.array_split(np.arange(n), 3):
            if len(indices) < 4:
                continue
            seg_ac = np.sqrt(np.mean(pulse_stack[:, indices] ** 2, axis=1))
            switches.append(set(_top2_candidate_pairs(seg_ac)) != global_top2)
        rank_stability = float(1.0 - np.mean(switches)) if switches else 0.0
    imbalance_mean = float(np.mean(imbalance))
    return OrderedDict([
        ("G_imbalance_mean", imbalance_mean),
        ("G_imbalance_p90", float(np.percentile(imbalance, 90))),
        ("G_imbalance_iqr", robust_iqr(imbalance)),
        ("G_rangeNorm_mean", float(np.mean(range_norm))),
        ("G_rangeNorm_p90", float(np.percentile(range_norm, 90))),
        ("G_ch_dc_cv", guarded_ratio(float(np.std(channel_dc)), abs(float(np.mean(channel_dc))), scale=channel_dc)),
        ("G_ch_dc_max_min_ratio", guarded_ratio(float(np.max(np.abs(channel_dc))), float(np.min(np.abs(channel_dc))), scale=channel_dc)),
        ("GCH_AC_RANGE_RATIO", guarded_ratio(float(np.ptp(channel_ac)), float(np.mean(channel_ac)), scale=channel_ac)),
        ("G_bp_corr_mean", float(np.mean(correlations))),
        ("G_bp_corr_min", float(np.min(correlations))),
        ("G_bp_corr_std", float(np.std(correlations))),
        ("G_2OF3_AC_SUPPORT", support),
        ("G_TOP2_TO_ALL_AC_RATIO", top2_ratio),
        ("G_TOP2_CORR_MIN", top2_corr),
        ("G_WEAK_CHANNEL_GAP", weak_gap),
        ("G_TOP1_TO_TOP2_AC_RATIO", top1_top2),
        ("G_TOP2_RANK_STABILITY", rank_stability),
        ("G_2OF3_PERIODICITY", float(np.median(periodicity))),
        ("G_ZONE_LAG_RMS_SEC", float(np.sqrt(np.mean(pair_lags ** 2)) / float(fs))),
    ]), imbalance, channel_ac


def _acc_candidate_features(acc_window, green_top2_pulse, green_top2_raw, fs,
                            selected_features=None):
    governed_names = [name for name in stage2_model_candidate_names() if name.startswith("ACC_")]
    names = (
        governed_names
        if selected_features is None
        else [name for name in selected_features if name in governed_names]
    )
    if not names:
        return OrderedDict()
    if acc_window is None or len(acc_window) < 4:
        return OrderedDict((name, 0.0) for name in names)
    acc = np.asarray(acc_window, dtype=np.float64)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)
    if acc.shape[1] < 3:
        acc = np.hstack([acc, np.zeros((len(acc), 3 - acc.shape[1]))])
    acc = acc[:, :3]
    acc = np.column_stack([finite_signal(acc[:, i]) for i in range(3)])
    magnitude = np.sqrt(np.sum(acc * acc, axis=1))
    baseline_window = max(3, int(round(0.8 * float(fs))))
    if baseline_window % 2 == 0:
        baseline_window += 1
    baseline = _median_filter_np(magnitude, baseline_window)
    motion = magnitude - baseline
    motion_mad = robust_mad(motion)
    clip_limit = max(8.0 * 1.4826 * motion_mad, _scale_floor(magnitude))
    motion = np.clip(motion, -clip_limit, clip_limit)
    motion_rms = float(np.sqrt(np.mean(motion ** 2)))
    mean_mag = float(np.mean(magnitude))
    jerk = np.abs(np.diff(magnitude))
    jerk_tail_count = max(1, int(np.ceil(0.1 * len(jerk)))) if len(jerk) else 0
    jerk_tail_mean = (
        float(np.mean(np.partition(jerk, len(jerk) - jerk_tail_count)[-jerk_tail_count:]))
        if jerk_tail_count else 0.0
    )
    relative_motion = guarded_ratio(motion_rms, mean_mag, scale=magnitude)
    green_names = {
        "ACC_GREEN_BP_CORR", "ACC_GREEN_REL_MOTION_GAP",
        "ACC_GREEN_MAX_LAG_CORR", "ACC_GREEN_PSD_SIMILARITY",
    }
    needs_green = bool(set(names) & green_names)
    if needs_green:
        green_top2_pulse = finite_signal(green_top2_pulse)
        green_top2_raw = finite_signal(green_top2_raw)
        n = min(len(motion), len(green_top2_pulse))
        acc_green_corr = (
            abs(guarded_corr(motion[:n], green_top2_pulse[:n])) if n >= 8 else 0.0
        )
        acc_green_max_lag_corr = (
            max_norm_xcorr(motion[:n], green_top2_pulse[:n], max(1, int(round(0.4 * fs))))
            if n >= 8 else 0.0
        )
        acc_green_psd_similarity = (
            spectral_power_cosine(motion[:n], green_top2_pulse[:n], fs) if n >= 16 else 0.0
        )
        green_ratio = guarded_ratio(
            float(np.sqrt(np.mean(green_top2_pulse ** 2))),
            abs(float(np.median(green_top2_raw))),
            scale=green_top2_raw,
        )
    else:
        acc_green_corr = 0.0
        acc_green_max_lag_corr = 0.0
        acc_green_psd_similarity = 0.0
        green_ratio = 0.0
    values = {
        "ACC_MAG_MEAN": mean_mag,
        "ACC_MAG_STD": float(np.std(magnitude)),
        "ACC_MAG_MAD": robust_mad(magnitude),
        "ACC_DYNAMIC_STD": float(np.sqrt(np.sum(np.var(acc, axis=0)))),
        "ACC_BP_RMS": motion_rms,
        "ACC_DIFF_MAD": robust_mad(np.diff(magnitude)) if len(magnitude) > 1 else 0.0,
        "ACC_MAG_P90": float(np.percentile(magnitude, 90)),
        "ACC_GRAVITY_RATIO": guarded_ratio(
            float(np.linalg.norm(np.mean(acc, axis=0))),
            mean_mag,
            scale=magnitude,
        ),
        "ACC_GREEN_BP_CORR": acc_green_corr,
        "ACC_REL_MOTION": relative_motion,
        "ACC_GREEN_REL_MOTION_GAP": float(
            abs(np.log1p(max(0.0, relative_motion)) - np.log1p(max(0.0, green_ratio)))
        ),
        "ACC_JERK_TAIL_MEAN_REL": guarded_ratio(
            jerk_tail_mean,
            mean_mag,
            scale=magnitude,
        ),
        "ACC_GREEN_MAX_LAG_CORR": acc_green_max_lag_corr,
        "ACC_GREEN_PSD_SIMILARITY": acc_green_psd_similarity,
    }
    return OrderedDict((name, float(values.get(name, 0.0))) for name in names)


def _extract_selected_optical_features(ir, ambient, g1, g2, g3, fs, selected_features):
    """Compute only the optical calculation families required by ``selected_features``."""
    requested_order = [
        str(name) for name in selected_features
        if str(name) != "mode" and not str(name).startswith("ACC_")
    ]
    requested = set(requested_order)
    unknown = [name for name in requested_order if not is_stage2_model_candidate(name)]
    if unknown:
        raise ValueError("unknown Stage2 model candidates: " + ", ".join(unknown))

    inputs = [finite_signal(x) for x in (ir, ambient, g1, g2, g3)]
    n = min(len(x) for x in inputs)
    if n < int(float(fs)):
        raise ValueError(f"window too short: n={n}, required>={int(float(fs))}")
    _, ambient_input, g1_input, g2_input, g3_input = [x[:n] for x in inputs]
    green_input = (g1_input + g2_input + g3_input) / 3.0
    features = OrderedDict()

    def keep(values):
        features.update((name, value) for name, value in values.items() if name in requested)

    quality_names = {"GREEN_FLAT_RATIO", "GREEN_SPIKE_RATIO", "AMB_FLAT_RATIO", "AMB_SPIKE_RATIO"}
    if requested & quality_names:
        keep(_quality_features(ambient_input, green_input))

    ambient_raw = _contact_raw_signal(ambient_input)
    g1_raw = _contact_raw_signal(g1_input)
    g2_raw = _contact_raw_signal(g2_input)
    g3_raw = _contact_raw_signal(g3_input)
    green_raw = (g1_raw + g2_raw + g3_raw) / 3.0
    g1_pulse = _pulse_signal(g1_raw, fs)
    g2_pulse = _pulse_signal(g2_raw, fs)
    g3_pulse = _pulse_signal(g3_raw, fs)
    green_pulse = (g1_pulse + g2_pulse + g3_pulse) / 3.0
    ambient_pulse = _pulse_signal(ambient_raw, fs)
    raw_stack = np.vstack([g1_raw, g2_raw, g3_raw])
    pulse_stack = np.vstack([g1_pulse, g2_pulse, g3_pulse])
    green_median_raw = np.median(raw_stack, axis=0)
    green_median_pulse = np.median(pulse_stack, axis=0)
    channel_ac = np.sqrt(np.mean(pulse_stack ** 2, axis=1))
    green_top2_raw, green_top2_pulse, _ = _tie_aware_top2_composite(
        raw_stack, pulse_stack, channel_ac
    )

    if "COMM_GREEN_AC" in requested:
        features["COMM_GREEN_AC"] = float(
            0.5 * np.sqrt(np.mean(green_pulse ** 2))
            + 0.5 * 1.4826 * robust_mad(green_pulse)
        )
    if "COMM_AMB_AC" in requested:
        features["COMM_AMB_AC"] = float(
            0.5 * np.sqrt(np.mean(ambient_pulse ** 2))
            + 0.5 * 1.4826 * robust_mad(ambient_pulse)
        )

    channel_sources = (
        ("GREEN", green_raw, green_pulse, {"GREEN_ROBUST_RANGE_RATIO", "GREEN_SEG_ACDC_CV"}),
        ("AMBX", ambient_raw, ambient_pulse, {"AMB_ROBUST_RANGE_RATIO", "AMB_SEG_ACDC_CV"}),
        (
            "GTOP2",
            green_top2_raw,
            green_top2_pulse,
            {
                "GTOP2_ROBUST_RANGE_RATIO", "GTOP2_SEG_ACDC_CV",
                "GTOP2_HALF_ACDC_DELTA", "GTOP2_SEG_ACDC_RANGE",
            },
        ),
    )
    fft_caches = {}
    for prefix, raw, pulse, extra_names in channel_sources:
        band_name = "AMB_BAND_ENERGY_RATIO" if prefix == "AMBX" else f"{prefix}_BAND_ENERGY_RATIO"
        channel_names = {
            f"{prefix}_DC_MEDIAN", f"{prefix}_DC_IQR", f"{prefix}_AC_RMS",
            f"{prefix}_AC_MAD", f"{prefix}_AC_DC_RATIO", f"{prefix}_DERIV_MAD",
            f"{prefix}_FFT_PEAK_MEDIAN_RATIO", f"{prefix}_DOM_FREQ",
            f"{prefix}_AUTO_CORR_PEAK", f"{prefix}_BAND_ENERGY_RATIO",
        }
        if not requested & (channel_names | extra_names | {band_name}):
            continue
        cache = compute_fft_cache(pulse, fs, fmin=0.5, fmax=5.0)
        fft_caches[prefix] = cache
        if prefix == "GREEN":
            values = {
                "GREEN_ROBUST_RANGE_RATIO": robust_range_ratio(raw),
                "GREEN_SEG_ACDC_CV": segment_acdc_cv(raw),
            }
        elif prefix == "AMBX":
            values = {
                "AMB_ROBUST_RANGE_RATIO": robust_range_ratio(raw),
                "AMB_SEG_ACDC_CV": segment_acdc_cv(raw),
            }
        else:
            values = {
                "GTOP2_ROBUST_RANGE_RATIO": robust_range_ratio(raw),
                "GTOP2_SEG_ACDC_CV": segment_acdc_cv(raw),
            }
            values.update(segment_stability_features(raw, "GTOP2"))
        keep(values)
        keep(_channel_candidate_features(raw, pulse, prefix, fs, cache))
        if band_name in requested:
            features[band_name] = band_energy_ratio_from_fft_cache(cache)

    if "GREEN_CORR" in requested:
        features["GREEN_CORR"] = guarded_corr(
            green_pulse,
            moving_average_filter(green_pulse, max(2, int(round(0.15 * float(fs))))),
        )
    if "GTOP2_zero_cross_rate" in requested:
        features["GTOP2_zero_cross_rate"] = zero_cross_rate(green_top2_pulse)
    if "GTOP2_abs_diff_ratio" in requested:
        features["GTOP2_abs_diff_ratio"] = guarded_ratio(
            float(np.mean(np.abs(np.diff(green_top2_pulse)))) if len(green_top2_pulse) > 1 else 0.0,
            float(np.mean(np.abs(green_top2_pulse - np.median(green_top2_pulse)))),
            scale=green_top2_pulse,
        )

    spatial_names = {
        "G_imbalance_mean", "G_imbalance_p90", "G_imbalance_iqr",
        "G_rangeNorm_mean", "G_rangeNorm_p90", "G_ch_dc_cv",
        "G_ch_dc_max_min_ratio", "GCH_AC_RANGE_RATIO", "G_bp_corr_mean",
        "G_bp_corr_min", "G_bp_corr_std", "G_2OF3_AC_SUPPORT",
        "G_TOP2_TO_ALL_AC_RATIO", "G_TOP2_CORR_MIN", "G_WEAK_CHANNEL_GAP",
        "G_TOP1_TO_TOP2_AC_RATIO", "G_TOP2_RANK_STABILITY",
        "G_2OF3_PERIODICITY", "G_ZONE_LAG_RMS_SEC",
    }
    imbalance = None
    if requested & (spatial_names | {"corr_Gmean_G_imbalance"}):
        spatial, imbalance, _ = _green_spatial_candidates(
            [g1_raw, g2_raw, g3_raw], [g1_pulse, g2_pulse, g3_pulse], fs
        )
        keep(spatial)

    cross_values = {}
    if "corr_Ambient_Gmean" in requested:
        cross_values["corr_Ambient_Gmean"] = guarded_corr(ambient_raw, green_raw)
    if "GREEN_AMB_BP_CORR" in requested:
        cross_values["GREEN_AMB_BP_CORR"] = guarded_corr(green_pulse, ambient_pulse)
    if "GREEN_AMB_ENV_CORR" in requested:
        cross_values["GREEN_AMB_ENV_CORR"] = guarded_corr(
            smooth_envelope(green_pulse, fs), smooth_envelope(ambient_pulse, fs)
        )
    if "AMB_AC_TO_GREEN_AC" in requested:
        cross_values["AMB_AC_TO_GREEN_AC"] = guarded_ratio(
            float(np.sqrt(np.mean(ambient_pulse ** 2))),
            float(np.sqrt(np.mean(green_pulse ** 2))),
            scale=green_pulse,
        )
    if "AMB_DC_TO_GREEN_DC" in requested:
        cross_values["AMB_DC_TO_GREEN_DC"] = guarded_ratio(
            float(np.median(ambient_raw)), abs(float(np.median(green_raw))), scale=green_raw
        )
    if "GREEN_AMB_LEAK_STABILITY" in requested:
        cross_values["GREEN_AMB_LEAK_STABILITY"] = ambient_green_leak_stability(
            ambient_raw, green_top2_raw
        )
    if "GREEN_AMB_SEG_CORR_RANGE" in requested:
        cross_values["GREEN_AMB_SEG_CORR_RANGE"] = segment_corr_range(
            ambient_raw, green_top2_raw
        )
    if "corr_Gmean_G_imbalance" in requested:
        cross_values["corr_Gmean_G_imbalance"] = guarded_corr(green_raw, imbalance)
    keep(cross_values)

    if "GTOP2_ROBUST_SKEWNESS" in requested:
        features["GTOP2_ROBUST_SKEWNESS"] = robust_quantile_skewness(green_top2_pulse)
    if "GTOP2_SPECTRAL_ENTROPY" in requested:
        top2_fft = fft_caches.get("GTOP2") or compute_fft_cache(
            green_top2_pulse, fs, fmin=0.5, fmax=5.0
        )
        features["GTOP2_SPECTRAL_ENTROPY"] = normalized_spectral_entropy(top2_fft)

    smooth_window = max(2, int(round(0.15 * float(fs))))
    median_names = {
        "GMEDIAN_AC_DC_RATIO", "GMEDIAN_CORR", "GMEDIAN_AUTO_CORR_PEAK",
        "GMEDIAN_FFT_PEAK_MEDIAN_RATIO",
    }
    if requested & median_names:
        if "GMEDIAN_AC_DC_RATIO" in requested:
            features["GMEDIAN_AC_DC_RATIO"] = guarded_ratio(
                float(np.sqrt(np.mean(green_median_pulse ** 2))),
                abs(float(np.median(green_median_raw))),
                scale=green_median_raw,
            )
        if "GMEDIAN_CORR" in requested:
            features["GMEDIAN_CORR"] = guarded_corr(
                green_median_pulse, moving_average_filter(green_median_pulse, smooth_window)
            )
        if "GMEDIAN_AUTO_CORR_PEAK" in requested:
            features["GMEDIAN_AUTO_CORR_PEAK"] = autocorr_periodicity_features(
                green_median_pulse, fs
            )[0]
        if "GMEDIAN_FFT_PEAK_MEDIAN_RATIO" in requested:
            features["GMEDIAN_FFT_PEAK_MEDIAN_RATIO"] = float(
                compute_fft_cache(green_median_pulse, fs, fmin=0.5, fmax=5.0)["peak_ratio"]
            )

    if "GTOP2_CORR" in requested:
        features["GTOP2_CORR"] = guarded_corr(
            green_top2_pulse, moving_average_filter(green_top2_pulse, smooth_window)
        )
    if "G_TOP2_ALL_CORR" in requested:
        features["G_TOP2_ALL_CORR"] = guarded_corr(green_top2_pulse, green_pulse)
    if "G_WEAK_TO_TOP2_CORR" in requested:
        weakest_ac = float(np.min(channel_ac))
        tolerance = max(abs(weakest_ac) * 1e-9, 1e-12)
        indices = np.flatnonzero(channel_ac - weakest_ac <= tolerance)
        features["G_WEAK_TO_TOP2_CORR"] = float(np.median([
            guarded_corr(pulse_stack[int(index)], green_top2_pulse) for index in indices
        ]))

    zone_frequency_names = {
        "G_ZONE_DOM_FREQ_MAD_HZ", "G_ZONE_HR_SUPPORT_RATIO",
        "G_PAIR_FREQ_GAP_MIN_HZ", "G_PAIR_FREQ_GAP_MEDIAN_HZ",
        "G_ZONE_PHASE_CONCENTRATION", "G_PAIR_SPECTRAL_CONSENSUS",
    }
    zone_ffts = None
    valid_spectrum = None
    zone_dom_freqs = None
    reference_hz = 0.0
    if requested & zone_frequency_names:
        zone_ffts = [compute_fft_cache(zone, fs, fmin=0.5, fmax=5.0) for zone in pulse_stack]
        zone_dom_freqs = np.asarray([float(cache["dom_freq"]) for cache in zone_ffts])
        valid_spectrum = np.asarray([
            _frequency_evidence_valid(raw_stack[index], pulse_stack[index], cache, fs)
            for index, cache in enumerate(zone_ffts)
        ], dtype=bool)
        valid_dom_freqs = zone_dom_freqs[valid_spectrum]
        reference_hz = float(np.median(valid_dom_freqs)) if len(valid_dom_freqs) else 0.0
        if "G_ZONE_DOM_FREQ_MAD_HZ" in requested:
            features["G_ZONE_DOM_FREQ_MAD_HZ"] = (
                float(np.mean(np.abs(valid_dom_freqs - reference_hz))) if len(valid_dom_freqs) else 0.0
            )
        if "G_ZONE_HR_SUPPORT_RATIO" in requested:
            features["G_ZONE_HR_SUPPORT_RATIO"] = (
                float(np.mean(valid_spectrum & (np.abs(zone_dom_freqs - reference_hz) <= 0.20)))
                if reference_hz > 0.0 else 0.0
            )

    pair_names = {
        "G_PAIR_PERIODICITY_MAX", "G_PAIR_PERIODICITY_MEDIAN",
        "G_PAIR_FREQ_GAP_MIN_HZ", "G_PAIR_FREQ_GAP_MEDIAN_HZ",
        "G_PAIR_ACDC_MEDIAN", "G_PAIR_AMB_ABS_CORR_MIN",
        "G_PAIR_AMB_ABS_CORR_MEDIAN", "G_PAIR_SPECTRAL_CONSENSUS",
    }
    pair_indices = ((0, 1), (1, 2), (2, 0))
    if requested & pair_names:
        periodicity, frequency_gaps, acdc, ambient_corr, spectral = [], [], [], [], []
        for left, right in pair_indices:
            pair_raw = 0.5 * (raw_stack[left] + raw_stack[right])
            pair_pulse = 0.5 * (pulse_stack[left] + pulse_stack[right])
            periodicity.append(autocorr_periodicity_features(pair_pulse, fs)[0])
            acdc.append(guarded_ratio(
                float(np.sqrt(np.mean(pair_pulse ** 2))),
                abs(float(np.median(pair_raw))),
                scale=pair_raw,
            ))
            ambient_corr.append(abs(guarded_corr(pair_pulse, ambient_pulse)))
            if zone_ffts is not None and valid_spectrum[left] and valid_spectrum[right]:
                frequency_gaps.append(abs(zone_dom_freqs[left] - zone_dom_freqs[right]))
                spectral.append(_spectral_power_cosine_from_cache(zone_ffts[left], zone_ffts[right]))
        pair_values = {
            "G_PAIR_PERIODICITY_MAX": float(np.max(periodicity)),
            "G_PAIR_PERIODICITY_MEDIAN": float(np.median(periodicity)),
            "G_PAIR_FREQ_GAP_MIN_HZ": float(np.min(frequency_gaps)) if frequency_gaps else 0.0,
            "G_PAIR_FREQ_GAP_MEDIAN_HZ": float(np.median(frequency_gaps)) if frequency_gaps else 0.0,
            "G_PAIR_ACDC_MEDIAN": float(np.median(acdc)),
            "G_PAIR_AMB_ABS_CORR_MIN": float(np.min(ambient_corr)),
            "G_PAIR_AMB_ABS_CORR_MEDIAN": float(np.median(ambient_corr)),
            "G_PAIR_SPECTRAL_CONSENSUS": float(np.median(spectral)) if spectral else 0.0,
        }
        keep(pair_values)

    residual_names = {"G_AMB_RESIDUAL_2OF3_PERIODICITY", "G_AMB_RESIDUAL_PAIR_CORR_MAX"}
    if requested & residual_names:
        residuals = [_ambient_projection_residual(zone, ambient_pulse) for zone in pulse_stack]
        keep({
            "G_AMB_RESIDUAL_2OF3_PERIODICITY": float(np.median([
                autocorr_periodicity_features(residual, fs)[0] for residual in residuals
            ])),
            "G_AMB_RESIDUAL_PAIR_CORR_MAX": float(np.max([
                guarded_corr(residuals[left], residuals[right]) for left, right in pair_indices
            ])),
        })
    if "G_ZONE_PHASE_CONCENTRATION" in requested:
        valid_zone_ffts = [cache for cache, valid in zip(zone_ffts, valid_spectrum) if valid]
        features["G_ZONE_PHASE_CONCENTRATION"] = (
            _phase_concentration(valid_zone_ffts, reference_hz) if len(valid_zone_ffts) >= 2 else 0.0
        )

    fixed_names = {name for name in requested if name.startswith("GZONE")}
    if fixed_names:
        keep(_fixed_position_zone_candidates(
            [g1_raw, g2_raw, g3_raw], [g1_pulse, g2_pulse, g3_pulse], ambient_pulse, fs
        ))

    missing = [name for name in requested_order if name not in features]
    if missing:
        raise RuntimeError("selective optical extractor missing: " + ", ".join(missing))
    ordered = OrderedDict()
    for name in requested_order:
        value = features[name]
        ordered[name] = float(value) if value is not None and np.isfinite(value) else 0.0
    preprocessed = {
        "g1_bp": g1_pulse, "g2_bp": g2_pulse, "g3_bp": g3_pulse,
        "g_top2_bp": green_top2_pulse, "g_top2_raw": green_top2_raw,
        "g_median_bp": green_median_pulse, "g_median_raw": green_median_raw,
        "amb_bp": ambient_pulse, "g_mean_bp": green_pulse,
        "g_mean_raw": green_raw, "amb_raw": ambient_raw,
    }
    return ordered, preprocessed


def extract_feature_pool_from_window(ir, ambient, g1, g2, g3, fs=25,
                                     return_preprocessed=False, selected_features=None):
    """Extract the governed optical Stage2 candidate pool.

    ACC candidates are assembled by :func:`assemble_stage2_feature_candidates`.
    """
    if selected_features is not None:
        ordered, preprocessed = _extract_selected_optical_features(
            ir, ambient, g1, g2, g3, fs, selected_features
        )
        return (ordered, preprocessed) if return_preprocessed else ordered

    inputs = [finite_signal(x) for x in (ir, ambient, g1, g2, g3)]
    n = min(len(x) for x in inputs)
    if n < int(float(fs)):
        raise ValueError(f"window too short: n={n}, required>={int(float(fs))}")
    _, ambient_input, g1_input, g2_input, g3_input = [x[:n] for x in inputs]
    green_input = (g1_input + g2_input + g3_input) / 3.0
    features = OrderedDict()
    features.update(_quality_features(ambient_input, green_input))

    ambient_raw = _contact_raw_signal(ambient_input)
    g1_raw = _contact_raw_signal(g1_input)
    g2_raw = _contact_raw_signal(g2_input)
    g3_raw = _contact_raw_signal(g3_input)
    green_raw = (g1_raw + g2_raw + g3_raw) / 3.0
    g1_pulse = _pulse_signal(g1_raw, fs)
    g2_pulse = _pulse_signal(g2_raw, fs)
    g3_pulse = _pulse_signal(g3_raw, fs)
    green_pulse = (g1_pulse + g2_pulse + g3_pulse) / 3.0
    ambient_pulse = _pulse_signal(ambient_raw, fs)
    raw_stack = np.vstack([g1_raw, g2_raw, g3_raw])
    pulse_stack = np.vstack([g1_pulse, g2_pulse, g3_pulse])
    green_median_raw = np.median(raw_stack, axis=0)
    green_median_pulse = np.median(pulse_stack, axis=0)

    features["COMM_GREEN_AC"] = float(
        0.5 * np.sqrt(np.mean(green_pulse ** 2))
        + 0.5 * 1.4826 * robust_mad(green_pulse)
    )
    features["COMM_AMB_AC"] = float(
        0.5 * np.sqrt(np.mean(ambient_pulse ** 2))
        + 0.5 * 1.4826 * robust_mad(ambient_pulse)
    )

    channel_ac = np.asarray([
        float(np.sqrt(np.mean(g1_pulse ** 2))),
        float(np.sqrt(np.mean(g2_pulse ** 2))),
        float(np.sqrt(np.mean(g3_pulse ** 2))),
    ])
    green_top2_raw, green_top2_pulse, _ = _tie_aware_top2_composite(
        raw_stack, pulse_stack, channel_ac
    )

    green_fft = compute_fft_cache(green_pulse, fs, fmin=0.5, fmax=5.0)
    ambient_fft = compute_fft_cache(ambient_pulse, fs, fmin=0.5, fmax=5.0)
    top2_fft = compute_fft_cache(green_top2_pulse, fs, fmin=0.5, fmax=5.0)
    median_fft = compute_fft_cache(green_median_pulse, fs, fmin=0.5, fmax=5.0)
    zone_ffts = [
        compute_fft_cache(zone, fs, fmin=0.5, fmax=5.0)
        for zone in pulse_stack
    ]

    for name, raw, pulse, prefix, cache in [
        ("green", green_raw, green_pulse, "GREEN", green_fft),
        ("ambient", ambient_raw, ambient_pulse, "AMBX", ambient_fft),
        ("green_top2", green_top2_raw, green_top2_pulse, "GTOP2", top2_fft),
    ]:
        if name == "green":
            features["GREEN_ROBUST_RANGE_RATIO"] = robust_range_ratio(raw)
            features["GREEN_SEG_ACDC_CV"] = segment_acdc_cv(raw)
        elif name == "ambient":
            features["AMB_ROBUST_RANGE_RATIO"] = robust_range_ratio(raw)
            features["AMB_SEG_ACDC_CV"] = segment_acdc_cv(raw)
        else:
            features["GTOP2_ROBUST_RANGE_RATIO"] = robust_range_ratio(raw)
            features["GTOP2_SEG_ACDC_CV"] = segment_acdc_cv(raw)
            features.update(segment_stability_features(raw, "GTOP2"))
        features.update(_channel_candidate_features(raw, pulse, prefix, fs, cache))

    features["GREEN_BAND_ENERGY_RATIO"] = band_energy_ratio_from_fft_cache(green_fft)
    features["GTOP2_BAND_ENERGY_RATIO"] = band_energy_ratio_from_fft_cache(top2_fft)
    features["AMB_BAND_ENERGY_RATIO"] = band_energy_ratio_from_fft_cache(ambient_fft)
    features["GREEN_CORR"] = guarded_corr(
        green_pulse,
        moving_average_filter(green_pulse, max(2, int(round(0.15 * float(fs))))),
    )
    features["GTOP2_zero_cross_rate"] = zero_cross_rate(green_top2_pulse)
    features["GTOP2_abs_diff_ratio"] = guarded_ratio(
        float(np.mean(np.abs(np.diff(green_top2_pulse)))) if len(green_top2_pulse) > 1 else 0.0,
        float(np.mean(np.abs(green_top2_pulse - np.median(green_top2_pulse)))),
        scale=green_top2_pulse,
    )

    spatial, imbalance, _ = _green_spatial_candidates(
        [g1_raw, g2_raw, g3_raw],
        [g1_pulse, g2_pulse, g3_pulse],
        fs,
    )
    features.update(spatial)
    features["corr_Ambient_Gmean"] = guarded_corr(ambient_raw, green_raw)
    features["GREEN_AMB_BP_CORR"] = guarded_corr(green_pulse, ambient_pulse)
    features["GREEN_AMB_ENV_CORR"] = guarded_corr(
        smooth_envelope(green_pulse, fs),
        smooth_envelope(ambient_pulse, fs),
    )
    green_rms = float(np.sqrt(np.mean(green_pulse ** 2)))
    ambient_rms = float(np.sqrt(np.mean(ambient_pulse ** 2)))
    features["AMB_AC_TO_GREEN_AC"] = guarded_ratio(
        ambient_rms, green_rms, scale=green_pulse
    )
    features["AMB_DC_TO_GREEN_DC"] = guarded_ratio(
        float(np.median(ambient_raw)),
        abs(float(np.median(green_raw))),
        scale=green_raw,
    )
    features["GREEN_AMB_LEAK_STABILITY"] = ambient_green_leak_stability(
        ambient_raw, green_top2_raw
    )
    features["GREEN_AMB_SEG_CORR_RANGE"] = segment_corr_range(
        ambient_raw, green_top2_raw
    )
    features["corr_Gmean_G_imbalance"] = guarded_corr(green_raw, imbalance)
    features["GTOP2_ROBUST_SKEWNESS"] = robust_quantile_skewness(green_top2_pulse)
    features["GTOP2_SPECTRAL_ENTROPY"] = normalized_spectral_entropy(top2_fft)

    # Three-zone robust evidence.  These candidates retain the mean and
    # AC-RMS top2 views, while adding median, pair-order-statistic, ambient-
    # residual, phase, and spectral-consensus views.  Every aggregation is
    # independent of the physical numbering of the three symmetric zones.
    smooth_window = max(2, int(round(0.15 * float(fs))))
    median_ac = float(np.sqrt(np.mean(green_median_pulse ** 2)))
    features["GMEDIAN_AC_DC_RATIO"] = guarded_ratio(
        median_ac,
        abs(float(np.median(green_median_raw))),
        scale=green_median_raw,
    )
    features["GMEDIAN_CORR"] = guarded_corr(
        green_median_pulse,
        moving_average_filter(green_median_pulse, smooth_window),
    )
    features["GMEDIAN_AUTO_CORR_PEAK"] = autocorr_periodicity_features(
        green_median_pulse, fs
    )[0]
    features["GMEDIAN_FFT_PEAK_MEDIAN_RATIO"] = float(median_fft["peak_ratio"])
    features["GTOP2_CORR"] = guarded_corr(
        green_top2_pulse,
        moving_average_filter(green_top2_pulse, smooth_window),
    )
    features["G_TOP2_ALL_CORR"] = guarded_corr(green_top2_pulse, green_pulse)
    weakest_ac = float(np.min(channel_ac))
    weakest_tolerance = max(abs(weakest_ac) * 1e-9, 1e-12)
    weakest_indices = np.flatnonzero(channel_ac - weakest_ac <= weakest_tolerance)
    features["G_WEAK_TO_TOP2_CORR"] = float(np.median([
        guarded_corr(pulse_stack[int(index)], green_top2_pulse)
        for index in weakest_indices
    ]))

    zone_dom_freqs = np.asarray(
        [float(cache["dom_freq"]) for cache in zone_ffts], dtype=np.float64
    )
    valid_spectrum = np.asarray([
        _frequency_evidence_valid(raw_stack[index], pulse_stack[index], cache, fs)
        for index, cache in enumerate(zone_ffts)
    ], dtype=bool)
    valid_dom_freqs = zone_dom_freqs[valid_spectrum]
    reference_hz = (
        float(np.median(valid_dom_freqs)) if len(valid_dom_freqs) else 0.0
    )
    features["G_ZONE_DOM_FREQ_MAD_HZ"] = (
        float(np.mean(np.abs(valid_dom_freqs - reference_hz)))
        if len(valid_dom_freqs) else 0.0
    )
    features["G_ZONE_HR_SUPPORT_RATIO"] = (
        float(np.mean(valid_spectrum & (np.abs(zone_dom_freqs - reference_hz) <= 0.20)))
        if reference_hz > 0.0 else 0.0
    )

    pair_indices = ((0, 1), (1, 2), (2, 0))
    pair_periodicity = []
    pair_freq_gaps = []
    pair_acdc = []
    pair_ambient_abs_corr = []
    pair_spectral_consensus = []
    for left, right in pair_indices:
        pair_raw = 0.5 * (raw_stack[left] + raw_stack[right])
        pair_pulse = 0.5 * (pulse_stack[left] + pulse_stack[right])
        pair_periodicity.append(autocorr_periodicity_features(pair_pulse, fs)[0])
        if valid_spectrum[left] and valid_spectrum[right]:
            pair_freq_gaps.append(abs(zone_dom_freqs[left] - zone_dom_freqs[right]))
        pair_acdc.append(guarded_ratio(
            float(np.sqrt(np.mean(pair_pulse ** 2))),
            abs(float(np.median(pair_raw))),
            scale=pair_raw,
        ))
        pair_ambient_abs_corr.append(abs(guarded_corr(pair_pulse, ambient_pulse)))
        if valid_spectrum[left] and valid_spectrum[right]:
            pair_spectral_consensus.append(
                _spectral_power_cosine_from_cache(zone_ffts[left], zone_ffts[right])
            )
    features["G_PAIR_PERIODICITY_MAX"] = float(np.max(pair_periodicity))
    features["G_PAIR_PERIODICITY_MEDIAN"] = float(np.median(pair_periodicity))
    features["G_PAIR_FREQ_GAP_MIN_HZ"] = (
        float(np.min(pair_freq_gaps)) if pair_freq_gaps else 0.0
    )
    features["G_PAIR_FREQ_GAP_MEDIAN_HZ"] = (
        float(np.median(pair_freq_gaps)) if pair_freq_gaps else 0.0
    )
    features["G_PAIR_ACDC_MEDIAN"] = float(np.median(pair_acdc))
    features["G_PAIR_AMB_ABS_CORR_MIN"] = float(np.min(pair_ambient_abs_corr))
    features["G_PAIR_AMB_ABS_CORR_MEDIAN"] = float(np.median(pair_ambient_abs_corr))

    residuals = [
        _ambient_projection_residual(zone, ambient_pulse) for zone in pulse_stack
    ]
    residual_periodicity = [
        autocorr_periodicity_features(residual, fs)[0] for residual in residuals
    ]
    residual_pair_corr = [
        guarded_corr(residuals[left], residuals[right])
        for left, right in pair_indices
    ]
    features["G_AMB_RESIDUAL_2OF3_PERIODICITY"] = float(
        np.median(residual_periodicity)
    )
    features["G_AMB_RESIDUAL_PAIR_CORR_MAX"] = float(np.max(residual_pair_corr))
    valid_zone_ffts = [
        cache for cache, is_valid in zip(zone_ffts, valid_spectrum) if is_valid
    ]
    features["G_ZONE_PHASE_CONCENTRATION"] = (
        _phase_concentration(valid_zone_ffts, reference_hz)
        if len(valid_zone_ffts) >= 2 else 0.0
    )
    features["G_PAIR_SPECTRAL_CONSENSUS"] = (
        float(np.median(pair_spectral_consensus))
        if pair_spectral_consensus else 0.0
    )
    features.update(_fixed_position_zone_candidates(
        [g1_raw, g2_raw, g3_raw],
        [g1_pulse, g2_pulse, g3_pulse],
        ambient_pulse,
        fs,
    ))

    ordered_names = [
        name for name in stage2_model_candidate_names()
        if not name.startswith("ACC_") and name != "mode"
    ]
    missing = [name for name in ordered_names if name not in features]
    if missing:
        raise RuntimeError("governed optical feature extractor missing: " + ", ".join(missing))
    ordered = OrderedDict()
    for name in ordered_names:
        value = features[name]
        ordered[name] = float(value) if value is not None and np.isfinite(value) else 0.0
    preprocessed = {
        "g1_bp": g1_pulse,
        "g2_bp": g2_pulse,
        "g3_bp": g3_pulse,
        "g_top2_bp": green_top2_pulse,
        "g_top2_raw": green_top2_raw,
        "g_median_bp": green_median_pulse,
        "g_median_raw": green_median_raw,
        "amb_bp": ambient_pulse,
        "g_mean_bp": green_pulse,
        "g_mean_raw": green_raw,
        "amb_raw": ambient_raw,
    }
    return (ordered, preprocessed) if return_preprocessed else ordered


def assemble_stage2_feature_candidates(
    ir,
    ambient,
    g1,
    g2,
    g3,
    *,
    mode,
    fs=25.0,
    acc_window=None,
    selected_features=None,
):
    if selected_features is not None:
        selected_features = [str(name) for name in selected_features]
        unknown = [name for name in selected_features if not is_stage2_model_candidate(name)]
        duplicates = [
            name for index, name in enumerate(selected_features)
            if name in selected_features[:index]
        ]
        if unknown:
            raise ValueError("unknown Stage2 model candidates: " + ", ".join(unknown))
        if duplicates:
            raise ValueError("duplicate Stage2 model candidates: " + ", ".join(duplicates))
        if selected_features == ["mode"]:
            return OrderedDict([("mode", float(mode))]), {}

    green_acc_names = {
        "ACC_GREEN_BP_CORR", "ACC_GREEN_REL_MOTION_GAP",
        "ACC_GREEN_MAX_LAG_CORR", "ACC_GREEN_PSD_SIMILARITY",
    }
    needs_optical = (
        selected_features is None
        or any(name != "mode" and not name.startswith("ACC_") for name in selected_features)
        or bool(set(selected_features or []) & green_acc_names)
    )
    if needs_optical:
        optical, preprocessed = extract_feature_pool_from_window(
            ir=ir,
            ambient=ambient,
            g1=g1,
            g2=g2,
            g3=g3,
            fs=fs,
            return_preprocessed=True,
            selected_features=selected_features,
        )
    else:
        optical, preprocessed = OrderedDict(), {}
    combined = OrderedDict([("mode", float(mode))])
    combined.update(optical)
    if selected_features is None or any(name.startswith("ACC_") for name in selected_features):
        combined.update(_acc_candidate_features(
            acc_window,
            preprocessed.get("g_top2_bp", np.zeros(0, dtype=np.float64)),
            preprocessed.get("g_top2_raw", np.zeros(0, dtype=np.float64)),
            fs,
            selected_features=selected_features,
        ))
    ordered_names = stage2_model_candidate_names() if selected_features is None else selected_features
    ordered = OrderedDict((name, float(combined.get(name, 0.0))) for name in ordered_names)
    if selected_features is None:
        validate_stage2_candidate_names(ordered.keys())
    return ordered, preprocessed


def stage2_diagnostic_fields(ir, ambient, g1, g2, g3, acc_window=None):
    ppg_arrays = [
        np.asarray(x, dtype=np.float64).reshape(-1)
        for x in (ir, ambient, g1, g2, g3)
    ]
    green_arrays = ppg_arrays[2:]
    ppg_invalid = int(sum(np.size(x) - np.isfinite(x).sum() for x in ppg_arrays))
    green_invalid = int(sum(np.size(x) - np.isfinite(x).sum() for x in green_arrays))
    acc_invalid = 0
    acc_available = acc_window is not None and len(acc_window) >= 4
    if acc_available:
        acc = np.asarray(acc_window, dtype=np.float64)
        acc_invalid = int(np.size(acc) - np.isfinite(acc).sum())
    return {
        "feature_pool_version": STAGE2_FEATURE_POOL_VERSION,
        "TOTAL_INVALID_COUNT": float(ppg_invalid + acc_invalid),
        "PPG_INVALID_COUNT": float(ppg_invalid),
        "GREEN_INVALID_COUNT": float(green_invalid),
        "ACC_AVAILABLE": float(bool(acc_available)),
    }


def extract_commercial_feature_overrides(ppg_window, acc_window, frequency, ppg_config):
    """Adapt a raw PPG/ACC window to the commercial float32 port.

    Returns an OrderedDict of the 8 ``COMMERCIAL_STAGE2_FIELDS`` computed by
    ``commercial_liveness_features.main``, overriding any Stage2 candidates that
    share the same field names.

    Inputs are passed as float32 to preserve full precision; the port internally
    applies F32() casts and ACC/4096 scaling exactly as the C reference does.
    """
    ppg = np.asarray(ppg_window)
    if frequency not in (25, 100):
        raise ValueError(f"commercial port requires frequency 25 or 100 Hz, got {frequency}")
    stride = frequency // 25
    expected_len = 125 * stride
    if ppg.shape[0] != expected_len:
        raise ValueError(
            f"commercial PPG window must be {expected_len} samples "
            f"at {frequency} Hz, got {ppg.shape[0]}"
        )
    if acc_window is None:
        raise ValueError("commercial ACC window is required but got None")
    acc = np.asarray(acc_window)
    if acc.shape[0] != expected_len:
        raise ValueError(
            f"commercial ACC window must be {expected_len} samples "
            f"at {frequency} Hz, got {acc.shape[0]}"
        )

    strided_ppg = np.nan_to_num(ppg[::stride], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    strided_acc = np.nan_to_num(acc[::stride], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    _, ambient, g1, g2, g3 = get_channels_from_window(strided_ppg, ppg_config)
    green = ((np.asarray(g1, dtype=np.float64)
              + np.asarray(g2, dtype=np.float64)
              + np.asarray(g3, dtype=np.float64)) / np.float64(3.0)).astype(np.float32)

    port_ppg = np.zeros((125, 4), dtype=np.float32)
    port_ppg[:, 0] = green
    port_ppg[:, 3] = ambient.astype(np.float32)

    result = _commercial_port_main(port_ppg, strided_acc)

    return OrderedDict(
        (name, float(value))
        for name, value in zip(COMMERCIAL_STAGE2_FIELDS, result)
    )


def extract_stage2_window(
    ppg_window,
    mode,
    fs=25.0,
    acc_window=None,
    use_stage2_ir=DEFAULT_USE_STAGE2_IR,
    selected_features=None,
):
    '''Return governed candidates, diagnostics, and shared intermediates.'''
    ppg = np.asarray(ppg_window, dtype=np.float64)
    if ppg.ndim == 1:
        ppg = ppg.reshape(-1, 1)
    if ppg.shape[1] < 3:
        raise ValueError(f'ppg_window requires at least 3 channels, got {ppg.shape[1]}')

    raw_ir, ambient, g1, g2, g3 = get_channels_from_window(ppg, mode)
    stage2_ir = apply_stage2_ir_policy(raw_ir, use_stage2_ir=use_stage2_ir)
    features, preprocessed = assemble_stage2_feature_candidates(
        stage2_ir,
        ambient,
        g1,
        g2,
        g3,
        mode=mode,
        fs=fs,
        acc_window=acc_window,
        selected_features=selected_features,
    )
    diagnostics = stage2_diagnostic_fields(
        raw_ir, ambient, g1, g2, g3, acc_window=acc_window
    )
    diagnostics['mode'] = int(mode)
    return features, diagnostics, preprocessed


def extract_window_features(ppg_window, fs=25.0, acc_window=None,
                            use_stage2_ir=DEFAULT_USE_STAGE2_IR, mode=None,
                            ppg_config=None, selected_features=None):
    if ppg_config is None:
        ppg_config = mode
    elif mode is not None and int(mode) != int(ppg_config):
        raise ValueError("mode and ppg_config disagree")
    if ppg_config is None:
        raise ValueError("ppg_config is required; automatic mode detection is disabled")
    resolved_mode = int(ppg_config)
    features, _, _ = extract_stage2_window(
        ppg_window,
        mode=resolved_mode,
        fs=fs,
        acc_window=acc_window,
        use_stage2_ir=use_stage2_ir,
        selected_features=selected_features,
    )
    expected_commercial_len = 125 * (int(fs) // 25)
    if len(ppg_window) == expected_commercial_len and acc_window is not None and (
        selected_features is None
        or any(name in COMMERCIAL_STAGE2_FIELDS for name in selected_features)
    ):
        with _commercial_warnings.catch_warnings():
            _commercial_warnings.simplefilter("ignore", RuntimeWarning)
            commercial = extract_commercial_feature_overrides(
                ppg_window, acc_window, frequency=int(fs), ppg_config=resolved_mode,
            )
        if selected_features is None:
            features.update(commercial)
        else:
            for name in selected_features:
                if name in commercial:
                    features[name] = commercial[name]
    return features

# =========================================================
# split 级别批量特征提取
# =========================================================

def _downsample_ppg(ppg, src_fs=100, tgt_fs=25):
    """Use fixed-phase 4-to-1 selection for 100 Hz to 25 Hz conversion.

    The retained samples are at source indices 0, 4, 8, ... .  No filtering,
    averaging, or interpolation is applied, matching the watch-side contract.
    """
    if src_fs == tgt_fs:
        return ppg
    if int(src_fs) != 100 or int(tgt_fs) != 25:
        raise ValueError(
            f"fixed-phase downsampling only supports 100 Hz to 25 Hz, "
            f"got {src_fs} Hz to {tgt_fs} Hz"
        )
    return np.asarray(ppg)[::4].astype(np.float64, copy=False)


def _commercial_only_feature_row(window, acc_seg, mode, frequency):
    """Return a feature dict with 8 commercial features + zero-filled model candidates.

    This is the fast path used by --commercial_only. It skips the full 126-feature
    Stage2 pipeline.  The commercial float32 port has a fixed 5-second contract, so
    it is used only for native 5-second windows.  For another valid window duration
    (for example 3 seconds / 75 points at 25 Hz), the same eight field names are
    calculated by the governed Stage2 implementation instead.  This preserves a
    duration-consistent feature vector without pretending that the fixed 5-second
    commercial formula was evaluated on an incomplete window.
    The remaining 118 model-candidate columns are filled with 0.0 so the output CSV
    keeps the complete column set expected by s04/s05/s06.
    """
    ppg = np.asarray(window, dtype=np.float64)
    if ppg.ndim == 1:
        ppg = ppg.reshape(-1, 1)

    feat = OrderedDict()
    # Start with zeros for ALL model candidates (ensures 126 columns always present)
    for name in stage2_model_candidate_names():
        feat[name] = 0.0

    native_port_length = 125 * (int(frequency) // 25)
    if int(frequency) in (25, 100) and len(ppg) == native_port_length:
        commercial = extract_commercial_feature_overrides(
            ppg, acc_seg, frequency=frequency, ppg_config=mode,
        )
        feat.update(commercial)  # override 8 zeros with exact port values
        diagnostics = {
            "TOTAL_INVALID_COUNT": 0.0,
            "PPG_INVALID_COUNT": 0.0,
            "GREEN_INVALID_COUNT": 0.0,
            "ACC_AVAILABLE": float(acc_seg is not None),
        }
    else:
        if int(frequency) == 25:
            ppg_25 = ppg
            acc_25 = None if acc_seg is None else np.asarray(acc_seg, dtype=np.float64)
        elif int(frequency) == 100:
            ppg_25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=25)
            acc_25 = None if acc_seg is None else _downsample_ppg(
                acc_seg, src_fs=100, tgt_fs=25
            )
        else:
            raise ValueError(f"unsupported PPG frequency for commercial_only: {frequency}")
        commercial, diagnostics, _ = extract_stage2_window(
            ppg_25,
            mode=mode,
            fs=25,
            acc_window=acc_25,
            selected_features=COMMERCIAL_STAGE2_FIELDS,
        )
        feat.update(commercial)

    feat["mode"] = int(mode)
    feat["feature_pool_version"] = STAGE2_FEATURE_POOL_VERSION
    feat.update(diagnostics)
    return feat


def _raise_sample_window_failures(sample_name, failures, attempted_windows):
    """Raise after every candidate window in one sample has been attempted."""
    if not failures:
        return
    examples = "; ".join(
        f"{position}: {type(exc).__name__}: {exc}"
        for position, exc in failures[:5]
    )
    raise RuntimeError(
        f"sample={sample_name}: feature extraction failed for "
        f"{len(failures)}/{attempted_windows} windows after all sample windows "
        f"were attempted; examples: {examples}"
    )


def _extract_rows_for_sample(sample, window_len, stride_len, fs,
                              target_aware_stride, stride_neg, stride_pos,
                              use_stage2_ir=DEFAULT_USE_STAGE2_IR,
                              commercial_only=False,
                              selected_features=None):
    """
    单样本全量抽窗特征。无合法窗口时返回空列表；读取或特征计算失败时，
    先尝试该样本的其余候选窗口，再抛出汇总异常。

    所有合法窗口直接进入 XGBoost 特征提取。3D 预切窗样本直接使用
    已有窗口；连续 100Hz 时序样本固定按 x[::4] 取为 25Hz 后滑窗。
    fs 参数为降采样后目标采样率 (25Hz)。

    commercial_only=True 时只提取 8 个商用特征 + 诊断字段，大幅加速。
    """
    FEATURE_FS = 25  # 特征提取统一采样率

    try:
        ppg = load_ppg(sample)
        acc = load_acc(sample)
        sample_frequency = get_sample_frequency(sample)
        mode = get_sample_ppg_config(sample)
    except Exception as e:
        sample_name = sample.get("sample_name", "unknown")
        raise RuntimeError(
            f"sample={sample_name}: input loading or metadata validation failed: "
            f"{type(e).__name__}: {e}"
        ) from e
    commercial_overrides_requested = (
        selected_features is None
        or any(name in COMMERCIAL_STAGE2_FIELDS for name in selected_features)
    )

    if is_prewindowed_signal(ppg):
        window_meta = load_grouped_window_metadata(sample)
        window_indices = list(window_meta.get("window_indices") or []) if window_meta else []
        window_labels = list(window_meta.get("window_labels") or []) if window_meta else []
        native_25hz = sample_frequency == FEATURE_FS
        ppg_src_fs = 25 if native_25hz else 100
        expected_native_window_len = int(round(
            float(window_len) * float(ppg_src_fs) / max(float(fs), 1.0)
        ))
        rows = []
        window_failures = []
        for win_idx in range(ppg.shape[0]):
            window_number = int(window_indices[win_idx]) if window_indices and win_idx < len(window_indices) else int(win_idx)
            window_target = int(window_labels[win_idx]) if window_labels and win_idx < len(window_labels) else int(sample["target"])
            raw_window = ppg[win_idx]
            if raw_window.shape[0] != expected_native_window_len:
                window_failures.append((
                    f"window_idx={win_idx}",
                    ValueError(
                        f"stored PPG window has {raw_window.shape[0]} samples, which does not "
                        f"match requested window length {expected_native_window_len} samples "
                        f"({float(window_len) / max(float(fs), 1.0):g}s at {ppg_src_fs} Hz)"
                    ),
                ))
                continue
            if raw_window.shape[0] < 2:
                window_failures.append((
                    f"window_idx={win_idx}",
                    ValueError(f"window has only {raw_window.shape[0]} samples"),
                ))
                continue
            if native_25hz:
                window = raw_window.astype(np.float64, copy=False)
            else:
                window = _downsample_ppg(raw_window, src_fs=100, tgt_fs=FEATURE_FS)
            acc_seg = None
            commercial_acc = None
            if acc is not None:
                try:
                    if is_prewindowed_signal(acc) and win_idx < acc.shape[0]:
                        raw_acc = acc[win_idx]
                        if raw_acc.shape[0] != expected_native_window_len:
                            raise ValueError(
                                f"stored ACC window has {raw_acc.shape[0]} samples, which does not "
                                f"match requested window length {expected_native_window_len} samples"
                            )
                        commercial_acc = raw_acc
                        acc_seg = raw_acc.astype(np.float64, copy=False) if native_25hz else _downsample_ppg(
                            raw_acc, src_fs=100, tgt_fs=FEATURE_FS
                        )
                    elif not is_prewindowed_signal(acc) and len(acc) > 0:
                        raw_start = int(win_idx * stride_len)
                        raw_acc = acc[raw_start:raw_start + window_len]
                        if len(raw_acc) > 0:
                            commercial_acc = raw_acc
                            acc_seg = raw_acc.astype(np.float64, copy=False) if native_25hz else _downsample_ppg(
                                raw_acc, src_fs=100, tgt_fs=FEATURE_FS
                            )
                except Exception as e:
                    window_failures.append((f"window_idx={win_idx}:acc", e))
                    print(f"ACC 处理失败: sample={sample.get('sample_name')}, "
                          f"window_idx={win_idx}, error={e}")
                    continue
            try:
                if commercial_only:
                    feat = _commercial_only_feature_row(
                        raw_window, commercial_acc, mode, ppg_src_fs
                    )
                else:
                    selective_kwargs = (
                        {"selected_features": selected_features}
                        if selected_features is not None else {}
                    )
                    feat, diagnostics, _ = extract_stage2_window(
                        window,
                        mode=mode,
                        fs=FEATURE_FS,
                        acc_window=acc_seg,
                        use_stage2_ir=use_stage2_ir,
                        **selective_kwargs,
                    )
                    expected_commercial_len = 125 * (ppg_src_fs // FEATURE_FS)
                    if (
                        commercial_overrides_requested
                        and commercial_acc is not None
                        and len(raw_window) == expected_commercial_len
                    ):
                        commercial_overrides = extract_commercial_feature_overrides(
                            raw_window,
                            commercial_acc,
                            ppg_src_fs,
                            mode,
                        )
                        if selected_features is None:
                            feat.update(commercial_overrides)
                        else:
                            for name in selected_features:
                                if name in commercial_overrides:
                                    feat[name] = commercial_overrides[name]
                    feat.update(diagnostics)
                feat["sample_name"] = sample["sample_name"]
                feat["h5_file"] = sample["h5_file"]
                feat["target"] = int(window_target)
                feat["start_100hz"] = int(window_number * stride_len)
                feat["start_sec"] = float(window_number * stride_len / max(fs, 1))
                feat["window_index"] = int(window_number)
                rows.append(feat)
            except Exception as e:
                window_failures.append((f"window_idx={win_idx}", e))
                print(f"特征提取失败: sample={sample.get('sample_name')}, "
                      f"window_idx={win_idx}, error={e}")
                continue
        _raise_sample_window_failures(
            sample.get("sample_name", "unknown"),
            window_failures,
            int(ppg.shape[0]),
        )
        return rows

    source_window_len = int(round((window_len / max(fs, 1)) * sample_frequency))
    if len(ppg) < source_window_len:
        return []

    # 检测是否 25Hz 原生数据
    native_25hz = sample_frequency == FEATURE_FS
    ppg_src_fs = 25 if native_25hz else 100

    sample_target = int(sample.get("target", 0))

    # 100Hz 固定保留索引 0,4,8,...；原生 25Hz 直接使用。
    if native_25hz:
        ppg_25 = ppg
        acc_25 = acc if (acc is not None and len(acc) > 0) else None
    else:
        ppg_25 = _downsample_ppg(ppg, src_fs=100, tgt_fs=FEATURE_FS)
        if acc is not None and len(acc) > 0:
            acc_25 = _downsample_ppg(acc, src_fs=100, tgt_fs=FEATURE_FS)
        else:
            acc_25 = None

    # 25Hz 下的窗长和步长（从调用参数推导，而非硬编码）
    window_sec_actual = window_len / max(fs, 1)
    stride_sec_actual = stride_len / max(fs, 1)
    win_25 = int(window_sec_actual * FEATURE_FS)
    stride_25 = int(stride_sec_actual * FEATURE_FS)
    if target_aware_stride:
        # stride_neg/pos 是从 fs=100 计算出来的原始值，需要换算到 25Hz
        stride_neg_25 = int(stride_neg * FEATURE_FS / max(fs, 1))
        stride_pos_25 = int(stride_pos * FEATURE_FS / max(fs, 1))
        stride_25 = stride_neg_25 if sample_target == 0 else stride_pos_25

    if len(ppg_25) < win_25:
        return []

    rows = []
    window_failures = []
    starts = list(range(0, len(ppg_25) - win_25 + 1, stride_25))
    for start in starts:
        window = ppg_25[start:start + win_25, :]
        try:
            acc_seg = None
            if acc_25 is not None and len(acc_25) > 0:
                acc_seg = align_acc_window(acc_25, len(ppg_25), start, win_25,
                                           fs_ppg=FEATURE_FS, fs_acc=FEATURE_FS)
            if native_25hz:
                commercial_ppg = window
                commercial_acc = acc_seg
                commercial_frequency = FEATURE_FS
            else:
                commercial_stride = int(sample_frequency // FEATURE_FS)
                commercial_start = int(start * commercial_stride)
                commercial_len = int(win_25 * commercial_stride)
                commercial_ppg = ppg[commercial_start:commercial_start + commercial_len]
                commercial_acc = None if acc is None else acc[
                    commercial_start:commercial_start + commercial_len
                ]
                commercial_frequency = sample_frequency
            if commercial_only:
                feat = _commercial_only_feature_row(
                    commercial_ppg, commercial_acc, mode, commercial_frequency
                )
            else:
                selective_kwargs = (
                    {"selected_features": selected_features}
                    if selected_features is not None else {}
                )
                feat, diagnostics, _ = extract_stage2_window(
                    window,
                    mode=mode,
                    fs=FEATURE_FS,
                    acc_window=acc_seg,
                    use_stage2_ir=use_stage2_ir,
                    **selective_kwargs,
                )
                if (
                    win_25 == 125
                    and commercial_overrides_requested
                    and commercial_acc is not None
                    and len(commercial_ppg) == 125 * (commercial_frequency // FEATURE_FS)
                ):
                    commercial_overrides = extract_commercial_feature_overrides(
                        commercial_ppg, commercial_acc, commercial_frequency, mode
                    )
                    if selected_features is None:
                        feat.update(commercial_overrides)
                    else:
                        for name in selected_features:
                            if name in commercial_overrides:
                                feat[name] = commercial_overrides[name]
                feat.update(diagnostics)
            feat["sample_name"] = sample["sample_name"]
            feat["h5_file"] = sample["h5_file"]
            feat["target"] = int(sample["target"])
            feat["start_100hz"] = int(start * (fs / FEATURE_FS))  # 映射回原始 fs 坐标
            feat["start_sec"] = float(start / FEATURE_FS)
            feat["window_index"] = int(start // max(stride_25, 1))
            rows.append(feat)
        except Exception as e:
            window_failures.append((f"start={start}", e))
            print(f"特征提取失败: sample={sample.get('sample_name')}, "
                  f"start={start}, error={e}")
            continue
    _raise_sample_window_failures(
        sample.get("sample_name", "unknown"),
        window_failures,
        len(starts),
    )
    return rows


def _worker_extract(args_tuple):
    """子进程入口。"""
    (sample, window_len, stride_len, fs,
     target_aware_stride, stride_neg, stride_pos,
     use_stage2_ir, commercial_only, selected_features) = args_tuple
    return _extract_rows_for_sample(
        sample, window_len, stride_len, fs,
        target_aware_stride, stride_neg, stride_pos,
        use_stage2_ir=use_stage2_ir,
        commercial_only=commercial_only,
        selected_features=selected_features,
    )


def extract_features_for_split(samples,
                               window_sec=5,
                               stride_sec=1,
                               fs=100,
                               target_aware_stride=False,
                               target_ratio=5.0,
                               use_stage2_ir=DEFAULT_USE_STAGE2_IR,
                               n_workers=None,
                               commercial_only=False,
                               selected_features=None):
    """
    提取特征池（样本级并行）。

    参数:
        samples: 样本列表
        window_sec: 窗口秒数
        stride_sec: 默认步长秒数
        fs: 采样率
        target_aware_stride: 是否启用target感知stride
        target_ratio: 目标正负样本比例 (neg/pos)
        n_workers: 并行 worker 数；None=自动(cpu_count-1)，1=单进程
        commercial_only: 仅提取商用 8 特征（跳过 126 项 Stage2 全量池）
    """
    window_len = int(window_sec * fs)
    stride_len = int(stride_sec * fs)
    stride_neg = int(1 * fs)
    stride_pos = int(3 * fs)

    original_samples = list(samples)
    n_workers = resolve_n_workers(n_workers, n_items=len(original_samples))

    # Sort heaviest first: reduces tail latency when last few workers are left
    # waiting on a single large continuous-signal sample.
    def _sample_weight(s):
        shape = s.get("ppg_shape")
        if shape and len(shape) >= 1 and isinstance(shape[0], (int, float)):
            return int(shape[0])
        return 0  # conservative: unknown size goes last

    ordered_sample_items = sorted(
        enumerate(original_samples),
        key=lambda item: _sample_weight(item[1]),
        reverse=True,
    )
    ordered_original_indices = [index for index, _sample in ordered_sample_items]
    ordered_samples = [sample for _index, sample in ordered_sample_items]
    if n_workers > 1 and len(ordered_samples) >= n_workers * 2:
        print(f"  s03 sample order: heaviest-first "
              f"(max_windows={_sample_weight(ordered_samples[0])}, "
              f"min_windows={_sample_weight(ordered_samples[-1])})", flush=True)

    args_list = [
        (s, window_len, stride_len, fs,
         target_aware_stride, stride_neg, stride_pos,
         use_stage2_ir, commercial_only, selected_features)
        for s in ordered_samples
    ]

    rows_by_sample = [[] for _ in original_samples]
    failures = []
    if n_workers == 1:
        for i, a in enumerate(args_list, 1):
            original_idx = ordered_original_indices[i - 1]
            try:
                rows_by_sample[original_idx] = _worker_extract(a)
            except Exception as exc:
                sample_name = original_samples[original_idx].get(
                    "sample_name", f"idx={original_idx}")
                failures.append((sample_name, exc))
                print(
                    f"  [ERROR] s03 worker failed sample={sample_name}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if len(args_list) >= 10 and (i % max(1, len(args_list) // 10) == 0 or i == len(args_list)):
                print(f"  s03 progress: {i}/{len(args_list)} samples", flush=True)
    else:
        pool_kwargs = {"max_workers": n_workers}
        mp_ctx = multiprocessing_context_from_env()
        if mp_ctx is not None:
            pool_kwargs["mp_context"] = mp_ctx
        completed = 0
        total = len(args_list)
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            future_to_idx = {
                ex.submit(_worker_extract, a): ordered_original_indices[i]
                for i, a in enumerate(args_list)
            }
            print(f"  s03 parallel extraction: {total} samples, workers={n_workers}", flush=True)
            # Intentionally no timeout: every submitted sample must finish or
            # raise an explicit error. Slow samples are never cancelled or
            # converted into silent empty results.
            for fut in as_completed(future_to_idx):
                sample_idx = future_to_idx[fut]
                try:
                    rows = fut.result()
                except Exception as exc:
                    sample_name = original_samples[sample_idx].get(
                        "sample_name", f"idx={sample_idx}")
                    failures.append((sample_name, exc))
                    print(
                        f"  [ERROR] s03 worker failed sample={sample_name}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    rows = []
                rows_by_sample[sample_idx] = rows
                completed += 1
                if completed % max(1, total // 10) == 0 or completed == total:
                    print(f"  s03 progress: {completed}/{total} samples", flush=True)

    if failures:
        examples = "; ".join(
            f"{name}: {type(exc).__name__}: {exc}"
            for name, exc in failures[:5]
        )
        raise RuntimeError(
            f"s03 feature extraction failed for {len(failures)}/"
            f"{len(original_samples)} samples after all samples were attempted; "
            f"examples: {examples}"
        )
    all_rows = []
    for rows in rows_by_sample:
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def infer_uniform_prewindowed_window_sec(samples):
    """Infer a common 3s/5s duration from prewindowed sample metadata.

    Only 3-D PPG records participate. Continuous recordings have no intrinsic
    window duration and retain the CLI configuration. A mixed-duration dataset is
    rejected because one shared model cannot use both feature distributions.
    """
    inferred = set()
    for sample in samples or []:
        shape = tuple(sample.get("ppg_shape") or ())
        if len(shape) != 3:
            continue
        try:
            frequency = int(sample.get("frequency"))
        except (TypeError, ValueError):
            continue
        matching = {
            seconds
            for seconds in (3, 5)
            if int(seconds * frequency) in shape[1:]
        }
        if len(matching) > 1:
            raise ValueError(
                f"sample={sample.get('sample_name', 'unknown')} has ambiguous "
                f"prewindowed PPG shape={shape} at {frequency} Hz"
            )
        inferred.update(matching)
    if len(inferred) > 1:
        raise ValueError(
            "mixed prewindowed durations detected (3s and 5s). Split them into "
            "separate training/deployment runs because they require different models."
        )
    return next(iter(inferred), None)


# =========================================================
# main
# =========================================================
def main(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--window_sec", type=int, default=5, choices=[3, 5],
                        help="Stage2 窗口秒数：3s (75点@25Hz) 或 5s (125点@25Hz)")
    parser.add_argument("--stride_sec", type=int, default=1)
    parser.add_argument("--use_stage2_ir", action=argparse.BooleanOptionalAction,
                        default=DEFAULT_USE_STAGE2_IR,
                        help="legacy compatibility flag; Stage2 model features are always ambient/green/ACC only")
    
    # target 感知 stride 参数
    # 注意：默认 False。train/deploy 都用统一 1s stride 才能保证窗口分布一致。
    # 启用此参数会让 target=0 用 1s、target=1 用 3s，制造 neg:pos 5:1 的窗口比，
    # 配合 s05 旧版 scale_pos_weight=neg/pos 会双倍偏置模型预测正类、抬高 FP，
    # 与"FP 代价高"的产品目标冲突。
    parser.add_argument("--target_aware_stride", action="store_true",
                        default=False,
                        help="[不推荐] 启用 target 感知 stride（pos=3s, neg=1s）。"
                             "与部署 1s stride 分布不一致，且与 scale_pos_weight 叠加双倍偏置。"
                             "保留仅供对照。")
    parser.add_argument("--target_ratio", type=float, default=5.0,
                        help="target 感知 stride 启用时的目标 neg/pos 比例")

    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")
    parser.add_argument(
        "--selected_feature_file",
        type=str,
        default=None,
        help="CSV with one 'feature' column; compute only these features in row order",
    )
    parser.add_argument("--commercial_only", action="store_true",
                        help="仅提取商用 8 特征，跳过 126 项 Stage2 全量池")

    raw_cli_args = sys.argv[1:] if args is None else []
    if args is None:
        args = parser.parse_args()

    if args.commercial_only and args.selected_feature_file:
        parser.error("--commercial_only conflicts with --selected_feature_file")
    selected_features = None
    if args.selected_feature_file:
        selected_features = load_direct_feature_csv(args.selected_feature_file)
        print(
            f"[direct] selective extraction enabled: {len(selected_features)} features; "
            "unselected calculation families will be skipped"
        )

    split_path = os.path.join(args.artifact_dir, "splits.json")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    all_samples = [
        sample
        for part in ("train", "valid", "test")
        for sample in split.get(part, [])
    ]
    inferred_window_sec = infer_uniform_prewindowed_window_sec(all_samples)
    if inferred_window_sec is not None and int(args.window_sec) != inferred_window_sec:
        if "--window_sec" in raw_cli_args:
            parser.error(
                f"--window_sec {args.window_sec} conflicts with prewindowed H5 "
                f"duration {inferred_window_sec}s"
            )
        print(
            f"[window] auto-detected {inferred_window_sec}s prewindowed H5 data; "
            f"overriding default --window_sec {args.window_sec}"
        )
        args.window_sec = inferred_window_sec

    feature_pool_frames = {}
    for part in ["train", "valid", "test"]:
        print("=" * 80)
        print(f"提取 {part} 特征")
        print("=" * 80)

        if args.target_aware_stride:
            print("  ⚠ target_aware_stride: 启用 (不推荐)")
            print("    target=0 stride=1s, target=1 stride=3s -> 制造窗口级不平衡")
            print("    若 s05 同时启用 scale_pos_weight=neg/pos 会双倍偏置正类，"
                  "与 FP 高代价目标冲突。")
        else:
            print(f"  target_aware_stride: 禁用 (统一 stride={args.stride_sec}s，"
                  f"与部署 1s stride 分布一致)")

        df = extract_features_for_split(
            samples=split[part],
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            fs=100,
            target_aware_stride=args.target_aware_stride,
            target_ratio=args.target_ratio,
            use_stage2_ir=args.use_stage2_ir,
            n_workers=args.n_workers,
            commercial_only=args.commercial_only,
            selected_features=selected_features,
        )

        out_path = os.path.join(args.artifact_dir, f"feature_pool_{part}.csv")
        df.to_csv(out_path, index=False)
        feature_pool_frames[part] = df

        print(f"{part} 特征提取完成: {len(df)} windows")
        print(f"保存到: {out_path}")

        if len(df) > 0 and "target" in df.columns:
            print(f"  target=0: {np.sum(df['target'].values == 0)}")
            print(f"  target=1: {np.sum(df['target'].values == 1)}")
            meta_cols = ["sample_name", "h5_file", "target", "start_100hz", "start_sec"]
            print(f"  特征列数: {len([c for c in df.columns if c not in meta_cols])}")

    analysis_outputs = export_feature_pool_analysis_plot(feature_pool_frames, args.artifact_dir)
    print(f"Stage2 特征池分析 PNG 已保存: {analysis_outputs['png']}")

if __name__ == "__main__":
    main()
