# s02_ir_dc_threshold.py
# -*- coding: utf-8 -*-

"""
步骤2：Stage1 IR DC/ACDC 固定阈值配置。

当前部署阈值固定为：
- dc_threshold = 1.5e6
- ac_dc_threshold = 1.0

本脚本不再做阈值搜参；只提取 train/valid 的 Stage1 primitive windows，
评估固定阈值表现，并写出 stage1_threshold.json 与 stage1_scatter.png。
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import resample_poly
from s03_extract_feature_pool import load_ppg as load_h5_ppg, flatten_prewindowed_signal

STAGE1_PRIMITIVE_SEC = 1.0
STAGE1_DECISION_SEC = 3.0
DEFAULT_MIN_DURATION_SEC = STAGE1_DECISION_SEC
STAGE1_FS = 5
STAGE1_GATE_K = int(round(STAGE1_DECISION_SEC / STAGE1_PRIMITIVE_SEC))
FIXED_DEPLOY_DC_THRESHOLD = 1.5e6
FIXED_DEPLOY_AC_DC_THRESHOLD = 1.0


def resolve_fixed_deploy_thresholds(_args=None):
    """Return the fixed Stage1 deployment thresholds; no threshold search is run."""
    return float(FIXED_DEPLOY_DC_THRESHOLD), float(FIXED_DEPLOY_AC_DC_THRESHOLD)


def _env_flag(name):
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def resolve_n_workers(n_workers=None, n_items=None, cap=4):
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


def load_ppg(sample):
    ppg = load_h5_ppg(sample)
    ppg = flatten_prewindowed_signal(ppg)
    return ppg


def _is_25hz_sample(sample):
    name = sample.get("sample_name", "") if isinstance(sample, dict) else str(sample)
    return "sleep_25hz" in name.lower()


def downsample_to_5hz(signal, fs_original=100, fs_target=5):
    if fs_original == fs_target:
        return signal
    gcd = np.gcd(fs_original, fs_target)
    up = fs_target // gcd
    down = fs_original // gcd
    return resample_poly(signal, up, down)


def _has_consecutive_pass(flags, k=STAGE1_GATE_K):
    count = 0
    for flag in flags:
        if bool(flag):
            count += 1
            if count >= k:
                return True
        else:
            count = 0
    return False


# =========================================================
# 单样本 → windows（worker 复用）
# =========================================================

def _extract_windows_from_sample(sample, min_duration_sec):
    win = int(round(STAGE1_PRIMITIVE_SEC * STAGE1_FS))
    stride = win
    try:
        ppg = load_ppg(sample)
    except Exception as e:
        print(f"读取失败 {sample.get('sample_name')}: {e}")
        return []

    ppg_fs = 25 if _is_25hz_sample(sample) else 100
    min_duration = int(min_duration_sec * ppg_fs)
    if len(ppg) < min_duration:
        return []

    ir = ppg[:, 0]
    ir_5hz = downsample_to_5hz(ir, ppg_fs, 5)
    rows = []
    for i in range(0, len(ir_5hz) - win + 1, stride):
        x = ir_5hz[i:i + win]
        # DC: min(邻均值) —— 对单点毛刺鲁棒
        if len(x) >= 2:
            neighbor_mean = (x[:-1] + x[1:]) / 2.0
            dc = float(np.min(neighbor_mean))
        else:
            dc = float(np.mean(x))
        # AC: 邻差 MAD —— 与 DC 同源（局部尺度），对单点抖动免疫
        # 旧: ac = float(np.std(x))  非鲁棒，单点冲击拉飞
        if len(x) >= 2:
            ac = float(np.median(np.abs(np.diff(x))))
        else:
            ac = 0.0
        acdc = float(ac / (np.abs(dc) + 1e-12))
        rows.append({
            "sample_name": sample["sample_name"],
            "h5_file": sample["h5_file"],
            "target": int(sample["target"]),
            "start_5hz": int(i),
            "window_sec": float(STAGE1_PRIMITIVE_SEC),
            "decision_sec": float(STAGE1_DECISION_SEC),
            "dc": dc,
            "ac": ac,
            "ac_dc_ratio": acdc,
        })
    return rows


def _worker_extract_sample(args_tuple):
    sample, min_duration_sec = args_tuple
    return _extract_windows_from_sample(sample, min_duration_sec)


def extract_stage1_windows(samples, min_duration_sec=DEFAULT_MIN_DURATION_SEC, n_workers=None):
    """样本级并行（n_workers=None 自动选 cpu-1；=1 单进程）。"""
    n_workers = resolve_n_workers(n_workers, n_items=len(samples))

    args_list = [(s, min_duration_sec) for s in samples]
    all_rows = []
    if n_workers == 1 or len(samples) <= 2:
        for a in args_list:
            all_rows.extend(_worker_extract_sample(a))
    else:
        pool_kwargs = {"max_workers": n_workers}
        mp_ctx = multiprocessing_context_from_env()
        if mp_ctx is not None:
            pool_kwargs["mp_context"] = mp_ctx
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            futures = {ex.submit(_worker_extract_sample, a): i for i, a in enumerate(args_list)}
            total = len(futures)
            for done_count, fut in enumerate(as_completed(futures), 1):
                sample_idx = futures[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    sample_name = samples[sample_idx].get("sample_name", f"idx={sample_idx}")
                    print(f"  [WARN] s02 worker failed sample={sample_name}: {e}", flush=True)
                    rows = []
                all_rows.extend(rows)
                if total >= 10 and (done_count % max(1, total // 10) == 0 or done_count == total):
                    print(f"  s02 progress: {done_count}/{total} samples", flush=True)
    return pd.DataFrame(all_rows)


# =========================================================
# 阈值评分：向量化
# =========================================================

def _prepare_arrays(df):
    """从 df 一次性提取评分所需的 numpy 数组。"""
    if len(df) == 0:
        return None

    dc = df["dc"].values.astype(np.float64)
    acdc = df["ac_dc_ratio"].values.astype(np.float64)
    target = df["target"].values.astype(np.int8)
    start_5hz = df["start_5hz"].values.astype(np.int64) if "start_5hz" in df.columns else np.arange(len(df))

    # sample_name → integer codes
    names = df["sample_name"].values
    uniq, inv = np.unique(names, return_inverse=True)
    n_samples = len(uniq)

    # 每个 sample 的 target（取该 sample 任意一行即可，因同 sample 内 target 恒定）
    sample_target = np.zeros(n_samples, dtype=np.int8)
    seen = np.zeros(n_samples, dtype=bool)
    for i, code in enumerate(inv):
        if not seen[code]:
            sample_target[code] = target[i]
            seen[code] = True

    return {
        "dc": dc,
        "acdc": acdc,
        "target": target,
        "sample_idx": inv.astype(np.int64),
        "start_5hz": start_5hz,
        "n_samples": int(n_samples),
        "sample_target": sample_target,
        "num_windows": int(len(df)),
    }


def _fast_eval_threshold(arrs, dc_th, acdc_th):
    """直接基于 arrs 的快速版本，返回与 eval_threshold 同结构 dict。"""
    if arrs is None:
        return {
            "dc_threshold": float(dc_th),
            "ac_dc_threshold": float(acdc_th),
            "window_target1_pass_rate": 0.0,
            "window_target0_pass_rate": 0.0,
            "window_target0_reject_rate": 0.0,
            "sample_target1_pass_rate": 0.0,
            "sample_target0_pass_rate": 0.0,
            "sample_target0_reject_rate": 0.0,
            "num_windows": 0,
            "num_samples": 0,
            "num_target1_samples": 0,
            "num_target0_samples": 0,
        }

    dc = arrs["dc"]
    acdc = arrs["acdc"]
    target = arrs["target"]
    sample_idx = arrs["sample_idx"]
    n_samples = arrs["n_samples"]
    sample_target = arrs["sample_target"]

    p_win = (dc > dc_th) & (acdc < acdc_th)

    t1_mask = target == 1
    t0_mask = target == 0
    n_t1 = int(t1_mask.sum())
    n_t0 = int(t0_mask.sum())
    w1 = float(p_win[t1_mask].mean()) if n_t1 > 0 else 0.0
    w0 = float(p_win[t0_mask].mean()) if n_t0 > 0 else 0.0

    # sample level: 1s primitive windows, 3 consecutive passes for a 3s decision.
    sample_pass = np.zeros(n_samples, dtype=bool)
    start_5hz = arrs.get("start_5hz", np.arange(len(p_win)))
    for code in range(n_samples):
        mask = sample_idx == code
        if not np.any(mask):
            continue
        order = np.argsort(start_5hz[mask])
        sample_pass[code] = _has_consecutive_pass(p_win[mask][order], STAGE1_GATE_K)

    n_s_t1 = int((sample_target == 1).sum())
    n_s_t0 = int((sample_target == 0).sum())
    s1 = float(sample_pass[sample_target == 1].mean()) if n_s_t1 > 0 else 0.0
    s0 = float(sample_pass[sample_target == 0].mean()) if n_s_t0 > 0 else 0.0

    return {
        "dc_threshold": float(dc_th),
        "ac_dc_threshold": float(acdc_th),

        "window_target1_pass_rate": float(w1),
        "window_target0_pass_rate": float(w0),
        "window_target0_reject_rate": float(1.0 - w0),

        "sample_target1_pass_rate": float(s1),
        "sample_target0_pass_rate": float(s0),
        "sample_target0_reject_rate": float(1.0 - s0),

        "num_windows": int(arrs["num_windows"]),
        "num_samples": int(n_samples),
        "num_target1_samples": int(n_s_t1),
        "num_target0_samples": int(n_s_t0),
    }


def eval_threshold(df, dc_th, acdc_th):
    """向后兼容：单次调用时内部组好 arrays 再算。"""
    arrs = _prepare_arrays(df)
    return _fast_eval_threshold(arrs, dc_th, acdc_th)


# =========================================================
# 由固定部署阈值派生训练门控阈值
# =========================================================

def make_train_threshold(deploy_dc, deploy_acdc, train_dc_ratio=0.90, train_acdc_margin=0.10):
    train_dc = float(deploy_dc * train_dc_ratio)
    train_acdc = float(deploy_acdc + train_acdc_margin)
    return {
        "dc_threshold": train_dc,
        "ac_dc_threshold": train_acdc,
        "rule": "宽松 Stage1，仅用于 second-stage 特征提取/筛选/训练，不用于最终部署评估",
        "train_dc_ratio": float(train_dc_ratio),
        "train_acdc_margin": float(train_acdc_margin),
    }


# =========================================================
# 可视化
# =========================================================

def plot_stage1_scatter(df_train, df_valid, deploy_dc, deploy_acdc, out_path):
    """画 DC vs AC/DC 散点图，不同 target 不同颜色，阈值虚线叠加。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scientific_figures import save_scientific_figure

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, df, title in [(ax1, df_train, "Train"), (ax2, df_valid, "Valid")]:
        if len(df) == 0:
            ax.set_title(f"{title} (no data)")
            continue

        t0 = df[df["target"] == 0]
        t1 = df[df["target"] == 1]

        ax.scatter(t0["dc"], t0["ac_dc_ratio"], c="#4C78A8", s=8, alpha=0.4,
                   label="target=0 (not-worn)", edgecolors="none")
        ax.scatter(t1["dc"], t1["ac_dc_ratio"], c="#E07B53", s=8, alpha=0.6,
                   label="target=1 (worn)", edgecolors="none")

        # 阈值线
        ax.axvline(x=deploy_dc, color="#2A9D8F", linestyle="--", linewidth=1.5,
                   label=f"dc={deploy_dc:.1e}")
        ax.axhline(y=deploy_acdc, color="#2A9D8F", linestyle="--", linewidth=1.5,
                   label=f"ac/dc={deploy_acdc:.4f}")

        # 标注通过区域
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_between([deploy_dc, xlim[1]], 0, deploy_acdc,
                        alpha=0.05, color="#2A9D8F")
        ax.text(deploy_dc * 1.05, deploy_acdc * 0.5, "PASS",
                fontsize=14, color="#2A9D8F", alpha=0.5, weight="bold")

        ax.set_xlabel("DC")
        ax.set_ylabel("AC/DC Ratio")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.suptitle(f"Stage1 IR Threshold: DC > {deploy_dc:.1e},  AC/DC < {deploy_acdc:.4f}",
                 fontsize=13)
    plt.tight_layout()
    source_rows = []
    for split_name, frame in (("train", df_train), ("valid", df_valid)):
        for row in frame[["target", "dc", "ac_dc_ratio"]].to_dict("records"):
            source_rows.append({"split": split_name, **row})
    artifact_dir = os.path.dirname(os.fspath(out_path))
    outputs = save_scientific_figure(
        fig, out_path, source_data=source_rows,
        core_conclusion="The fixed Stage1 IR gate is visualized against train and validation distributions without tuning on test.",
        panel_map={"a": "Train IR DC versus AC/DC.", "b": "Validation IR DC versus AC/DC."},
        inputs=[
            path for path in [
                os.path.join(artifact_dir, "stage1_train_windows.csv"),
                os.path.join(artifact_dir, "stage1_valid_windows.csv"),
            ] if os.path.isfile(path)
        ],
        split="train_valid",
        n_definition="one Stage1 decision window per source row",
        statistics={"threshold": "fixed deployment gate", "interval": "none"},
        reviewer_risks=["Window observations are correlated within samples."],
    )
    plt.close(fig)
    print(f"散点图已保存: {out_path}")
    return outputs


# =========================================================
# main
# =========================================================

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--min_duration_sec", type=float, default=DEFAULT_MIN_DURATION_SEC)
    parser.add_argument("--train_dc_ratio", type=float, default=0.90)
    parser.add_argument("--train_acdc_margin", type=float, default=0.10)
    parser.add_argument("--n_workers", type=int,
                        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
                        help="并行 worker 数")

    if args is None:
        args = parser.parse_args()

    split_path = os.path.join(args.artifact_dir, "splits.json")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    os.makedirs(args.artifact_dir, exist_ok=True)

    print("=" * 80)
    print(f"提取 Stage1 train/valid 窗口 (n_workers={args.n_workers})")
    print("=" * 80)

    df_train = extract_stage1_windows(split["train"], min_duration_sec=args.min_duration_sec,
                                       n_workers=args.n_workers)
    df_valid = extract_stage1_windows(split["valid"], min_duration_sec=args.min_duration_sec,
                                       n_workers=args.n_workers)

    df_train.to_csv(os.path.join(args.artifact_dir, "stage1_train_windows.csv"), index=False)
    df_valid.to_csv(os.path.join(args.artifact_dir, "stage1_valid_windows.csv"), index=False)

    print(f"train windows: {len(df_train)}")
    print(f"valid windows: {len(df_valid)}")

    if len(df_train) == 0:
        raise RuntimeError("Stage1 train 窗口为空。")

    # Fixed deployment thresholds; this stage does not perform threshold search.
    deploy_dc, deploy_acdc = resolve_fixed_deploy_thresholds(args)
    threshold_source = "fixed"

    train_metrics = eval_threshold(df_train, deploy_dc, deploy_acdc)
    valid_metrics = eval_threshold(df_valid, deploy_dc, deploy_acdc)

    train_threshold = make_train_threshold(
        deploy_dc, deploy_acdc,
        train_dc_ratio=args.train_dc_ratio,
        train_acdc_margin=args.train_acdc_margin
    )

    train_gate_metrics_on_train = eval_threshold(
        df_train, train_threshold["dc_threshold"], train_threshold["ac_dc_threshold"]
    )
    train_gate_metrics_on_valid = eval_threshold(
        df_valid, train_threshold["dc_threshold"], train_threshold["ac_dc_threshold"]
    )

    result = {
        "dc_threshold": float(deploy_dc),
        "ac_dc_threshold": float(deploy_acdc),

        "deploy_stage1_threshold": {
            "dc_threshold": float(deploy_dc),
            "ac_dc_threshold": float(deploy_acdc),
            "primitive_window_sec": float(STAGE1_PRIMITIVE_SEC),
            "decision_sec": float(STAGE1_DECISION_SEC),
            "decision_windows": int(STAGE1_GATE_K),
            "rule": "1s window: dc > dc_threshold and ac_dc_ratio < ac_dc_threshold",
            "sample_rule": "sample_pass = any 3 consecutive 1s windows pass",
            "target_requirement": "target=1 sample pass rate must be 100%",
            "threshold_source": threshold_source,
            "search_source": threshold_source,
        },

        "train_stage1_threshold": train_threshold,

        "deploy_train_metrics": train_metrics,
        "deploy_valid_metrics": valid_metrics,
        "train_gate_metrics_on_train": train_gate_metrics_on_train,
        "train_gate_metrics_on_valid": train_gate_metrics_on_valid,

        "notes": [
            "deploy_stage1_threshold 用于最终部署和端到端 test 评估",
            "train_stage1_threshold 用于第二阶段 train/valid 特征提取、特征筛选、模型训练",
            "test 不参与任何特征筛选、模型阈值选择或后处理搜参"
        ]
    }

    print("\n" + "=" * 80)
    print("Stage1 固定阈值结果")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 画散点图
    plot_path = os.path.join(args.artifact_dir, "stage1_scatter.png")
    plot_stage1_scatter(df_train, df_valid, deploy_dc, deploy_acdc, plot_path)

    out_path = os.path.join(args.artifact_dir, "stage1_threshold.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
