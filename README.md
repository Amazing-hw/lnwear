# Wearing Liveness Detection Pipeline

这是一个基于手表 PPG + ACC 信号的佩戴状态 / 活体检测项目。代码把流程拆成三个层级：

1. **Stage1：IR DC/ACDC 流式粗筛**  
   用 1s primitive window 和 3s decision gate 快速过滤明显非佩戴片段。

2. **Stage2：3s/1s 单窗模型**  
   对通过 Stage1 的窗口提取特征，用 XGBoost 输出逐窗口佩戴概率。

3. **Stage3：状态机后处理**  
   基于 Stage2 概率做平滑、阈值和连续窗口确认，控制非佩戴误触和佩戴响应时延。

项目的核心评估原则是：**先看 Stage2 窗口级能力，再看 Stage3 端到端状态机表现**。`s05` 固化单窗阈值，`s07` 只搜后处理状态机参数，两者不要混在同一个目标里。

## 代码结构

```text
new_codex/
  README.md
  SINGLE_WINDOW_98_FEATURE_OPTIMIZATION_PLAN.md

  s01_data_split.py
    扫描 H5，过滤 PPG shape 不符合要求的样本，按 sample/record 分层切分 train/valid/test。

  s02_ir_dc_threshold.py
    固定 Stage1 IR DC/ACDC 阈值为 3.6e6 / 0.35，导出 primitive window 统计和散点图。

  s03_extract_feature_pool.py
    提取 Stage2 特征池；3D 预切窗直接逐个使用已有 3s 窗口，连续时序才按 3s/1s 滑窗。
    默认跳过每条样本前 3 个 Stage2 窗口。
    默认 Stage2 不使用 IR，开启 --use_stage2_ir 后才保留真实 IR。

  s04_feature_selection.py
    做特征清洗、稳定性筛选、相关性/VIF、Permutation、SHAP、FP proxy 和候选子集搜索。

  s05_train_final_model.py
    训练 XGBoost，做概率校准和单窗阈值选择。
    支持复杂度受限搜参：显式控制特征数、树数、深度、正则、节点预算等。

  s06_deploy_eval.py
    模拟部署流式推理，输出 sample/window/state-machine 三类指标。
    导出逐窗口 NPZ 缓存、窗口错误分层报告、hard negatives 和部署产物。

  s07_postprocess_optimize.py
    只读取逐窗口 NPZ，不重跑模型；在 valid 上搜索状态机参数，并可在 test 上 replay 复验。

  s08_run_pipeline.py
    一键编排训练、搜参、评估和部署导出流程；商用 baseline 对比作为显式可选步骤。

  s09_commercial_compare.py
    当前项目方案与商业 AdaBoost baseline 对比。

  s10_generalization_audit.py
    只读取已有评估 artifacts，汇总窗口级、样本级、分层和 hard-negative 诊断，输出商用泛化审计报告。
```

## 数据要求

默认数据目录是 `dataset/`，可通过 `--dataset_dir` 指定。项目会扫描目录下的 `.h5` 文件。

旧版连续/预切窗 sample group 至少包含：

```text
sample_group/
  ppg          PPG 原始数据，当前筛选要求 shape[0] == 40
  target       样本标签，0=非佩戴，1=佩戴
  ppg_config   可选元数据；当前流程不再按 ppg_config 过滤
```

`ppg` 支持三种形态：

```text
(40, T)              连续时序，后续步骤按 window/stride 滑窗
(N_win, 40, T_win)   sample 内已经切好的窗口
record/window_group  一个 H5 含多条 record，每条 record 下按窗口 group 存放 ppg
```

当 `ppg` 是 3D 预切窗时，`s03/s06/s09` 会直接逐个使用已有窗口，不再在每个 sample 内二次滑窗；`s01` 仍按 sample/group 做 train/valid/test 切分。

当前新 H5 结构也支持一个 H5 文件里包含多条数据，每条数据下包含多个 3s 窗口 group。窗口 group 名称必须能从末尾解析出窗口编号和 label：

```text
record_a/
  anything_w0_1/
    ppg        (40, 300)
    acc        (3, 300)
  anything_w1_1/
    ppg        (40, 300)
    acc        (3, 300)
  anything_w20_1/
    ppg        (40, 300)
    acc        (3, 300)
```

解析规则是按 `_` 分割后的倒数第二段匹配 `w数字`，最后一段是 `label`。例如 `xxx_w20_1` 表示第 20 个窗口，label=1。H5 内部保存顺序不重要；`s01/s03/s06/s07` 会按 `w` 后的数字排序，`skip_initial_windows=3` 也是在排序后跳过前三个窗口。窗口是从原始信号按 3s 窗长、1s stride 预先截取的，因此后续不会重新滑窗。

切分是 sample 级，不是 window 级。训练、验证、测试应避免同一条采集、同一人或同一 H5 的相邻样本跨 split；如果你的 H5 语义代表同一次采集，建议重点检查 split 结果。

## 环境依赖

建议 Python 3.9+。

```bash
pip install numpy scipy pandas scikit-learn xgboost joblib h5py matplotlib pytest
```

常用依赖：

```text
numpy, scipy, pandas, scikit-learn, xgboost, joblib, h5py, matplotlib, pytest
```

## 默认关键参数

```text
Stage1 primitive window: 1s
Stage1 decision gate:    连续 3 个 primitive 通过后开启 Stage2
Stage2 window:           3s (也支持 5s，通过 --window_sec 5 切换)
Stage2 stride:           1s
skip_initial_windows:    3
use_stage2_ir:           false
postprocess latency:     first_worn_output_p95 <= 6s
```

Stage2 窗长选择：
- `--window_sec 3`（默认）：75 点@25Hz，响应快，适合实时佩戴检测
- `--window_sec 5`：125 点@25Hz，频域分辨率更高（0.2Hz），适合需要精确心率频段的场景

`use_stage2_ir=false` 只影响 Stage2：特征提取前把 IR 信号置零。Stage1 始终使用真实 IR 做 DC/ACDC 门控。

## 快速开始

主流程运行（含 XGBoost 模型搜参；不含商用 baseline 对比、逐窗 NPZ 缓存导出、s07 后处理搜参和 s10 泛化审计）：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

这条命令会一次性完成：数据切分、Stage1 固定阈值配置、Stage2 特征提取（预切窗直接使用，连续时序按 `window_sec/stride_sec` 滑窗）、特征筛选、候选特征子集搜索、XGBoost 复杂度受限搜参、test 端到端评估、部署产物和部署配方导出。

**完整搜参 + 后处理 + 部署导出**（一条命令，包含特征数搜索 + 状态机后处理搜参）：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts \
    --model_search_feature_counts "8,10,12,15" \
    --export_window_cache --optimize_postprocess
```

此命令依次完成：特征提取 → 特征筛选 → 对每个 k 值独立搜参选最优 → 导出 NPZ 缓存 → 后处理状态机搜参 → test 评估 → 部署产物导出。最终交付物可直接交给嵌入式同事。

legacy s06 状态机优化、逐窗 NPZ 缓存导出和 s07 后处理状态机搜参很耗时，默认不跑。需要时再显式打开：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --optimize
```

或：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --export_window_cache \
  --optimize_postprocess
```

泛化审计只读取已有评估产物，不重新训练模型。需要做商用部署诊断时显式打开：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --run_generalization_audit \
  --stop_after s10_audit
```

只打印命令，不执行：

```bash
python s08_run_pipeline.py --dry_run
```

推荐先看 dry-run，确认每一步参数都符合预期。

## 一键流水线

`s08_run_pipeline.py` 默认执行顺序：

```text
s01                 数据扫描与 train/valid/test 切分
s02                 Stage1 固定阈值配置
s03                 Stage2 特征池提取
s04                 特征筛选
s04_search          候选特征子集搜索
s05                 XGBoost 模型训练
s06_opt             旧版 s06 状态机优化参考（默认不跑；需 --optimize）
s06_eval            端到端部署评估
s10_audit           商用泛化审计（默认不跑；需 --run_generalization_audit）
s06_xpt             导出部署产物
s06_feat/s06_cb     导出部署特征脚本和部署配方
```

默认终点是 `s06_cb`，即跑到部署配方导出后停止；`s06_opt/s06_cache/s06_replay_cache/s07_post/s10_audit` 和商用方案对比都暂不纳入默认全流程。需要 legacy s06 状态机优化时显式加 `--optimize`；需要后处理搜参时显式加 `--export_window_cache --optimize_postprocess`；需要泛化审计时显式加 `--run_generalization_audit --stop_after s10_audit`；需要商业 baseline 对比时，再显式加 `--commercial_compare --stop_after s09_cmp`。

如果 `--stop_after` 直接指向可选步骤，例如 `s07_post`、`s09_cmp` 或 `s10_audit`，`s08` 会把该目标视为显式请求，并自动打开必要前置步骤；默认不带这些 stop target 时仍保持精简主流程。

推荐一条命令（默认 3s 窗口，自动多 k 搜参）：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --window_sec 3 \
  --stride_sec 1 \
  --skip_initial_windows 3 \
  --no-use_stage2_ir \
  --max_features 20 \
  --model_search \
  --model_search_feature_counts "10,12,15,18,20" \
  --max_model_nodes 500 \
  --split test
```

5s 窗口版本：
```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --window_sec 5 \
  --max_features 20 \
  --model_search_feature_counts "10,12,15,18,20"
```

## 复杂度受限模型搜参

模型搜参默认开启，同时默认启用**特征数量搜参**（测试 k ∈ {10,12,15,18,20} 选出最优特征数）。如果只想快速跑固定默认 XGBoost 参数和固定特征数，可以在 `s08_run_pipeline.py` 上显式加 `--no-model_search`。默认策略是 `staged_group_cv`：只把 `total_nodes <= max_model_nodes` 作为硬约束，在预算内用 train 内部 group CV 选择窗口级泛化性能最稳的模型和特征数。

### 搜索空间

| 参数 | 候选值 | 候选数 |
|---|---|---|
| 特征数量 k | 10, 12, 15, 18, 20 | 5 |
| n_estimators | 20, 25, 30, 35, 40, 45, 50, 55, 60 | 9 |
| max_depth | 2, 3, 4 | 3 |
| learning_rate | 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10 | 7 |
| min_child_weight | 10, 15, 20, 25, 30, 40, 50 | 7 |
| reg_lambda | 5, 8, 10, 12, 16, 20, 30 | 7 |
| reg_alpha | 0, 0.5, 1, 1.5, 2, 3 | 6 |
| subsample | 0.70, 0.75, 0.80, 0.85, 0.90 | 5 |
| colsample_bytree | 0.70, 0.75, 0.80, 0.85, 0.90 | 5 |
| **总计** | | **~8.9M × 5k** |

通过 `--model_search_feature_counts ""` 可禁用特征数搜参，仅用 `--max_features` 固定值。

一键流水线开启示例：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --max_features 20 \
  --model_search_feature_counts "10,12,15,18,20" \
  --max_model_nodes 500 \
  --model_search_strategy staged_group_cv \
  --model_search_max_candidates 600 \
  --model_search_stage2_top_k 80 \
  --model_search_cv_folds 3 \
  --model_search_cv_repeats 2 \
  --model_search_random_state 42 \
  --model_search_n_estimators 20,25,30,35,40,45,50,55,60 \
  --model_search_max_depth 2,3,4 \
  --model_search_learning_rate 0.025,0.03,0.04,0.05,0.06,0.08,0.10 \
  --model_search_min_child_weight 10,15,20,25,30,40,50 \
  --model_search_reg_lambda 5,8,10,12,16,20,30 \
  --model_search_reg_alpha 0,0.5,1,1.5,2,3 \
  --model_search_subsample 0.70,0.75,0.80,0.85,0.90 \
  --model_search_colsample_bytree 0.70,0.75,0.80,0.85,0.90
```

搜参选择策略：

```text
1. 对每个 k ∈ {10,12,15,18,20}，从 ranked_features.json 取 top-k 特征。
2. 每个 k 内，从参数空间按固定 random_state 抽样最多 600 个 XGBoost 候选。
3. Stage A 在 train 内部 group split 上预筛，保留 top 80。
4. Stage B 用 3 folds x 2 repeats 的 group CV 复评候选。
5. 在 max_model_nodes 预算内，选最优 (k × params) 组合。
6. 最优特征集 + 最优模型 → model_bundle.pkl → 自动传递到所有部署产物。
7. 多 k 搜参只用 train 内部 group-CV 做模型/特征数选择；valid 仍按 sample group 拆成 calibration split 和 threshold split，用于概率校准和单窗阈值固化。
```

完整候选结果写入 `artifacts/model_search_results.csv`。最佳参数、特征数、valid calibration/threshold split 和搜索稳定性摘要写入 `artifacts/final_model_config.json` 和 `artifacts/model_bundle.pkl`。

## 分步运行

### 1. 数据切分

```bash
python s01_data_split.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --valid_size 0.15 \
  --test_size 0.15 \
  --random_state 42
```

输出：

```text
artifacts/splits.json
```

### 2. Stage1 固定阈值配置

```bash
python s02_ir_dc_threshold.py \
  --artifact_dir artifacts
```

当前 Stage1 部署阈值固定，不做搜参：

```text
dc_threshold = 3.6e6
ac_dc_threshold = 0.35
```

输出：

```text
artifacts/stage1_threshold.json
artifacts/stage1_train_windows.csv
artifacts/stage1_valid_windows.csv
artifacts/stage1_scatter.png
```

### 3. Stage2 特征池提取

```bash
python s03_extract_feature_pool.py \
  --artifact_dir artifacts \
  --window_sec 3 \
  --stride_sec 1 \
  --skip_initial_windows 3 \
  --no-use_stage2_ir
```

输出：

```text
artifacts/feature_pool_train.csv
artifacts/feature_pool_valid.csv
artifacts/feature_pool_test.csv
```

如果要让 Stage2 使用 IR：

```bash
python s03_extract_feature_pool.py --artifact_dir artifacts --use_stage2_ir
```

注意：`s03/s05/s06/s09` 必须使用同一个 Stage2 IR 策略。

### 4. 特征筛选

```bash
python s04_feature_selection.py \
  --artifact_dir artifacts \
  --max_features 15 \
  --min_fold_auc 0.55 \
  --deployment_score_weight 0.25 \
  --fp_cost_weight 0.25 \
  --fp_proxy_recall_floor 0.95
```

如果运行日志长时间停在 `STEP 1/5: 数据清洗/VIF 开始`，优先用下面命令确认瓶颈是否来自 VIF：

```bash
python s04_feature_selection.py \
  --artifact_dir artifacts \
  --max_features 15 \
  --skip_vif
```

输出：

```text
artifacts/selected_features.json
```

如需候选子集搜索：

```bash
python s04_feature_selection.py \
  --artifact_dir artifacts \
  --max_features 15 \
  --run_subset_search \
  --subset_search_max_features 15
```

### 5. 最终模型训练

```bash
python s05_train_final_model.py \
  --artifact_dir artifacts \
  --window_sec 3 \
  --step_sec 1 \
  --no-use_stage2_ir \
  --threshold_objective fbeta \
  --threshold_beta 0.5 \
  --calibration_method isotonic
```

输出：

```text
artifacts/final_model.json
artifacts/final_model_config.json
artifacts/model_bundle.pkl
```

`s05` 的职责是训练模型、校准概率、选择并固化单窗 `window_model_threshold`。后处理状态机参数不在这里搜。

### 6. 导出逐窗口 NPZ

这一步默认不在 `s08_run_pipeline.py` 主流程中执行。需要做后处理参数搜参时，可以单独运行，或在 s08 中显式加 `--export_window_cache`。

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --split valid \
  --window_sec 3 \
  --stride_sec 1 \
  --skip_initial_windows 3 \
  --window_output_root window_outputs \
  --export_window_cache
```

输出目录：

```text
artifacts/window_outputs/valid/
```

每条样本一个 `.npz`，包含：

```text
sample_name
target
window_start_sec
window_end_sec
stage1_enabled
prob_raw
pred_raw
quality
ood_rate
mode
fallback
model_threshold
window_sec
stride_sec
skip_initial_windows
use_stage2_ir
model_fingerprint_json
feature_names_json
```

### 7. 后处理搜参与 test replay

这一步默认不在 `s08_run_pipeline.py` 主流程中执行。需要时可以单独运行，或在 s08 中显式加 `--export_window_cache --optimize_postprocess`。

如果已经完成主流程或已经开启 `model_search` 训练好了当前模型，后续只想基于当前 `artifacts` 导出逐窗口 NPZ 并做后处理搜参，推荐直接复用已有模型产物，不重跑 `s01-s05`：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts \
  --skip s01,s02,s03,s04,s04_search,s05 \
  --export_window_cache \
  --optimize_postprocess \
  --postprocess_split valid \
  --split test \
  --stop_after s07_post
```

这条命令的实际执行顺序是：

```text
s06_cache         使用当前 model_bundle.pkl 导出 valid 逐窗口 NPZ
s06_replay_cache  使用同一个模型导出 test 逐窗口 NPZ
s07_post          在 valid 缓存上搜索后处理参数，并在 test 缓存上 replay
```

它会复用当前目录下的模型和配置，例如：

```text
artifacts/model_bundle.pkl
artifacts/final_model_config.json
artifacts/stage1_threshold.json
artifacts/selected_features.json
```

输出包括：

```text
artifacts/window_outputs/valid/
artifacts/window_outputs/test/
artifacts/postprocess_opt/postprocess_optimized.json
artifacts/postprocess_opt/postprocess_search_results.csv
artifacts/postprocess_opt/postprocess_replay_valid_to_test.json
```

注意：如果训练主流程使用过非默认参数，例如 `--use_stage2_ir`、不同的 `--window_sec`、`--stride_sec` 或 `--skip_initial_windows`，这里必须保持一致。当前默认是 `--no-use_stage2_ir`、`window_sec=3`、`stride_sec=1`、`skip_initial_windows=3`。

先导出 valid 和 test 两个 split 的窗口缓存：

```bash
python s06_deploy_eval.py --artifact_dir artifacts --split valid --export_window_cache
python s06_deploy_eval.py --artifact_dir artifacts --split test --export_window_cache
```

再在 valid 上搜后处理参数，并在 test 上 replay：

```bash
python s07_postprocess_optimize.py \
  --artifact_dir artifacts \
  --split valid \
  --cache_root window_outputs \
  --replay_split test \
  --max_sample_fp_rate 0.02 \
  --max_false_worn_event_rate 0.02 \
  --max_first_worn_output_p95_sec 6.0 \
  --fp_cost 4.0
```

输出：

```text
artifacts/postprocess_opt/postprocess_optimized.json
artifacts/postprocess_opt/postprocess_search_results.csv
artifacts/postprocess_opt/postprocess_replay_valid_to_test.json
```

`postprocess_optimized.json` 是 valid 上选出的配置；`postprocess_replay_valid_to_test.json` 同时记录 valid selection metrics 和 test replay metrics。

### 8. 部署评估与错误分析

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --split test \
  --method state_machine \
  --window_sec 3 \
  --stride_sec 1 \
  --skip_initial_windows 3
```

主要输出：

```text
artifacts/end_to_end_eval_test_state_machine.json
artifacts/error_stratification_test_state_machine.json
artifacts/hard_negatives_test_state_machine.json
artifacts/window_error_analysis_test_state_machine.csv
artifacts/window_error_analysis_test_state_machine.json
artifacts/per_sample_summary.csv
```

### 9. 泛化审计

这一步只读取已有 artifacts，不重新训练模型，也不改变模型或阈值。建议在准备商用部署或发现 test 表现异常时运行：

```bash
python s10_generalization_audit.py \
  --artifact_dir artifacts \
  --split test \
  --method state_machine \
  --min_support 10
```

也可以通过 `s08` 在 `s06_eval` 后自动接上：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --run_generalization_audit \
  --stop_after s10_audit
```

输出：

```text
artifacts/generalization_audit/summary.json
artifacts/generalization_audit/summary.md
artifacts/generalization_audit/window_strata.csv
artifacts/generalization_audit/sample_strata.csv
artifacts/generalization_audit/action_items.csv
```

审计会汇总窗口级 accuracy/precision/recall/FP rate/FN rate、样本级 false-worn event rate、positive sample first-worn latency P50/P95，并按 `mode/h5_file/sample_name/record/window_index/time_bin/quality_bin/ood_bin` 分层；如果存在 `subject_id/device_id/session_id`，也会自动纳入分层。小样本分层会标记 `low_support`，避免被少量样本误导。

### 10. 商业 baseline 对比

```bash
python s09_commercial_compare.py \
  --artifact_dir artifacts \
  --split test \
  --method state_machine \
  --no-use_stage2_ir
```

输出：

```text
artifacts/commercial_compare/
```

## 结果怎么看

建议按这个顺序看：

1. **样本级数据是否合理**  
   看 `splits.json` 的样本数、正负比例、H5 分布。确认没有 window 级泄漏或同源样本跨 split。

2. **Stage1 是否过严或过松**  
   看 `stage1_threshold.json` 和 `stage1_scatter.png`。Stage1 过严会导致佩戴样本没机会进入 Stage2。

3. **Stage2 单窗能力**  
   看 `end_to_end_eval_*` 里的 `window_model_summary`。这是模型窗口级能力，不看状态机。

4. **泛化审计汇总**
   看 `generalization_audit/summary.md` 和 `generalization_audit/action_items.csv`。这里会把 Stage2 单窗、端到端状态机、hard negatives、mode、H5/record、时间位置、quality/OOD 分层放在一起，优先判断性能掉点来自模型、数据、阈值还是状态机。

5. **窗口错误分层报告**
   看 `window_error_analysis_*`：
   - `error_type`：FP/FN 集中情况。
   - `prob_bin`：高置信 FP 或低置信 FN。
   - `time_bin`：是否集中在前几个窗口或状态切换早期。
   - `mode`：是否某个硬件通道模式拖累。
   - `ood_bin` / `quality_bin`：是否来自低质量或分布外窗口。

6. **hard negatives**
   看 `hard_negatives_*`。非佩戴误识别为佩戴是敏感错误，应优先处理高置信 FP。

7. **后处理 valid/test 分离**
   看 `postprocess_replay_valid_to_test.json`。valid 上选参数，test 上只复验，不参与选择。

8. **商业 baseline 对比**
   看 `commercial_compare/`，确认当前方案相对 5s/1s AdaBoost baseline 的收益和代价。

## 商用性能优化路线

第一优先级是泛化审计。先运行 `s10_generalization_audit.py`，确认当前指标是否代表真实部署：看 FP/FN 是否集中在某个 `mode`、H5/record、人/设备/session、低质量/OOD、早期窗口或 hard negatives。没有这个结论时，不建议继续盲目扩大模型或状态机搜参。

第二优先级是 hard negative 数据闭环和特征增强。如果 FP 集中在 hard negatives，优先补负样本、增强 FP proxy 特征和质量/OOD 特征；如果 FN 集中在低质量或 OOD，优先补对应正样本场景，或加强 Stage1/quality gating。

第三优先级是校准、阈值和状态机约束优化。窗口级能力稳定后，再调整概率校准、单窗阈值、state-machine 连续窗口确认、释放条件和延迟约束，避免用状态机掩盖模型本身的问题。

第四优先级是端侧一致性和上线监控。部署前固定 golden vectors，确认端侧特征、阈值、IR 使用策略、窗口排序和 skip 前 3 个窗口与训练评估完全一致；上线后记录 mode/quality/OOD/hard-negative 触发分布，监控数据漂移。

## 98% 单窗目标的排查路径

如果 Stage2 raw window accuracy 没到 98%，不要先盲目扩大模型，建议按顺序排查：

1. 看 `window_error_analysis_*` 中 FP/FN 是否集中在某个 `mode`、时间段、概率段、OOD 段。
2. 如果 FP 高，优先看 hard negatives 和 FP proxy 特征；非佩戴误触比 FN 更敏感。
3. 如果 FN 高，检查正样本召回低的窗口是否来自 Stage1 过严、低质量窗口或特定特征缺失。
4. 如果 train 高 valid 低，优先检查 split 泄漏、特征漂移和过拟合特征。
5. 特征筛选稳定后，保留默认模型搜参做复杂度受限参数比较；需要快速排查流程时再临时加 `--no-model_search`。
6. 单窗达标后，再运行 `s07` 搜后处理，不要用状态机掩盖单窗模型问题。

更详细的特征优化计划见：

```text
SINGLE_WINDOW_98_FEATURE_OPTIMIZATION_PLAN.md
```

## 主要产物清单

```text
artifacts/
  splits.json
  stage1_threshold.json
  stage1_train_windows.csv
  stage1_valid_windows.csv
  stage1_scatter.png

  feature_pool_train.csv
  feature_pool_valid.csv
  feature_pool_test.csv
  selected_features.json
  ranked_features.json
  feature_diagnostics.csv

  final_model.json
  final_model_config.json
  model_bundle.pkl
  model_search_results.csv

  window_outputs/{split}/
    manifest.csv
    manifest.json
    *.npz

  postprocess_opt/
    postprocess_optimized.json
    postprocess_search_results.csv
    postprocess_replay_valid_to_test.json

  end_to_end_eval_{split}_{method}.json
  error_stratification_{split}_{method}.json
  hard_negatives_{split}_{method}.json
  window_error_analysis_{split}_{method}.csv
  window_error_analysis_{split}_{method}.json
  per_sample_summary.csv

  generalization_audit/
    summary.json
    summary.md
    window_strata.csv
    sample_strata.csv
    action_items.csv

  deploy_package/
  deploy_cookbook.json
  deploy_xgboost.json
  deploy_feature_extractor.py
  deploy_selected_feature_formulas.json
  golden_vectors.json

  feature_subset_search/
    subset_candidates.csv
    subset_eval_valid.csv
    subset_eval_summary.json
    best_subset_features.json

  commercial_compare/
  report_plots/
    s04_feature_selection_report.png
    s05_training_report.png
```

## 特征工程

当前候选特征池 ~170 个（含元数据），来自 3 个批次的扩展：

### 特征分组

| 信号源 | 通道 | 特征组 | 说明 |
|---|---|---|---|
| PPG Green | G_mean, G1, G2, G3 | 单通道 (DC/AC/FFT/autocorr) | 每通道 10 个基础特征 |
| PPG Green | G1/G2/G3 空间 | 不均衡/空间向量/相关性 | 120° 对称 LED 空间特征（16 个） |
| PPG Green | G1/G2/G3 per-ch | 通道间共识 (min/max/range/cv) | 倾斜方向检测（32 个） |
| PPG IR | IR | 单通道 + FFT harmonic + SNR | 红外脉搏通道（15 个） |
| PPG Ambient | Ambient | 单通道 + spectral + 波形 | 环境光抑制（8 个） |
| ACC | X/Y/Z 三轴 | 基础 (per-axis) + 幅值 + 震颤 | 加速度检测（19 个） |
| ACC-PPG | cross | 相干性 + BP 相关 | 运动-脉搏交叉（4 个） |
| 跨通道 | GREEN-IR-AMB | 相关性/比值/泄漏 | 光学串扰建模（14 个） |
| 信号质量 | GREEN/IR | 饱和度/削顶率 | 传感器接触质量（4 个） |
| 通用 | — | Hjorth/Entropy/Deriv/Temporal | 波形形态学（24 个） |
| 元数据 | — | SIG_LEN/SIG_SEC/mode 等 | 窗口元信息（4 个） |

### Tier 历史

| Tier | 新增内容 | 增量 |
|---|---|---|
| 原始 | 基础 PPG + 空间 + 跨通道 | ~76 |
| Tier 1 | ACC per-axis + 震颤检测 | ~16 |
| Tier 2 | IR FFT harmonic/SNR + AMB spectral + 信号质量 + ACC 姿态 | ~16 |
| Tier 3 | G1/G2/G3 per-channel + 通道间 consensus + dropout indicators | ~65 |
| 冗余剔除 | AC_RMS (保留 AC_MAD) + AUTO_CORR_LAG_SEC + Hjorth_Complexity 等 | -21 |

### 特征筛选

`s04_feature_selection.py` 负责从 ~170 候选特征中选出 `max_features` 个（默认 15）。流程：

```text
clean_features_by_train（缺失/低方差/高相关/VIF）
  → fast_group_preselection（按特征组预筛 Top4）
  → cross_validate_importance（5-fold Permutation + SHAP）
  → 按 deployment_score 排序选取 max_features
```

## 测试与验收

语法检查：

```bash
python -m py_compile s01_data_split.py s02_ir_dc_threshold.py s03_extract_feature_pool.py s04_feature_selection.py s05_train_final_model.py s06_deploy_eval.py s07_postprocess_optimize.py s08_run_pipeline.py s09_commercial_compare.py s10_generalization_audit.py
```

运行测试：

```bash
python -B -m pytest D:\wearing_liveness\new\tests .\test_model_search_config.py .\test_window_error_reports.py .\test_generalization_audit.py .\test_deploy_feature_extractor.py -q --rootdir D:\wearing_liveness\new\new_codex --basetemp .\.pytest_tmp_all -o cache_dir=.pytest_cache_all
```

如果直接从上级目录跑 pytest，在某些 Windows 权限环境里可能会因为默认 Temp 或 `.pytest_cache` 权限导致退出阶段异常。上面的命令把 rootdir、basetemp 和 cache_dir 都固定到当前项目目录，比较稳。

## 常见问题

### 找不到 H5

显式指定数据目录：

```bash
python s08_run_pipeline.py --dataset_dir D:\path\to\dataset --artifact_dir artifacts
```

### 后处理没有输入

先导出窗口缓存：

```bash
python s06_deploy_eval.py --artifact_dir artifacts --split valid --export_window_cache
```

### Stage2 IR 策略不一致

默认推荐 `--no-use_stage2_ir`。如果开启 IR，`s03/s05/s06/s09` 全链路都要使用 `--use_stage2_ir`。

### 终端中文显示乱码

脚本和 README 使用 UTF-8。PowerShell 可尝试：

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
```

## 部署注意事项

部署侧必须保持以下内容和训练评估一致：

```text
Stage1 dc/acdc thresholds
Stage1 1s primitive + 3s decision gate
Stage2 window_sec / stride_sec
skip_initial_windows
use_stage2_ir
selected feature order
fill values
clip bounds
XGBoost model
window_model_threshold
postprocess state machine params
```

如果 `use_stage2_ir=false`，部署侧进入 Stage2 特征提取前也要把 IR 信号置零；Stage1 仍继续使用真实 IR。`s08` 导出的 `deploy_feature_extractor.py` 会把 `model_bundle.pkl["meta"]` 中的 `USE_STAGE2_IR`、`DEFAULT_FS` 和 `DEFAULT_WINDOW_SEC` 固化到脚本常量里，工程侧不需要另外猜测窗长或 IR 使用策略。

部署端不会在导出时再裁剪 IR 相关特征。`deploy_feature_extractor.py` 的 `FEATURE_ORDER` 必须严格等于 `model_bundle.pkl["feature_names"]`，`FILL_VALUES`、`CLIP_BOUNDS` 和 `WINDOW_MODEL_THRESHOLD` 也都以 `model_bundle.pkl` 为准。若希望减少端侧特征数量，必须通过 `--model_search_feature_counts` 在训练/搜参阶段选出更小的特征集并重新训练模型，不能在部署导出阶段删除特征。

`s08_run_pipeline.py` 在默认部署导出流程中会校验 `model_bundle.pkl`、`deploy_feature_extractor.py`、`deploy_xgboost.json`、`deploy_cookbook.json` 和 `deploy_package/model_params.json` 的特征顺序、阈值、fill/clip 配置是否一致；若发现旧产物或部署文件漂移，会直接报错中断。

默认部署导出还会生成 `artifacts/golden_vectors.json`。它包含固定合成窗口的 `FEATURE_ORDER`、fill/clip 后特征向量、XGBoost 概率和窗口级阈值标签，用于工程侧 C/Rust/端侧实现做 golden-vector 对齐。上线前应要求端侧对同一批 golden vectors 输出完全一致或在约定浮点误差内一致。

窗口级工程化识别的核心推理产物是前两个文件；`golden_vectors.json` 是上线前做端侧一致性验收的校验产物：

```text
deploy_feature_extractor.py
final_model.json
golden_vectors.json
```

`deploy_feature_extractor.py` 是自包含脚本：它内联了所需的 Stage2 窗口级预处理、`use_stage2_ir` 策略和特征计算逻辑，不依赖 `s03_extract_feature_pool.py`、`s08_run_pipeline.py` 或其他训练/评估脚本。工程侧用它按 `FEATURE_ORDER` 生成特征向量，用 `final_model.json` 计算窗口佩戴概率，再用脚本内置的 `WINDOW_MODEL_THRESHOLD` 或 `classify_probability(probability)` 得到窗口级 0/1 识别结果。

### 训练与推理的预处理管道

训练侧（s05）和推理侧（s06 / deploy_feature_extractor.py）执行相同的预处理语义：

```text
训练侧 (s05):
  raw features (from s03 CSV)
    → ① clip_outliers(k=1.5, IQR-based)
        用 train 的 Q1-1.5*IQR / Q3+1.5*IQR 裁剪极端值
        valid 用 train 的裁剪边界，避免 valid IQR 泄漏
    → ② prepare_fill_values
        对裁剪后的 train 计算每个特征的中位数
    → ③ apply_fill: inf → NaN → fillna(train median)
    → ④ XGBoost 训练

推理侧 (s06 apply_preprocess):
  raw features (from s03 extract_feature_pool_from_window)
    → ① inf → NaN
    → ② fillna(fill_values)
    → ③ clip(clip_bounds)
    → ④ XGBoost 推理

推理侧 (deploy_feature_extractor.py, 独立部署脚本):
  raw window signals
    → ① 按 USE_STAGE2_IR 对 Stage2 IR 输入保留或置零
    → ② standalone feature extraction
    → ③ inf/None → fill_values (FILL_VALUES dict)
    → ④ clip(clip_bounds) (CLIP_BOUNDS dict)
    → ⑤ 返回特征向量
```

关键差异说明：

```text
训练侧先 clip 再 fill，推理侧先 fill 再 clip。
fill_values 是 train median（通常落在 clip 范围内），因此顺序差异在绝大多数情况下不影响结果。

训练侧的 fill_values 是在 clip_outliers 之后重新计算的（s05 prepare_fill_values），
而非复用 s04 clean_features_by_train 的 fill_values。
s04 的 fill_values 仅供诊断参考，实际部署使用的是 model_bundle.pkl 中的 fill_values。
```

### clip_outliers 参数

```text
方法: IQR-based 异常值裁剪
k = 1.5 (标准 Tukey 参数)
训练时从 train 学边界 → 用 train 边界裁剪 train 和 valid
valid 不自算 IQR（防止数据泄漏）
推理侧通过 clip_bounds JSON 应用相同的裁剪逻辑。
```

### s03 特征提取中的 NaN/inf 处理

```text
s03 extract_feature_pool_from_window 末尾：
  if v is None or not np.isfinite(v):
      feat[k] = 0.0

这是特征提取层面的"兜底"处理，所有无法计算的值（除零、log(0)、空信号等）
在写入特征池 CSV 之前就被替换为 0.0。

这意味着：
  - s04/s05 从 CSV 读取特征时已经看不到 NaN/inf（大部分已被 s03 消除）
  - 下游的 inf→NaN→fill 链是防御性的额外保护（处理 s03 未覆盖的边界情况）
  - 0.0 可能偏离特征分布的合理默认值，但会被 s05 clip_outliers 收敛到 IQR 下界

训练和推理使用相同的 s03 代码，因此这一行为在两端一致，
不构成训练-部署 gap。
```

### 部署产物的预处理信息

三个部署产物都包含完整的 fill + clip 信息：

```text
1. deploy_package/model_params.json
   {
     "fill_values": {...},          ← 每个特征的 train median
     "clip_bounds": {...},          ← 每个特征的 [lower, upper] IQR 边界
   }

2. deploy_xgboost.json
   {
     "fill_values": {...},
     "clip_bounds": {...},
     "preprocess_order": ["select feature_order", "fill NaN/inf with fill_values", "clip by clip_bounds"],
     ...
   }

3. deploy_feature_extractor.py（独立 Python 脚本）
   - FILL_VALUES = {...}            ← 硬编码字典
   - CLIP_BOUNDS = {...}            ← 硬编码字典
   - _clean_value() 先 fill 再 clip
   - 零外部依赖（仅 numpy + scipy）
```

### 状态机后处理参数

```text
alpha = 0.4          EMA 平滑系数（越小越平滑）
median_k = 1         中值滤波窗口（1=不滤波）
T_on = 0.75          从未佩戴→佩戴的 score 阈值
T_off = 0.35         从佩戴→未佩戴的 score 阈值
K_on = 5             连续超过 T_on 次数才触发佩戴
K_off = 3            连续低于 T_off 次数才触发未佩戴
cooldown_sec = 5     两次状态翻转之间的最小间隔（防止频繁抖动）
```
