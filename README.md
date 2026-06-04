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
    扫描 H5，过滤 ppg_config/channel 不符合要求的样本，按 sample 分层切分 train/valid/test。

  s02_ir_dc_threshold.py
    固定 Stage1 IR DC/ACDC 阈值为 3.6e6 / 0.35，导出 primitive window 统计和散点图。

  s03_extract_feature_pool.py
    按 3s/1s 滑窗提取 Stage2 特征池，默认跳过每条样本前 3 个 Stage2 窗口。
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
    一键编排 s01 到 s09，外层暴露主要实验参数。

  s09_commercial_compare.py
    当前项目方案与商业 AdaBoost baseline 对比。
```

## 数据要求

默认数据目录是 `dataset/`，可通过 `--dataset_dir` 指定。项目会扫描目录下的 `.h5` 文件。

当前代码至少依赖每个 sample group 中包含：

```text
sample_group/
  ppg          PPG 原始数据，当前筛选要求 shape[0] == 40
  target       样本标签，0=非佩戴，1=佩戴
  ppg_config   当前筛选要求 ppg_config == 65
```

`ppg` 同时支持两种形态：

```text
(40, T)              连续时序，后续步骤按 window/stride 滑窗
(N_win, 40, T_win)   sample 内已经切好的窗口
```

当 `ppg` 是 3D 预切窗时，`s03/s06/s09` 会直接逐个使用已有窗口，不再在每个 sample 内二次滑窗；`s01` 仍按 sample/group 做 train/valid/test 切分。

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
Stage2 window:           3s
Stage2 stride:           1s
skip_initial_windows:    3
use_stage2_ir:           false
postprocess latency:     first_worn_output_p95 <= 6s
```

`use_stage2_ir=false` 只影响 Stage2：特征提取前把 IR 信号置零。Stage1 始终使用真实 IR 做 DC/ACDC 门控。

## 快速开始

完整运行：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

只打印命令，不执行：

```bash
python s08_run_pipeline.py --dry_run --stop_after s09_cmp
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
s06_opt             旧版 s06 状态机优化参考
s06_cache           导出 valid 逐窗口 NPZ
s06_replay_cache    导出 replay split 逐窗口 NPZ，默认 test
s07_post            valid 搜后处理参数，并 replay test
s06_eval            端到端部署评估
s06_feat/s06_cb     导出部署特征脚本和部署配方
s09_cmp             商业方案对比
```

常用命令：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --window_sec 3 \
  --stride_sec 1 \
  --skip_initial_windows 3 \
  --no-use_stage2_ir \
  --max_features 15 \
  --postprocess_split valid \
  --split test \
  --commercial_split test
```

## 复杂度受限模型搜参

模型搜参默认关闭。建议在特征筛选方案稳定后再开启，用来比较“更小/更稳的 XGBoost 参数”，而不是靠增大模型复杂度冲指标。

一键流水线开启示例：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --max_features 12 \
  --model_search \
  --max_model_nodes 260 \
  --model_search_fp_cost 2.0 \
  --model_search_size_cost 0.1 \
  --model_search_valid_fraction 0.5 \
  --model_search_n_estimators 20,30 \
  --model_search_max_depth 2,3 \
  --model_search_learning_rate 0.03,0.05 \
  --model_search_min_child_weight 30,50 \
  --model_search_reg_lambda 10,20 \
  --model_search_reg_alpha 1,2 \
  --model_search_subsample 0.7,0.8 \
  --model_search_colsample_bytree 0.7,0.8
```

搜参评分：

```text
score = window_accuracy - fp_cost * window_fp_rate - size_cost * size_ratio
size_ratio = total_nodes / max_model_nodes
```

超过 `--max_model_nodes` 的候选会被过滤。完整候选结果写入：

```text
artifacts/model_search_results.csv
```

最佳参数、节点数和模型复杂度信息写入：

```text
artifacts/final_model_config.json
artifacts/model_bundle.pkl
```

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
  --skip_initial_windows 3 \
  --export_window_cache
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

### 9. 商业 baseline 对比

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

4. **窗口错误分层报告**  
   看 `window_error_analysis_*`：
   - `error_type`：FP/FN 集中情况。
   - `prob_bin`：高置信 FP 或低置信 FN。
   - `time_bin`：是否集中在前几个窗口或状态切换早期。
   - `mode`：是否某个硬件通道模式拖累。
   - `ood_bin` / `quality_bin`：是否来自低质量或分布外窗口。

5. **hard negatives**  
   看 `hard_negatives_*`。非佩戴误识别为佩戴是敏感错误，应优先处理高置信 FP。

6. **后处理 valid/test 分离**  
   看 `postprocess_replay_valid_to_test.json`。valid 上选参数，test 上只复验，不参与选择。

7. **商业 baseline 对比**  
   看 `commercial_compare/`，确认当前方案相对 5s/1s AdaBoost baseline 的收益和代价。

## 98% 单窗目标的排查路径

如果 Stage2 raw window accuracy 没到 98%，不要先盲目扩大模型，建议按顺序排查：

1. 看 `window_error_analysis_*` 中 FP/FN 是否集中在某个 `mode`、时间段、概率段、OOD 段。
2. 如果 FP 高，优先看 hard negatives 和 FP proxy 特征；非佩戴误触比 FN 更敏感。
3. 如果 FN 高，检查正样本召回低的窗口是否来自 Stage1 过严、低质量窗口或特定特征缺失。
4. 如果 train 高 valid 低，优先检查 split 泄漏、特征漂移和过拟合特征。
5. 特征筛选稳定后，再开启 `--model_search` 做复杂度受限参数比较。
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

  deploy_package/
  deploy_cookbook.json
  deploy_xgboost.json
  deploy_feature_extractor.py

  commercial_compare/
  report_plots/
```

## 测试与验收

语法检查：

```bash
python -m py_compile s01_data_split.py s02_ir_dc_threshold.py s03_extract_feature_pool.py s04_feature_selection.py s05_train_final_model.py s06_deploy_eval.py s07_postprocess_optimize.py s08_run_pipeline.py s09_commercial_compare.py
```

运行测试：

```bash
python -B -m pytest D:\wearing_liveness\new\tests .\test_model_search_config.py .\test_window_error_reports.py -q --rootdir D:\wearing_liveness\new\new_codex --basetemp .\.pytest_tmp_all -o cache_dir=.pytest_cache_all
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
XGBoost model
window_model_threshold
postprocess state machine params
```

如果 `use_stage2_ir=false`，部署侧进入 Stage2 特征提取前也要把 IR 信号置零；Stage1 仍继续使用真实 IR。
