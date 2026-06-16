# Wearing Liveness Detection Pipeline

## Staged E2E 自动优化

用于把“窗口准确率”“FP/hard-negative 风险”和“端到端后处理”拆成三个互不覆盖的自动化阶段：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --staged_e2e_optimize \
  --model_search_full_top_k 2
```

产物会分别写入：

```text
artifacts/staged_e2e/01_accuracy_first
artifacts/staged_e2e/02_fp_safe_hard_negative
artifacts/staged_e2e/03_e2e_postprocess
```

部署配方导出会额外生成 `deploy_performance_profile.json`，记录最终 `FEATURE_ORDER`、FFT source 数量、可靠性特征使用情况、XGBoost 树/节点数和端侧中间量复用建议。

## 当前默认流程准则

- 默认 Stage2 窗长是 `5s`、stride 是 `1s`；如需兼容旧 3s 窗口，显式传 `--window_sec 3`。
- 对 grouped/pre-windowed H5，窗口会按名称里的 `w数字` 排序，排序后跳过前三个窗口；预切窗口直接使用，不会二次滑窗。
- IR 只用于 Stage1 DC/ACDC 门控；Stage2 特征筛选、XGBoost、s06/s07、部署脚本和 golden vectors 都只允许环境光、三绿光和 ACC 特征。
- 默认主命令会做特征数量搜索 `8,10,12,15,18` 和 XGBoost 参数搜索，不导出 NPZ、不跑 s07 后处理搜参、不跑 s10 审计：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

### 运行场景速查

先按目标选择命令，再用 `--dry_run` 预览真实执行步骤。除特别说明外，默认都是 Stage2 `5s` 窗口、`1s` stride、Stage2 不使用 IR、特征数量搜索 `8,10,12,15,18`、XGBoost `staged_group_cv` 搜参、最大节点数 `500`。默认运行预算档是 `--runtime_profile balanced`：最多抽样 `180` 个 XGBoost 候选、Stage B 保留 `24` 个候选、local swap 最多 `8` 个候选。

1. **只检查命令是否会串通，不执行训练**

   适用条件：首次改参数、换数据目录、准备长时间训练前。

   ```bash
   python s08_run_pipeline.py \
     --dataset_dir dataset \
     --artifact_dir artifacts \
     --dry_run
   ```

   作用：只打印 `s01` 到默认终点 `s06_cb` 的命令，不读取 H5、不写训练产物。

2. **默认部署候选训练与导出**

   适用条件：已有原始 H5 数据，希望得到一套可部署模型、部署特征脚本和部署配方。

   ```bash
   python s08_run_pipeline.py \
     --dataset_dir dataset \
     --artifact_dir artifacts
   ```

   自动步骤：`s01 -> s02 -> s03 -> s04 -> s04_search -> s05 -> s06_eval -> s06_xpt/s06_feat/s06_cb`。默认不跑 NPZ 缓存、`s07` 后处理搜参、`s10` 审计和 `s09` 商业对比。流程结束会输出 `[RUNTIME] step elapsed summary`，用于判断主要耗时阶段。

3. **优先拉高 Stage2 单窗口准确率**

   适用条件：当前主要问题是单窗口模型能力下降，先排除特征质量、特征数量和模型参数问题。

   ```bash
   python s08_run_pipeline.py \
     --dataset_dir dataset \
     --artifact_dir artifacts \
     --accuracy_first_optimize \
     --model_search_full_top_k 2 \
     --stop_after s05
   ```

   自动条件：阈值目标使用 `accuracy`；不自动打开 hard negative、NPZ、`s07` 后处理或 `s10` 审计；最终关注 `model_search_results.csv`、`final_model_config.json` 和 `model_bundle.pkl`。

4. **高 FP 风险 / object-worn hard negative 训练**

   适用条件：主要风险是非人体佩戴、物体表面反光、松戴等负样本被识别为佩戴。

   ```bash
   python s08_run_pipeline.py \
     --dataset_dir dataset \
     --artifact_dir artifacts \
     --hard_negative_optimize \
     --hard_negative_weight 3.0 \
     --hard_negative_top_percentile 0.1 \
     --stop_after s05
   ```

   自动条件：启用 train-only OOF hard negative mining；用于训练层面压低高风险 FP。只想先看模型改善时停在 `s05`，暂不跑后处理。

5. **完整 hard negative + NPZ + 后处理闭环**

   适用条件：模型已经有可用窗口能力，需要同时考虑端到端准确率、误识别率和响应时延。

   ```bash
   python s08_run_pipeline.py \
     --dataset_dir dataset \
     --artifact_dir artifacts \
     --hard_negative_optimize \
     --export_window_cache \
     --optimize_postprocess \
     --postprocess_split valid \
     --split test \
     --postprocess_search_budget 2000 \
     --stop_after s07_post
   ```

   自动步骤：先导出 valid/test 逐窗 NPZ，再用 valid 搜 `s07` 状态机参数，并在 test 上 replay。`--postprocess_search_budget` 用于限制后处理候选数量，避免搜参时间过长。

6. **只基于已有模型做后处理搜参**

   适用条件：`model_bundle.pkl` 已存在，不想重新跑数据切分、特征提取、筛选和训练。

   ```bash
   python s08_run_pipeline.py \
     --dataset_dir dataset \
     --artifact_dir artifacts \
     --skip s01,s02,s03,s04,s04_search,s05 \
     --export_window_cache \
     --optimize_postprocess \
     --postprocess_split valid \
     --split test \
     --postprocess_search_budget 2000 \
     --stop_after s07_post
   ```

   前置条件：`artifacts/model_bundle.pkl`、特征配置和 split 产物必须已经存在。

7. **三阶段 staged E2E 自动优化**

   适用条件：需要同时保留三套目标明确的实验产物，便于比较单窗准确率、FP 安全性和最终后处理表现。

   ```bash
   python s08_run_pipeline.py \
     --dataset_dir dataset \
     --artifact_dir artifacts \
     --staged_e2e_optimize \
     --model_search_full_top_k 2
   ```

   自动产物：`artifacts/staged_e2e/01_accuracy_first`、`02_fp_safe_hard_negative`、`03_e2e_postprocess`。三阶段不要混用同一个目录，避免目标函数覆盖。

8. **泛化审计**

   适用条件：已有 `s06` 评估产物，需要按人群、样本类型、窗口质量等分层看弱点。

   ```bash
   python s08_run_pipeline.py \
     --dataset_dir dataset \
     --artifact_dir artifacts \
     --run_generalization_audit \
     --stop_after s10_audit
   ```

   输出：`s10` 只读取已有评估/窗口产物，不重新训练模型。

9. **商业 baseline 对比**

   适用条件：已有项目模型和可评估 split，需要比较商业阈值/AdaBoost baseline 与当前 XGBoost 部署链路。

   ```bash
   python s08_run_pipeline.py \
     --dataset_dir dataset \
     --artifact_dir artifacts \
     --commercial_compare \
     --stop_after s09_cmp
   ```

   关键条件：`s09` 会使用同一 `window_sec/stride_sec/skip_initial_windows` 对项目模型做对比；如需保留逐窗概率细节，加 `--keep_window_probs`。

10. **兼容旧 3s 窗口实验**

    适用条件：只用于响应速度或历史结果对照；不要和 5s 产物混在同一部署结论里。

    ```bash
    python s08_run_pipeline.py \
      --dataset_dir dataset \
      --artifact_dir artifacts_3s \
      --window_sec 3 \
      --model_search_feature_counts "8,10,12,15,18"
    ```

    约束：训练、评估、NPZ、后处理和部署导出必须使用同一个 `window_sec`。最终生产推荐仍以 5s 作为默认候选。

11. **固定特征数量做快速对照**

    适用条件：已经决定部署特征数，例如强制不超过 15 个或 18 个，只想验证模型参数。

    ```bash
    python s08_run_pipeline.py \
      --dataset_dir dataset \
      --artifact_dir artifacts \
      --max_features 15 \
      --model_search_feature_counts 15
    ```

    约束：`--model_search_feature_counts` 传单个值时应与 `--max_features` 一致，否则容易误读最终入选特征数。

### 部署友好特征池

s03 已去 scipy 化（FIR 带通 + numpy median + np.correlate），所有特征计算均为 C 可移植运算。无部署过滤器——全部特征参与筛选。训练与部署通过 s03 import 使用完全相同代码，数值一致。

- 只跑到模型搜参 + train-only hard negative 回流训练，不导出 NPZ、不跑后处理：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --hard_negative_optimize \
  --stop_after s05
```

- 完整 hard negative 闭环、NPZ、s07 后处理搜参和 s10 审计：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --hard_negative_optimize
```

这是一个基于手表 PPG + ACC 信号的佩戴状态 / 活体检测项目。代码把流程拆成三个层级：

1. **Stage1：IR DC/ACDC 流式粗筛**  
   用 1s primitive window 和 3s decision gate 快速过滤明显非佩戴片段。

2. **Stage2：5s/1s 单窗模型**  
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
    固定 Stage1 IR DC/ACDC 阈值为 1.5e6 / 1.0，导出 primitive window 统计和散点图。

  s03_extract_feature_pool.py
    提取 Stage2 特征池；3D 预切窗直接逐个使用已有窗口，连续时序默认按 5s/1s 滑窗。
    默认跳过每条样本前 3 个 Stage2 窗口。
    Stage2 特征池固定不包含 IR 派生特征；IR 只用于 Stage1 DC/ACDC 门控。

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

当前新 H5 结构也支持一个 H5 文件里包含多条数据，每条数据下包含多个预切窗口 group。窗口 group 名称必须能从末尾解析出窗口编号和 label：

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

解析规则是按 `_` 分割后的倒数第二段匹配 `w数字`，最后一段是 `label`。例如 `xxx_w20_1` 表示第 20 个窗口，label=1。H5 内部保存顺序不重要；`s01/s03/s06/s07` 会按 `w` 后的数字排序，`skip_initial_windows=3` 也是在排序后跳过前三个窗口。预切窗口后续不会重新滑窗；连续时序默认才按 5s 窗长、1s stride 截取。

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
Stage2 window:           5s (也支持 3s，通过 --window_sec 3 切换)
Stage2 stride:           1s
skip_initial_windows:    3
use_stage2_ir:           false
postprocess latency:     first_worn_output_p95 <= 6s
```

Stage2 窗长选择：
- `--window_sec 5`（默认）：125 点@25Hz，频域分辨率更高（0.2Hz），适合需要精确心率频段的场景
- `--window_sec 3`：75 点@25Hz，响应快，适合实时佩戴检测

Stage1 始终使用真实 IR 做 DC/ACDC 门控；Stage2 从特征池源头只保留环境光、绿光和 ACC 特征。`--use_stage2_ir` 仅保留为旧命令兼容项，不会让 IR 派生特征进入特征筛选、模型训练或部署导出。

## 快速开始

主流程运行（含 XGBoost 模型搜参；不含商用 baseline 对比、逐窗 NPZ 缓存导出、s07 后处理搜参和 s10 泛化审计）：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

这条命令会一次性完成：数据切分、Stage1 固定阈值配置、Stage2 特征提取（预切窗直接使用，连续时序按 `window_sec/stride_sec` 滑窗）、特征筛选、候选特征子集搜索、XGBoost 复杂度受限搜参、test 端到端评估、部署产物和部署配方导出。

**完整搜参 + 后处理 + 部署导出**（一条命令，包含特征数搜索 + 状态机后处理搜参）：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts \
    --full_optimize
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

推荐一条命令（默认 5s 窗口，自动多 k 搜参）：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --max_features 18 \
  --model_search_feature_counts "8,10,12,15,18" \
  --feature_search_local_swap \
  --feature_search_swap_tail_size 3 \
  --feature_search_swap_pool_size 8 \
      --feature_search_swap_max_candidates 8 \
  --max_model_nodes 500 \
  --split test
```

3s 窗口版本：
```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --window_sec 3 \
  --max_features 18 \
  --model_search_feature_counts "8,10,12,15,18"
```

## 复杂度受限模型搜参

模型搜参默认开启，同时默认启用**特征数量搜参**。s08 先用默认参数快速评估每个 k（无 model_search），选出最优 k 后再对该 k 做完整模型搜参。这比每个 k 都跑完整搜参减少约 80% 耗时。

### 运行预算档与日志策略

`s08` 提供三档运行预算，优先通过 `--runtime_profile` 控制默认耗时；单项参数仍可显式覆盖。

| 预算档 | 使用场景 | max_candidates | Stage B top_k | local swap candidates | postprocess budget |
|---|---|---:|---:|---:|---:|
| `fast` | 快速冒烟、参数连通性、粗排查 | 120 | 16 | 4 | 240 |
| `balanced`（默认） | 日常训练与部署候选 | 180 | 24 | 8 | 240 |
| `thorough` | 最终验收、追求极限准确率 | 360 | 48 | 12 | 2000 |

控制台日志默认走 compact 思路：每个阶段只打印入口命令、阶段结果和耗时摘要；XGBoost 搜参明细写入 `model_search_results.csv`，最终配置写入 `final_model_config.json`。`s05` 的 IQR 异常值裁剪会预先为候选特征学习一次 train clip bounds，并在多 k/local-swap 中复用；控制台只打印裁剪摘要和 top-N 异常特征，不再逐次展开所有特征。

### 搜索空间

| 参数 | 候选值 | 候选数 |
|---|---|---|
| 特征数量 k | 8, 10, 12, 15, 18 | 5 |
| n_estimators | 20, 25, 30, 35, 40, 45, 50, 55, 60 | 9 |
| max_depth | 2, 3, 4 | 3 |
| learning_rate | 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10 | 7 |
| min_child_weight | 10, 15, 20, 25, 30, 40, 50 | 7 |
| reg_lambda | 5, 8, 10, 12, 16, 20, 30 | 7 |
| reg_alpha | 0, 0.5, 1, 1.5, 2, 3 | 6 |
| subsample | 0.70, 0.75, 0.80, 0.85, 0.90 | 5 |
| colsample_bytree | 0.70, 0.75, 0.80, 0.85, 0.90 | 5 |
| **总计** | | **~8.9M × 5k** |

若要固定特征数量，传单个 k 与 `--max_features` 一致即可：`--max_features 15 --model_search_feature_counts 15`。

一键流水线开启示例：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --runtime_profile balanced \
  --max_features 18 \
  --model_search_feature_counts "8,10,12,15,18" \
  --max_model_nodes 500 \
  --model_search_strategy staged_group_cv \
  --model_search_max_candidates 180 \
  --model_search_stage2_top_k 24 \
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
1. 对每个 k ∈ {8,10,12,15,18}，从 ranked_features.json 取 top-k 特征。
2. 每个 k 内，从参数空间按固定 random_state 抽样候选；balanced 默认最多 180 个，thorough 为 360 个。
3. Stage A 在 train 内部 group split 上预筛；balanced 默认保留 top 24，thorough 保留 top 48。
4. Stage B 用 3 folds x 2 repeats 的 group CV 复评候选。
5. 在 max_model_nodes 预算内，选最优 (k × params) 组合。
6. 最优特征集 + 最优模型 → model_bundle.pkl → 自动传递到所有部署产物。
7. 多 k 搜参只用 train 内部 group-CV 做模型/特征数选择；valid 仍按 sample group 拆成 calibration split 和 threshold split，用于概率校准和单窗阈值固化。
```

完整候选结果写入 `artifacts/model_search_results.csv`。最佳参数、特征数、valid calibration/threshold split 和搜索稳定性摘要写入 `artifacts/final_model_config.json` 和 `artifacts/model_bundle.pkl`。

最终验收或需要最大化 98% 单窗目标时，可显式恢复重预算：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_thorough \
  --runtime_profile thorough \
  --model_search_full_top_k 2
```

### 窗口准确率优先优化命令

如果当前目标是先提升 Stage2 单窗口 window accuracy，而不是优先压低误报或自动做 hard-negative 回流，推荐使用下面这条命令：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --accuracy_first_optimize \
  --model_search_strategy staged_group_cv \
  --model_search_feature_counts "8,10,12,15,18" \
  --model_search_full_top_k 2 \
  --stop_after s05
```

这条命令会在部署友好的 Stage2 特征池内做特征数量搜索、候选子集搜索和 XGBoost 参数搜索。特征数量仍限制在 `8,10,12,15,18`，排序以 group-aware CV 的窗口准确率为第一目标，CV 方差、FP rate、特征数和节点数只作为后续 tie-breaker。`--model_search_full_top_k 2` 会对 quick 评估前 2 个特征数做完整搜参，降低 quick 阶段误选 k 的风险；最终仍以最后一次完整搜参产物作为部署候选。新增的 top2 绿光形态、5s 分段稳定度、环境光/绿光泄漏稳定度和 ACC×绿光耦合特征都是普通标量特征，端侧只需要按最终 `FEATURE_ORDER` 计算入选项。

它和 `--hard_negative_optimize` 的区别是：`--accuracy_first_optimize` 会把阈值目标切到 `accuracy`，聚焦窗口准确率，不自动导出 NPZ、不跑 s07 后处理搜参、不跑 s10 审计，也不自动提高 hard negative 权重；`--hard_negative_optimize` 面向非人体佩戴/物体佩戴误报风险，会自动启用 train-only OOF hard negative mining、严格 precision/FP 约束、窗口缓存、后处理搜参和泛化审计。

### P0/P1 准确率提升闭环

当前 Stage2 特征池采用“候选变丰富、最终仍少特征”的策略：s03 会生成更多部署友好的三绿光、环境光泄漏和 ACC×绿光标量特征，s04/s05 再通过候选子集搜索、特征数量搜索和 XGBoost group-CV 决定最终保留哪些特征。最终部署仍只计算 `model_bundle.pkl["feature_names"]` 里入选的少数特征，不会在部署阶段临时裁剪。

新增候选重点包括：
- 三绿光空间/排名稳定：`G_TOP1_TO_TOP2_AC_RATIO`、`G_TOP2_RANK_STABILITY`、`G_TOP2_SWITCH_RATE`、`G_SPATIAL_VMAG_RANGE`。
- 环境光泄漏稳定：`GREEN_AMB_SEG_CORR_RANGE`、`GREEN_AMB_LEAK_STABILITY`。
- ACC 与绿光低成本耦合：`ACC_STILL_GREEN_MISMATCH`、`ACC_TO_GTOP2_AC_RATIO`、`ACC_STILL_X_GREEN_STABILITY`、`ACC_DIFF_TO_GTOP2_DIFF_RATIO`。

如果你的主要风险是“物体佩戴/非人体佩戴被识别为佩戴”，优先运行：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --hard_negative_optimize
```

这条命令会在 train split 内用 group-aware OOF 预测挖 hard negatives，并对这些负窗口加权重训；valid/test 只用于阈值、后处理和评估，不会回流训练。若数据或文件名中包含 `negative_type=object_worn`、`scene_type=object_worn*`、`subject_type=non_human` 等字段或标记，`hard_negative_mining_train.csv` 会保留这些上下文，`final_model_config.json` 会记录 `object_worn_hard_negatives` 和 `object_worn_fraction`，s10 审计会给出 `object_worn_false_positive_cluster` 的 P0 动作项。

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
dc_threshold = 1.5e6
ac_dc_threshold = 1.0
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
  --window_sec 5 \
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

不要再为 Stage2 打开 IR。`--use_stage2_ir` 仅为旧命令兼容保留；Stage2 特征池始终只包含环境光、绿光和 ACC。

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
  --window_sec 5 \
  --step_sec 1 \
  --no-use_stage2_ir \
  --threshold_objective accuracy \
  --calibration_method isotonic
```

默认训练目标优先保证单窗口 accuracy；如果真实运行中主要错误来自“非人体佩戴在物体上”这类高舆情风险 FP，再显式把窗口阈值切到 precision 约束模式：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --threshold_objective precision_constrained \
  --threshold_min_precision 0.995 \
  --model_search_fp_cost 4.0 \
  --full_optimize \
  --max_sample_fp_rate 0.005 \
  --max_false_worn_event_rate 0.005 \
  --postprocess_fp_cost 8.0
```

这会让 s05 的窗口阈值选择优先满足高 precision，再让 s07 后处理搜参更强地惩罚 false-worn event。若召回下降，先检查这些 object-worn hard negatives 是否已充分进入 train/valid，而不是降低 FP 约束。

### Stage2 训练分布策略：不要盲目全量训练

当前部署链路是两阶段：Stage1 先用 IR DC/ACDC gate 决定哪些窗口有机会进入 Stage2，Stage2 XGBoost 只在这些窗口上输出佩戴概率。因此，Stage2 的默认训练分布应尽量贴近真实部署输入：**Stage1 通过或接近通过的窗口**。不建议为了“看起来覆盖全量数据”而直接把所有明显 Stage1 失败的 easy negatives 混入 Stage2 训练，否则会带来训练/部署分布错位：模型可能学到很多部署时根本看不到的简单负样本，窗口级 accuracy 变好，但对 object-worn、松戴、物体表面反光这类真正会穿过 Stage1 的 hard negatives 反而不够敏感。

更推荐的策略是：

```text
Stage2 final training set =
  positives that pass the normal Stage1/skip policy
  + negatives that pass Stage1
  + object-worn / hard-negative windows that pass or nearly pass Stage1
```

也就是说，object-worn 不应该作为普通负样本“随机混进去”，而应该作为高风险 hard-negative 子类被固定纳入 train/valid/test，并在泛化审计里单独看 FP rate / false-worn event rate。如果数据里能提供 `negative_type=object_worn`、`scene_type=object_worn` 或类似字段，建议按这个字段做分层审计；如果暂时没有字段，至少在 `sample_name` / H5 分组命名中保留可解析标记，方便 s10 和人工复盘。

什么时候可以考虑“全量数据”？

- 用于 Stage1 阈值审计：可以看全量 negatives 中哪些会穿过 Stage1。
- 用于 hard-negative mining：从全量 negatives 里挖出 Stage1 pass / near-pass 的 object-worn 窗口，再加入 Stage2 训练。
- 不建议直接用于最终 Stage2 模型训练，除非有显式采样/加权，保证 easy negatives 不淹没 object-worn hard negatives。

### Hard negative 数据闭环与命令

这里的 hard negative 指真实标签为未佩戴/非人体佩戴，但 Stage2 或状态机给出高佩戴概率、甚至输出佩戴事件的样本。当前最需要优先处理的是 `object-worn`：手表戴在物体、桌面、布料、塑料、假体或其他非人体介质上却被识别为佩戴。它比“人手上但识别为未佩戴”的舆情风险更高，因此训练、阈值和后处理都应优先约束这类 false positive。

如果数据能提供 `negative_type=object_worn`、`scene_type=object_worn`、`subject_type=non_human` 等字段，建议保留到 s01/s03 后续表格中；如果暂时没有元数据，至少在 H5 record、`sample_name` 或文件名中保留 `object_worn` / `non_human` / `hard_negative` 标记，方便 s06/s10 分层复盘。

推荐的高 FP 风险完整训练命令：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --hard_negative_optimize
```

这条命令会同时做特征数量搜索、XGBoost 参数搜索、train-only OOF hard negative 挖掘、hard negative 加权重训、valid/test 窗口缓存导出、s07 后处理搜参、s06 端到端评估、s10 泛化审计和部署产物导出。窗口模型阶段会自动切到 `precision_constrained` 和 `threshold_min_precision=0.995`，训练阶段只从 `feature_pool_train.csv` 挖 hard negatives 并写出 `hard_negative_mining_train.csv` / `hard_negative_training_weights.csv`，后处理阶段再用 `max_false_worn_event_rate=0.005` 与 `postprocess_fp_cost=8.0` 约束样本级误触发。

如果只想先跑到模型搜参 + hard negative 回流训练为止，暂时不导出 NPZ、不跑 s07 后处理搜参、不跑 s10 审计，使用：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --hard_negative_optimize \
  --stop_after s05
```

这会停在 `s05`，产物重点看：

```text
artifacts/final_model.json
artifacts/final_model_config.json
artifacts/model_bundle.pkl
artifacts/model_search_results.csv
artifacts/hard_negative_mining_train.csv
artifacts/hard_negative_training_weights.csv
```

如果需要调整回流强度，可以额外加：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --hard_negative_optimize \
  --hard_negative_weight 4.0 \
  --hard_negative_top_percentile 0.15
```

如果模型已经训练完，只想基于当前结果导出窗口缓存并做后处理搜参，使用：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --skip s01,s02,s03,s04,s04_search,s05 \
  --export_window_cache \
  --optimize_postprocess \
  --postprocess_split valid \
  --split test \
  --max_sample_fp_rate 0.005 \
  --max_false_worn_event_rate 0.005 \
  --postprocess_fp_cost 8.0 \
  --stop_after s07_post
```

如果只想单独生成 hard negative 与错误分析文件，运行：

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --split test \
  --method state_machine \
  --window_sec 5 \
  --stride_sec 1 \
  --skip_initial_windows 3 \
  --no-use_stage2_ir
```

重点查看这些输出：

```text
artifacts/hard_negatives_test_state_machine.json
artifacts/window_error_analysis_test_state_machine.csv
artifacts/window_error_analysis_test_state_machine.json
artifacts/error_stratification_test_state_machine.json
artifacts/end_to_end_eval_test_state_machine.json
```

复盘顺序建议：

1. 先看 `hard_negatives_test_state_machine.json`，确认高置信 FP 是否集中在 object-worn、反光物体、松戴、低质量或某个 mode。
2. 再看 `window_error_analysis_test_state_machine.csv`，筛选 `target == 0` 且 `pred_raw == 1` 或 `pred_state_machine == 1` 的窗口，按 `prob_raw`、`sample_name`、`h5_file`、`window_index` 排序。
3. 然后看 `error_stratification_test_state_machine.json`，确认 FP 是否集中在 `mode`、H5/record、早期窗口、quality/OOD 分层。
4. 最后看 `end_to_end_eval_test_state_machine.json` 中的 `sample_fp_rate`、`false_worn_event_rate` 和首个佩戴输出延迟，避免只看窗口 accuracy。

泛化审计命令：

```bash
python s10_generalization_audit.py \
  --artifact_dir artifacts \
  --split test \
  --method state_machine \
  --min_support 10
```

或在 s08 里显式追加审计：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --run_generalization_audit \
  --stop_after s10_audit
```

审计后重点看：

```text
artifacts/generalization_audit/summary.md
artifacts/generalization_audit/action_items.csv
artifacts/generalization_audit/window_strata.csv
artifacts/generalization_audit/sample_strata.csv
```

数据回流原则：

- 优先把 object-worn false positives 作为固定高风险负样本组加入下一轮 train/valid/test，而不是只随机增加 easy negatives。
- train/valid/test 必须按 sample/record/H5 或更强的 subject/device/session 分组隔离，避免同一物体场景泄漏到多个 split。
- 如果 object-worn FP 仍集中在某些材质、反光、低质量或特定 mode，优先补这些场景的数据和特征审计，不要简单降低 precision 约束。
- 如果窗口 accuracy 很高但 `false_worn_event_rate` 偏高，应优先检查后处理参数、warmup、连续触发条件和 hard negative 分层，而不是只继续扩大 XGBoost 搜参。

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
  --window_sec 5 \
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

注意：如果训练主流程使用过非默认窗参数，例如不同的 `--window_sec`、`--stride_sec` 或 `--skip_initial_windows`，这里必须保持一致。Stage2 不再使用 IR 派生特征，`--use_stage2_ir` 不作为后处理一致性参数。

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
  --fp_cost 1.5 \
  --search_budget 240
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
  --window_sec 5 \
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

如果窗口错误报告中包含三绿光可靠性特征，审计会额外生成 `green_support_bin`、`green_top2_corr_bin`、`green_weak_gap_bin`、`green_stability_bin` 等分层；当 FP 集中在 `<2of3`、`low_corr`、`large_gap` 或 `low_stability` 时，`action_items.csv` 会提示优先检查 hard negatives 和三绿光可靠性特征。`summary.json` 也会记录最终模型选中了多少个三绿光可靠性特征。

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
| PPG Green | G1/G2/G3 reliability | 2-of-3 support/top2 corr/weak gap | 三绿光可靠性候选特征 |

三绿光通道只通过特征表达，不增加端侧硬投票或多模型结构。新增可靠性候选包括 `G_2OF3_AC_SUPPORT`、`G_TOP2_TO_ALL_AC_RATIO`、`G_TOP2_CORR_MIN`、`G_WEAK_CHANNEL_GAP` 和 `G_SPATIAL_STABILITY_SCORE`。它们会和其他普通 float 特征一起参与 `8,10,12,15,18` 特征数量搜参；只有被最终模型选中时才会进入 `FEATURE_ORDER`、`deploy_feature_extractor.py` 和 golden vectors。
| PPG IR | IR | Stage1 DC/ACDC gate only | 不进入 Stage2 特征池 |
| PPG Ambient | Ambient | 单通道 + spectral + 波形 | 环境光抑制（8 个） |
| ACC | X/Y/Z 三轴 | 基础 (per-axis) + 幅值 + 震颤 | 加速度检测（19 个） |
| ACC-PPG | cross | 相干性 + BP 相关 | 运动-脉搏交叉（4 个） |
| 跨通道 | GREEN-AMB / ACC-GREEN | 相关性/比值/泄漏 | Stage2 不使用 IR 派生特征 |
| 信号质量 | GREEN/AMB | 饱和度/削顶率 | 传感器接触质量 |
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

`s04_feature_selection.py` 负责从 ~170 候选特征（IR 已剔除）中选出 `max_features` 个（默认 15）。流程：

```text
clean_features_by_train（缺失/低方差/高相关/VIF）
  → fast_group_preselection（按特征组预筛 Top4）
  → cross_validate_importance（5-fold Permutation + SHAP）
  → 按 deployment_score 排序选取 max_features
```

各组特征的入选上限（GROUP_LIMITS_DEFAULT）：

| 特征组 | Limit | 特征组 | Limit |
|---|---|---|---|
| commercial_baseline | 8 | green_stats | 2 |
| green_spatial | 2 | green_3ch_consistency | 2 |
| frequency | 2 | waveform_morphology | 3 |
| signal_complexity | 2 | signal_quality | 2 |
| acc_features | 1 | acc_per_axis | 1 |
| acc_tremor | 1 | acc_orientation | 1 |
| ambient_stats | 1 | amb_cross | 1 |
| ir_g_amplitude | 1 | ir_g_correlation | 1 |
| spatial_coupling | 1 | mode | 1 |
| other | 2 | meta | 0 |

绿光、环境光和 ACC 相关槽位参与 Stage2 特征筛选。IR 相关组（ir_stats/ir_g_*/ambient_stage1/ACC_IR 等）不再作为 Stage2 候选特征；如旧产物中仍含 IR 特征，需要重新运行 s03-s05。

## 测试与验收

语法检查：

```bash
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in pathlib.Path('.').glob('*.py')]; print('ast parse ok')"
```

运行测试：

```bash
python -m pytest -q --basetemp "%TEMP%\wearing_liveness_pytest_all"
```

部署公式和端到端烟测门禁：

```bash
python -m pytest test_deploy_feature_extractor.py test_end_to_end_pipeline_guard.py -q --basetemp "%TEMP%\wearing_liveness_pytest_deploy_guard"
```

这组测试会检查所有 `s03` 可导出的窗口特征都有部署公式，并用合成 grouped H5 跑到 `s08 -> s06_cb`，确认部署脚本、XGBoost JSON、部署配方、golden vectors 和一致性校验都能通过。以后如果再次出现 `No deploy formula registered for selected features: ...`，应先在这组测试里失败，而不是等真实数据跑到导出阶段才发现。

如果在某些 Windows 权限环境里遇到项目目录 `.pytest_tmp_*` 或 `__pycache__` 无法写入，优先使用上面的 `%TEMP%` basetemp 和 `ast.parse` 语法检查命令；它们不会依赖项目目录下新建测试临时目录或写 `.pyc` 文件。

真实数据最终验收仍需要在数据机上运行完整默认流程并保存日志：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts > full_pipeline.log 2>&1
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --run_generalization_audit --stop_after s10_audit > generalization_audit.log 2>&1
```

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

如果后处理搜参阶段出现 `worker caches not initialized`，说明运行的 `s07_postprocess_optimize.py` 不是当前修复后的版本，或旧进程/旧代码仍在执行。当前版本会通过 `ProcessPoolExecutor` initializer 显式把 NPZ caches 注入每个 worker；并且只有成功评估的候选才会写入 `postprocess_search_results.csv`，不会再因为全部 worker 失败触发 `UnboundLocalError: metrics`。

### Stage2 IR 策略不一致

Stage2 现在固定不使用 IR 派生特征。`--use_stage2_ir` 仅为兼容旧命令保留，不建议使用，也不会把 IR 特征带入最终模型；如 `model_bundle.pkl` 中仍有 IR 特征，部署导出会报错并要求重新生成。

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

Stage2 的部署策略是“训练前过滤，不在导出阶段裁剪”：
1. **s03 源头**：Stage2 特征池输出只包含环境光、绿光和 ACC；`SQI_*` 这类泛化名字也不会再隐式使用 IR。
2. **s04/s05 防线**：如果读取到旧 CSV、旧 `ranked_features.json` 或旧 `selected_features.json`，会再次移除 IR 派生候选，防止旧产物污染训练。
3. **s08 部署导出**：`deploy_feature_extractor.py` 的 `FEATURE_ORDER` 严格等于 `model_bundle.pkl["feature_names"]`，不会静默裁剪。若旧 bundle 仍含 IR 特征，导出会直接报错，要求重新运行 s03-s05。

嵌入端只需按 `FEATURE_ORDER` 计算环境光、绿光和 ACC 特征向量；IR 通道只属于 Stage1 门控，不属于 Stage2 窗口级 XGBoost 输入。

`s08_run_pipeline.py` 在默认部署导出流程中会校验 `model_bundle.pkl`、`deploy_feature_extractor.py`、`deploy_xgboost.json`、`deploy_cookbook.json` 和 `deploy_package/model_params.json` 的特征顺序、阈值、fill/clip 配置是否一致；若发现旧产物或部署文件漂移，会直接报错中断。

默认部署导出还会生成 `artifacts/golden_vectors.json`。它包含固定合成窗口的 `FEATURE_ORDER`、fill/clip 后特征向量、XGBoost 概率和窗口级阈值标签，用于工程侧 C/Rust/端侧实现做 golden-vector 对齐。上线前应要求端侧对同一批 golden vectors 输出完全一致或在约定浮点误差内一致。

窗口级工程化识别的核心推理产物是前两个文件；`golden_vectors.json` 是上线前做端侧一致性验收的校验产物：

```text
deploy_feature_extractor.py
final_model.json
golden_vectors.json
```

`deploy_feature_extractor.py` 是自包含脚本：它内联了所需的 Stage2 窗口级预处理和特征计算逻辑，不依赖 `s03_extract_feature_pool.py`、`s08_run_pipeline.py` 或其他训练/评估脚本。工程侧用它按 `FEATURE_ORDER` 生成环境光、绿光和 ACC 特征向量，用 `final_model.json` 计算窗口佩戴概率，再用脚本内置的 `WINDOW_MODEL_THRESHOLD` 或 `classify_probability(probability)` 得到窗口级 0/1 识别结果。

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
    → ① Stage2 忽略 IR，只计算环境光、绿光和 ACC 特征
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
   - 零训练脚本依赖（仅 numpy）
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
