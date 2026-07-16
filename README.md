# 手表佩戴/活体检测流水线

本项目是一个面向手表佩戴/活体检测的训练、评估和部署导出流水线。把原始 H5 中的 PPG/ACC 信号处理成 Stage2 窗口级特征，用 XGBoost 训练窗口模型，再用端到端评估和可选的状态机后处理控制误触发、响应时延和部署复杂度。

首次处理新数据时，先生成完整特征排序：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

默认采用人工固化特征流程，运行到 `s04` 后暂停。直接在
`artifacts/manual_feature_selection.csv` 的 `selected` 列填写 `1`，保存后再恢复训练。除 `selected` 外的列属于不可变契约，不应修改。
选择数量、信号类别和 FFT 类别完全由用户决定；工程代价只给警告，不修改选择。
需要无人值守基线时显式使用：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_auto \
  --feature_selection_mode auto
```

先不训练、只看实际会执行哪些步骤：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --dry_run
```

## 1. 核心结论

- 默认 `feature_selection_mode=manual`，在 `s04` 输出完整排序后暂停，不会隐式选择特征或进入训练。
- `--feature_selection_mode auto` 才执行候选子集搜索、特征数量搜索、模型训练和部署导出。
- 不加 `--auto_optimize_e2e` 时，阈值目标仍是 `accuracy`，`threshold_min_precision=0.95`，`model_search_fp_cost=2.0`，不会自动导出窗口 NPZ，也不会跑 `s07` 后处理搜索。
- 加 `--auto_optimize_e2e` 后，会自动切换为 `feature_selection_mode=auto`，并进入产品指标优先的端到端优化链路：启用模型搜索、窗口 NPZ 导出、`s07` 后处理搜索，并把阈值目标改成 `precision_constrained`。
- `--with_postprocess` 只是打开后处理搜索；它不会改成 auto E2E 的精度约束策略，也不会写 `auto_optimize/` 汇总。
- 商用 8 特征全部进入 126 项受治理特征池；其中 6 项映射到同公式规范特征，`COMM_GREEN_AC` 和 `COMM_AMB_AC` 作为独立候选。特征名 `mode` 为兼容既有 CSV/模型继续保留，但其值直接等于 H5 的 `ppg_config`；若选中，必须重点审计跨 subject/device/session/config 泛化，防止硬件捷径。
- `frequency` 和 `ppg_config` 是采样率与绿光布局的唯一来源，不再通过样本名、窗口长度或信号方差推断。三种配置均归一成三个固定物理光区，同一区域内部等权平均；现有均值、中位数、top2、2-of-3 和 pair 特征继续提供鲁棒证据，15 项固定位置候选保留单边翘起和局部漏光的方向信息。
- 新增 7 项互补候选：top2 稳健偏度与谱熵、ACC jerk 尾部均值、ACC–PPG 有限延迟相关与 PSD 相似度、三光区 2-of-3 周期性和三对时延 RMS。
- v8 保留原有 20 项三光区鲁棒候选：逐点中位波形、按 AC-RMS 能量排序且对并列值无编号偏置的 top2 复合波形、三种两区组合的周期性/主频差/ACDC/环境相关顺序统计、环境光线性投影残差、三光区相位集中度和频谱共识。RMS 只用于衡量脉动能量，不等同于信号质量；频率、相位和谱共识只使用同时通过相对 AC 幅值、频谱峰值与自相关三重有效性门控的光区。另为每个固定光区提供相对 DC、相对 AC、AC/DC、周期性和环境光绝对相关性五类候选；它们带固定位置与跨硬件泛化风险标记，只由完整排序和人工 CSV 决定是否使用。
- 手动特征会冻结为 `manual_selected_features.json`，随后驱动模型搜参、复杂度约束、hard-negative 候选和 C 部署特征顺序。
- hard-negative 候选只使用 train OOF 误报；只有 valid accuracy 不下降且 FPR 不恶化才接受，否则自动回滚参考模型。
- Stage1 与 Stage2 是两条并行方法：Stage2 对全量合法窗口持续完成特征、模型和 EMA/投票/状态机更新；Stage1 只在最终对外输出端做快速门控，不过滤 Stage2 数据，也不暂停或重置 Stage2 状态。

## 2. 三阶段架构

```text
原始 H5
  → s01 数据扫描与 train/valid/test 切分
  ├→ s02 Stage1 IR DC/ACDC 固定阈值快速门控 ───────────────┐
  └→ s03 Stage2 全量特征池提取
  → s04 稳定性特征筛选与候选子集搜索
  → s05 XGBoost 训练、校准、阈值选择、模型搜索
  → s06 Stage2 持续后处理 ───────────────────────────────┤
                                                      └→ Stage1 AND Stage2 最终输出
  → s06 部署式端到端评估与部署产物导出
  → s07 可选：基于窗口 NPZ 的后处理状态机搜索
  → s06 可选：泛化审计
```

### Stage1

`s02_ir_dc_threshold.py` 使用 IR DC/ACDC 做快速门控。固定部署阈值为
`dc_threshold=0.1e6`（100,000）、`ac_dc_threshold=1.0`；训练侧仍记录按
`train_dc_ratio=0.90` 派生的 `dc_threshold=0.09e6`（90,000），用于 Stage1 独立统计。
它不是 Stage2 数据过滤器，也不是最终 XGBoost 输入特征的一部分。

### Stage2

`s03_extract_feature_pool.py` 从所有合法窗口中提取环境光、三路绿光和 ACC 特征，train/valid/test 均不按 Stage1 过滤。推理时 Stage2 模型及后处理也始终持续执行；`stage1_gate_flags` 只用于生成最终融合输出。当前 Stage2 固定不使用 IR 派生特征；`--use_stage2_ir` 只保留为兼容旧命令的参数，不建议作为新实验方向。

### Stage3

`s06_deploy_eval.py` 可以直接用窗口概率做端到端评估。`s07_postprocess_optimize.py` 在已导出的逐窗 NPZ 上搜索状态机参数，用于降低样本级 FP、false-worn event 和响应时延风险。

## 3. 默认人工流程做什么

默认命令：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

默认会运行：

```text
s01 → s02 → s03 → s04 → 暂停，等待人工固化特征
```

`s04` 输出完整 126 特征的 `feature_ranking_full.json/csv`、
`manual_feature_selection.csv` 和
`feature_pool_completeness.json`。默认流程不会用 Top-K、分组上限或 FFT 上限覆写人工选择。

创建人工文件后恢复：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts \
  --feature_selection_mode manual \
  --manual_feature_file artifacts/manual_feature_selection.csv \
  --skip s01,s02,s03,s04 \
  --stop_after s06_cb
```

恢复后会严格按 CSV 中选中特征的名称、行顺序和数量训练固定特征集合；默认执行 staged group-CV 模型搜参、最大深度 2–5 和 train OOF hard-negative 候选，然后生成 ROC/PR、端到端评估、错误分析和部署产物。该恢复命令默认不搜索特征数量，也不运行后处理参数搜索。

### 3.1 如何人工给出最终训练特征

人工筛选的唯一输入文件是：

```text
artifacts/manual_feature_selection.csv
```

该文件由 `s04` 根据同目录下的 `feature_ranking_full.json` 自动生成，包含完整特征池及其排序、稳定性、漂移、误报风险和部署成本信息。人工选择时只允许修改 `selected` 列：

```text
selected = 1    选中该特征
selected = 0    不选该特征
```

必须同时满足以下约束：

- 只能选择 `eligible=1` 的行；`eligible=0` 的特征即使写成 `selected=1` 也会被拒绝。
- 除 `selected` 外，不能修改任何单元格、列名、列顺序、行顺序或行数。
- 不要在 CSV 中增加备注列；需要记录选择理由时，应在实验记录或独立说明文件中记录。
- `selected` 只能填写整数 `0` 或 `1`，不能使用 `yes`、`true`、空格或公式。
- 至少选择一个特征。manual 模式不限制特征数量、信号类别或 FFT 数量，也不会使用 Top-K、group cap 或 local-swap 覆写人工结果。
- CSV 必须与当前 `feature_ranking_full.json` 配套。重新运行 `s03/s04` 或升级特征池后，应使用新生成的 CSV 重新选择，不能沿用旧文件。

加载时会核对 CSV schema、特征池版本、完整排序 SHA256、每一行不可变字段、train/valid 列是否存在以及特征是否具备候选资格。任何契约字段被 Excel/WPS 或人工误改，训练都会明确报错，而不是静默回退到自动选择。

### 3.2 人工筛选时如何阅读各列

| 列 | 含义 | 人工筛选建议 |
|---|---|---|
| `rank` | 当前完整排序名次 | 用于确定初始候选范围，不应简单等同于最终 Top-K |
| `feature` | 唯一特征名 | 最终模型、部署导出和 C 特征顺序都使用该名称 |
| `eligible` | 是否允许人工选择 | 只选择值为 `1` 的行 |
| `group` | 特征所属信号/机理类别 | 用于避免只选同一类高度冗余特征 |
| `ranking_score` | train 内部 group-CV 分离能力、FP proxy 和部署适配度的综合分数 | 适合作为第一轮候选排序依据 |
| `train_group_fold_auc_mean` | train 内部分组折上的单特征平均 AUC | 应看其与 `0.5` 的距离；明显低于 `0.5` 也可能是稳定的反向判别特征 |
| `valid_auc` | valid 上的单特征 AUC | 只作泛化旁证，不参与完整排序分数，也不能单独用于反复选特征 |
| `fp_proxy_sample_fp_rate` | train-only 样本级误报代理 | 越高越需要谨慎，尤其是非佩戴误报敏感场景 |
| `valid_psi` | train/valid 分布漂移 | `>0.25` 通常需要重点检查数据源、mode 或设备捷径 |
| `deployment_cost` | 端侧计算成本 | 同等准确率下优先成本较低者 |
| `signal_source` / `preprocessing` / `formula` / `unit` | 信号来源、预处理、定义和单位 | 用于判断机理是否互补、是否可解释 |
| `fft` / `buffer_samples` / `accumulator` / `c_operators` | FFT、缓存和 C 实现需求 | 用于审计内存、时延和工程实现复杂度 |
| `risk_flags` | 高漂移、低方差、实验性高成本等风险 | 有风险不代表绝对不能选，但必须用消融和分层指标证明增益 |
| `commercial_8_member` | 是否属于保留的商用八特征映射 | 仅表示来源关系，不是自动入选规则 |

完整排名的核心证据来自 train 内部按 `sample_name` 分组的评估；valid 指标用于诊断旁证，test 不参与人工筛选。对于单特征 AUC，XGBoost 可以利用正向或反向单调关系，因此不应把 AUC 小于 `0.5` 的特征直接判为无效。更重要的是跨 group-fold 的稳定性、与其他已选特征的互补性以及 valid 漂移是否可接受。

### 3.3 面向单窗准确率的推荐人工组合

不建议机械选择排名前 N。第一轮可以分别构造约 12、16、20 项的候选集合，并尽量覆盖以下互补机理：

| 机理类别 | 建议初始数量 | 目的 |
|---|---:|---|
| 绿光接触强度、AC/DC | 2–3 | 描述总体接触和脉动幅度 |
| 三光区空间一致性及弱区异常 | 3–5 | 处理单边翘起、局部漏光和部分光区失效 |
| 固定位置光区候选 | 0–3 | 利用稳定区域顺序描述局部翘起/漏光；必须审计 device/mode 捷径 |
| top2、逐点中位等稳健复合 | 2–4 | 在一个光区异常时保留可靠脉动信息 |
| 周期、主频、相关性、时延 | 2–4 | 描述生理周期一致性，抑制随机光学扰动 |
| ACC 运动强度 | 1–2 | 区分运动伪影和稳定佩戴 |
| ACC–PPG 耦合 | 1–2 | 描述运动污染是否同步进入光学信号 |
| 环境光影响或投影残差 | 1–2 | 识别外界光线泄漏及其对三光区的影响 |
| `mode` | 0–1 | 可作为候选，但必须审计跨 subject/device/session/mode 泛化 |

以上数量只是构造对照实验的起点，不是强制上限。相同机理下高度相关的多个均值、最大值或相近频域统计通常不应全部选入；优先选择跨分组稳定、漂移较低且能解释不同失败模式的代表特征。带 `experimental_high_cost` 的相位或频谱共识特征建议逐项做消融，只有在 group-CV 和独立 valid 上均有稳定增益时才保留。

建议准备三套人工候选：

```text
manual_feature_selection_core.csv       约 12 项，低成本核心组合
manual_feature_selection_robust.csv     约 16 项，增加三光区可靠性与运动耦合
manual_feature_selection_expanded.csv   约 20 项，加入经审计的频域/相位候选
```

这三份文件都必须从同一次 `s04` 生成的原始 `manual_feature_selection.csv` 完整复制，只修改 `selected`。为避免模型、报告和部署产物互相覆盖，正式对照时应为每套候选使用独立的 `artifact_dir`；可以先复制完整的 s01–s04 产物目录，再分别编辑各目录中的 CSV。不要在同一目录连续训练多个候选后再比较残留文件。

### 3.4 保存 CSV 后恢复训练

使用默认文件恢复：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts \
  --feature_selection_mode manual \
  --manual_feature_file artifacts/manual_feature_selection.csv \
  --skip s01,s02,s03,s04 \
  --stop_after s06_cb
```

也可以显式指定同一排序对应的另一份人工候选 CSV：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts_robust \
  --feature_selection_mode manual \
  --manual_feature_file artifacts_robust/manual_feature_selection_robust.csv \
  --skip s01,s02,s03,s04 \
  --stop_after s06_cb
```

训练开始前会把实际选中的名称和顺序冻结到：

```text
artifacts/manual_selected_features.json
```

该冻结文件随后统一驱动 XGBoost 训练、模型复杂度检查、hard-negative 候选、错误分析、模型 bundle、C 部署特征顺序和 golden vectors。manual 模式下不会再执行特征数量搜索或 local-swap；最终特征及其数量完全由 CSV 中的 `selected` 决定。

人工候选对照时至少比较以下冻结指标：

- train 内部 repeated group-CV 的平均单窗 accuracy 和标准差；
- valid 单窗 accuracy、FP、FN、precision、recall 和阈值；
- 按 mode、H5、时间位置、三光区可靠性、质量和 OOD 分层的错误率；
- 总节点数、FFT 来源数、最大缓存长度和端侧耗时；
- 最终只对确定的候选运行一次独立 test，禁止根据 test 结果反复修改 `selected`。

默认关键参数：

```text
window_sec:                    5
stride_sec:                    1
automatic_edge_window_trim:   3（读入并排序后直接使用 [3:-3]）
skip_initial_windows:          0（仅作为额外兼容裁剪）
use_stage2_ir:                 false
max_features:                  仅用于 auto 模式；manual 不限数量
feature_selection_mode:        manual
model_search:                  true
model_search_feature_counts:   8,10,12,15,18（仅 auto；manual 忽略）
model_search_strategy:         staged_group_cv
model_search_max_depth:        2,3,4,5
model_search_n_workers:        默认跟随全局 n_workers（当前上限 4）
max_model_nodes:               500
mine_hard_negatives:           manual 恢复时为 true
threshold_objective:           accuracy
threshold_min_precision:       0.95
model_search_fp_cost:          2.0
runtime_profile:               balanced
effective stop_after:          s04（没有人工文件时）
optimize:                      false
export_window_cache:           false
optimize_postprocess:          false
```

### 3.5 当前并行策略

项目中的“并行”包含三个不同层次，不能混为一谈：

1. **Stage1/Stage2 业务语义并行**：Stage2 对全量合法窗口持续计算特征、模型概率和后处理状态；Stage1 只在最终输出处做快速门控。两条方法互不筛除数据，也不会互相暂停或重置，但单个样本内不要求为 Stage1 和 Stage2 各启动一个 CPU 线程。
2. **数据与评估任务并行**：s01 按 H5 文件、s02/s03/s06 按样本、s04 按 `seed × group-fold`、s07 按后处理网格点使用多进程。
3. **s05 模型候选并行**：single-split Stage A 以及 staged group-CV 的 Stage A/Stage B 在外层并行评估独立 XGBoost 候选；每个候选内部默认 `n_jobs=1`，避免嵌套线程抢占。

主流程统一使用：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --n_workers 4
```

`--model_search_n_workers` 默认继承 `--n_workers`，也可以单独降低：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --n_workers 4 \
  --model_search_n_workers 2
```

并行安全约束：

- worker 数会自动限制到实际任务数，空闲任务不会额外创建线程。
- `WL_FORCE_SERIAL=1` 会强制所有支持该开关的阶段回退到单 worker，便于排查环境问题。
- 主流程默认设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1` 和 `VECLIB_MAXIMUM_THREADS=1`；如果调用者已经显式设置，则保留调用者的值。
- `WL_INNER_N_JOBS` 默认未设置，此时单个 XGBoost 候选使用 1 个内部线程；如果显式提高该值，应相应降低 `--model_search_n_workers`，避免两层并行相乘。
- s03 并行 worker 可以乱序完成，但主进程会按原始样本索引重组后再写 CSV，保证相同输入和配置下行顺序确定。
- s05 候选可以乱序完成，但记录按候选输入索引重组；候选异常会终止搜索，不会静默删除失败候选后继续选择模型。
- 小任务自动串行：s01 只有一个 H5、s02/s03/s06 不超过两个样本或 s04 不超过四个 fold 任务时，跳过进程池开销。

manual 模式下，CSV 冻结的特征名称和数量不会改变；并行只作用于这组固定特征上的模型参数候选，不会重新搜索特征数或执行 local-swap。

默认不会运行：

```text
s06_opt                        legacy 状态机参数优化
s06_cache / s06_replay_cache   逐窗 NPZ 缓存导出
s07_post                       后处理状态机搜索
s06_audit                      s06 内嵌泛化审计
auto_optimize/                 auto E2E 汇总产物
```

## 4. 是否使用 `--auto_optimize_e2e`

### 默认 manual：先排序，再固化

首次运行使用默认命令生成完整排序：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

特点：

- 排名分数来自 train 内部 group-CV，valid 指标只作诊断旁证。
- 人工文件冻结特征名称和顺序，不允许隐式 Top-K、local-swap 或自动回退。
- 适合审计特征机理、部署成本和分布漂移。

若目标是直接得到无人值守基线、部署脚本、XGBoost JSON 和 golden vectors，使用：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_auto \
  --feature_selection_mode auto
```

### 使用 `--with_postprocess`：只加后处理搜索

当窗口模型已经可用，但需要用状态机降低误触发或控制响应延迟时：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --feature_selection_mode auto \
  --with_postprocess
```

这等价于打开：

```text
--export_window_cache
--optimize_postprocess
```

特点：

- 会导出 valid/test 逐窗 NPZ。
- 固定在 `valid` 上搜索 `s07` 参数，并在 `split`（通常为 `test`）上做冻结 replay；禁止使用 test 选参。
- 不会把 `threshold_objective` 自动改成 `precision_constrained`。
- 不会写 `artifacts/auto_optimize/auto_optimization_summary.json`。

### 使用 `--auto_optimize_e2e`：产品指标优先的自动闭环

当你已经明确当前目标不是单窗口 accuracy，而是综合权衡样本级准确率、召回、FP、false-worn event、首次佩戴输出时延和部署成本时：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --auto_optimize_e2e
```

`--auto_optimize_e2e` 会自动设置：

```text
model_search:             true
export_window_cache:      true
optimize_postprocess:     true
ranking_objective:        balanced
threshold_objective:      precision_constrained
threshold_min_precision:  0.97   # 未手动传参时
model_search_fp_cost:     4.0    # 未手动传参时
fp_cost_weight:           ≥0.35  # 未手动传参时
postprocess_fp_cost:      ≥4.0   # 未手动传参时
```

auto 结束后会写：

```text
artifacts/auto_optimize/auto_optimization_summary.json
artifacts/auto_optimize/candidate_scores.csv
artifacts/auto_optimize/candidate_manifest.json
```

推荐使用场景：

- 你要优先降低非佩戴误识别为佩戴的风险。
- 你要用 valid 选择状态机参数，再在 test replay 验证。
- 你要把最终选择依据沉淀成机器可读的 summary/manifest。
- 你接受训练时间比默认流程更长。

不推荐使用场景：

- 第一次跑新数据，尚未确认 Stage2 窗口模型是否正常。
- 只是想快速导出一版部署模型。
- 主要目标是冲单窗口 raw accuracy。
- 数据量很小，valid/test replay 指标不稳定。

### auto 与默认流程的差异速查

| 项目 | 默认流程 | `--with_postprocess` | `--auto_optimize_e2e` |
|---|---:|---:|---:|
| `threshold_objective` | `accuracy` | `accuracy` | `precision_constrained` |
| `threshold_min_precision` | `0.95` | `0.95` | `0.97` |
| `model_search_fp_cost` | `2.0` | `2.0` | `4.0` |
| 导出窗口 NPZ | 否 | 是 | 是 |
| 运行 `s07` | 否 | 是 | 是 |
| 写 `auto_optimize/` | 否 | 否 | 是 |
| 推荐用途 | 基线/部署产物 | 后处理调参 | 产品指标自动选择 |

手动传入的参数优先级更高。例如：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --auto_optimize_e2e \
  --threshold_min_precision 0.995 \
  --model_search_fp_cost 5.0
```

这会保留你手动指定的 `0.995` 和 `5.0`。

## 5. 场景化使用方式

### 5.1 只预览命令

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --dry_run
```

用途：确认路径、参数、步骤和下游命令，不读取 H5，不训练。

### 5.2 默认两阶段训练与部署导出

第一阶段生成排序 CSV 并暂停：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts
```

只修改 `artifacts/manual_feature_selection.csv` 的 `selected` 列后，执行第二阶段：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts \
  --manual_feature_file artifacts/manual_feature_selection.csv \
  --skip s01,s02,s03,s04
```

第二阶段默认完成充分模型搜参、hard-negative 候选、评估、错误分析和部署导出；不会搜索特征数量，也不会运行后处理参数搜索。

输出重点：

```text
artifacts/model_bundle.pkl
artifacts/final_model.json
artifacts/final_model_config.json
artifacts/deploy_feature_extractor.py
artifacts/deploy_xgboost.json
artifacts/deploy_cookbook.json
artifacts/golden_vectors.json
artifacts/deploy_package/
artifacts/report_plots/                  # 全部图表自动生成
artifacts/feature_embedding_report/      # 特征嵌入可视化报告
artifacts/error_plots/                  # 错误样本时序图
```

### 5.3 快速检查到某一步

只跑到特征筛选：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --stop_after s04
```

只跑到模型训练：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts \
  --manual_feature_file artifacts/manual_feature_selection.csv \
  --skip s01,s02,s03,s04 \
  --stop_after s05
```

### 5.4 快速调试，不做模型搜索

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_debug \
  --feature_selection_mode auto \
  --no-model_search \
  --model_search_feature_counts 15 \
  --max_features 15
```

用途：缩短调试时间。正式结果建议恢复默认 `--model_search`。

### 5.5 固定特征数量

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_k15 \
  --feature_selection_mode auto \
  --max_features 15 \
  --model_search_feature_counts 15
```

注意：固定特征数时，`--max_features` 和 `--model_search_feature_counts` 建议保持一致，避免误读最终入选特征数量。

### 5.6 只优化单窗口 accuracy

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_accuracy \
  --feature_selection_mode auto \
  --accuracy_first_optimize \
  --stop_after s05
```

该模式会：

```text
threshold_objective = accuracy
ranking_objective = window_accuracy
deployment_score_weight = 0
fp_cost_weight = 0
```

适合先排查 Stage2 raw window 能力，不建议直接把它当最终产品指标选择。

### 5.7 后处理搜索

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --feature_selection_mode auto \
  --with_postprocess \
  --postprocess_split valid \
  --split test \
  --postprocess_search_budget 240
```

如果只想基于已有模型重新跑后处理：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --skip s01,s02,s03,s04,s04_search,s04_embed,s05 \
  --export_window_cache \
  --optimize_postprocess \
  --postprocess_split valid \
  --split test \
  --stop_after s07_post
```

前提：`artifacts/` 下已经存在 split、特征池、模型和配置产物。

### 5.8 auto E2E 自动优化

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_auto \
  --auto_optimize_e2e \
  --postprocess_split valid \
  --split test
```

如果你只想验证 auto 会打开哪些步骤：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_auto \
  --auto_optimize_e2e \
  --stop_after s07_post \
  --dry_run
```

### 5.9 更快或更彻底的预算档

快速档：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_fast \
  --feature_selection_mode auto \
  --runtime_profile fast
```

更彻底搜索：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_thorough \
  --feature_selection_mode auto \
  --runtime_profile thorough
```

预算档会影响模型搜索候选数、后处理搜索预算等；你显式传入的参数会覆盖预算档默认值。

### 5.10 3s 窗口实验

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_3s \
  --feature_selection_mode auto \
  --window_sec 3 \
  --stride_sec 1 \
  --model_search_feature_counts 8,10,12,15,18
```

不要把 3s 和 5s 的产物混在同一个结论里。训练、评估、后处理、部署导出必须使用同一个 `window_sec`。

### 5.11 泛化审计

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --feature_selection_mode auto \
  --run_generalization_audit \
  --stop_after s06_audit
```

当 `--stop_after s06_audit` 被指定时，`s08` 会自动打开 `--run_generalization_audit`。

## 6. `s08_run_pipeline.py` 步骤控制

可用 `--stop_after`：

```text
s01
s02
s03
s04
s04_search
s04_embed
s05
s05_viz
s06_opt
s06_cache
s06_replay_cache
s07_post
s06_eval
s06_tree_viz
s06_audit
s06_xpt
s06_feat
s06_plot
s06_cb
```

示例：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts_auto --feature_selection_mode auto --stop_after s06_eval
```

可用 `--skip` 跳过已有产物对应步骤：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_auto \
  --feature_selection_mode auto \
  --skip s01,s02,s03 \
  --stop_after s05
```

自动联动规则：

- `--stop_after s06_cache`、`s06_replay_cache` 或 `s07_post` 会按需打开 `--export_window_cache`。
- `--stop_after s07_post` 会按需打开 `--optimize_postprocess`。
- `--stop_after s06_audit` 会按需打开 `--run_generalization_audit`。

这些是 step target 的便利联动，不等同于 `--auto_optimize_e2e`。

## 7. 数据格式

默认数据目录是 `dataset/`，可以用 `--dataset_dir` 指定。项目扫描目录下的 `.h5` 文件。

常见 H5 结构：

```text
sample_group/
  ppg          PPG 数据
  target       标签，0=非佩戴，1=佩戴
  acc          可选 ACC 数据
  frequency    必填，取值 25 或 100
  ppg_config   必填，取值 0、1 或 2
```

`ppg` 支持：

```text
(40, T)              连续时序，由 s03/s06 按 window_sec/stride_sec 滑窗
(N_win, 40, T_win)   已预切窗口，后续逐窗口使用
record/window_group  一个 H5 中包含多条 record，每条 record 下有多个窗口 group
```

预切窗口 group 示例：

```text
record_a/
  frequency    100
  ppg_config   1
  anything_w0_1/
    ppg        (40, 300)
    acc        (3, 300)
  anything_w1_1/
    ppg        (40, 300)
    acc        (3, 300)
```

窗口 group 名称需要能解析出窗口编号和 label。形如 `xxx_w20_1` 表示第 20 个窗口，label=1。读取一条数据并按窗口编号排序后，PPG、ACC、窗口编号和标签统一直接使用 `[3:-3]`，自动删除前三包和后三包；总窗口数不超过 6 时该条数据不产生 Stage2 窗口。

`frequency=25` 时 Stage2 直接使用原数据，`frequency=100` 时先降采样到 25 Hz；Stage1 分别从对应原始采样率降到 5 Hz。`ppg_config` 使用零基通道编号生成固定三光区：

- `0`：`g1=ch3`，`g2=ch4`，`g3=ch5`；
- `1`：`g1=(ch3+ch9)/2`，`g2=(ch4+ch10)/2`，`g3=(ch5+ch11)/2`；
- `2`：`g1=(ch6+ch9+ch12)/3`，`g2=(ch7+ch10+ch13)/3`，`g3=(ch8+ch11+ch14)/3`。

grouped-window 数据也允许把两个字段写在每个窗口子 group 中，但所有窗口必须一致。缺失、非法或分组内不一致的数据会整条跳过，并分别统计 `missing_frequency`、`invalid_frequency`、`missing_ppg_config`、`invalid_ppg_config` 和 `inconsistent_metadata`；不会回退到旧自动判断。

## 8. 脚本索引

### `s01_data_split.py`

作用：扫描 H5，按 sample/group 切分 train/valid/test。

常用命令：

```bash
python s01_data_split.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --valid_size 0.15 \
  --test_size 0.15 \
  --random_state 42
```

### `s02_ir_dc_threshold.py`

作用：使用部署 DC=`0.1e6`、AC/DC=`1.0` 生成 Stage1 固定门控统计和阈值产物；
默认训练门控 DC 按 0.90 比例派生为 `0.09e6`。

```bash
python s02_ir_dc_threshold.py \
  --artifact_dir artifacts \
  --min_duration_sec 1
```

### `s03_extract_feature_pool.py`

作用：依据 H5 `frequency` 和 `ppg_config` 提取 126 项 Stage2 特征池 CSV，包括元数据特征 `mode`、三个固定位置绿光区、环境光、ACC、三光区鲁棒共识/局部异常特征、15 项固定位置候选，以及商用 8 特征的规范映射/独立 AC 候选；不使用 IR 派生特征。

```bash
python s03_extract_feature_pool.py \
  --artifact_dir artifacts \
  --window_sec 5 \
  --stride_sec 1 \
  --skip_initial_windows 0 \
  --no-use_stage2_ir
```

不推荐新实验使用 `--target_aware_stride`，它会让训练窗口分布和部署 1s stride 分布不一致。

### `s04_feature_selection.py`

作用：清洗并完整排序 126 项特征，计算稳定性、FP proxy 和 C 工程元数据，并导出 CSV 选择接口。manual 模式不应用分组或数量上限；CSV 中只允许修改 `selected` 列。

```bash
python s04_feature_selection.py \
  --artifact_dir artifacts \
  --max_features 15 \
  --min_fold_auc 0.55 \
  --deployment_score_weight 0.25 \
  --fp_cost_weight 0.25 \
  --ranking_objective balanced \
  --run_subset_search
```

`--ranking_objective window_accuracy` 会放松部分 group caps，适合单窗口 accuracy 排查。

### `s05_train_final_model.py`

作用：训练 XGBoost、校准概率、选择窗口阈值、执行复杂度受限模型搜索。

```bash
python s05_train_final_model.py \
  --artifact_dir artifacts \
  --manual_feature_file artifacts/manual_feature_selection.csv \
  --threshold_objective accuracy \
  --model_search \
  --mine_hard_negatives \
  --max_model_nodes 500
```

FP 更敏感的手动阈值策略：

```bash
python s05_train_final_model.py \
  --artifact_dir artifacts \
  --threshold_objective precision_constrained \
  --threshold_min_precision 0.97 \
  --model_search_fp_cost 4.0
```

`--mine_hard_negatives` 产生独立候选和 `hard_negative_decision.json`；不满足 valid 无退化规则时不会替换参考模型。

### `s06_deploy_eval.py`

作用：用部署语义做端到端评估，导出部署包、窗口缓存和错误分析。

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --split test \
  --method state_machine \
  --window_sec 5 \
  --stride_sec 1 \
  --skip_initial_windows 0 \
  --export_deploy
```

导出 `s07` 所需窗口 NPZ：

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --split valid \
  --export_window_cache \
  --window_output_root window_outputs
```

### `s07_postprocess_optimize.py`

作用：读取窗口 NPZ，搜索状态机参数，不重新训练窗口模型。

```bash
python s07_postprocess_optimize.py \
  --artifact_dir artifacts \
  --split valid \
  --cache_root window_outputs \
  --fp_cost 4.0 \
  --max_window_fp_rate 0.01 \
  --max_sample_fp_rate 0.02 \
  --max_false_worn_event_rate 0.02 \
  --max_first_worn_output_p95_sec 3.0 \
  --search_budget 240 \
  --replay_split test
```

### `s08_run_pipeline.py`

作用：主控编排脚本。推荐优先使用它，而不是手动串所有阶段。

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

`s06_deploy_eval.py` 也内嵌泛化审计入口，用于读取已有评估产物并输出分层审计结果：

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --generalization_audit \
  --split test \
  --method state_machine \
  --min_support 10
```

## 9. 主要产物

```text
artifacts/
  splits.json
  stage1_threshold.json

  feature_pool_train.csv
  feature_pool_valid.csv
  feature_pool_test.csv
  ranked_features.json
  selected_features.json
  feature_diagnostics.csv

  final_model.json
  final_model_config.json
  model_bundle.pkl
  model_search_results.csv

  end_to_end_eval_test_state_machine.json
  error_stratification_test_state_machine.json
  window_error_analysis_test_state_machine.csv
  per_sample_xgboost_windows.csv
  per_sample_statemachine_windows.csv
  per_sample_final_prediction.csv

  deploy_feature_extractor.py
  deploy_selected_feature_formulas.json
  deploy_xgboost.json
  deploy_cookbook.json
  golden_vectors.json
  deploy_package/

  window_outputs/
  postprocess_opt/
  postprocess_opt/postprocess_search_summary.png
  auto_optimize/
  generalization_audit/
  generalization_audit/audit_strata_heatmap.png
  generalization_audit/audit_ranked_error_bars.png
  generalization_audit/audit_latency_distribution.png
  report_plots/
  report_plots/s04_shap_importance.png
  report_plots/s04_feature_selection_report.png
  report_plots/s05_roc_pr_curves.png
  report_plots/s05_training_report.png
  report_plots/s05_threshold_fp_recall_tradeoff.png
  report_plots/s06_deploy_report.png
  report_plots/s06_tree_feature_usage.png
  error_plots/

  feature_embedding_report/
  feature_embedding_report/embedding_report.md
  feature_embedding_report/embedding_summary.json
  feature_embedding_report/embedding_source_data.csv
  feature_embedding_report/pca_2d.png / pca_3d.png
  feature_embedding_report/tsne_2d.png / tsne_3d.png
  feature_embedding_report/umap_2d.png / umap_3d.png（需 umap-learn）
  feature_embedding_report/embedding_panel_2d / embedding_panel_3d
  feature_embedding_report/selected_feature_correlation_heatmap
  feature_embedding_report/selected_feature_split_auc_heatmap
  feature_embedding_report/pca_loading_top_features
  feature_embedding_report/feature_distribution_XX_特征名 (每个入选特征一张)
```

重要文件解释：

- `model_bundle.pkl`：训练侧完整 bundle，包含模型、特征顺序、fill/clip、阈值和元数据。
- `final_model.json`：XGBoost 模型 JSON。
- `deploy_feature_extractor.py`：自包含 Python 特征提取脚本，已内联训练侧受治理特征引擎，不依赖项目源码。
- `deploy_xgboost.json`：部署侧模型和预处理配置。
- `deploy_cookbook.json`：部署配方说明。
- `golden_vectors.json`：端侧 golden-vector 对齐用例。
- `postprocess_opt/postprocess_replay_valid_to_test.json`：后处理在 valid 选参、test replay 的结果。
- `auto_optimize/auto_optimization_summary.json`：仅 `--auto_optimize_e2e` 写入的自动选择摘要。

## 10. 图片输出说明（全部默认自动生成）

流水线会在各阶段自动生成以下分析图，全部为 600 DPI PNG。核心定量图同时输出
`*_source_data.csv`、`*_figure_manifest.json`、`*_figure_qa.json`；manifest 记录结论、
面板含义、split、样本量口径、输入哈希和风险，QA 校验 DPI、像素尺寸及非空内容。

### 10.1 数据切分与 Stage1 门控

| 图片 | 说明 |
|---|---|
| `report_plots/s01_split_analysis.png` | train/valid/test 的样本数量、正负类别构成和佩戴比例；用于在训练前发现切分失衡 |
| `stage1_scatter.png` | IR DC vs AC/DC 散点图，左栏 train、右栏 valid。蓝色为非佩戴、橙色为佩戴，青色虚线标注固定部署阈值及 PASS 区 |

### 10.2 特征池与特征筛选（s03-s04）

| 图片 | 说明 |
|---|---|
| `report_plots/s03_feature_pool_analysis.png` | 四栏展示各 split 窗口覆盖、有限值率、126 项特征的可解释物理分组，以及 train-only 标准化类别分离度；该分离度只用于诊断，不替代 grouped-valid 排名 |
| `report_plots/s04_feature_selection_report.png` | 三栏：左栏 Top 20 特征排名柱状（已选深色标记 + 综合得分折线），右上选入特征的群组分布，右下 Top 12 特征 FP Proxy 风险双柱 |
| `report_plots/s04_shap_importance.png` | 四栏 SHAP 报告：左上 Top 20 特征 mean(\|SHAP\|) 柱状（已选特征深色），右上 Train vs Valid SHAP 散点 + Spearman ρ 标注，左下 Top-K 重叠率柱状，右下可疑 Train-only 强特征红色柱状 |

### 10.3 训练与 ROC/PR 曲线（s05）

| 图片 | 说明 |
|---|---|
| `report_plots/s05_training_report.png` | 三栏：上图全宽阈值选择曲线（precision/recall/F-beta/F1 vs 阈值，虚线标出选中阈值），左下三组验证指标 precision/recall 柱状，右下校准方法信息框 |
| `report_plots/s05_threshold_fp_recall_tradeoff.png` | 低误判上线用阈值权衡图：左侧展示 threshold vs precision/recall/false-positive-rate，右侧展示 recall vs false-positive-rate，并标出当前选中阈值。同步导出 CSV |
| `report_plots/s05_roc_pr_curves.png` | 四栏 2×2：左上完整 ROC 曲线（FPR vs TPR, AUC 标注），右上完整 PR 曲线（Recall vs Precision, iso-F1 等高虚线），左下放大 ROC（FPR 0-0.2），右下放大 PR（高精度区域）。数据来自 valid 分割 |

### 10.4 部署评估报告图（s06）

| 图片 | 说明 |
|---|---|
| `report_plots/s06_deploy_report.png` | 六栏 2×3 仪表盘：左上 Stage 漏斗（输入→Stage1通过→正类），中上样本混淆矩阵热力图，右上窗口概率分布直方图（红=target0，绿=target1），左下 sample/window-model/state-stream 三项指标柱状对比，中下 Top 误报样本概率横向柱状，右下 Stage1 分层错误 FP/FN 柱状 |
| `report_plots/s06_tree_feature_usage.png` | 四栏 2×2 XGBoost 树结构分析：左上特征分裂次数柱状（Top 15），右上特征平均 Gain 柱状（Top 15），左下特征平均 Cover 柱状（Top 15），右下各树节点数分布直方图 + 总节点数/均值/中位数标注 |
| `error_plots/{样本名}.png` | 仅对预测错误的样本逐一样本生成 4 行时序图：①真实标签、②窗口级 XGBoost 概率、③状态机 EMA Score、④状态机最终输出标签 |

### 10.5 后处理搜索图（s07）

| 图片 | 说明 |
|---|---|
| `postprocess_opt/postprocess_search_summary.png` | 六栏 2×3 后处理参数网格搜索结果：联合比较因果中值滤波、EMA、迟滞阈值、K-on/K-off、冷却时间，以及 final-state、majority voting、any-worn 三种样本聚合策略；按 valid 的窗口准确率、FPR、P95 新增延迟和状态翻转数选择，test 只做冻结回放 |

### 10.6 泛化审计图（s06_audit）

| 图片 | 说明 |
|---|---|
| `generalization_audit/audit_strata_heatmap.png` | 按维度×层级分别展示 accuracy/precision/recall/fp_rate 的热力图（红-绿渐变），低支持度层级加灰色横线标记。截断至 Top 50 层级 |
| `generalization_audit/audit_ranked_error_bars.png` | 按 window/sample 分层把 FP/FN 错误数排序展示，优先暴露最需要补数、加 hard negative 或单独调阈值的 stratum。同步导出 CSV |
| `generalization_audit/audit_latency_distribution.png` | 正样本 first-worn 输出延迟分布与负样本 false-worn 风险摘要，用于同时检查出值速度和误戴风险。同步导出 CSV |

### 10.7 特征嵌入可视化（s04_embed，默认自动生成）

| 图片 | 说明 |
|---|---|
| `feature_embedding_report/pca_2d.png` | PCA 2D 散点图（全部数据） |
| `feature_embedding_report/pca_3d.png` | PCA 3D 散点图（全部数据） |
| `feature_embedding_report/tsne_2d.png` | t-SNE 2D 散点图（全部数据） |
| `feature_embedding_report/tsne_3d.png` | t-SNE 3D 散点图（全部数据） |
| `feature_embedding_report/umap_2d.png` | UMAP 2D 散点图（全部数据，需 umap-learn） |
| `feature_embedding_report/umap_3d.png` | UMAP 3D 散点图（全部数据，需 umap-learn） |
| `feature_embedding_report/pca_2d_balanced.png` | PCA 2D 散点图（正样本降采样至与负样本同数量） |
| `feature_embedding_report/pca_3d_balanced.png` | PCA 3D 散点图（正负样本平衡） |
| `feature_embedding_report/tsne_2d_balanced.png` | t-SNE 2D 散点图（正负样本平衡） |
| `feature_embedding_report/tsne_3d_balanced.png` | t-SNE 3D 散点图（正负样本平衡） |
| `feature_embedding_report/umap_2d_balanced.png` | UMAP 2D 散点图（正负样本平衡，需 umap-learn） |
| `feature_embedding_report/umap_3d_balanced.png` | UMAP 3D 散点图（正负样本平衡，需 umap-learn） |
| `feature_embedding_report/embedding_panel_2d.png` | PCA+t-SNE+UMAP 并排 2D 面板（全部数据） |
| `feature_embedding_report/embedding_panel_3d.png` | PCA+t-SNE+UMAP 并排 3D 面板（全部数据） |
| `feature_embedding_report/embedding_panel_2d_balanced.png` | PCA+t-SNE+UMAP 并排 2D 面板（正负样本平衡） |
| `feature_embedding_report/embedding_panel_3d_balanced.png` | PCA+t-SNE+UMAP 并排 3D 面板（正负样本平衡） |
| `feature_embedding_report/embedding_source_data_balanced.csv` | 平衡版本的降维坐标和元数据 |
| `feature_embedding_report/selected_feature_correlation_heatmap.png` | 入选特征 Pearson 相关性热图，用于检查冗余和特征簇 |
| `feature_embedding_report/selected_feature_split_auc_heatmap.png` | 入选特征在 train/valid/test 各 split 的单变量 AUC separation 热图，用于检查分布漂移和泛化稳定性 |
| `feature_embedding_report/pca_loading_top_features.png` | PCA 前两主成分 loading 贡献最高的特征条形图，用于解释降维分离来源 |
| `feature_embedding_report/feature_distribution_{序号}_{特征名}.png` | 每个入选特征在 target=0/1 两类的分布对比箱线图 + 散点覆盖 |

UMAP 需安装 `umap-learn` 包。如未安装，报告仍会输出 PCA 和 t-SNE 图，并在 `embedding_summary.json` 中记录 UMAP 跳过原因。

## 11. 部署交付要点

端侧需要保持以下内容与训练评估一致：

```text
Stage1 DC/ACDC 阈值
H5 frequency（25 或 100）
H5 ppg_config（0、1 或 2）及固定三光区映射
Stage2 window_sec
Stage2 stride_sec
读入后固定删除首尾各 3 个窗口（`[3:-3]`）
FEATURE_ORDER
fill_values
clip_bounds
XGBoost 模型参数
WINDOW_MODEL_THRESHOLD
可选：postprocess 状态机参数
```

Python 最小运行交付只需要两个项目文件：

```text
artifacts/deploy_feature_extractor.py
artifacts/final_model.json
```

目标 Python 环境需要安装 NumPy、SciPy 和 XGBoost。`deploy_feature_extractor.py` 内嵌人工 CSV 最终选择的 `FEATURE_ORDER`、fill/clip、窗口阈值和完整特征计算代码；它不读取 `s03_extract_feature_pool.py` 或其他 artifact JSON。部署侧可调用 `extract_features_from_ppg(ppg, acc, frequency, ppg_config)`，由脚本完成 100→25 Hz 降采样（25 Hz 直接使用）、三光区映射和特征提取；也可在已有 `g1/g2/g3` 时调用 `extract_features(...)`。随后用 XGBoost 加载 `final_model.json` 即可推理。

用于审计、固件移植和数值对齐的推荐完整交付文件：

```text
artifacts/deploy_feature_extractor.py
artifacts/final_model.json
artifacts/deploy_xgboost.json
artifacts/deploy_cookbook.json
artifacts/golden_vectors.json
artifacts/deploy_package/
```

上线前必须用 `golden_vectors.json` 对齐端侧实现。端侧对同一批输入应输出相同的特征向量、窗口概率和阈值分类结果，或在约定浮点误差范围内一致。

### 11.1 上线验收 checklist

面向手表量产或准量产交付时，不建议只用默认命令作为最终结论。上线验收至少完成：

- 使用 `--auto_optimize_e2e` 跑完产品指标优先链路，并确认 `threshold_objective=precision_constrained`、`threshold_min_precision` 达到项目要求。
- 在独立 test 集和外部泛化集上同时检查 sample-level FP rate、false-worn event rate、FN rate、first-worn latency P95。
- 按 `subject_id`、`device_id`、`session_id`、`mode`、环境光、运动状态、佩戴松紧、肤色/皮肤状态分层审计；缺少元信息时不能宣称跨人/跨设备泛化已验证。
- 复查 `report_plots/s05_threshold_fp_recall_tradeoff.png`、`generalization_audit/audit_ranked_error_bars.png`、`generalization_audit/audit_latency_distribution.png`，确认当前阈值不是只追求窗口 accuracy。
- 使用 `golden_vectors.json` 在端侧逐项对齐特征顺序、fill/clip、窗口概率、阈值分类和状态机输出。
- 在目标硬件上实测单窗口耗时、首个佩戴输出延迟、内存、功耗，并保留版本号、模型 fingerprint 和部署参数。
- 对 object-worn、桌面遮挡、强运动、弱绿光、强环境光、冷启动前几窗等高风险负样本建立回归集，新增数据后重跑 s06/s07/s06_audit。

## 12. 指标阅读顺序

建议先看：

1. `end_to_end_eval_test_state_machine.json`
2. `window_model_summary`
3. `per_sample_summary.csv`
4. `window_error_analysis_test_state_machine.csv`
5. `error_stratification_test_state_machine.json`
6. `postprocess_opt/postprocess_replay_valid_to_test.json`，如果运行了后处理
7. `auto_optimize/auto_optimization_summary.json`，如果运行了 auto E2E
8. `generalization_audit/summary.md`，如果运行了审计

排查顺序：

- 先确认 Stage2 raw window 能力是否正常。
- 再看 FP/FN 是否集中在特定 split、record、mode、窗口位置或低质量信号。
- 最后再调 `s07` 状态机参数，不要用后处理掩盖窗口模型本身的问题。

## 13. 环境依赖

建议 Python 3.9+。

```bash
pip install numpy scipy pandas scikit-learn xgboost joblib h5py matplotlib pillow pytest
```

常用依赖：

```text
numpy
scipy
pandas
scikit-learn
xgboost
joblib
h5py
matplotlib
pillow
pytest
```

Windows PowerShell 中文显示异常时可尝试：

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
```

## 14. 验证命令

语法检查：

```bash
python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in pathlib.Path('.').glob('*.py')]; print('ast parse ok')"
```

主流程 dry-run：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --dry_run
```

auto E2E dry-run：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_auto \
  --auto_optimize_e2e \
  --stop_after s07_post \
  --dry_run
```

部署与端到端守卫测试：

```bash
python -m pytest test_deploy_feature_extractor.py test_end_to_end_pipeline_guard.py -q
```

模型搜索和配置测试：

```bash
python -m pytest test_model_search_config.py -q
```

如果当前目录的 `.pytest_cache` 或临时目录权限异常，可以指定系统临时目录：

```bash
python -m pytest test_model_search_config.py -q --basetemp "%TEMP%\wearing_liveness_pytest"
```

## 15. 常见问题

### 找不到 H5

确认数据目录中存在 `.h5`：

```bash
python s08_run_pipeline.py --dataset_dir D:\path\to\dataset --artifact_dir artifacts
```

如果只是复用已有产物，可以跳过 `s01`：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --skip s01
```

### 后处理没有输入

`s07` 需要窗口 NPZ。先导出：

```bash
python s06_deploy_eval.py \
  --artifact_dir artifacts \
  --split valid \
  --export_window_cache \
  --window_output_root window_outputs
```

或直接使用：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts \
  --feature_selection_mode auto \
  --with_postprocess
```

### `--auto_optimize_e2e` 不生效

确认命令是双横线：

```text
--auto_optimize_e2e
```

检查 dry-run 输出中是否出现：

```text
--threshold_objective precision_constrained
--threshold_min_precision 0.97
--model_search_fp_cost 4.0
s07_postprocess_optimize.py
[auto_e2e] would write artifacts\auto_optimize\auto_optimization_summary.json
```

如果没有这些内容，说明当前命令没有进入 auto E2E 分支。

### 不加 auto 时为什么没有 `auto_optimize/`

这是预期行为。`auto_optimize/` 只由 `--auto_optimize_e2e` 写入。默认流程只生成常规训练、评估、部署产物和图表。

### `feature_dict[name]` 报 `list indices must be integers or slices, not str`

这通常表示旧代码或旧生成的部署脚本把 `extract_features()` 返回的 list 当 dict 使用。当前 `s08` 的 golden vector 导出会把 list 转成：

```python
feature_dict = dict(zip(selected, feature_vector))
```

如果仍报错，重新导出部署产物和 golden vectors：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts \
  --manual_feature_file artifacts/manual_feature_selection.csv \
  --skip s01,s02,s03,s04,s04_search,s04_embed,s05,s05_viz \
  --stop_after s06_cb
```

### 旧快捷入口还能用吗

不推荐，也不会由 `s08` 接受：

```text
--hard_negative_optimize
--staged_e2e_optimize
```

替代方式：

- 单窗口能力排查：`--accuracy_first_optimize`
- 后处理搜索：`--with_postprocess`
- 产品指标自动闭环：`--auto_optimize_e2e`

## 16. 推荐工作流

第一次跑新数据：

```bash
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts --dry_run
python s08_run_pipeline.py --dataset_dir dataset --artifact_dir artifacts
```

窗口模型不达标：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts \
  --manual_feature_file artifacts/manual_feature_selection.csv \
  --skip s01,s02,s03,s04 \
  --accuracy_first_optimize \
  --stop_after s05
```

窗口模型可用，开始做端到端产品指标：

```bash
python s08_run_pipeline.py \
  --artifact_dir artifacts \
  --manual_feature_file artifacts/manual_feature_selection.csv \
  --skip s01,s02,s03,s04 \
  --with_postprocess
```

需要自动汇总产品指标并选择当前候选：

```bash
python s08_run_pipeline.py \
  --dataset_dir dataset \
  --artifact_dir artifacts_auto \
  --auto_optimize_e2e
```

部署交付前：

```bash
python -m pytest test_deploy_feature_extractor.py test_end_to_end_pipeline_guard.py -q
```
