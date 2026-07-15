# Stage1 与 Stage2 并行执行设计

> 历史设计记录：其中提到的独立商业基线随后已从活动项目删除；当前仅保留商用八特征映射与候选特征。

## 1. 背景与结论

商用设备中存在两条完整、并行执行的判断链路：

- Stage1：IR DC/ACDC 固定阈值，提供低延迟快速判断。
- Stage2：PPG/ACC 特征、模型和后处理，始终对完整时序持续执行。

Stage1 不拥有 Stage2 的计算生命周期。Stage1 关闭时只能屏蔽对外读取，不能停止
Stage2 特征、模型概率、EMA、投票或状态机更新；Stage1 打开时直接读取当前已经稳定的
Stage2 结果。

## 2. 现状问题

当前实现把 Stage1 同时当成数据过滤器和计算开关：

- `s03_extract_feature_pool.py` 删除 Stage1 未通过的样本/窗口，Stage2 训练集不是全量。
- `s06_deploy_eval.py` 在 Stage1 关闭时写入空特征并跳过模型计算。
- Stage2 window metrics 只统计 Stage1 通过窗口，无法衡量独立模型能力。
- 后处理收到的是被 Stage1 截断的概率流，打开门控后需要重新积累状态。
- `s09_commercial_compare.py` 的商业基线也只使用 Stage1 通过窗口训练和推理。

## 3. 方案比较

### 方案 A：完整解耦（采用）

Stage2 全量训练、全量推理、全量后处理；Stage1 只生成逐窗门控标志，最终对外状态为
`stage1_gate AND stage2_state`。同时报告 Stage1、独立 Stage2 和最终融合输出三套指标。

优点是训练、验证、测试和部署语义完全一致，Stage1 打开时无 Stage2 预热延迟。代价是
Stage2 计算量不再因 Stage1 关闭而下降，这与真实商用执行方式一致。

### 方案 B：只把训练集改为全量（不采用）

训练使用全量数据，但部署仍在 Stage1 关闭时跳过 Stage2。实现较小，但产生明显的
train/serve skew，无法满足并行执行背景。

### 方案 C：同时维护全量模型和门控模型（不采用）

维护两套特征池和模型。实验灵活，但增加产物、选择、部署与验收复杂度，没有商用需求支撑。

## 4. 数据流

```text
原始 PPG/ACC
  ├─ Stage1 IR 阈值流 ───────────────→ stage1_gate[t]
  └─ Stage2 全量特征 → 模型概率 → EMA/投票/状态机 → stage2_state[t]

独立 Stage2 输出：stage2_state[t]
商用对外输出：    stage1_gate[t] AND stage2_state[t]
```

Stage1 关闭不会向 Stage2 注入零概率、清空 EMA、暂停投票或重置状态。只有数据读取失败、
窗口无效或真实特征提取异常才允许 Stage2 产生缺失窗口。

## 5. 训练与特征选择

- `s03` 对 train/valid/test 的所有合法窗口提取 Stage2 特征，不调用 Stage1 作为过滤条件。
- Stage1 阈值 JSON 继续生成并用于独立 Stage1 分析和最终融合评估。
- Stage1 门控结果不进入 111 项 Stage2 候选特征，避免模型学习门控捷径。
- `s04/s05` 的排名、人工 CSV、模型搜索和 hard-negative mining 全部基于全量 Stage2 窗口。
- 数据加载失败、窗口长度不足和不可计算特征仍按原有错误策略处理，不属于 Stage1 过滤。

## 6. 推理、后处理与输出

- 预切窗和连续时序推理都对每个合法 Stage2 窗口提取特征并预测概率。
- Stage1 同步计算逐窗 `stage1_gate_flags`。为兼容已有缓存，旧字段
  `stage2_enabled_flags` 暂时保留为同值别名，但不再表示 Stage2 是否执行。
- 后处理使用完整概率流持续更新，生成独立 `stage2_states`。
- 对外状态单独计算为 `output_states = stage2_states AND stage1_gate_flags`。
- Stage1 从关闭变为打开时，不重置 Stage2 状态；输出立即反映当前 Stage2 状态。

## 7. 指标与报告

必须明确区分三种口径：

1. `stage1_only`：固定阈值快速方法的通过率、准确率和错误分层。
2. `stage2_independent`：全量窗口模型与持续后处理指标，不按 Stage1 过滤。
3. `fused_output`：`Stage1 AND Stage2` 的最终商用输出、误报和首次输出延迟。

窗口错误分析、缓存 manifest 和 PNG 报告保留 Stage1 标志用于分层，但不得从独立 Stage2
指标中删除 Stage1 关闭窗口。

## 8. 后处理搜索

`s07` 使用全量 Stage2 概率流搜索 EMA、投票、迟滞和确认参数，搜索过程中不暂停状态。
候选选择首先满足独立 Stage2 的鲁棒性约束，同时输出同一候选在最终融合口径下的指标。
test 仍只做冻结参数回放，不参与搜索。

## 9. 商业基线

`s09` 的八特征 AdaBoost 同样使用全部合法窗口训练和预测。Stage1 标志只用于生成最终
融合输出和分层报告，保证商业基线与当前 Stage2 模型使用相同的并行语义。

## 10. 兼容与验收

- 旧缓存的 `stage2_enabled_flags` 可读取，新缓存同时写入语义明确的 `stage1_gate_flags`。
- 新缓存、评估 JSON 和部署配方增加并行语义版本或明确字段说明，避免误读旧产物。
- 验收必须证明：Stage1 全关闭时 Stage2 仍产生完整概率和状态；门控打开后无需预热即可
  读取已有状态；全量特征池窗口数不因 Stage1 阈值变化而改变；最终对外输出仍受 Stage1
  屏蔽；全部图表继续只输出 PNG。
