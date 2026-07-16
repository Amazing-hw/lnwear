# 默认人工特征训练与关闭后处理搜参设计

状态：用户已确认（2026-07-16）

## 目标

默认主流程采用人工特征选择的两阶段运行方式：第一阶段运行到完整特征排序并暂停；用户修改 CSV 后，第二阶段严格基于人工选择的特征进行充分模型训练、搜参、hard-negative 优化、评估和部署导出。默认不执行任何后处理参数搜索。

## 默认流程

### 第一阶段：生成排序 CSV

- `feature_selection_mode=manual`。
- 运行 `s01`–`s04`，生成完整 `manual_feature_selection.csv` 后暂停。
- 不运行自动候选子集搜索，不自动决定特征数量，不进入模型训练。

### 人工选择

- 用户只修改 `manual_feature_selection.csv` 的 `selected` 列。
- 选中特征的名称、顺序和数量由 CSV 唯一决定。
- 其他列继续作为不可修改的排序与版本契约。

### 第二阶段：训练与部署

- 使用 `--skip s01,s02,s03,s04` 和同一 artifact 目录恢复。
- `s05` 严格读取人工 CSV，不搜索特征数量。
- 默认执行 staged group-CV XGBoost 搜参，最大深度候选为 2、3、4、5，并按配置并行评估候选。
- manual 恢复训练默认启用 train OOF hard-negative 候选；只有 valid 不退化时才接受。
- 随后执行模型评估、错误样本分析以及模型、独立特征脚本和部署配方导出。

## 后处理策略

- 默认 `optimize=false`，不运行 `s06_opt` legacy 状态机参数搜索。
- 默认 `export_window_cache=false`，不为后处理搜索导出窗口缓存。
- 默认 `optimize_postprocess=false`，不运行 `s07_post`。
- 默认流程不隐式打开上述开关。
- 后处理实现和 `--with_postprocess`、`--optimize`、`--optimize_postprocess` 等显式入口继续保留，供未来主动恢复；只有用户显式指定时才运行。

## 验收

- 默认 dry-run 明确停在 `s04`，且不包含 `s06_opt`、窗口缓存和 `s07_post` 命令。
- manual CSV 恢复 dry-run 包含充分模型搜参、hard-negative、评估和部署导出，不包含特征数量搜索或后处理搜参。
- 导出模型和独立特征脚本的 `FEATURE_ORDER` 与人工 CSV 完全一致。
- README 首要运行说明与代码默认值一致。
- 全项目测试、编译、错误级静态检查和差异检查通过。

