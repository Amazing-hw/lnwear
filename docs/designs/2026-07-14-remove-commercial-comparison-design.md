# 删除商用对比功能设计

## 目标

从活动项目中完整删除独立商用 AdaBoost baseline 及其与当前模型的对比链路，同时保留
商用八特征在 Stage2 受治理特征池中的候选身份、公式、元数据、排序和人工 CSV 选择能力。

## 采用方案

采用“删除活动功能、保留历史记录”的方案：

- 删除 `s09_commercial_compare.py`。
- 删除 `s08_run_pipeline.py` 中的 `s09_cmp` 步骤、嵌入调用、自动启用逻辑和全部
  `--commercial_*` 命令行参数。
- 删除商业对比专属测试，包括 s09 行为、对比 CSV、对比 PNG 和流水线入口测试。
- 删除 `pipeline_acceptance.py` 中商业对比图的条件验收逻辑。
- 删除 README 中商业对比运行命令、产物目录、图表和步骤说明。
- 历史设计/计划文档不改写原始决策，只在相关活动设计文档中说明该功能于
  2026-07-14 被移除；历史内容不再代表当前可运行能力。

## 必须保留

以下内容属于特征池治理契约，不属于商用模型对比功能，必须完整保留：

- `stage2_feature_catalog.py` 中 `COMMERCIAL_8_FEATURE_MAPPING`。
- `COMM_GREEN_AC`、`COMM_AMB_AC` 以及其余六项规范映射特征。
- `commercial_8_member`、`commercial_original_name` 元数据字段。
- `s03_extract_feature_pool.py` 中商用特征公式和输出。
- `s04_feature_selection.py` 中八特征候选集合、完整性报告和映射元数据。
- 人工选择 CSV 中用于识别商用八特征成员的列。
- 验证上述公式、映射、元数据、可选择性和特征池完整性的测试。

## 边界与兼容性

- `--commercial_compare`、`--commercial_split`、`--commercial_fp_cost`、
  `--commercial_keep_window_probs` 和 `--stop_after s09_cmp` 均不再受支持。
- 已经生成在旧 artifact 目录中的 `commercial_compare/` 不主动删除；新代码不再读取、
  生成或验收这些历史产物。
- `commercial_8_baseline` 可继续作为“八特征候选子集”的内部名称，但不得再描述为可运行的
  AdaBoost 对比模型或流水线阶段。
- 当前 Stage1/Stage2 并行架构、模型训练、后处理和部署产物不改变。

## 测试与验收

- 先添加/调整契约测试，要求 CLI 帮助和 dry-run 不再包含 s09 或商业对比参数、步骤和产物。
- 删除只测试已移除模块的测试；保留并运行所有商用特征池映射和公式测试。
- 扫描活动 Python/README，确保不存在 `s09_cmp`、`commercial_compare` 或
  `s09_commercial_compare.py` 运行入口。
- 运行相关聚焦测试、完整 pytest、生产文件 `py_compile`、Pylint errors-only、三种 dry-run
  和 `git diff --check`。
- 所有保留绘图仍为 PNG-only，不使用浏览器。

