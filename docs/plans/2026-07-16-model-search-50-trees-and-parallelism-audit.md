# 模型搜参 50 棵树上限与并行策略审计实施计划

## 目标

用单一配置源把所有 XGBoost 搜参候选限制在 50 棵以内，并通过代码审计与测试确认各阶段采用受控外层并行、单模型默认单线程。

## 文件范围

- 新建 `model_search_limits.py`：保存最大树数、默认候选和统一解析校验函数。
- 修改 `s05_train_final_model.py`：从统一配置读取默认候选，并在构建搜索轴时强制校验。
- 修改 `s08_run_pipeline.py`：从统一配置读取默认 CLI 值，参数解析后立即拒绝大于 50 的候选。
- 修改 `test_model_search_config.py`：覆盖默认网格、s05 直接入口、s08 CLI 入口和 dry-run。
- 修改 `README.md`：同步最大 50 棵和逐阶段并行说明。

## 实施步骤

### 1. 先增加失败测试

- 将默认候选断言改为 `[20,25,30,35,40,45,50]`，明确 `max(...) == 50`。
- 增加 `s05.build_model_search_axes()` 对 `51` 抛出 `ValueError` 的测试。
- 增加 `s08 --model_search_n_estimators 20,51 --dry_run` 返回非零且输出“maximum is 50”的测试。
- 增加默认 `s08` dry-run 不出现 `55,60` 的测试。

运行：

```text
python -m pytest test_model_search_config.py -k "n_estimators or search_budget" -q --basetemp .pytest_tmp_tree_limit_red -p no:cacheprovider
```

预期：旧默认网格和缺少硬校验导致新增测试失败。

### 2. 实现统一硬上限

- 在 `model_search_limits.py` 定义：

```python
MAX_MODEL_SEARCH_N_ESTIMATORS = 50
DEFAULT_MODEL_SEARCH_N_ESTIMATORS = (20, 25, 30, 35, 40, 45, 50)
```

- 提供 `parse_model_search_n_estimators(raw)`：去重并保持输入顺序；空值、非整数和大于 50 的值抛出 `ValueError`。
- `s05` 的 `DEFAULT_MODEL_SEARCH_SPACE["n_estimators"]` 由统一常量生成，`build_model_search_axes()` 使用统一解析函数。
- `s08` 的默认参数由统一常量生成；`parse_args()` 后调用统一解析函数并通过 `ArgumentParser.error()` 返回清晰 CLI 错误。

### 3. 验证并行策略

- 运行现有并行契约测试，确认 s03 顺序重组、s05 候选线程池、worker 继承、强制串行回退和 s07 进程池缓存初始化。
- 逐项复核 s01–s07 的 worker 分辨逻辑、进程/线程池入口和小任务串行阈值。
- 统一把 s01–s07 的 worker 数限制到实际任务数，并为 s07 增加 `WL_FORCE_SERIAL` 回退，避免空闲进程。
- 确认 `s04/s05/s06` 的单模型 `n_jobs=1` 默认值，以及 `s08` 的 BLAS/OpenMP 线程上限，避免嵌套并行。

运行：

```text
python -m pytest test_prewindowed_h5.py test_model_search_config.py test_s07_postprocess_optimize.py -q --basetemp .pytest_tmp_parallel_contract -p no:cacheprovider
```

### 4. 同步文档并完整验收

- README 默认模型搜索树数更新为最大 50。
- README 明确各阶段并行粒度、小任务串行优化、`WL_FORCE_SERIAL` 回退和单模型线程限制。
- 执行全量测试、Python 编译、生产代码错误级 Pylint、`git diff --check` 和 s08 dry-run。

通过标准：

```text
所有测试通过；默认/自定义搜参均不可能超过 50 棵；生产代码无错误级静态问题；并行契约通过。
```
