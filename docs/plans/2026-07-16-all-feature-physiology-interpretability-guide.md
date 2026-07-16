# 全特征生理意义与可解释性手册实施计划

## 目标

生成 `FEATURE_INTERPRETABILITY_GUIDE.md`，逐项覆盖 `stage2_interpretable_v8` 的 126 个模型候选特征，并提供可直接用于人工 CSV 筛选的解释、风险和工程建议。

## 文件范围

- 新建 `FEATURE_INTERPRETABILITY_GUIDE.md`：完整人工筛选手册。
- 新建 `test_feature_interpretability_guide.py`：自动检查特征覆盖、顺序、固定字段和 README 入口。
- 修改 `README.md`：增加手册链接。

## 实施步骤

### 1. 建立失败的文档契约测试

- 从 `stage2_feature_catalog.model_candidate_names()` 获取唯一特征列表。
- 解析手册中的 `### \`FEATURE_NAME\`` 标题。
- 断言标题数量为 126、顺序完全一致、无重复和额外名称。
- 对每个特征段落断言存在以下字段：定义/公式、生理与物理意义、佩戴判别预期、鲁棒性价值、混淆与泛化风险、工程与人工筛选建议。
- 断言 README 包含 `FEATURE_INTERPRETABILITY_GUIDE.md` 链接。

运行：

```text
python -m pytest test_feature_interpretability_guide.py -q --basetemp .pytest_tmp_feature_guide_red -p no:cacheprovider
```

预期：手册文件尚不存在，测试失败。

### 2. 编写手册公共部分

- 说明文档边界：特征是统计证据而非单一生理因果。
- 定义三光区、GREEN、GMEDIAN、GTOP2、AMB/AMBX、ACC、AC/DC、带通、周期性和相关性。
- 给出人工筛选流程：先看稳定性和分层漂移，再看互补性，最后看端侧成本。
- 增加 126 项顺序索引。

### 3. 编写 126 项逐项说明

- 严格按照 catalog 顺序编写三级标题。
- 公式引用 `feature_record(name)["formula"]` 并结合 `s03` 实际实现转换为可读表达。
- 每项写满六个固定字段。
- 对无稳定单调方向的特征明确要求以 grouped-valid 和外部泛化集验证。
- 对 `mode`、固定位置、FFT、环境相关和 ACC 特征分别标注 shortcut、映射、成本、共同驱动和非活体充分条件风险。

### 4. 增加人工筛选总结与 README 入口

- 给出建议的核心、鲁棒、扩展三层选择逻辑，但不替用户自动确定特征数量。
- 给出同族冗余控制建议和商用 8 特征映射说明。
- 在 README 人工选择章节增加手册链接。

### 5. 验收

运行：

```text
python -m pytest test_feature_interpretability_guide.py -q --basetemp .pytest_tmp_feature_guide -p no:cacheprovider
python -m pytest -q --basetemp .pytest_tmp_full_feature_guide -p no:cacheprovider
python -m py_compile *.py
git diff --check
```

通过标准：126 项标题与 catalog 完全一致，每项六字段齐全，README 可发现，全量测试与静态检查无回归。
