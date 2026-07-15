# 三光区鲁棒特征扩展设计

## 目标

在不修改 `mode` 和原始通道到三个中心对称光区映射的前提下，增强单边翘起、局部外光、
公共环境光和运动污染场景下的佩戴活体特征。保留现有 91 项候选，新增三光区稳健候选，
由完整排序和人工 CSV 决定最终训练特征与数量。

## 不变约束

- `get_channels_from_window` 仍是唯一布局适配器，输出 `g1/g2/g3`。
- 所有新增空间特征对 `g1/g2/g3` 任意排列保持不变。
- 不使用光区编号、绝对方向、顺逆时针或最差光区索引。
- 不修改现有 `GREEN_*`、`GTOP2_*` 和商用八特征公式。
- 所有候选必须有限、可解释、可导出到 C；所有绘图继续 PNG-only。

## 信号表示

每个光区先独立执行有限值处理、毛刺修复、滚动中位数去趋势和轻度平滑，然后并行生成：

1. `GREEN`：三个光区逐点均值，保留整体能量和商用平均语义。
2. `GTOP2`：按整窗 pulse RMS 选择两个区域并逐点均值，保留强度优先分支。
3. `GMEDIAN`：三个光区逐点中位数，抑制一个异常区域。
4. 三种 pair：`mean(g1,g2)`、`mean(g2,g3)`、`mean(g3,g1)` 全部计算；不额外硬选
   “最佳 pair”，只输出三对指标的排列不变顺序统计。

RMS 只视为 AC 强度证据，不视为通道质量结论。融合后的特征不能替代三个光区的原始
空间一致性特征。

## 新增候选

### 中位数与 top2 补充

1. `GMEDIAN_AC_DC_RATIO`：中位数 pulse RMS / 中位数 raw DC。
2. `GMEDIAN_CORR`：`corr(gmedian_pulse, moving_average(gmedian_pulse, 0.15s))`。
3. `GMEDIAN_AUTO_CORR_PEAK`：中位数 pulse 在 40–180 bpm 延迟范围的最大归一化自相关。
4. `GMEDIAN_FFT_PEAK_MEDIAN_RATIO`：中位数 pulse 频带峰值/中位频谱比。
5. `GTOP2_CORR`：在 top2 合成 pulse 上复用商用平滑相关公式。
6. `G_TOP2_ALL_CORR`：top2 pulse 与三光区均值 pulse 的相关性。
7. `G_WEAK_TO_TOP2_CORR`：AC 最弱区域 pulse 与 top2 pulse 的相关性。

### 三光区主频共识

8. `G_ZONE_DOM_FREQ_MAD_HZ`：三个光区 0.5–5 Hz 主频相对中位主频的平均绝对偏差；三点场景下不用传统 median absolute deviation，避免“两区一致、一区偏离”时错误返回 0。
9. `G_ZONE_HR_SUPPORT_RATIO`：主频落在三个光区中位主频 ±0.20 Hz 内的光区比例。

### 三种 pair 全评估

每对都计算 pair pulse 周期性、pair AC/DC、两区域主频差以及 pair pulse 与 ambient pulse
的绝对相关。只输出以下顺序统计：

10. `G_PAIR_PERIODICITY_MAX`。
11. `G_PAIR_PERIODICITY_MEDIAN`。
12. `G_PAIR_FREQ_GAP_MIN_HZ`。
13. `G_PAIR_FREQ_GAP_MEDIAN_HZ`。
14. `G_PAIR_ACDC_MEDIAN`。
15. `G_PAIR_AMB_ABS_CORR_MIN`。
16. `G_PAIR_AMB_ABS_CORR_MEDIAN`。

最大/最小统计回答“是否存在一对可信光区”，中位数统计回答“是否具备至少 2-of-3
空间支持”。不能只用单个最佳值作为佩戴证据。

### 环境光残差与实验候选

对每个光区 pulse 使用受保护的线性投影：

`residual_i = zone_i - cov(zone_i, ambient)/guarded_var(ambient) * ambient`

原信号继续保留，残差只用于新增对照特征：

17. `G_AMB_RESIDUAL_2OF3_PERIODICITY`：三个残差周期强度的中位数。
18. `G_AMB_RESIDUAL_PAIR_CORR_MAX`：三对残差相关性的最大值。
19. `G_ZONE_PHASE_CONCENTRATION`：三个光区在中位主频 FFT bin 上的相位集中度，标记为
    experimental/high-cost。
20. `G_PAIR_SPECTRAL_CONSENSUS`：三对归一化频带功率谱余弦相似度的中位数，标记为
    experimental/high-cost。

特征池版本升级，候选数量由 91 增至 111，仍满足 80–120 项治理约束。

## 数值与工程规则

- 平坦信号、短信号、零 ambient 方差和非有限输入全部返回有限值。
- 比值使用信号尺度保护分母；相关性使用方差保护。
- 相位集中度在频谱能量不足或有效主频缺失时返回 0。
- pair 和三光区统计只使用 sort/median/min/max/MAD，不暴露组合身份。
- 复用已有每区 pulse、FFT 和 ambient pulse 缓存，禁止为同一信号重复 FFT。
- 新候选在 catalog 中声明公式、预处理、C 操作符、范围、成本和风险标志。

## 验收

- 测试先行，先证明旧特征池缺少全部 20 项候选。
- 证明所有新增空间特征在六种光区排列下完全不变。
- 合成场景覆盖：三路干净周期、单区幅度衰减、单区毛刺、单区外光、三路公共 ambient、
  三路无脉搏、异步光区和频率不一致。
- 验证中位数和至少一对 pair 在单区异常时保持稳定，同时 2-of-3 中位统计不会被单路强
  RMS 噪声误导。
- 验证 catalog、提取顺序、有限输出、人工 CSV、C 合同和部署特征脚本完全一致。
- 运行聚焦测试、完整 pytest、生产文件 `py_compile`、Pylint errors-only、三种 dry-run、
  `git diff --check` 和旧语义扫描。
