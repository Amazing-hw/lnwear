# Visual Analysis and PPT Report Design

Date: 2026-07-10
Status: Confirmed, pending implementation

## 1. Purpose

The wearing-liveness pipeline needs a consistent visual evidence package for two uses:

1. Dense daily analysis for data quality, feature review, model interpretation, and error diagnosis.
2. Clean 16:9 PNG pages that can be inserted directly into an English-language PPT presentation.

The visual report must complement the manual feature-selection workflow. Dataset-level figures cover all relevant samples and windows, while feature-level figures are generated only for the final manually selected training features.

## 2. Figure Contract

Core conclusion: the final wearing-liveness model is supported by traceable data quality, stable and interpretable selected features, defensible performance, and localized error analysis.

Figure archetype: quantitative grid with four asymmetric summary pages.

Backend: Python only, using Matplotlib and Seaborn for drawing, rendering, exporting, and visual QA.

Output contract:

- English text only.
- PNG only.
- White background.
- Analysis figures use dynamic dimensions based on content density.
- PPT summary pages use exactly 1920 x 1080 pixels at 16:9.
- Every quantitative figure has a corresponding source-data CSV.
- Every output is indexed in a machine-readable manifest.
- Text overlap, clipping, blank-image, and unreadable-layout checks are mandatory.

Evidence hierarchy:

1. Data integrity and split composition.
2. Final selected-feature separation and stability.
3. Model discrimination, calibration, threshold behavior, and feature usage.
4. Error concentration, hard negatives, state behavior, and latency risk.

## 3. Architecture

Add an independent module:

```text
s10_visual_report.py
```

The module reads existing artifacts and never retrains a model. It supports:

```powershell
python s10_visual_report.py --artifact_dir artifacts --profile all
python s10_visual_report.py --artifact_dir artifacts --profile analysis
python s10_visual_report.py --artifact_dir artifacts --profile ppt
python s10_visual_report.py --artifact_dir artifacts --sections data,features
```

`s08_run_pipeline.py` invokes it automatically after the final model, selected feature list, and end-to-end evaluation artifacts are available. The module remains independently runnable so styling or report changes do not require rerunning feature extraction or training.

The new pipeline step is:

```text
s10_visual
```

Default `s08` execution includes `s10_visual` before the final deployment cookbook step. It can be skipped explicitly or targeted with `--stop_after s10_visual`.

The analysis profile produces the complete diagnostic figure set. The PPT profile
does not create a mechanical 16:9 duplicate of every analysis figure; it composes
the same source evidence into four curated summary pages.

## 4. Inputs

The visual module reads whichever of the following artifacts exist and records missing optional inputs in the manifest:

- `splits.json`
- `stage1_threshold.json`
- `stage1_train_windows.csv`
- `stage1_valid_windows.csv`
- `feature_pool_train.csv`
- `feature_pool_valid.csv`
- `feature_pool_test.csv`
- `feature_ranking_full.json`
- `manual_selected_features.json`
- `model_bundle.pkl`
- `final_model_config.json`
- `model_search_results.csv`
- `end_to_end_eval_{split}_state_machine.json`
- `window_error_analysis_{split}_state_machine.csv`
- `error_stratification_{split}_state_machine.json`
- `hard_negatives_{split}_state_machine.json`
- optional postprocess replay and generalization-audit artifacts

Required inputs are validated before rendering. Dataset overview plots may be produced before manual selection, but selected-feature, model, error, and PPT reports require a finalized manual feature file and final model artifacts.

Section-level requirements are explicit:

- `data`: split, Stage1, and feature-pool artifacts.
- `features`: all `data` inputs plus the final manual feature file.
- `model`: final manual feature file, model bundle, configuration, and feature pools.
- `errors`: end-to-end evaluation and window-error artifacts.
- `ppt`: all four sections.

When `--profile ppt` is requested without an explicit `--sections`, all four
sections are loaded automatically. An explicit section subset is valid only
for `analysis`; partial PPT pages are not generated.

Test data is used only for final visualization and generalization diagnosis. It does not influence feature ranking, manual feature validation, model selection, calibration, or threshold selection.

Source labels shown in figures use stable basenames or generated source IDs. Full
local filesystem paths are never rendered into PNG files.

## 5. Output Structure

```text
artifacts/visual_report/
  analysis/
    data_overview/
    selected_features/
    model_interpretation/
    errors_robustness/
  ppt/
    data_quality_overview.png
    selected_feature_evidence.png
    model_performance.png
    error_robustness.png
  source_data/
  visual_report_manifest.json
  visual_qa.json
```

Analysis filenames use stable numeric prefixes so their reading order remains deterministic. Feature names are sanitized only for filenames; titles retain the exact model feature name.

## 6. Shared Visual System

### 6.1 Typography

- Font family: Arial, Helvetica, DejaVu Sans fallback.
- Letter spacing remains default.
- Analysis base font: 10-12 pt, dynamically increased for sparse panels.
- PPT title: 30-36 pt.
- PPT panel title: 20-24 pt.
- PPT axis and tick text: at least 16 pt.
- Legends are frameless and preferably shared across panels.
- Long feature names wrap at semantic separators such as underscores; font size is reduced only after wrapping.

### 6.2 Semantic Colors

- Train: `#3B6FB6`
- Valid: `#D89C3D`
- Test: `#3D8B6D`
- Not-worn / target 0: `#C65A5A`
- Worn / target 1: `#3A8F5D`
- Neutral/reference: `#6F7782`
- Warning/OOD: `#A45A9C`
- Threshold/reference lines: `#252A31`

The same role uses the same color in every figure. Red and green are reinforced with line style, markers, or hatching where grayscale interpretation matters.

### 6.3 Layout Rules

- Top and right spines are hidden.
- Grid lines are omitted or limited to low-alpha horizontal guides.
- Legends never cover data; use shared top legends, direct labels, or a legend-only area.
- Axes that invite comparison use identical limits.
- Figure sizes are computed from feature count, category count, and longest label length.
- No nested decorative panels or visual effects unrelated to the evidence.
- Each PPT page has one dominant evidence panel and smaller supporting panels.

## 7. Analysis Figure Set

### 7.1 Data Overview

1. `01_split_sample_window_counts.png`
   - Sample counts and Stage2 window counts by train/valid/test and class.
   - Shows exact `n` labels.

2. `02_class_balance_by_split.png`
   - Sample-level and window-level class proportions.
   - Includes imbalance ratios.

3. `03_sample_duration_windows_per_sample.png`
   - Duration and windows-per-sample distributions by split and class.

4. `04_stage1_funnel_pass_rates.png`
   - Input samples, Stage1 pass, Stage2 emitted windows, and final positive output.
   - Includes target-specific Stage1 pass/reject rates.

5. `05_source_mode_composition.png`
   - H5 source, hardware mode, fallback, and OOD composition.
   - High-cardinality sources use ranked horizontal bars with an explicit remainder category.

6. `06_data_quality_missing_ood.png`
   - Missing feature rate, invalid feature counts, OOD rate, and fallback reason summary.

### 7.2 Final Selected Features

Only features in the final `manual_selected_features.json` are plotted.

For every selected feature, generate:

1. `feature_{index}_{name}_distribution.png`
   - A two-row by three-column analytic plate; columns are train, valid, and test.
   - Top row: violin plus compact box and controlled jitter, with target encoded consistently by color and marker.
   - Bottom row: target-specific ECDF curves under the same split-specific x limits.
   - Annotation block: sample/window `n`, univariate AUC, robust effect size, missing rate, PSI, and KS statistic.

2. `feature_{index}_{name}_density.png`
   - Three split panels with identical x limits and target-specific density curves.
   - Falls back to histogram/ECDF when a distribution is discrete or nearly constant.

Generate selected-feature summaries:

3. `selected_feature_effect_size_auc.png`
   - Ranked effect size and AUC with train/valid/test comparison.

4. `selected_feature_split_stability.png`
   - Heatmap of AUC, PSI, mean shift, missing rate, and direction consistency.

5. `selected_feature_correlation.png`
   - Train correlation heatmap with hierarchical ordering.
   - Upper triangle may display values only when readable.

6. `selected_feature_group_cost.png`
   - Feature-group composition, scale-dependence flags, and deployment cost.

### 7.3 Model Interpretation

1. `model_tree_feature_usage.png`
   - XGBoost gain, cover, and split count for final selected features.

2. `model_shap_summary.png`
   - SHAP beeswarm or compact distribution summary for the final raw XGBoost model.
   - Contributions are computed with XGBoost native `pred_contribs=True`; the external `shap` package is not required.
   - Uses a bounded sample for rendering only; all summary statistics use the full eligible dataset.

3. `model_shap_dependence_{feature}.png`
   - Dependence plots for the highest-impact selected features.
   - Number of plots is bounded by the smaller of five or selected feature count.
   - Contributions explain the raw XGBoost margin; calibration behavior is shown separately in the calibration panel.

4. `model_feature_direction_summary.png`
   - Signed class effect, SHAP direction, and monotonic trend summary.

5. `model_probability_distribution.png`
   - Probability distributions by class and split.

6. `model_roc_pr_calibration.png`
   - ROC, precision-recall, and calibration panels with split labels and sample counts.

7. `model_threshold_tradeoff.png`
   - Threshold versus precision, recall, FPR, F1/F0.5, and selected operating point.

8. `model_search_complexity.png`
   - Accuracy/FP/model-node trade-off when model-search artifacts exist.

### 7.4 Errors and Robustness

1. `error_fp_fn_by_split_mode.png`
   - FP/FN counts and rates by split and hardware mode.

2. `error_by_time_quality_ood.png`
   - Error rate by window time, signal-quality bucket, Stage1 enable state, and OOD bucket.

3. `hard_negative_probability_profiles.png`
   - Ranked high-risk negative samples with mean/max probability and false-worn duration.

4. `state_stage_comparison.png`
   - Stage1, raw Stage2, and state-machine accuracy/precision/recall/FPR comparison.

5. `latency_false_worn_distribution.png`
   - First-worn output latency and false-worn event duration distributions.

6. `error_temporal_trace_{sample}.png`
   - Representative FP and FN traces: target, Stage1 enable, raw probability, EMA score, threshold lines, and final state.
   - The report selects a bounded, deterministic set of the highest-risk examples.

## 8. PPT Summary Pages

All PPT pages are exactly 1920 x 1080 PNG files with safe margins and large text.

### 8.1 `data_quality_overview.png`

Core conclusion: the train/valid/test data structure and Stage1 filtering are visible and auditable.

Panel map:

- Hero: sample and window counts by split and class.
- Support: Stage1 funnel, duration/window distributions, source/mode/OOD composition.

### 8.2 `selected_feature_evidence.png`

Core conclusion: the manually selected features provide class separation while retaining cross-split stability and manageable deployment cost.

Panel map:

- Hero: selected-feature AUC/effect-size stability heatmap.
- Support: correlation heatmap, feature group/cost summary, compact distribution panels for the strongest features.

When many features are selected, the page dynamically reduces per-feature annotations but never removes a selected feature from the heatmap.

### 8.3 `model_performance.png`

Core conclusion: the final model discriminates and calibrates the classes at a traceable threshold.

Panel map:

- Hero: ROC and precision-recall curves.
- Support: probability distribution, calibration, threshold trade-off, and top feature usage.

### 8.4 `error_robustness.png`

Core conclusion: remaining failure modes are localized by split, mode, time, signal quality, OOD status, and state-machine behavior.

Panel map:

- Hero: FP/FN risk heatmap or ranked strata.
- Support: hard-negative summary, latency/false-worn distributions, and Stage1/Stage2/state comparison.

## 9. Source Data and Manifest

Every figure writes a CSV into `source_data/` containing the exact plotted values. Large per-window tables may be summarized, but the manifest records the source artifact and aggregation rule.

`visual_report_manifest.json` contains one entry per figure:

```json
{
  "figure_id": "selected_feature_split_stability",
  "profile": "analysis",
  "path": "analysis/selected_features/selected_feature_split_stability.png",
  "source_data": "source_data/selected_feature_split_stability.csv",
  "inputs": ["feature_pool_train.csv", "feature_pool_valid.csv", "feature_pool_test.csv"],
  "selected_features_hash": "...",
  "model_fingerprint": "...",
  "width_px": 2400,
  "height_px": 1600,
  "qa_status": "pass"
}
```

## 10. Strict Visual QA

Each rendered PNG is reopened and checked before the step completes.

### 10.1 Pre-render checks

- Required columns and finite data exist.
- Label lengths and category counts are measured.
- Figure size, margins, tick density, wrapping, and legend placement are selected dynamically.
- Empty or single-valued inputs select an explicit fallback chart instead of producing a misleading density plot.

### 10.2 Renderer checks

Use the Matplotlib renderer bounding boxes to verify:

- Title, subtitle, panel labels, axes labels, tick labels, legends, annotations, and colorbars remain within the canvas.
- Adjacent tick labels do not overlap along their axis after rotation/wrapping.
- Registered layout-critical text objects do not overlap above a small tolerance; expected text inside bars, heatmap cells, or data marks is exempt.
- Legends do not overlap plotted data regions when placed inside an axis.
- Long feature labels fit after wrapping and dynamic sizing.

### 10.3 Pixel checks

Reopen each PNG with Pillow and verify:

- Exact PPT dimensions for the four PPT pages.
- Image is not blank or nearly uniform.
- Non-white content occupies a reasonable fraction of the canvas.
- Content does not touch the outer safety margin.
- PNG can be decoded and has the expected RGB/RGBA mode.

### 10.4 Retry and failure policy

On QA failure, rerender using deterministic escalation steps:

1. Increase figure dimensions or subplot spacing.
2. Move legends to a dedicated area.
3. Wrap labels and reduce tick density.
4. Reduce annotation density while preserving essential values.
5. Reduce font size only down to the profile-specific minimum.

If a figure still fails after the bounded retry sequence, `s10_visual_report.py` exits nonzero. It does not silently omit the figure. `visual_qa.json` records the failed rule, attempts, final bounding boxes, and suggested correction.

## 11. CLI

Proposed arguments:

```text
--artifact_dir PATH
--profile {analysis,ppt,all}
--sections LIST
--split {train,valid,test}
--dpi INT
--max_shap_dependence INT
--max_error_traces INT
--strict_qa / --no-strict_qa
--overwrite / --no-overwrite
```

Defaults:

```text
profile: all
sections: data,features,model,errors
split: test
dpi: 200 for 1920x1080 PPT geometry; 300 for analysis outputs
max_shap_dependence: 5
max_error_traces: 12
strict_qa: true
overwrite: true
```

PPT pixel dimensions are controlled by figure inches and DPI together and asserted after export.

`--dpi` applies to analysis outputs. PPT rendering uses fixed geometry
`figsize=(9.6, 5.4)` at 200 DPI so the exported image is always 1920 x 1080.

## 12. Error Handling

- Missing manual feature file: `features`, `model`, and `ppt` fail with an actionable message; `--sections data` remains valid.
- Missing test feature pool: `features`, `model`, and `ppt` fail; explicitly requested `--sections data` may still run when its inputs are complete.
- Native XGBoost contribution export failure: model interpretation fails and reports the exact booster/model compatibility error.
- Missing optional postprocess artifacts: postprocess-only panels are marked skipped, while raw model and default state-machine plots continue.
- Feature in model bundle but absent from a split: fail and list the split and feature.
- Output path collision: overwrite by default; manifest is regenerated atomically.

## 13. Tests

1. Output tree and stable filenames are generated.
2. Feature-level plots are generated only for final manually selected features.
3. All selected features appear in the summary heatmap and correlation plot.
4. Train/valid/test are visually and numerically represented in feature distributions.
5. Test data does not affect ranking or model-selection artifacts.
6. Every PNG has a corresponding source-data CSV and manifest entry.
7. PPT pages are exactly 1920 x 1080.
8. Synthetic long labels trigger wrapping without clipping.
9. Many selected features trigger dynamic sizing without overlap.
10. Constant and sparse features use fallback plots and do not crash KDE.
11. Deliberate overlap and clipping fixtures fail strict QA.
12. Blank-image fixtures fail pixel QA.
13. `s08 --dry_run` contains one `s10_visual` command after final model/evaluation readiness.
14. Standalone `s10_visual_report.py` reruns without touching model artifacts.
15. Existing user modifications outside the report output tree are never overwritten.
16. `--sections data` runs without a manual feature file, while feature/model/PPT sections reject its absence.
17. Native XGBoost contribution plots use the exact final feature order and do not require the external `shap` package.

## 14. Acceptance Criteria

- The report covers dataset quality, final selected features, model interpretation, and errors/robustness.
- Per-feature figures are limited to the final manually selected training features.
- Every selected-feature distribution distinguishes train, valid, and test.
- All labels, titles, legends, and annotations are English and visually unobstructed.
- Analysis and PPT versions are generated automatically and independently rerunnable.
- PPT summary pages are directly usable as 16:9 PNG slides.
- All plotted data is traceable through source CSVs and the manifest.
- Strict QA prevents blank, clipped, or overlapping figures from being reported as successful.
