# Manual Feature Selection, Model Optimization, and Scientific Figures Design

> Superseded for feature count and manual-selection format by
> `2026-07-13-mode-feature-and-csv-selection-design.md`. This document retains
> the historical rationale; the active v8 workflow has 126 candidates, uses a
> CSV-only selection contract, and normalizes green layouts to three symmetric zones.

Date: 2026-07-11
Status: Approved design, pending implementation plan

## 1. Goal

Build a traceable end-to-end workflow that:

- ranks every valid Stage2 candidate using multiple independent forms of evidence;
- exposes the complete ranking in an Excel-first manual selection interface;
- trains exactly the features selected by the user, without count or category limits;
- compares baseline, searched, and train-OOF hard-negative XGBoost candidates;
- optimizes causal postprocessing for high window accuracy, low false-worn risk, and fast stable output;
- preserves strict train/valid/test separation;
- exports C engineering contracts and explicit cost warnings; and
- produces complete, publication-quality 600 DPI PNG figures with source data and manifests.

This design extends the existing `stage2_interpretable_v2` work. Where it conflicts with earlier designs, this document takes precedence for commercial-feature inclusion, manual selection freedom, model acceptance, and figure outputs.

## 2. Acceptance Targets

The primary validation objective is maximum valid-split window accuracy, subject to:

- valid window false-positive rate at or below 1% when achievable;
- postprocessing-added first stable worn-output P95 latency at or below 3 seconds;
- XGBoost total node count at or below 500;
- hard-negative retraining must not reduce valid window accuracy and must improve or preserve valid false-positive rate; and
- test data is used only once after features, model, calibration, threshold, and postprocessing are frozen.

The latency clock starts at the first valid Stage2 probability, not at signal acquisition. The 5-second Stage2 input window is therefore outside this latency metric.

If no candidate meets the false-positive or latency target, the pipeline may still produce an analysis candidate, but the acceptance report must mark the corresponding requirement as failed.

## 3. End-to-End Data Flow

```text
s03 versioned Stage2 feature pool
  -> s04 complete evidence-based ranking
  -> Excel/CSV manual selection package
  -> user sets selected=1
  -> selection contract validation and frozen JSON
  -> s05 baseline/search/hard-negative candidates
  -> valid-only model, calibration, and threshold selection
  -> s06 raw-H5 deployment replay and immutable caches
  -> s07 causal postprocessing search on valid caches
  -> frozen model and postprocessing
  -> one read-only test evaluation
  -> C contracts, acceptance report, 600 DPI PNG/CSV/manifests
```

No stage may silently crop, reorder, alias, replace, or append features selected by the user.

## 4. Complete Feature Pool

### 4.1 Governed candidates

The governed candidate pool increases from 81 to 83 independent formulas:

- the existing 81 catalog candidates remain;
- `COMM_GREEN_AC` is added using the commercial mixed RMS/MAD green AC formula; and
- `COMM_AMB_AC` is added using the commercial mixed RMS/MAD ambient AC formula.

All 83 candidates must:

- have a catalog record and stable order;
- be produced by the shared Stage2 window assembler;
- be finite on supported pathological inputs;
- include physical group, formula, preprocessing, units, signal source, numerical guard, C operators, buffer requirement, FFT flag, accumulator recommendation, and parity tolerances; and
- appear in the complete ranking even when ineligible for final selection.

### 4.2 Commercial eight-feature membership

The original commercial eight-feature surface is represented without adding exact duplicate columns:

| Commercial name | Governed candidate |
|---|---|
| `GREEN_CORR` | `GREEN_CORR` |
| `GREEN_AC` | `COMM_GREEN_AC` |
| `AMB_AC` | `COMM_AMB_AC` |
| `ACC_YSUM` | `ACC_MAG_MEAN` |
| `GREEN_DC` | `GREEN_DC_MEDIAN` |
| `AMB_DC` | `AMBX_DC_MEDIAN` |
| `GREEN_XCORR` | `GREEN_AUTO_CORR_PEAK` |
| `FFT_PEAK_MEDIAN_RATIO` | `GREEN_FFT_PEAK_MEDIAN_RATIO` |

Catalog records expose `commercial_8_member` and `commercial_original_name`. The Excel interface can filter the full commercial set directly. `s09` retains its original names and AdaBoost implementation for historical comparison.

### 4.3 Completeness audit

The feature audit must prove:

- catalog count and ordered extractor keys match exactly;
- all candidates have non-`other` semantic groups;
- all candidates have formula and C metadata;
- all 83 candidates appear exactly once in ranking JSON, CSV, and Excel;
- diagnostic fields and removed aliases do not enter the model matrix; and
- all commercial mappings are complete and unambiguous.

## 5. Evidence-Based Ranking

`s04` ranks the entire governed pool. It may mark candidates ineligible because of explicit numerical or data-quality failures, but it must retain those candidates in the audit output with reasons.

Ranking evidence includes:

1. numerical validity, missingness, variance, and finite-value behavior;
2. grouped cross-validation permutation importance;
3. train and valid univariate AUC/separation;
4. train/valid stability, SHAP consistency, and rank consistency;
5. distribution drift metrics;
6. false-positive proxy and hard-negative risk;
7. correlation/redundancy penalty; and
8. C deployment cost.

The combined score must be reproducible from exported component columns. Raw components, normalized components, weights, eligibility status, and ineligibility reasons remain visible. No implicit Top-K filtering occurs before the complete ranking is exported.

## 6. Excel-First Manual Selection Interface

### 6.1 Outputs

`s04` generates:

- `manual_feature_selection.xlsx` as the primary user interface;
- `manual_feature_selection.csv` as a version-control and audit equivalent;
- `feature_ranking_full.json` and `feature_ranking_full.csv` as immutable ranking sources; and
- `manual_selected_features.json` after the user selection is validated and frozen.

### 6.2 Workbook structure

`Feature Selection` contains one row for every governed candidate. The first column is a `selected` 0/1 dropdown. Remaining columns include:

- combined rank and feature name;
- eligibility and reasons;
- semantic group, formula, unit, signal source, preprocessing path;
- commercial-eight membership and original commercial name;
- train/valid AUC, grouped-CV importance, SHAP, stability, drift, FP proxy, and redundancy;
- deployment cost, FFT source, buffer samples, accumulator, C operators, tolerances; and
- recommendation and review-risk notes.

The sheet uses frozen headers, filters, stable column widths, wrapped text, conditional formatting, and protected contract columns. Recommended features may be highlighted, but all `selected` cells default to 0.

`Selection Summary` reports selected count, group distribution, signal sources, FFT sources, buffers, operators, accumulator requirements, and engineering-risk warnings.

`Instructions & Contract` records schema version, feature-pool version, ranking path and SHA256, workbook generation time, selection instructions, and immutable fields.

### 6.3 Selection freedom and validation

The user may select any non-empty set of eligible governed candidates, with no restriction on:

- feature count;
- semantic category count or coverage;
- signal source; or
- FFT source count.

C cost, FFT count, buffer use, and feature count are warnings only. Training must not silently modify the selection.

The importer rejects only contract errors:

- empty selection;
- unknown, duplicate, ineligible, missing, or non-computable features;
- invalid selected values;
- workbook schema, feature-pool version, or ranking SHA mismatch; or
- changed immutable contract fields.

Errors identify the worksheet, row, field, and feature. Selected feature order follows workbook row order and is preserved throughout training and deployment.

## 7. Model Training and Candidate Selection

### 7.1 Candidate families

For the frozen manual feature order, `s05` evaluates:

- the default baseline XGBoost;
- parameter-search candidates; and
- a train-OOF hard-negative weighted retraining candidate based on the best eligible pre-mining model.

All candidates use exactly the same manually selected input surface.

### 7.2 Data isolation

- Train fits models, preprocessing values, clip bounds, OOF predictions, and hard-negative weights.
- Grouped cross-validation keeps windows from the same sample in one fold.
- Valid selects model candidate, calibration method, and probability threshold.
- Test is not loaded for any selection, search, calibration, threshold, hard-negative, or postprocessing decision.

### 7.3 Hard-negative policy

Hard negatives are high-confidence negative-class errors from train OOF predictions only. The pipeline exports the selected rows, probabilities, strata, weights, and OOF fold provenance.

The weighted candidate is accepted only when, on valid:

- window accuracy is not lower than the comparison model; and
- false-positive rate improves or is unchanged.

Otherwise the pipeline atomically retains the better pre-mining model and records the rollback reason. A failed mining or retraining job cannot overwrite an existing valid model.

### 7.4 Candidate decision order

1. Reject non-finite predictions and models with more than 500 total tree nodes.
2. Prefer candidates meeting valid window false-positive rate at or below 1%.
3. Within the feasible set, maximize valid window accuracy.
4. Break close ties by lower false-positive rate, higher recall, fewer nodes, and lower cross-validation variance, in that order.
5. If no model meets the false-positive target, select the best analysis candidate by lowest false-positive rate and then accuracy, but mark deployment acceptance as failed.

Outputs include the full candidate leaderboard, CV variability, thresholds, calibration, node counts, hard-negative comparison, accepted/rolled-back decision, and artifact fingerprints.

## 8. C Engineering Contract

C friendliness is measured and reported, but feature selection is not blocked by feature-cost budgets.

The deployment package exports, in manual feature order:

- selected catalog metadata and commercial membership;
- preprocessing and operator inventory;
- buffer, FFT, numeric-range, accumulator, and tolerance requirements;
- raw golden-vector feature values before fill/clip;
- model feature order, fill values, clip bounds, and tree node count; and
- explicit risk flags for expensive feature surfaces.

The hard model constraint is at most 500 total XGBoost nodes. Unknown formulas, missing C metadata, stale pool versions, or mismatched feature order are contract failures.

## 9. Causal Postprocessing

`s07` searches postprocessing parameters only on immutable valid caches bound to the frozen model fingerprint, feature order, threshold, window geometry, stride, and skip count.

The causal chain is:

```text
Stage1 gate
  -> XGBoost probability
  -> causal median filter
  -> EMA
  -> T_on/T_off hysteresis
  -> K_on/K_off confirmation
  -> cooldown
  -> stable worn state
```

Selection rules are:

- postprocessing-added first stable worn-output P95 latency at or below 3 seconds;
- false-worn/window false-positive risk at or below 1% when achievable;
- maximize valid streaming-window accuracy inside those constraints;
- break ties by lower latency, fewer state flips, and simpler parameters; and
- never use future probabilities.

Stride must be passed explicitly to confirmation and cooldown logic. If no postprocessing candidate meets the constraints, preserve the frozen window model, mark postprocessing acceptance failed, and do not conceal the failure by sacrificing recall.

Robustness evaluation covers worn/remove/rewear transitions, probability spikes, repeated false positives, Stage1 closure/recovery, empty windows, missing ACC, quality degradation, OOD, mixed-label grouped windows, alternate strides, short samples, and initial skipped windows.

Test caches replay the frozen postprocessing once and never enter search.

## 10. Scientific Figure Contract

### 10.1 Backend and export

All visual outputs use Python and matplotlib exclusively. Required figures are 600 DPI PNG on a white background with consistent sans-serif typography, colorblind-accessible encodings, restrained semantic colors, readable final-size text, and compact lowercase panel labels.

Every quantitative figure has:

- a source-data CSV;
- a machine-readable figure manifest;
- explicit split and sample/window counts;
- metric, center, interval, and variability definitions;
- input paths and SHA256 values;
- core conclusion and panel evidence map;
- DPI, dimensions, generation time, and reviewer-risk notes; and
- a visible indication when test is the final read-only evaluation.

### 10.2 Evidence families

Required figure families are:

1. Feature-pool completeness, group composition, ranking evidence decomposition, train/valid stability, drift, redundancy, C operator/buffer cost, and commercial-eight placement.
2. Manual-selection audit: selected versus unselected ranks, groups, signals, FFT/buffer/operator cost, and warnings.
3. Training and search: candidate accuracy/FPR/nodes, CV variability, ROC/PR, threshold tradeoff, calibration, importance, and SHAP consistency.
4. Hard negatives: OOF error probabilities, mined-negative strata, before/after valid metrics, and acceptance/rollback evidence.
5. Postprocessing: accuracy/FPR/latency Pareto surface, parameter heatmaps, first-output distribution, state flips, and false-worn risk.
6. End-to-end robustness: Stage1-to-state funnel, window/stream/sample metrics, strata, OOD, missing ACC, short samples, and error timelines.
7. Commercial comparison using the original eight-feature AdaBoost baseline.
8. A scientific-report overview figure using an asymmetric mixed-modality layout. The hero evidence is frozen-model window accuracy, false-positive rate, and added response latency; supporting panels show manual selection, hard-negative impact, postprocessing Pareto behavior, and final test acceptance.

The figure set is narrative rather than a collection of decorative dashboards. Each panel must provide unique evidence for its stated conclusion.

### 10.3 Figure QA

Automated and manual QA checks:

- PNG exists, is nonblank, and records 600 DPI metadata;
- expected pixel dimensions agree with the figure size;
- labels, legends, and ticks stay inside canvas bounds;
- colors remain distinguishable without relying on red/green alone;
- plotted values match source CSV values;
- required manifest fields and input hashes exist; and
- all mandatory figure/source/manifest triples are present.

Missing mandatory report outputs mark the scientific-report acceptance as incomplete even when model tests pass.

## 11. Error Handling and Artifact Integrity

- Selection, training, cache, and final-model artifacts are written atomically.
- Search, hard-negative, plotting, or cache failures cannot overwrite the last valid frozen model.
- Stale versions, fingerprints, rankings, or cache contracts fail with actionable messages.
- C cost overruns are explicit warnings and never silently alter selected features.
- Model node-budget violations reject only the offending candidate.
- Analysis candidates and accepted deployment candidates have visibly different status.

## 12. Test Strategy

Tests cover six layers:

1. Exact 83-candidate catalog/extractor equality, commercial mapping, finite pathological-input behavior, metadata, and complete ranking coverage.
2. Workbook generation, dropdown validation, formatting, protected contract fields, CSV equality, selected-order preservation, and tamper detection.
3. Arbitrary non-empty feature counts and categories with no silent crop, append, substitution, or reorder.
4. Deterministic baseline/search/hard-negative selection, node-budget rejection, valid constraints, atomic acceptance, and rollback.
5. Postprocessing latency, FPR, causality, stride, robustness cases, immutable cache provenance, and test isolation.
6. Figure 600 DPI metadata, size, nonblank pixels, bounds, source-data equality, manifests, and mandatory-output completeness.

The final regression gate includes compile checks, focused contract tests, a synthetic end-to-end run, and the complete pytest suite.

## 13. Final Acceptance Report

The final report has separate pass/fail sections:

- `Feature pool`: 83 independent formulas, commercial mapping, complete ranking, metadata, and finite behavior.
- `Selection`: exact Excel-selected order, contract hashes, and no silent changes.
- `Model`: valid candidate decision, FPR, accuracy, CV variability, node count, and hard-negative decision.
- `Postprocess`: added P95 latency, FPR, streaming accuracy, flips, and robustness.
- `Test`: one frozen read-only evaluation.
- `C readiness`: feature/operator/buffer/FFT/tolerance contract, tree nodes, and warnings.
- `Figures`: mandatory PNG/CSV/manifest completeness and QA.

No aggregate success label may hide a failed section.
