# Manual Feature Training and Scientific Figures Implementation Plan

> Historical plan. The active CSV-only contract is defined in
> `2026-07-13-mode-feature-and-csv-selection.md`; the active pool is now v8 with
> 126 governed candidates and fixed-position three-zone features;
> Excel/OpenPyXL steps below are no longer operational requirements.

> Implementation checklist for the wearing-liveness pipeline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved 83-feature ranking, Excel-first manual selection, valid-only model/hard-negative decision, causal postprocessing acceptance, and 600 DPI scientific reporting workflow.

**Architecture:** Keep `s03-s09` as the operational pipeline, add small contract-focused modules for Excel selection and scientific figure output, and make existing stages consume those contracts. Preserve the current versioned catalog, model bundle, cache provenance, and deployment package boundaries.

**Tech Stack:** Python 3, NumPy, pandas, openpyxl, XGBoost, scikit-learn, matplotlib, Pillow, pytest.

---

### Task 1: Add the Two Independent Commercial AC Candidates

**Files:**
- Modify: `stage2_feature_catalog.py`
- Modify: `s03_extract_feature_pool.py`
- Modify: `test_interpretable_feature_pool.py`

- [ ] Add a failing test asserting catalog/extractor count 83, `COMM_GREEN_AC` and `COMM_AMB_AC` formulas, commercial-name mapping, finite values, and ordered candidate equality.
- [ ] Run `python -m pytest test_interpretable_feature_pool.py -q` and confirm failure because the catalog still has 81 candidates.
- [ ] Add catalog metadata fields `commercial_8_member` and `commercial_original_name` to every record, with mappings for the eight commercial members.
- [ ] Compute the two new features from the governed pulse signals using `0.5 * RMS + 0.5 * 1.4826 * MAD`.
- [ ] Run focused feature, deployment, and prewindowed tests.

### Task 2: Build the Excel Manual Selection Contract

**Files:**
- Create: `manual_feature_selection.py`
- Modify: `s04_feature_selection.py`
- Modify: `s05_train_final_model.py`
- Modify: `s08_run_pipeline.py`
- Create: `test_manual_feature_selection_excel.py`

- [ ] Add failing tests for an `.xlsx` workbook with `Feature Selection`, `Selection Summary`, and `Instructions & Contract`; 83 ordered rows; 0/1 validation; frozen/filterable headers; protected immutable cells; commercial columns; and an equivalent CSV.
- [ ] Add failing importer tests for arbitrary non-empty selections, workbook-order preservation, no category/count/FFT limits, empty/unknown/duplicate/ineligible rejection, and ranking SHA/version tamper detection.
- [ ] Implement `export_manual_selection_workbook(ranking_payload, output_dir)` using openpyxl and an immutable ranking SHA256.
- [ ] Implement `load_manual_selection_workbook(path, ranking_path, train_columns, valid_columns)` returning selected features plus provenance and engineering warnings.
- [ ] Make `s04` export the workbook after complete ranking and make `s05` accept `.xlsx`, freeze the validated result to `manual_selected_features.json`, and preserve order.
- [ ] Update `s08` manual pause/resume messages and defaults to point to the Excel file while retaining JSON compatibility.
- [ ] Run the Excel contract tests and existing manual-mode tests.

### Task 3: Make Ranking Coverage and Components Auditable

**Files:**
- Modify: `s04_feature_selection.py`
- Modify: `test_s04_progress_diagnostics.py`
- Modify: `test_interpretable_feature_pool.py`

- [ ] Add failing tests that every catalog candidate appears exactly once in ranking JSON/CSV/Excel, including ineligible rows.
- [ ] Add failing tests that component scores, normalized scores, weights, eligibility, reasons, commercial membership, and C metadata are present and sufficient to recompute the combined score.
- [ ] Refactor ranking export to left-join diagnostics onto catalog order rather than dropping cleaned-out features.
- [ ] Export an explicit feature-pool completeness JSON with catalog, extracted, ranked, eligible, and ineligible counts.
- [ ] Run all s04, feature governance, and deployment-policy tests.

### Task 4: Add Deterministic Baseline/Search/Hard-Negative Acceptance

**Files:**
- Modify: `s05_train_final_model.py`
- Modify: `test_model_search_config.py`
- Create: `test_model_candidate_acceptance.py`

- [ ] Add failing unit tests for candidate rejection above 500 nodes or with non-finite predictions; feasible selection by FPR then accuracy; tie breaks; and analysis-only status when no candidate reaches 1% FPR.
- [ ] Add failing tests that hard-negative retraining is accepted only when valid accuracy does not fall and FPR does not rise, otherwise baseline/search artifacts remain selected with a rollback reason.
- [ ] Implement pure `select_model_candidate(records, max_nodes=500, max_fpr=0.01)` and `accept_hard_negative_candidate(reference, candidate)` functions.
- [ ] Integrate the functions after valid calibration/threshold evaluation and write `model_candidate_leaderboard.csv/json` plus `hard_negative_decision.json` atomically.
- [ ] Ensure train OOF provenance remains the only mining input and test is never loaded during selection.
- [ ] Run model-search, hard-negative, deployment, and end-to-end tests.

### Task 5: Enforce Postprocessing Accuracy/FPR/Latency Acceptance

**Files:**
- Modify: `s07_postprocess_optimize.py`
- Modify: `test_s07_postprocess_optimize.py`

- [ ] Add failing tests for added latency measured from first valid Stage2 probability, P95 <= 3 seconds, FPR <= 1%, maximum streaming accuracy, tie breaks by latency/flips/simplicity, and analysis-only fallback.
- [ ] Implement pure constraint annotation and deterministic selection helpers.
- [ ] Record first-output latency, false-worn events, state flips, acceptance status, and selection reason in search CSV/JSON.
- [ ] Prevent failed postprocessing acceptance from replacing the frozen window-model configuration.
- [ ] Run cache, stride, causal replay, and postprocessing tests.

### Task 6: Introduce a Shared 600 DPI Scientific Figure Contract

**Files:**
- Create: `scientific_figures.py`
- Modify: `s04_feature_selection.py`
- Modify: `s05_train_final_model.py`
- Modify: `s06_deploy_eval.py`
- Modify: `s07_postprocess_optimize.py`
- Modify: `s08_run_pipeline.py`
- Modify: `s09_commercial_compare.py`
- Create: `test_scientific_figure_contract.py`
- Modify: `test_report_plots.py`

- [ ] Add failing tests for 600 DPI PNG metadata, nonblank pixels, expected dimensions, source CSV, manifest fields, input SHA256, panel evidence, split/n/statistics metadata, and visible read-only test status.
- [ ] Implement a shared matplotlib theme, `save_scientific_figure`, source-data export, figure manifest writer, and QA checker.
- [ ] Route all required s04-s09 summary plots, postprocessing plots, generalization plots, and error timelines through the shared exporter; replace 120/180 DPI outputs with 600 DPI.
- [ ] Export source CSVs for every quantitative summary and mark missing mandatory triples as report-incomplete.
- [ ] Add feature-pool, manual-selection, hard-negative, postprocessing, and scientific overview figures with one explicit conclusion and unique evidence per panel.
- [ ] Run plotting tests and inspect representative PNGs with Pillow for nonblank content and bounds.

### Task 7: Integrate Acceptance Reporting and Documentation

**Files:**
- Create: `pipeline_acceptance.py`
- Modify: `s08_run_pipeline.py`
- Modify: `README.md`
- Modify: `test_end_to_end_pipeline_guard.py`

- [ ] Add a failing synthetic end-to-end test requiring separate Feature pool, Selection, Model, Postprocess, Test, C readiness, and Figures statuses.
- [ ] Implement acceptance aggregation without hiding failed sections and write JSON, Markdown, and CSV summaries.
- [ ] Update README commands for Excel selection, resume training, hard-negative/model acceptance, latency definition, C warnings, and figure locations.
- [ ] Run synthetic end-to-end guard and verify expected model, contracts, workbook, figures, CSVs, and manifests.

### Task 8: Complete Verification

**Files:**
- Test: all modified and new test files

- [ ] Run `python -m compileall -q .`.
- [ ] Run focused contract suites for catalog, Excel, model decision, postprocessing, figures, deployment, and end-to-end behavior.
- [ ] Run `python -m pytest -q` and require zero failures.
- [ ] Run a synthetic 83-feature inventory check for ordered keys, finite values, groups, formulas, C metadata, and commercial mapping.
- [ ] Run `git diff --check`, review the final diff, and preserve unrelated user changes.
