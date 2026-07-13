# Interpretable Stage2 Feature Pool Implementation Plan

> Implementation checklist for the interpretable Stage2 feature pool. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 208-column Stage2 candidate surface with a versioned 80-120 feature catalog that is interpretable, numerically robust, and explicitly portable to C.

**Architecture:** Add a focused catalog module as the policy source of truth, keep signal computation in `s03_extract_feature_pool.py`, and make `s04` plus deployment export consume catalog metadata rather than maintaining independent lists. A shared window assembler eliminates the current batch/deployment duplication. Version markers force stale artifacts to be regenerated.

**Tech Stack:** Python 3, NumPy, pandas, pytest, XGBoost, JSON deployment artifacts.

---

### Task 1: Lock the candidate policy with failing tests

**Files:**
- Create: `test_interpretable_feature_pool.py`
- Modify: `test_deploy_feature_extractor.py`
- Modify: `test_deployment_friendly_features.py`

- [ ] Add tests asserting an ordered catalog exists, the generated window candidate keys equal it, the count is 80-120, no feature maps to `other`, and forbidden constants/aliases/identifiers are absent.
- [ ] Add tests asserting diagnostic-only fields are excluded by `s04.get_feature_cols` while `mode` remains available in feature-pool rows.
- [ ] Add tests for constant, NaN/Inf, isolated spike, step transition, and low-amplitude inputs.
- [ ] Add tests asserting every candidate exposes preprocessing and C metadata.
- [ ] Run `python -m pytest test_interpretable_feature_pool.py -q` and confirm the tests fail because the catalog and new policy do not exist.

### Task 2: Add the Stage2 feature catalog

**Files:**
- Create: `stage2_feature_catalog.py`
- Test: `test_interpretable_feature_pool.py`

- [ ] Define `FEATURE_POOL_VERSION`, ordered feature records, diagnostic-only fields, removal reasons, semantic groups, units, preprocessing paths, numerical guards, and C operator metadata.
- [ ] Expose helpers for ordered candidate names, feature lookup, group lookup, candidate filtering, and catalog validation.
- [ ] Run the focused tests and confirm catalog-only tests pass while extractor equality tests still fail.

### Task 3: Refactor preprocessing and the shared window assembler

**Files:**
- Modify: `s03_extract_feature_pool.py`
- Test: `test_interpretable_feature_pool.py`
- Test: `test_prewindowed_h5.py`

- [ ] Split preprocessing into quality-before-repair, minimally cleaned raw, pulsatile detrended, and ACC motion paths.
- [ ] Replace the short-window 65-tap forward-backward FIR dependency with median detrending plus bounded smoothing for model candidates.
- [ ] Use scale-aware safe ratios and guarded correlations in new candidate formulas.
- [ ] Remove computation of individual G1/G2/G3 expansions, consensus expansion, channel identifiers, constants, aliases, and unstable formulas.
- [ ] Add one shared assembler that combines optical, ACC, Stage1 diagnostic, and row metadata fields.
- [ ] Make `extract_window_features` and both batch extraction paths call the shared assembler.
- [ ] Run focused extraction, deployment, and prewindowed tests.

### Task 4: Make ranking consume the catalog

**Files:**
- Modify: `s04_feature_selection.py`
- Modify: `test_s04_progress_diagnostics.py`
- Modify: `test_deployment_friendly_features.py`
- Test: `test_interpretable_feature_pool.py`

- [ ] Replace stale `FEATURE_GROUPS` and duplicate deployment allowlists with catalog-derived groups.
- [ ] Treat `mode`, invalid counters, pool version, and row identifiers as metadata.
- [ ] Reject unknown candidate fields and missing/mismatched pool versions.
- [ ] Include catalog metadata and removal rationale in feature diagnostics and ranking exports.
- [ ] Run all `s04` and deployment-policy tests.

### Task 5: Version training and deployment artifacts

**Files:**
- Modify: `s05_train_final_model.py`
- Modify: `s06_deploy_eval.py`
- Modify: `s08_run_pipeline.py`
- Modify: `test_deploy_feature_extractor.py`
- Modify: `test_end_to_end_pipeline_guard.py`

- [ ] Persist `feature_pool_version` in selection outputs, model bundle, final config, formulas, cookbook, and deployment package.
- [ ] Reject stale selection files and model bundles with an instruction to rerun from `s03`.
- [ ] Build selected-feature formulas and group/cost summaries from the catalog.
- [ ] Export `stage2_feature_catalog.json` and `stage2_c_contract.json` for selected features.
- [ ] Extend golden vectors with raw selected feature values and per-feature absolute/relative tolerances.
- [ ] Run deployment and end-to-end guard tests.

### Task 6: Documentation and complete verification

**Files:**
- Modify: `README.md`
- Modify: `SINGLE_WINDOW_98_FEATURE_OPTIMIZATION_PLAN.md`

- [ ] Document the pool version, removed feature classes, preprocessing paths, C parity workflow, and mandatory rerun from `s03`.
- [ ] Run `python -m compileall .`.
- [ ] Run `python -m pytest -q`.
- [ ] Generate a synthetic feature inventory and verify count, group coverage, finite values, and formula/catalog coverage.
- [ ] Review `git diff` to ensure existing unrelated user changes are preserved.
