# Three-Zone Correctness and Deployment Alignment Implementation Plan

> Historical completed plan for the v7/111 stage. The active implementation is
> v8/126; use README and the 2026-07-16 fixed-position/deployment designs.

> **Execution:** Use TDD in the current dirty workspace; do not commit or push.

**Goal:** Make three-zone features strictly permutation-invariant and frequency-valid, align every deployment recipe with the governed extractor, and remove commercial comparison functionality while retaining commercial features.

**Architecture:** Keep the frozen mode adapter and 111-feature surface. Centralize tie-aware pair selection and frequency validity in `s03`, use the catalog as the only deployment formula source, and remove the optional s09 comparison branch end to end.

**Tech Stack:** Python, NumPy, pytest, JSON/CSV deployment contracts, existing PNG reporting infrastructure.

**Status (2026-07-15):** Completed. Final project verification: 257 tests passed; production modules compiled; Pylint errors-only, three pipeline dry-runs, PNG-only/stale-reference scans, and `git diff --check` passed.

---

### Task 1: Freeze v7 three-zone numerical contracts

**Files:** `test_interpretable_feature_pool.py`, `stage2_feature_catalog.py`, `s03_extract_feature_pool.py`

- [x] Add failing tests for v7/111, exact-RMS-tie permutation invariance, zero/tiny-noise invalid frequency, and two-valid-one-flat consensus.
- [x] Run the tests and confirm failures expose index-dependent top2, false dominant frequency, and stale v6.
- [x] Implement tie-aware optimal-pair aggregation and reuse it in the main and spatial extractors.
- [x] Implement guarded frequency-evidence validity and valid-pair aggregation.
- [x] Update catalog formulas, costs, risk flags, and version; run focused tests to green.

### Task 2: Make deployment recipes catalog-backed

**Files:** `test_deploy_feature_extractor.py`, `test_deployment_friendly_features.py`, `s08_run_pipeline.py`, `s06_deploy_eval.py`

- [x] Add failing tests requiring all 111 cookbook recipes to be matched and current preprocessing descriptions to exclude legacy step/medfilt/bandpass operations.
- [x] Add a failing test proving subset deployment costs use feature records rather than a group overwrite.
- [x] Replace duplicate cookbook formulas with catalog payloads and accurate shared intermediates.
- [x] Synchronize the s06 formula export preprocessing description.
- [x] Replace group-cost lookup in subset scoring with per-feature catalog cost; run contract tests to green.

### Task 3: Remove commercial comparison functionality

**Files:** `test_end_to_end_pipeline_guard.py`, `test_pipeline_acceptance.py`, `test_model_search_config.py`, `pipeline_acceptance.py`, `s08_run_pipeline.py`, `README.md`; delete `s09_commercial_compare.py` and `test_s09_grouped_label_guard.py`

- [x] Add a failing integrity test that active CLI, pipeline steps, acceptance figures and README contain no commercial comparison entry while catalog commercial mappings remain.
- [x] Remove s09 execution, arguments, step aliases, optional figure acceptance and documentation/artifact sections.
- [x] Delete the comparison module and comparison-only tests; retain catalog mapping assertions.
- [x] Run pipeline, README and catalog contract tests to green.

### Task 4: Documentation and complete verification

**Files:** `README.md`, active design/plan documents, changed production and test modules.

- [x] Correct RMS-quality and preprocessing language and document v7 invalid-frequency semantics.
- [x] Run focused feature/deployment/pipeline tests.
- [x] Run complete pytest with a workspace basetemp.
- [x] Run py_compile and Pylint errors-only on changed production modules.
- [x] Run manual, auto and auto-E2E dry-runs and confirm PNG-only formats.
- [x] Run stale-reference scans and `git diff --check`; preserve the dirty worktree without commit/push.
