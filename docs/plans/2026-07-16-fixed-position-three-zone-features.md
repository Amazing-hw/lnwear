# Fixed-Position Three-Zone Features Implementation Plan

> **Execution workflow:** Implement inline with test-driven development and verification checkpoints. Do not create extra framework paths or commit without explicit user approval.

**Goal:** Expand the governed Stage2 pool from 111 to 126 candidates with 15 normalized, fixed-position green-zone features while retaining one shared multi-layout model and manual CSV selection.

**Architecture:** Keep `get_channels_from_window` as the only hardware-layout adapter and preserve its mode mappings. Compute position-aware candidates only after the three canonical zones exist, register every formula and engineering field in the catalog, and let the existing catalog-driven ranking/deployment pipeline propagate selected candidates.

**Tech Stack:** Python, NumPy, pandas, pytest, JSON/CSV deployment contracts.

---

### Task 1: Freeze the v8/126 feature contract

**Files:**
- Modify: `test_interpretable_feature_pool.py`
- Modify: `stage2_feature_catalog.py`

- [ ] Add a failing test requiring `stage2_interpretable_v8`, exactly 126 ordered candidates, and the exact 15 `GZONE{1,2,3}_*` names.
- [ ] Add assertions that each new record belongs to `green_position`, declares source, formula, unit, C operators, deployment cost, bounds where applicable, and the three position/generalization risk flags.
- [ ] Run the focused catalog tests and confirm failure because v7 contains 111 candidates.
- [ ] Register the 15 catalog records, bump the pool version, and widen the governed-count ceiling from 120 to 140.
- [ ] Re-run the focused catalog tests and confirm they pass.

### Task 2: Implement position-aware extraction

**Files:**
- Modify: `test_interpretable_feature_pool.py`
- Modify: `s03_extract_feature_pool.py`

- [ ] Add failing numerical tests for the five formulas on three unequal synthetic zones.
- [ ] Add a failing position-sensitivity test showing that swapping zone 1 and zone 2 swaps their `GZONE` feature values while existing invariant features remain unchanged.
- [ ] Add failing finite-value tests for constant, zero, tiny-amplitude, and one-bad-zone inputs.
- [ ] Implement one helper that consumes the three raw zones, three pulse zones, ambient pulse, and sample rate, returning the 15 ordered candidates with guarded arithmetic.
- [ ] Insert the helper output into `extract_feature_pool_from_window` without changing existing green, median, top2, pair, commercial, or ACC formulas.
- [ ] Re-run the focused extraction tests and confirm they pass.

### Task 3: Verify canonical and grouped-layout equivalence

**Files:**
- Modify: `test_interpretable_feature_pool.py`
- Modify: `s03_extract_feature_pool.py` comments/docstrings only if the test exposes stale semantics

- [ ] Extend the existing layout test so mode1 three-zone signals and mode2 grouped green channels mapped into the same three zones produce identical `GZONE` features.
- [ ] Assert the fixed zone ordering is preserved rather than permutation-normalized.
- [ ] Run the layout tests and confirm they pass without changing mode mappings.
- [ ] Update the channel-adapter docstring from “no absolute direction” to stable fixed physical-region semantics.

### Task 4: Synchronize governance, deployment, and documentation

**Files:**
- Modify: `test_manual_feature_selection_csv.py`
- Modify: `test_deploy_feature_extractor.py`
- Modify: `test_pipeline_acceptance.py`
- Modify: `README.md`

- [ ] Add a manual-CSV test selecting at least one `GZONE` candidate and preserving its catalog order in the frozen selection.
- [ ] Add a deployment test selecting `GZONE` candidates and checking Python `FEATURE_ORDER`, selected formula JSON, Stage2 catalog, C contract, and golden-vector order.
- [ ] Update active v7/111 acceptance fixtures and README statements to v8/126 while leaving historical design documents unchanged.
- [ ] Document that position candidates are optional, normalized, shared across 3/6/9 layouts, and subject to mode/device generalization audit.

### Task 5: Acceptance and regression

**Files:** all modified production, tests, and active documentation.

- [ ] Run focused tests for the catalog, extraction, CSV selection, deployment export, and acceptance report.
- [ ] Run `python -m py_compile stage2_feature_catalog.py s03_extract_feature_pool.py s04_feature_selection.py s05_train_final_model.py s06_deploy_eval.py s08_run_pipeline.py pipeline_acceptance.py`.
- [ ] Run the complete pytest suite with cacheprovider disabled and a workspace basetemp.
- [ ] Run `git diff --check` and scan active code/README for stale v7/111 claims.
- [ ] Review the final diff to confirm mode mappings, commercial-eight formulas, manual feature count behavior, and unrelated user changes remain untouched.
