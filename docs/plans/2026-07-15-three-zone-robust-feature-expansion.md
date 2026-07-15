# Three-Zone Robust Feature Expansion Implementation Plan

> **Execution:** Implement with local TDD checkpoints; do not commit or push.

**Goal:** Expand the governed Stage2 pool from 91 to 111 candidates with permutation-invariant median, pair-consensus, ambient-residual, phase, and spectral features for unilateral lift and local-light robustness.

**Architecture:** Keep the existing mode adapter and all existing features unchanged. Reuse independently preprocessed `g1/g2/g3` signals to build a pointwise median representation and all three pair representations, then emit only permutation-invariant order statistics and guarded residual/FFT features.

**Tech Stack:** Python, NumPy, pytest, existing feature catalog/C-contract infrastructure.

**Status (2026-07-15):** Implemented, then superseded by the v7 correctness/alignment follow-up plan. The checked boxes below are the completed execution record; current verification evidence is recorded in the follow-up plan.

---

### Task 1: Freeze the 111-candidate catalog contract

**Files:**
- Modify: `test_interpretable_feature_pool.py`
- Modify: `stage2_feature_catalog.py`

- [x] Add a failing test asserting `FEATURE_POOL_VERSION == "stage2_interpretable_v6"`, exactly 111 ordered candidates, and the exact 20 names from the approved design.
- [x] Assert each new record has a non-empty formula/operator list, finite deployment cost, bounded-range metadata where applicable, and `experimental_high_cost` risk flags on phase/spectral consensus.
- [x] Run `python -m pytest test_interpretable_feature_pool.py::test_three_zone_robust_expansion_is_governed -q -p no:cacheprovider --basetemp .pytest_tmp_zone_catalog_red`; expect failure because v6 and the names do not exist.
- [x] Add the 20 catalog entries without changing existing entry order; use groups `green_top2_contact`, `pulse_shape`, `green_spatial`, `ambient_cross`, and `frequency` as appropriate.
- [x] Assign C operators including `median3`, `three_pair_loop`, `autocorrelation`, `correlation`, `safe_ratio`, `rfft`, `complex_phase`, and `cosine_similarity`; mark only the last two candidates high-cost.
- [x] Run the focused catalog test; expect pass.

### Task 2: Add guarded reusable three-zone primitives

**Files:**
- Modify: `test_interpretable_feature_pool.py`
- Modify: `s03_extract_feature_pool.py`

- [x] Add failing unit tests for helpers that compute ambient projection residuals, phase concentration, and pair spectral cosine, covering flat, non-finite, clean periodic, and ambient-contaminated signals.
- [x] Run the helper tests and confirm failure because the helpers are absent.
- [x] Extend `compute_fft_cache` to retain the complex RFFT in `complex_spec` while preserving all existing keys.
- [x] Implement `_ambient_projection_residual(zone_pulse, ambient_pulse)` using guarded centered covariance/variance and finite replacement.
- [x] Implement `_phase_concentration(fft_caches, reference_hz)` as the magnitude of the mean unit phasor at the nearest common FFT bin, returning zero for missing energy/reference.
- [x] Implement `_spectral_power_cosine_from_cache(left, right)` using aligned band power and a guarded norm.
- [x] Run the helper tests plus existing FFT/autocorrelation tests; expect pass.

### Task 3: Extract median, top2-complement, and all-pair features

**Files:**
- Modify: `test_interpretable_feature_pool.py`
- Modify: `s03_extract_feature_pool.py`

- [x] Add failing extraction tests asserting all 20 new keys are finite and ordered exactly like the catalog.
- [x] Add a six-permutation test covering every new spatial candidate.
- [x] Add synthetic robustness tests:
  - one zone attenuated or spiked leaves median/pair periodicity closer to clean than the three-zone mean;
  - two clean zones plus one ambient-contaminated zone retain strong residual 2-of-3 periodicity;
  - mismatched zone frequencies increase frequency MAD and reduce support;
  - asynchronous zones reduce phase concentration;
  - three flat zones return finite zeros where evidence is undefined.
- [x] Run these tests and confirm expected missing-key/count failures.
- [x] In `extract_feature_pool_from_window`, compute `gmedian_raw/pulse`, per-zone FFT caches, per-zone periodicity/dom-frequency, and three pair raw/pulse arrays once.
- [x] Emit the seven median/top2 complement features with guarded ratios/correlations.
- [x] Emit the two frequency-consensus features using MAD and a fixed ±0.20 Hz support tolerance.
- [x] Emit the seven all-pair order-statistic features from three-element arrays; do not expose pair identities.
- [x] Emit two ambient-residual features and the two high-cost experimental features, reusing cached FFTs.
- [x] Remove the existing dead `if False` expressions in top2 waveform construction without changing behavior.
- [x] Run the focused robustness/permutation tests; expect pass.

### Task 4: Synchronize feature governance and user-facing documentation

**Files:**
- Modify: `test_deploy_feature_extractor.py`
- Modify: `test_deployment_friendly_features.py`
- Modify: `test_manual_feature_selection_csv.py`
- Modify: `README.md`
- Modify: `docs/designs/2026-07-15-three-zone-robust-feature-expansion-design.md` only if implementation names differ (names should not differ)

- [x] Update count/version assertions from 91/v5 to 111/v6 while retaining all commercial-eight mapping assertions.
- [x] Add contract assertions that selected new candidates appear in manual CSV, generated deployment extractor order, C contract, cost summary, and finite replay.
- [x] Run the affected contract tests first and observe failures from stale counts/metadata.
- [x] Update README feature-pool count and document the five parallel three-zone evidence surfaces: mean, RMS-top2, median, all-pair, and ambient residual.
- [x] Run the contract tests; expect pass.

### Task 5: Acceptance and regression

**Files:** all changed production, tests, and active documentation.

- [x] Run focused tests:
  `python -m pytest test_interpretable_feature_pool.py test_deploy_feature_extractor.py test_deployment_friendly_features.py test_manual_feature_selection_csv.py -q -p no:cacheprovider --basetemp .pytest_tmp_zone_focused`.
- [x] Run the complete suite with a workspace basetemp and confirm every test passes.
- [x] Run `python -m py_compile stage2_feature_catalog.py s03_extract_feature_pool.py s04_feature_selection.py s05_train_final_model.py s06_deploy_eval.py s08_run_pipeline.py`.
- [x] Run Pylint errors-only on changed production modules.
- [x] Run manual, auto, and auto-E2E dry-runs and confirm plot formats remain PNG-only.
- [x] Run `git diff --check` and scan active code/README for stale v5/91 feature-pool claims.
- [x] Preserve the dirty worktree and do not commit or push.
