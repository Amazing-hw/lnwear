# Mode Feature and CSV Manual Selection Implementation Plan

**Goal:** Add `mode` to the selectable Stage2 feature surface and migrate manual selection from XLSX to a strict CSV-only contract.

**Architecture:** The feature catalog remains the single source of truth for model candidates. `manual_feature_selection.py` owns CSV generation and immutable validation; s04 exports it, s05 consumes it, and pipeline acceptance verifies its frozen provenance.

**Tech Stack:** Python, pandas/csv, pytest, XGBoost pipeline, JSON/SHA256 provenance.

---

### Task 1: Define failing feature-contract tests

- [x] Assert `mode` is a governed model candidate with deployable metadata.
- [x] Assert extraction output and feature-pool analysis include the updated catalog count.
- [x] Run focused tests and verify failure because `mode` is diagnostic-only.

### Task 2: Define failing CSV-selection tests

- [x] Replace workbook tests with CSV generation, selection, immutable-field, stale-ranking, invalid-value, duplicate-row, missing-column, and XLSX-rejection tests.
- [x] Run focused tests and verify failure because training still loads XLSX.

### Task 3: Implement catalog and extraction changes

- [x] Add `mode` to `FEATURE_CATALOG`, remove its diagnostic-only exclusion, bump the pool version, and attach engineering/risk metadata.
- [x] Ensure feature extraction returns the catalog-ordered `mode` value without duplicating it.
- [x] Run focused catalog/extraction tests to green.

### Task 4: Implement CSV-only manual selection

- [x] Replace workbook export/load functions with strict CSV equivalents.
- [x] Freeze schema version, pool version, ranking SHA256, row order, and immutable values.
- [x] Reject non-CSV inputs and preserve exact selected order/count/provenance.
- [x] Run focused CSV tests to green.

### Task 5: Migrate callers, acceptance, and documentation

- [x] Update s04, s05, s08, acceptance messages, README commands, artifact lists, and comments.
- [x] Remove active OpenPyXL/XLSX references and rename the old test module.
- [x] Run integration and integrity tests.

### Task 6: Verify the complete project

- [x] Run full pytest.
- [x] Run Python compilation and Pylint errors-only.
- [x] Run `git diff --check` and scan for stale XLSX/manual-workbook references.
