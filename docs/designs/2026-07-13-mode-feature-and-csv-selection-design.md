# Mode Feature and CSV Manual Selection Design

## Goal

Allow the detected acquisition `mode` to participate in the governed Stage2 feature ranking and final manual selection, and replace the XLSX manual-selection workflow with a CSV-only workflow.

## Feature contract

- Add integer `mode` (0/1/2) to the governed feature catalog as an acquisition-context feature.
- Keep the existing value produced by `detect_green_mode`; do not one-hot encode it.
- Include it in ranking, selection, training, model fingerprints, deployment order, and C parity whenever selected.
- Mark it as scale-independent, non-FFT, constant-time, and expose an explicit shortcut/generalization risk flag in its catalog metadata and documentation.
- Continue reporting `mode` as row metadata where useful; being metadata no longer excludes it from the model candidate surface.

## CSV selection contract

- `s04` writes only `artifacts/manual_feature_selection.csv` for manual selection.
- Users may edit only the `selected` column using values 0 or 1.
- Every other header, row, feature, ranking value, and engineering field is immutable and checked against the current ranking JSON.
- Contract columns freeze the CSV schema version, feature-pool version, and ranking SHA256 on every row, preventing stale or modified selection files from being accepted.
- `s05` rejects XLSX paths with a clear CSV-only error and freezes the selected names, exact order, count, file hash, and ranking hash in `manual_selected_features.json`.

## Compatibility and verification

- Remove active XLSX generation, parsing, dependencies, CLI defaults, messages, tests, and README examples.
- Existing XLSX selection files require a fresh `s04` run to generate the CSV.
- Update governed-feature counts and integrity/acceptance checks.
- Verify focused tests, full pytest, compilation, Pylint errors-only, diff whitespace, and residual XLSX references.
