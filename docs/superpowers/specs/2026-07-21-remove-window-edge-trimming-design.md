# Remove Window Edge Trimming Design

## Goal

Remove the former behavior that discarded the first three and last three
windows of every sample. Remove the separate `skip_initial_windows` concept as
well, so every training, evaluation, caching, and deployment path processes all
legal windows in their original order.

## Scope

This change covers the active Python pipeline and active documentation:

- continuous signals windowed by `s03` and `s06`;
- array-based pre-windowed H5 samples;
- grouped-window H5 samples and their window metadata;
- serial and parallel feature extraction;
- direct evaluation, state-machine optimization, and postprocess cache export;
- pipeline CLI command construction;
- deployment metadata and cookbook window configuration;
- tests and current operational documentation.

Dated design and plan documents remain historical records. They are not
rewritten unless a current-document index incorrectly presents old trimming
behavior as active.

## Required Behavior

### Window Selection

For a sample with `N` valid, ordered windows, the pipeline processes all `N`
windows. No implicit leading or trailing window is removed.

- Continuous signals use every start returned by the normal sliding-window
  range, beginning at start 0.
- Array pre-windowed inputs use every stored window from index 0 through
  `N - 1`.
- Grouped-window inputs remain sorted by the numeric `w` index and retain every
  recognized window.
- Original grouped-window indices and labels remain aligned through feature
  extraction, evaluation, reports, and caches.

A sample with at least one legal window is never converted to an empty sample
because of edge trimming. A sample with no legal window follows the ordinary
existing empty-input or validation behavior; there is no special edge-trim
exception or skip reason.

### Removal Of Explicit Skipping

The `skip_initial_windows` concept is removed rather than retained as a
deprecated no-op. This includes:

- CLI options in `s03_extract_feature_pool.py`, `s06_deploy_eval.py`, and
  `s08_run_pipeline.py`;
- public and internal Python function parameters;
- worker tuple fields and subprocess command arguments;
- postprocess NPZ cache required fields and provenance comparisons;
- deployment package and cookbook `window_config` fields;
- tests, comments, and current documentation.

Callers that still pass `--skip_initial_windows` or the removed Python keyword
receive the normal unknown-option or unexpected-keyword error. Old caches and
deployment artifacts containing this field are not treated as the new active
contract and must be regenerated.

## Component Changes

### `s03_extract_feature_pool.py`

- Remove `EDGE_WINDOW_TRIM`, `DEFAULT_SKIP_INITIAL_WINDOWS`,
  `NoUsableWindowsAfterEdgeTrim`, and their helper functions.
- Make grouped-window sorting return every recognized item.
- Load every pre-windowed PPG and ACC window without slicing.
- Preserve every grouped metadata position.
- Use the complete sliding-window start list for continuous samples.
- Remove skip-related parameters from extraction APIs and worker payloads.
- Remove the non-fatal `no_windows_after_edge_trim` aggregation path.
- Update module comments and CLI help to describe all-window processing.

Real sample-read and feature-computation failures remain explicit and retain
their existing fail-after-all-samples behavior.

### `s06_deploy_eval.py`

- Stop importing the former trim helper.
- Evaluate every pre-windowed item and every continuous sliding-window step.
- Remove skip-related parameters from inference, optimization, evaluation, and
  public prediction APIs.
- Remove the CLI option and all call propagation.
- Remove the field from deployment metadata, cookbook output, window caches,
  and cache collection metadata.

The probability, prediction, quality, timestamp, window-index, and target
arrays must remain length-aligned after invalid feature windows are removed by
the existing explicit failure handling.

### `s07_postprocess_optimize.py`

- Remove `skip_initial_windows` from required cache fields and contract keys.
- Remove its parsing and non-negative validation.
- Compare cache provenance using the remaining active window contract fields.

### `s08_run_pipeline.py`

- Remove the top-level CLI option.
- Stop passing the option to `s03` and every `s06` command.
- Ensure dry-run output contains no automatic edge-trim or initial-skip
  reference.

### Documentation

Update `README.md`, `docs/README.md`, and
`SINGLE_WINDOW_98_FEATURE_OPTIMIZATION_PLAN.md` to state that all legal windows
are processed in their sorted order. Remove the special short-sample
edge-trim behavior and all documented skip options or fields.

## Compatibility And Artifact Migration

This is an intentional window-population change. Feature CSVs, rankings, model
selection results, trained models, evaluation reports, caches, and deployment
packages created under the old 3+3 trimming contract are stale. They must be
regenerated from `s03` onward.

The feature formulas and governed feature-catalog version do not change solely
because of window selection. The data population and resulting trained model
do change.

## Error Handling

- Missing or invalid H5 metadata remains a scan-time skip with an explicit
  reason.
- H5 read failures and feature-computation failures remain explicit errors.
- A continuous signal shorter than one complete configured window produces no
  feature rows under the existing short-input behavior.
- A recognized pre-windowed or grouped-window sample processes every available
  window; no edge-count threshold applies.

## Verification

Implementation follows test-driven development. Focused regression tests must
first fail under the old behavior and then prove:

1. The former trim helper and constants no longer exist.
2. Six grouped windows produce six windows instead of a non-fatal empty result.
3. Six array pre-windowed windows produce six windows.
4. A continuous input with six legal windows produces starts 0 through 5.
5. Ten grouped windows preserve all numeric indices and aligned labels in `s03`
   and `s06`.
6. Continuous `s06` evaluation begins at time 0 and includes the last legal
   window.
7. `--skip_initial_windows` is rejected by all active CLIs and absent from
   `s08` dry-run commands.
8. New caches, deployment metadata, and cookbooks contain no
   `skip_initial_windows` field.
9. Active code and operational documentation contain no edge-trim symbols,
   `[3:-3]` contract, or initial-window skip option.

Final verification includes focused tests, the complete pytest suite, Python
compilation, Pylint errors-only, all pipeline dry-run modes, current-document
checks, and `git diff --check`.

## Acceptance Criteria

- Every legal window is used for training and evaluation.
- Continuous, array pre-windowed, and grouped-window inputs have identical
  all-window semantics.
- No active CLI, Python API, cache schema, deployment artifact, comment, or
  operational document retains `skip_initial_windows` or automatic 3+3 edge
  trimming.
- Window indices, labels, timestamps, predictions, and quality metadata remain
  aligned.
- Existing unrelated uncommitted work is preserved.
