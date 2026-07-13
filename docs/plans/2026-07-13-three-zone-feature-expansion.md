# Three-Zone Feature Expansion Implementation Plan

**Goal:** Add seven explainable, deployment-oriented Stage2 candidates without violating the three-zone rotational symmetry contract.

**Architecture:** Raw layout normalization stays in `get_channels_from_window`; feature helpers operate only on normalized zone signals and orientation-invariant aggregates. The governed catalog remains the single ordering and deployment metadata source.

**Tech Stack:** Python, NumPy, pandas, pytest, JSON/CSV contracts, matplotlib PNG reporting.

---

### Task 1: Lock the three-zone geometry contract

- [x] Add tests proving equivalent mode-1 and mode-2 raw layouts yield identical `g1/g2/g3` zones.
- [x] Add tests proving every new spatial candidate is unchanged by all zone permutations.
- [x] Run tests and confirm failure because the new candidates do not exist.

### Task 2: Add pulse-shape and zone-consistency candidates

- [x] Add failing clean/flat/noisy tests for robust skewness, normalized spectral entropy, two-of-three periodicity, and pairwise lag RMS.
- [x] Implement guarded helper functions and connect them to the governed optical extractor.
- [x] Add catalog formulas, bounded ranges, C operators, buffer sizes, and costs.
- [x] Run focused optical and catalog tests to green.

### Task 3: Add ACC–PPG motion-artifact candidates

- [x] Add failing tests for jerk-tail sensitivity, delayed motion coupling, PSD similarity, and missing ACC.
- [x] Reuse the bounded correlation and FFT paths to implement the three candidates.
- [x] Add catalog metadata and run focused ACC tests to green.

### Task 4: Migrate versioned contracts and documentation

- [x] Bump the pool version to v5 and update expected counts from 84 to 91.
- [x] Correct stale layout comments and document three-zone rotational invariance and new feature meanings.
- [x] Verify CSV selection and acceptance reports consume the new catalog dynamically.

### Task 5: Complete verification

- [x] Run focused feature, selection, deployment, pipeline, and figure tests.
- [x] Run the complete pytest suite.
- [x] Run compilation, Pylint errors-only, `git diff --check`, and stale-version scans.
