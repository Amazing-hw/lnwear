# Manual Model Search Efficiency Implementation Plan

**Goal:** Reduce wall-clock time after manual feature selection without changing the frozen features or final model-selection semantics.

**Architecture:** Increase safe outer parallelism by flattening Stage B and hard-negative fold work, eliminate repeated threshold loops, and reuse versioned metric caches. Preserve deterministic aggregation and retrain the selected final model from source data.

**Tech Stack:** Python, NumPy, XGBoost, concurrent.futures, pytest, JSON.

---

### Task 1: Threshold evaluation

**Files:** `s05_train_final_model.py`, `test_model_search_config.py`

- [x] Add a failing parity test comparing the public evaluator with a literal 181-threshold reference over random, tied and non-finite probabilities.
- [x] Implement sorted cumulative-count threshold evaluation and keep the same accuracy/FP/F1/precision/recall tie-break.
- [x] Run the focused parity test.

### Task 2: Stage B fold-level parallelism

**Files:** `s05_train_final_model.py`, `test_model_search_config.py`

- [x] Add a failing test requiring Stage B to submit `candidate_count × fold_count` fold tasks to `ordered_thread_map`.
- [x] Flatten fold work, aggregate deterministically, and run per-candidate full fits through the worker map.
- [x] Verify model-search records, fold counts and chosen params remain stable.

### Task 3: Hard-negative fold parallelism

**Files:** `s05_train_final_model.py`, `test_model_search_config.py`

- [x] Add a failing test that observes multiple OOF tasks and ordered aggregation.
- [x] Add `n_workers` to hard-negative mining and forward the s05 outer-worker setting.
- [x] Verify serial and parallel OOF outputs match.

### Task 4: Versioned resumable metric cache

**Files:** `s05_train_final_model.py`, `s08_run_pipeline.py`, `test_model_search_config.py`

- [x] Add failing tests for cache hit, corrupted-cache fallback and changed-data invalidation.
- [x] Implement full-data/CV fingerprinting, atomic JSON reads/writes and Stage A/Stage B metric caches.
- [x] Add `--model_search_cache/--no-model_search_cache` and forward it from s08.
- [x] Record schema, cache root, hits and misses in model-search summary.

### Task 5: Manual defaults, timing and dead CLI cleanup

**Files:** `s05_train_final_model.py`, `s08_run_pipeline.py`, `README.md`, `test_model_search_config.py`, `test_parallel_execution_contract.py`

- [x] Add failing dry-run tests for manual balanced 120/12, fast 80/12, thorough 360/48 and explicit override precedence.
- [x] Apply manual-profile defaults only during a valid manual resume.
- [x] Add Stage A/Stage B/best-refit/hard-negative runtime fields.
- [x] Remove `model_search_stage1_top_k` parser/forwarding/docs references.
- [x] Document fit-count estimates, cache behavior and 98-worker Stage B semantics.

### Task 6: Verification

- [x] Run focused model-search, manual-flow and parallel tests.
- [x] Run Python compilation and all relevant CLI help commands.
- [x] Run the full pytest suite and `git diff --check`.
