# Remove Model-Search Worker Cap Implementation Plan

**Goal:** Allow an explicitly requested global worker count such as 98 to reach and be honored by s05 model-candidate search, bounded only by available candidate tasks or the serial escape hatch.

**Architecture:** Keep the existing outer `ThreadPoolExecutor` and inner XGBoost `n_jobs=1` design. Remove only the fixed default cap from `resolve_model_search_workers`; retain positive-integer normalization, `WL_FORCE_SERIAL`, and `n_items` bounding. Keep s08 inheritance unchanged.

**Tech Stack:** Python, pytest, concurrent.futures, XGBoost.

---

### Task 1: Lock the requested concurrency contract

**Files:**
- Modify: `test_parallel_execution_contract.py`
- Modify: `test_model_search_config.py`

- [ ] Add an assertion that `resolve_model_search_workers(98, n_items=200) == 98`.
- [ ] Add a dry-run assertion that global `--n_workers 98` appears as `--model_search_n_workers 98`.
- [ ] Run the focused tests and verify they fail because s05 currently resolves 98 to 4.

### Task 2: Remove the fixed s05 cap

**Files:**
- Modify: `s05_train_final_model.py:290`

- [ ] Change `resolve_model_search_workers` so explicit worker counts are normalized with `max(1, int(n_workers))`, without a fixed cap.
- [ ] Retain `WL_FORCE_SERIAL` and `min(resolved, n_items)` behavior.
- [ ] Run the focused tests and verify they pass.

### Task 3: Synchronize documentation and regression coverage

**Files:**
- Modify: `README.md:238`
- Modify: `README.md:257`

- [ ] Replace the documented four-worker maximum with task-count bounding semantics.
- [ ] Explain that 98 outer workers can consume substantial memory and that inner candidate threads remain 1 by default.
- [ ] Run all parallel/model-search tests, then the full project test suite.
- [ ] Run a real s08 dry-run with `--n_workers 98` and directly resolve each stage at 200 tasks to confirm effective values.
