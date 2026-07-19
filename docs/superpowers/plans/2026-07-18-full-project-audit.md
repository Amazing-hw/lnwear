# Full Project Audit Implementation Plan

> **For agentic workers:** Execute this checklist inline. Do not delegate or use subagents for this audit.

**Goal:** Audit all active code, comments, and documentation, fix confirmed defects, and verify the documented pipeline entry points without requiring production datasets.

**Architecture:** Treat the root README and `docs/README.md` as the active contract, and treat dated design/plan files as historical records. Use repository tests and dry-run modes to validate contracts; when a defect is found, reproduce it with a focused failing test before applying the smallest production or documentation fix.

**Tech Stack:** Python 3, pytest, AST/bytecode compilation, Pylint errors-only, Markdown link checks, Git whitespace checks.

---

### Task 1: Inventory and contract audit

**Files:**
- Review: `README.md`
- Review: `docs/README.md`
- Review: all root `*.py` files
- Review: all repository `*.md` files

- [x] Confirm the active modules, tests, CLI entry points, dependencies, and repository status.
- [x] Scan code and Markdown for malformed text, stale active-stage references, broken local links, and unresolved maintenance markers.
- [x] Compare comments and help text with the active README contract.

### Task 2: Baseline verification

**Files:**
- Test: all root `test_*.py` files
- Verify: all root production `*.py` files

- [x] Compile every root Python file and require exit code 0.
- [x] Run the complete pytest suite with an isolated workspace basetemp and require zero failures.
- [x] Run Pylint in errors-only mode when installed and require zero errors.
- [x] Run documented pipeline dry-runs and CLI help smoke checks and require exit code 0.

### Task 3: Confirmed defect repair

**Files:**
- Modify: only files implicated by reproducible failures from Tasks 1-2
- Test: the closest matching `test_*.py` file, or a new focused integrity test

- [x] For each confirmed behavioral defect, add a focused regression test and run it to observe the expected failure.
- [x] Implement the minimum root-cause fix and rerun the focused test to green.
- [x] Correct documentation or comments only where they contradict the active implementation or expose a broken command/link.

### Task 4: Final verification

**Files:**
- Verify: all changed files and all active project entry points

- [x] Repeat full compilation, pytest, Pylint errors-only, Markdown checks, dry-runs, and `git diff --check`.
- [x] Review the final diff for unintended changes.
- [x] Report test counts, commands exercised, changes made, and any remaining data/environment-dependent risks.
