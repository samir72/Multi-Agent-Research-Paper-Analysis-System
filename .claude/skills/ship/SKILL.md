---
name: ship
description: Scan for secrets, update docs, run tests, commit and push
---

1. Run `git diff --staged` and `git status` (`-u` for untracked, never `-uall`). Scan every changed/new file for API keys, tokens, `.env` values, or credentials — even in files whose names look innocuous.
2. Check docs for staleness. This repo has no `docs/` directory — its root-level `*.md` files (`README.md`, `AGENTS.md`, `CLAUDE.md`, and the topic `BUGFIX_*.md`/`*_SUMMARY.md` files) **are** the docs surface, and `README.md`/`AGENTS.md` have been missed by this step before (staleness sat undetected across two prior `/ship` runs until caught by a manual audit) — do not stop at `CLAUDE.md` alone. Concretely:
   - `grep` every changed function/class signature by name across `README.md` `AGENTS.md` `CLAUDE.md` (e.g. `grep -rn "run_workflow(\|filter_by_agent(" *.md`) — a call-shape example that no longer matches the real signature is stale.
   - If a test file was added/removed or its test count changed, `pytest <file> --collect-only -q` for the real count and check it against `README.md`'s directory-tree comments (`tests/ ├── test_x.py # ... (N tests)`) and its "Test Coverage" section — both drift independently and have both been wrong at once.
   - New/removed/renamed files (scripts, config, modules) belong in `README.md`'s project-structure tree if a sibling file is already listed there.
   - **Don't edit dated changelog/version-history entries** (e.g. "### Version 2.13", "Previous Updates (Early 2025)") — those are historical snapshots of what was true *then* and stay accurate as-is, even when current reality has since moved on. Only fix sections making a *current-state* claim (project structure tree, "Current Test Suite", live usage examples, "Key Files"). When unsure which kind a line is, check whether it sits under a dated/versioned heading.
   - Illustrative before/after pseudocode in a bugfix writeup (e.g. `BUGFIX_MSGPACK_SERIALIZATION.md`-style "WRONG / CORRECT" snippets) teaches a pattern, not a literal current signature — leave it unless it's explicitly labeled as quoting a real file (e.g. `# app.py:431`).
3. Run the test suite (`pytest tests/ -v` in this repo) and confirm it's not worse than before the change. Report actual pass/fail counts, don't assume green.
4. Stage only the relevant files by name (never `git add -A`/`.`), write a commit message focused on *why*, and push — only after explicit confirmation per this repo's git safety rules.
