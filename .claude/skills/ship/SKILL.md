---
name: ship
description: Scan for secrets, update docs, run tests, commit and push
---

1. Run `git diff --staged` and `git status` (`-u` for untracked, never `-uall`). Scan every changed/new file for API keys, tokens, `.env` values, or credentials — even in files whose names look innocuous.
2. Check `CLAUDE.md`, `README.md`, and any files under `docs/` for references made stale by this change (file paths, function names, test counts, env var defaults). Update what's actually stale — don't rewrite unrelated sections.
3. Run the test suite (`pytest tests/ -v` in this repo) and confirm it's not worse than before the change. Report actual pass/fail counts, don't assume green.
4. Stage only the relevant files by name (never `git add -A`/`.`), write a commit message focused on *why*, and push — only after explicit confirmation per this repo's git safety rules.
