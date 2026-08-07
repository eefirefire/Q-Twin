# Task 11: Repo and code cleanup

- **Docstrings**: all 28 scripts in `qtwin/scripts/` open with a module
  docstring explaining what the script does and why -- verified directly
  (checked every file starts with `"""`), not assumed from having written
  most of them following this pattern throughout the project.
- **Dead/experimental code**: none found. Every script in the repo produces
  a committed output referenced from `docs/` or `models/` -- no orphaned
  one-off files were left in `scripts/`.
- **README**: was stale (still titled "Week 1", claimed `models/`/`app/`
  were empty placeholders) -- rewritten to describe the current pipeline,
  see `qtwin/README.md`.
- **Commit history**: 28 commits, each with a descriptive message covering
  what changed and why (this project's established pattern). NOT squashed
  or rewritten -- that would rewrite already-pushed public history, which
  is a destructive operation outside what should be done without explicit
  sign-off, and the existing history is already readable rather than
  confusing (no "wip"/"fix typo" noise commits to clean up).
