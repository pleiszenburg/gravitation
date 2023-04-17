# Changes

## 0.2.0 (2023-XX-XX)

- FEATURE: Only benchmark stage 1, the computation of forces, O(N^2)
- FEATURE: Expose number of point masses per benchmark more directly on CLI
- FIX: CLI bug in `benchmark` command would prevent pointing to another interpreter for the workers
- FIX: Dependency changed from `asciiplotlib` to `termplotlib`
- FIX: Make CLI run with recent versions of `click`
- DEV: Renamed `realtimeview` sub-command to more suitable `view` command
- DEV: Cleanup of packaging and setup, moved to `pyproject.toml` as far as possible
- DEV: Refactor of entire code base, modernization, typing
- DEV: Environment varable `GRAVITATION_DEBUG` can be set to `1`, enabling run-time type checking
- DOCS: All comments translated into English
- DOCS: Even longer wish list

## 0.0.2 (2019-02-13)

- FIX: Removed dependency to `unbuffer` on Unix-like systems
- FIX: Allow to run on Windows, untested though
- DOCS: Spelling, grammar and details

## 0.0.1 (2019-02-01)

Initial release.
