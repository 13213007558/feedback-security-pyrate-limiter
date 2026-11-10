# Feedback task starting environment

Upstream: https://github.com/vutran1710/PyrateLimiter
Upstream commit: `341b2c81f869fb53aa893f07f02182745248c062`
Prepared on macOS with CPython 3.12.11. No external database, queue, or cloud account is required.
The dependency file pins the installed runtime and test packages for the selected baseline. It is not a claim that all optional upstream test suites are installed.

From the repository root:

```sh
export NO_INTERNET=1 TZ=UTC
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r .feedback/requirements.txt -e .
.venv/bin/python -m pytest -q -o addopts= tests/test_algorithm.py tests/test_fixed_window.py tests/test_gcra.py -k 'not redis and not postgres and not multiprocessing'
```

Scope note: The scope is the command below, not all optional upstream integrations.

Observed baseline (2026-11-10): `119 passed, 1 skipped, 13 deselected, 4 warnings in 2.63s`
Whole-command wall time: 3.068 seconds, exit code 0. This is one local measurement, not a runtime guarantee. The first install is excluded.

This seed adds preparation metadata only. Application sources and upstream tests are unchanged; there is no task solution in this commit.
