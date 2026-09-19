#!/usr/bin/env python
"""Run every test module and report a combined result.

    python tests/run_all.py

Works without pytest. GPU-dependent tests skip themselves when no device is present;
set SPARSEST_TEST_HEAVY=1 to additionally load the full training arrays.
"""

import importlib
import pathlib
import sys

MODULES = ["test_model", "test_data", "test_search"]


def main():
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    total = failed = 0

    for name in MODULES:
        print(f"\n=== {name} ===")
        module = importlib.import_module(name)
        tests = [v for k, v in sorted(vars(module).items()) if k.startswith("test_")]
        for test in tests:
            total += 1
            try:
                test()
            except Exception as exc:  # noqa: BLE001 - runner reports and continues
                failed += 1
                print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
            else:
                print(f"ok   {test.__name__}")

    print(f"\n{'=' * 40}\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
