"""Run backend parity and performance checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(
        [
            "test.test_backend_parity.VerifyNetParityTests",
            "test.test_backend_parity.VerifyNetBenchmarkTests",
        ]
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
