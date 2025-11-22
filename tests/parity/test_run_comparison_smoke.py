import os
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_run_comparison_smoke():
    if not os.environ.get("RUN_PARITY_SMOKE"):
        pytest.skip("Set RUN_PARITY_SMOKE=1 to run the smoke parity comparison")

    cmd = [sys.executable, "run_comparison.py", "--smoke"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"run_comparison smoke failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
