"""
Tests for subprocess isolation helper.
"""

import subprocess
import sys
import time
from colab_mcp.utils.process import launch_isolated_subprocess, terminate_process


def test_launch_isolated_subprocess_terminates_cleanly():
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    proc = launch_isolated_subprocess(cmd)
    assert proc.poll() is None
    terminate_process(proc, timeout=0.5)
    assert proc.poll() is not None


def test_isolated_subprocess_devnull_redirection():
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('test'); sys.stderr.write('err')"]
    proc = launch_isolated_subprocess(cmd)
    proc.wait(timeout=3.0)
    assert proc.returncode == 0


def test_terminate_process_already_exited():
    cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]
    proc = launch_isolated_subprocess(cmd)
    proc.wait(timeout=3.0)
    # Should not raise exception
    terminate_process(proc, timeout=0.5)
    assert proc.poll() == 0


def test_terminate_process_handles_timeout_escalation(monkeypatch):
    cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    proc = launch_isolated_subprocess(cmd)
    try:
        # Verify terminate_process handles unresponsive processes by killing
        terminate_process(proc, timeout=0.1)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_isolated_subprocess_custom_environment():
    cmd = [sys.executable, "-c", "import os, sys; sys.exit(0 if os.environ.get('TEST_VAR') == '1' else 1)"]
    env = dict(sys.modules["os"].environ)
    env["TEST_VAR"] = "1"
    proc = launch_isolated_subprocess(cmd, env=env)
    proc.wait(timeout=3.0)
    assert proc.returncode == 0
