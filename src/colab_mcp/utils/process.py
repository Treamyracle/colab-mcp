"""
Subprocess isolation utilities.
Guarantees that child processes cannot inherit or corrupt server stdio streams.
"""

import subprocess
import sys
from typing import IO, Any, List, Optional, Union


def launch_isolated_subprocess(
    cmd: List[str],
    *,
    stdin: Optional[Union[int, IO[Any]]] = subprocess.DEVNULL,
    stdout: Optional[Union[int, IO[Any]]] = subprocess.DEVNULL,
    stderr: Optional[Union[int, IO[Any]]] = subprocess.DEVNULL,
    close_fds: bool = True,
    **kwargs: Any,
) -> subprocess.Popen:
    """
    Launches a child subprocess strictly isolated from parent stdio streams.
    Defaults to DEVNULL for stdin, stdout, and stderr with close_fds=True.
    """
    return subprocess.Popen(
        cmd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        close_fds=close_fds,
        **kwargs,
    )


def terminate_process(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    """
    Politely terminates a child process with escalation to kill if unresponsive.
    """
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except OSError:
                pass
