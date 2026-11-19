"""
Stdio Stdout Logging Isolation and Management.
Guarantees zero unsolicited stdout pollution to preserve MCP JSON-RPC framing.
"""

import io
import logging
import sys
from typing import Any, Optional

from colab_mcp.utils.redaction import RedactingFormatter

_original_stdout: io.TextIOBase = sys.stdout
_is_stdout_protected: bool = False


class StderrProxy(io.TextIOBase):
    """
    Proxy replacing sys.stdout that redirects all text writes and print statements
    to sys.stderr, while preserving access to the underlying binary buffer for FastMCP.
    """

    def __init__(self, real_stdout: io.TextIOBase, target_stderr: io.TextIOBase) -> None:
        super().__init__()
        self._real_stdout = real_stdout
        self._target_stderr = target_stderr

    @property
    def buffer(self) -> Any:
        """Expose the real underlying binary stdout buffer for stdio_server."""
        return getattr(self._real_stdout, "buffer", None)

    @property
    def encoding(self) -> str:
        return getattr(self._real_stdout, "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self._real_stdout, "errors", "replace")

    def write(self, s: str) -> int:
        """Redirect text writes to stderr."""
        return self._target_stderr.write(s)

    def flush(self) -> None:
        self._target_stderr.flush()

    def isatty(self) -> bool:
        return self._target_stderr.isatty()

    def fileno(self) -> int:
        return self._target_stderr.fileno()


def protect_stdout() -> None:
    """
    Replaces sys.stdout with StderrProxy to guarantee stdout purity.
    Idempotent: safe to call multiple times.
    """
    global _original_stdout, _is_stdout_protected
    if not _is_stdout_protected:
        _original_stdout = sys.stdout
        sys.stdout = StderrProxy(_original_stdout, sys.stderr)
        _is_stdout_protected = True


def unprotect_stdout() -> None:
    """Restores the original sys.stdout (primarily for testing teardown)."""
    global _original_stdout, _is_stdout_protected
    if _is_stdout_protected:
        sys.stdout = _original_stdout
        _is_stdout_protected = False


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures root and subsystem loggers to emit exclusively to sys.stderr
    with automated credential redaction.
    """
    protect_stdout()

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers that might point to stdout
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Attach stderr stream handler
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    formatter = RedactingFormatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    # Ensure third-party noisy loggers route to stderr without stdout pollution
    for logger_name in ("httpx", "httpcore", "websockets", "google", "playwright", "asyncio"):
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.handlers = []
        noisy_logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    """
    Factory function returning a logger configured for stderr emission.
    """
    if not _is_stdout_protected:
        protect_stdout()
    return logging.getLogger(name)
