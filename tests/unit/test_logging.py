"""
Tests for stdio stdout logging isolation and stderr redirection.
"""

import io
import logging
import sys
import pytest

from colab_mcp.utils.logging import (
    StderrProxy,
    get_logger,
    protect_stdout,
    setup_logging,
    unprotect_stdout,
)


@pytest.fixture(autouse=True)
def reset_logging_state():
    unprotect_stdout()
    yield
    unprotect_stdout()



def test_protect_stdout_redirects_print_to_stderr(capsys):
    setup_logging()
    print("Test print message")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Test print message" in captured.err


def test_logger_writes_to_stderr(capsys):
    setup_logging()
    logger = get_logger("test_unit_logging")
    logger.info("Unit test log message")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Unit test log message" in captured.err


def test_buffer_preservation_for_fastmcp():
    protect_stdout()
    assert hasattr(sys.stdout, "buffer")
    assert sys.stdout.buffer is not None
    # Verify buffer write executes without throwing
    test_bytes = b"BINARY_JSONRPC_FRAME\n"
    sys.stdout.buffer.write(test_bytes)
    sys.stdout.buffer.flush()


def test_get_logger_stream_is_stderr():
    setup_logging()
    root = logging.getLogger()
    assert len(root.handlers) > 0
    stream_handler = root.handlers[0]
    assert isinstance(stream_handler, logging.StreamHandler)
    assert stream_handler.stream == sys.stderr


def test_stdout_pure_when_logging(capsys):
    setup_logging(level=logging.DEBUG)
    logger = get_logger("pure_stdout_logger")
    logger.debug("Debug msg")
    logger.info("Info msg")
    logger.warning("Warning msg")
    logger.error("Error msg")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Debug msg" in captured.err
    assert "Info msg" in captured.err
    assert "Warning msg" in captured.err
    assert "Error msg" in captured.err


def test_logger_formatting_structure(capsys):
    setup_logging()
    logger = get_logger("format_check_logger")
    logger.info("Formatting test payload")
    captured = capsys.readouterr()
    err_output = captured.err
    assert "[INFO]" in err_output
    assert "[format_check_logger]" in err_output
    assert "Formatting test payload" in err_output


def test_multiline_traceback_stays_on_stderr(capsys):
    setup_logging()
    logger = get_logger("traceback_logger")
    try:
        raise ValueError("Simulated failure for traceback test")
    except ValueError:
        logger.exception("An exception occurred")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Simulated failure for traceback test" in captured.err
    assert "Traceback" in captured.err


def test_unicode_and_special_characters(capsys):
    setup_logging()
    logger = get_logger("unicode_logger")
    logger.info("Unicode test: 🚀 Colab MCP 漢字 - éàçüö")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Unicode test: 🚀 Colab MCP 漢字 - éàçüö" in captured.err


def test_idempotent_protect_stdout():
    protect_stdout()
    proxy1 = sys.stdout
    protect_stdout()
    proxy2 = sys.stdout
    assert proxy1 is proxy2
    assert isinstance(proxy1, StderrProxy)


def test_fileno_returns_stderr_fd():
    protect_stdout()
    try:
        assert sys.stdout.fileno() == sys.stderr.fileno()
    except io.UnsupportedOperation:
        # Some testing captures (like pytest capsys) do not support fileno()
        pass


def test_log_level_filtering(capsys):
    setup_logging(level=logging.WARNING)
    logger = get_logger("level_filter_logger")
    logger.debug("Ignored debug")
    logger.info("Ignored info")
    logger.warning("Emitted warning")
    captured = capsys.readouterr()
    assert "Ignored debug" not in captured.err
    assert "Ignored info" not in captured.err
    assert "Emitted warning" in captured.err
