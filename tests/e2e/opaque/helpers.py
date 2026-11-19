"""
Opaque-box helpers for MCP ClientSession CallToolResult parsing,
.ipynb construction, and execution-time M3-M5 capability gates.

ZERO imports from `colab_mcp.kernel` or `colab_mcp.notebook`.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Union

import pytest
import typer
from mcp import ClientSession
from mcp.types import CallToolResult, ImageContent, TextContent

from colab_mcp.cli import app as cli_app
from colab_mcp.server import create_server


def extract_blocks_text(result_or_blocks: Union[CallToolResult, List[Any]]) -> str:
    """Extract concatenated text from TextContent blocks of a CallToolResult or block list."""
    blocks = (
        result_or_blocks.content
        if isinstance(result_or_blocks, CallToolResult)
        else result_or_blocks
    )
    parts: List[str] = []
    for block in blocks:
        if isinstance(block, TextContent) or getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def extract_image_blocks(result_or_blocks: Union[CallToolResult, List[Any]]) -> List[ImageContent]:
    """Extract ImageContent blocks from a CallToolResult or block list."""
    blocks = (
        result_or_blocks.content
        if isinstance(result_or_blocks, CallToolResult)
        else result_or_blocks
    )
    images: List[ImageContent] = []
    for block in blocks:
        if isinstance(block, ImageContent) or getattr(block, "type", None) == "image":
            images.append(block)
    return images


def parse_tool_json(result: CallToolResult) -> Dict[str, Any]:
    """Assert CallToolResult succeeded and parse its JSON payload."""
    if result.is_error:
        err_text = extract_blocks_text(result)
        raise AssertionError(f"Expected tool success, got is_error=True: {err_text}")
    if result.structured_content is not None:
        return dict(result.structured_content)
    raw = extract_blocks_text(result)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except Exception as exc:
        raise AssertionError(f"Expected JSON tool output, got {raw!r}: {exc}") from exc


def parse_tool_blocks(result: CallToolResult) -> List[Any]:
    """Assert CallToolResult succeeded and return chronological content blocks."""
    if result.is_error:
        err_text = extract_blocks_text(result)
        raise AssertionError(f"Expected tool success, got is_error=True: {err_text}")
    return list(result.content)


def assert_tool_error(result: CallToolResult, *substrings: str) -> str:
    """Assert CallToolResult failed (is_error=True) and optionally match substrings."""
    assert result.is_error is True, (
        f"Expected tool error (is_error=True), got success: {extract_blocks_text(result)!r}"
    )
    err_text = extract_blocks_text(result)
    lowered = err_text.lower()
    for sub in substrings:
        assert sub.lower() in lowered, f"Expected {sub!r} in error text: {err_text!r}"
    return err_text


# ---------------------------------------------------------------------------
# .ipynb v4 Builder Helpers
# ---------------------------------------------------------------------------

def make_code_cell(source: Union[str, List[str]]) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source],
    }


def make_markdown_cell(source: Union[str, List[str]]) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }


def make_raw_cell(source: Union[str, List[str]]) -> Dict[str, Any]:
    return {
        "cell_type": "raw",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }


def make_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: List[Dict[str, Any]]) -> Path:
    nb = make_notebook(cells)
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Execution-time Capability Gates (In-flight M3-M5 compatibility)
# ---------------------------------------------------------------------------

async def require_tools(session: ClientSession, required: Iterable[str]) -> Set[str]:
    """Skip at execution time if any required MCP tool is not yet registered."""
    resp = await session.list_tools()
    registered = {t.name for t in resp.tools}
    missing = [name for name in required if name not in registered]
    if missing:
        pytest.skip(f"Tools not yet registered (in-flight M3-M5): {missing}")
    return registered


def get_registered_cli_commands() -> Set[str]:
    """Return registered CLI command names cleanly across single-command & multi-command Typer apps."""
    cmd = typer.main.get_command(cli_app)
    if hasattr(cmd, "commands"):
        return set(cmd.commands.keys())
    name = getattr(cmd, "name", None)
    return {name} if name else set()


def require_cli_command(command_name: str) -> None:
    """Skip at execution time if the M5 CLI command is not yet registered."""
    registered = get_registered_cli_commands()
    if command_name not in registered:
        pytest.skip(f"CLI command '{command_name}' not yet registered (M5)")


def require_m5_server_features() -> None:
    """Skip at execution time if M5 server.py features (host/port/lifespan/redaction) are not yet wired."""
    sig = inspect.signature(create_server)
    if "host" not in sig.parameters or "port" not in sig.parameters:
        pytest.skip("M5 create_server(host, port, lifespan, redaction) not yet implemented")
