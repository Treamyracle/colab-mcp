"""
MCP Server implementation for Google Colab automation (MCP SDK v2).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, AsyncIterator

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel

from colab_mcp.auth.tools import register_auth_tools
from colab_mcp.files.tools import register_file_tools
from colab_mcp.kernel.tools import (
    clear_kernel_clients,
    register_kernel_tools,
    reset_execution_registry,
)
from colab_mcp.notebook.runner import register_notebook_tools
from colab_mcp.runtime.session_manager import get_session_manager
from colab_mcp.runtime.tools import register_runtime_tools
from colab_mcp.utils.logging import get_logger, setup_logging
from colab_mcp.utils.redaction import redact_credentials, redact_data

logger = get_logger(__name__)

VALID_TRANSPORTS = frozenset({"stdio", "sse"})
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _normalize_log_level(log_level: str) -> str:
    if not isinstance(log_level, str):
        raise ValueError("log_level must be a string.")
    normalized = log_level.strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Unsupported log_level '{log_level}'. Valid options: {sorted(VALID_LOG_LEVELS)}"
        )
    return normalized


def _validate_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError(f"port must be an integer between 1 and 65535, got {port!r}.")
    return port


def _redact_content_block(block: Any) -> Any:
    """Redact credentials from text content blocks without corrupting ImageContent.data."""
    if isinstance(block, TextContent):
        safe_text = redact_credentials(block.text)
        if safe_text != block.text:
            return block.model_copy(update={"text": safe_text})
        return block
    if isinstance(block, ImageContent):
        return block
    if isinstance(block, dict):
        if block.get("type") == "image":
            return dict(block)
        return redact_data(block)
    if hasattr(block, "text") and isinstance(getattr(block, "text", None), str):
        safe_text = redact_credentials(block.text)
        if safe_text != block.text and hasattr(block, "model_copy"):
            return block.model_copy(update={"text": safe_text})
    return block


def _redact_tool_output(result: Any) -> Any:
    """Recursively redact sensitive credentials from any MCPServer tool dispatch output."""
    if isinstance(result, CallToolResult):
        redacted_content = [_redact_content_block(b) for b in result.content]
        redacted_structured = (
            redact_data(result.structured_content)
            if result.structured_content is not None
            else None
        )
        return result.model_copy(
            update={
                "content": redacted_content,
                "structured_content": redacted_structured,
            }
        )
    if isinstance(result, tuple):
        if len(result) == 2 and isinstance(result[1], dict):
            blocks = _redact_tool_output(result[0])
            structured = redact_data(result[1])
            return (blocks, structured)
        return tuple(_redact_tool_output(item) for item in result)
    if isinstance(result, list):
        return [_redact_tool_output(item) for item in result]
    if isinstance(result, (TextContent, ImageContent)):
        return _redact_content_block(result)
    if isinstance(result, BaseModel):
        dumped = redact_data(result.model_dump())
        try:
            return type(result).model_validate(dumped)
        except Exception:
            return dumped
    if isinstance(result, dict):
        return redact_data(result)
    if isinstance(result, str):
        return redact_credentials(result)
    return result


def _is_domain_error(exc: BaseException) -> bool:
    """Return True if *exc* is an actionable colab_mcp or input validation exception."""
    if type(exc).__module__.startswith("colab_mcp."):
        return True
    return isinstance(exc, (ValueError, TypeError, FileNotFoundError, PermissionError, TimeoutError, RuntimeError))


def _install_egress_redaction(mcp_server: MCPServer) -> None:
    """Wrap MCPServer ToolManager.call_tool with a global egress credential redaction filter."""
    tool_manager = getattr(mcp_server, "_tool_manager", None)
    if tool_manager is None:
        return
    original_call_tool = tool_manager.call_tool
    if getattr(original_call_tool, "_colab_egress_redaction_installed", False):
        return

    @wraps(original_call_tool)
    async def _redacted_call_tool(
        name: str,
        arguments: dict[str, Any],
        context: Any = None,
        convert_result: bool = False,
    ) -> Any:
        try:
            raw_result = await original_call_tool(
                name,
                arguments,
                context=context,
                convert_result=convert_result,
            )
            return _redact_tool_output(raw_result)
        except UnexpectedToolError as exc:
            cause = exc.__cause__
            if cause is not None and _is_domain_error(cause):
                safe_cause = redact_credentials(str(cause))
                raise ToolError(f"{exc}: {safe_cause}") from None
            safe_msg = redact_credentials(str(exc))
            if safe_msg != str(exc):
                raise UnexpectedToolError(safe_msg) from None
            raise
        except ToolError as exc:
            safe_msg = redact_credentials(str(exc))
            if safe_msg != str(exc):
                raise ToolError(safe_msg) from None
            raise
        except Exception as exc:
            safe_msg = redact_credentials(str(exc))
            if safe_msg != str(exc):
                try:
                    raise type(exc)(safe_msg) from None
                except Exception:
                    raise ToolError(safe_msg) from None
            raise

    setattr(_redacted_call_tool, "_colab_egress_redaction_installed", True)
    tool_manager.call_tool = _redacted_call_tool  # type: ignore[method-assign]


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
    """Manage server startup and guaranteed clean shutdown of kernel/session resources."""
    try:
        yield
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await clear_kernel_clients()
            except Exception as exc:
                logger.warning("Error clearing kernel clients during shutdown: %s", exc)
            try:
                reset_execution_registry()
            except Exception as exc:
                logger.warning("Error resetting execution registry during shutdown: %s", exc)
            try:
                await get_session_manager().shutdown()
            except Exception as exc:
                logger.warning("Error shutting down session manager during shutdown: %s", exc)


def create_server(
    name: str = "colab-mcp",
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "INFO",
) -> MCPServer:
    """Create and configure an MCPServer instance exposing all Colab MCP tools."""
    normalized_level = _normalize_log_level(log_level)
    _validate_port(port)
    _ = host  # Validated/passed through run_server(transport="sse", host=..., port=...)
    setup_logging(level=getattr(logging, normalized_level, logging.INFO))

    server = MCPServer(
        name=name,
        instructions=(
            "Google Colab Model Context Protocol server enabling cloud runtime management, "
            "Jupyter kernel code execution, and nbformat v4 notebook orchestration."
        ),
        log_level=normalized_level,  # type: ignore[arg-type]
        lifespan=lifespan,
    )

    register_auth_tools(server)
    register_runtime_tools(server)
    register_kernel_tools(server)
    register_notebook_tools(server)
    register_file_tools(server)
    _install_egress_redaction(server)

    return server


def run_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "INFO",
) -> None:
    """Validate transport/network configuration and run the MCPServer."""
    if not isinstance(transport, str):
        raise ValueError("transport must be a string ('stdio' or 'sse').")
    normalized_transport = transport.strip().lower()
    if normalized_transport not in VALID_TRANSPORTS:
        raise ValueError(
            f"Unsupported transport '{transport}'. Valid options: {sorted(VALID_TRANSPORTS)}"
        )
    validated_port = _validate_port(port)
    normalized_level = _normalize_log_level(log_level)
    setup_logging(level=getattr(logging, normalized_level, logging.INFO))

    server = create_server(
        host=host,
        port=validated_port,
        log_level=normalized_level,
    )
    if normalized_transport == "sse":
        server.run(transport=normalized_transport, host=host, port=validated_port)  # type: ignore[arg-type]
    else:
        server.run(transport=normalized_transport)  # type: ignore[arg-type]


# Default singleton server instance
mcp = create_server()


__all__ = [
    "create_server",
    "lifespan",
    "mcp",
    "run_server",
]
