"""
MCP host auto-configurator (`colab-mcp setup-mcp`).

Safely adds, updates, previews, or removes the ``colab`` MCP server entry across
Claude Desktop, Claude Code, Cursor, Antigravity, and VS Code config files.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

SUPPORTED_MCP_CLIENTS: tuple[str, ...] = (
    "claude-desktop",
    "claude-code",
    "cursor",
    "antigravity",
    "vscode",
)

SUPPORTED_COMMAND_MODES: tuple[str, ...] = ("auto", "uvx", "python")

DEFAULT_UVX_REPO = "git+https://github.com/Treamyracle/colab-mcp.git"


class SetupTargetResult(BaseModel):
    """Configuration outcome for one MCP client host."""

    client: str
    config_path: str
    root_key: str
    server_name: str
    action: str
    dry_run: bool
    config_snippet: Dict[str, Any] = Field(default_factory=dict)


class SetupMCPResponse(BaseModel):
    """Overall result from ``colab-mcp setup-mcp``."""

    status: str
    results: List[SetupTargetResult] = Field(default_factory=list)


def resolve_client_config_path(client: str, custom_path: Optional[Path] = None) -> tuple[Path, str]:
    """Return ``(config_file_path, top_level_servers_key)`` for a supported MCP host."""
    if custom_path is not None:
        key = "servers" if client == "vscode" else "mcpServers"
        return custom_path.expanduser().resolve(), key

    home = Path.home()
    normalized = client.strip().lower()
    if normalized == "claude-desktop":
        if sys.platform == "win32":
            appdata = Path(os.environ.get("APPDATA") or (home / "AppData" / "Roaming"))
            return appdata / "Claude" / "claude_desktop_config.json", "mcpServers"
        if sys.platform == "darwin":
            return (
                home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
                "mcpServers",
            )
        return home / ".config" / "Claude" / "claude_desktop_config.json", "mcpServers"
    if normalized == "claude-code":
        return home / ".claude.json", "mcpServers"
    if normalized == "cursor":
        return home / ".cursor" / "mcp.json", "mcpServers"
    if normalized == "antigravity":
        return home / ".gemini" / "antigravity" / "mcp_config.json", "mcpServers"
    if normalized == "vscode":
        return home / ".vscode" / "mcp.json", "servers"

    valid = ", ".join(SUPPORTED_MCP_CLIENTS)
    raise ValueError(f"Unsupported MCP client '{client}'. Choose from: auto, {valid}.")


def build_mcp_server_entry(command_mode: str = "auto") -> Dict[str, Any]:
    """Build the command/args dictionary for launching ``colab-mcp`` over stdio."""
    mode = command_mode.strip().lower()
    if mode not in SUPPORTED_COMMAND_MODES:
        valid = ", ".join(SUPPORTED_COMMAND_MODES)
        raise ValueError(f"Unsupported command mode '{command_mode}'. Choose from: {valid}.")
    if mode == "uvx":
        return {
            "command": "uvx",
            "args": ["--from", DEFAULT_UVX_REPO, "colab-mcp", "run"],
        }
    if mode == "python":
        return {
            "command": sys.executable,
            "args": ["-m", "colab_mcp", "run", "--transport", "stdio"],
        }
    binary = shutil.which("colab-mcp")
    if binary:
        return {"command": binary, "args": ["run"]}
    return {
        "command": sys.executable,
        "args": ["-m", "colab_mcp", "run", "--transport", "stdio"],
    }


def detect_installed_clients() -> List[str]:
    """Return supported MCP clients whose config file or parent config directory already exists."""
    detected: List[str] = []
    for client in SUPPORTED_MCP_CLIENTS:
        cfg_path, _ = resolve_client_config_path(client)
        if cfg_path.is_file() or cfg_path.parent.is_dir():
            detected.append(client)
    return detected


def _configure_single_target(
    client: str,
    *,
    config_path: Optional[Path] = None,
    server_name: str = "colab",
    command_mode: str = "auto",
    dry_run: bool = False,
    remove: bool = False,
) -> SetupTargetResult:
    target_path, root_key = resolve_client_config_path(client, custom_path=config_path)

    existing: Dict[str, Any] = {}
    if target_path.is_file():
        raw_text = target_path.read_text(encoding="utf-8").strip()
        if raw_text:
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Existing config at '{target_path}' must be a JSON object, got {type(parsed).__name__}."
                )
            existing = parsed

    servers_section = existing.get(root_key)
    if servers_section is None:
        servers_section = {}
    elif not isinstance(servers_section, dict):
        raise ValueError(
            f"Key '{root_key}' in '{target_path}' must be a JSON object."
        )
    else:
        servers_section = dict(servers_section)

    if remove:
        existed = server_name in servers_section
        servers_section.pop(server_name, None)
        action = "removed" if existed else "not_present"
        snippet: Dict[str, Any] = {}
    else:
        entry = build_mcp_server_entry(command_mode)
        if client == "vscode":
            entry = {"type": "stdio", **entry}
        action = "updated" if server_name in servers_section else "added"
        servers_section[server_name] = entry
        snippet = {root_key: {server_name: entry}}

    updated = dict(existing)
    updated[root_key] = servers_section

    if not dry_run and not (remove and action == "not_present"):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_name(f"{target_path.name}.tmp")
        temp_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_path, target_path)

    return SetupTargetResult(
        client=client,
        config_path=str(target_path),
        root_key=root_key,
        server_name=server_name,
        action=action,
        dry_run=dry_run,
        config_snippet=snippet,
    )


def configure_mcp_clients(
    *,
    client: str = "auto",
    config_path: Optional[str] = None,
    server_name: str = "colab",
    command_mode: str = "auto",
    dry_run: bool = False,
    remove: bool = False,
) -> SetupMCPResponse:
    """Configure one or auto-detected MCP clients with ``colab-mcp``."""
    custom = Path(config_path) if config_path else None
    normalized = client.strip().lower()

    if normalized == "auto":
        if custom is not None:
            targets = ["claude-desktop"]
        else:
            targets = detect_installed_clients()
            if not targets:
                return SetupMCPResponse(status="no_clients_detected", results=[])
    else:
        if normalized not in SUPPORTED_MCP_CLIENTS:
            valid = ", ".join(SUPPORTED_MCP_CLIENTS)
            raise ValueError(f"Unsupported MCP client '{client}'. Choose from: auto, {valid}.")
        targets = [normalized]

    results = [
        _configure_single_target(
            target,
            config_path=custom,
            server_name=server_name,
            command_mode=command_mode,
            dry_run=dry_run,
            remove=remove,
        )
        for target in targets
    ]
    return SetupMCPResponse(
        status="dry_run" if dry_run else ("removed" if remove else "configured"),
        results=results,
    )


__all__ = [
    "DEFAULT_UVX_REPO",
    "SUPPORTED_COMMAND_MODES",
    "SUPPORTED_MCP_CLIENTS",
    "SetupMCPResponse",
    "SetupTargetResult",
    "build_mcp_server_entry",
    "configure_mcp_clients",
    "detect_installed_clients",
    "resolve_client_config_path",
]
