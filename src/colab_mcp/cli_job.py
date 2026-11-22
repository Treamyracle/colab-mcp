"""
Ephemeral one-shot Python script execution helpers for ``colab-mcp job``.

Handles shebang stripping, ``sys.argv`` / ``__main__`` wrapping with exact
filename line numbers via ``compile()``, CPython-compatible ``SystemExit`` code
extraction, and runtime browser attach URL formatting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from mcp.types import TextContent

from colab_mcp.cli_output import CLIExitCode

_JOB_EXIT_SENTINEL = "__COLAB_MCP_JOB_EXIT__="


def strip_shebang(source: str) -> str:
    """Remove a leading ``#!...`` line so shebang scripts execute cleanly remotely."""
    if not source.startswith("#!"):
        return source
    nl_idx = source.find("\n")
    return source[nl_idx + 1 :] if nl_idx != -1 else ""


def build_job_payload(
    script_path: Path,
    script_args: Sequence[str],
) -> str:
    """Wrap *script_path* source so it executes remotely with native ``python`` semantics."""
    raw_source = script_path.read_text(encoding="utf-8")
    body = strip_shebang(raw_source)
    basename = os.path.basename(str(script_path))
    argv_literal = [basename, *[str(arg) for arg in script_args]]

    return (
        "import sys, warnings\n"
        f"sys.argv = {argv_literal!r}\n"
        "__name__ = '__main__'\n"
        f"__file__ = {basename!r}\n"
        "warnings.filterwarnings('ignore', message='To exit: use')\n"
        "try:\n"
        f"    exec(compile({body!r}, {basename!r}, 'exec'), globals())\n"
        "except SystemExit as _colab_sys_exit:\n"
        "    _code = _colab_sys_exit.code\n"
        "    if _code is None:\n"
        "        _code = 0\n"
        "    elif isinstance(_code, bool) or not isinstance(_code, int):\n"
        "        print(str(_code), file=sys.stderr)\n"
        "        _code = 1\n"
        f"    print(f'{_JOB_EXIT_SENTINEL}{{_code}}', file=sys.stderr, flush=True)\n"
    )


def extract_job_result(
    blocks: Sequence[Any],
    has_error: bool,
) -> tuple[List[Any], int]:
    """
    Strip internal exit-code sentinel lines from *blocks* and compute the CLI exit code.

    CPython conventions preserved for ``sys.exit``:
      - ``sys.exit()`` / ``sys.exit(None)`` / ``sys.exit(0)`` -> ``0``
      - ``sys.exit(<int>)`` -> ``<int>``
      - ``sys.exit('msg')`` -> ``1``
      - Uncaught kernel exception / timeout -> ``CLIExitCode.EXECUTION_ERROR`` (``13``)
    """
    cleaned: List[Any] = []
    parsed_code: Optional[int] = None

    for block in blocks:
        is_text = isinstance(block, TextContent) or getattr(block, "type", None) == "text"
        if not is_text:
            cleaned.append(block)
            continue

        text = str(getattr(block, "text", ""))
        kept_lines: List[str] = []
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if stripped.startswith(_JOB_EXIT_SENTINEL):
                raw_val = stripped[len(_JOB_EXIT_SENTINEL) :]
                try:
                    parsed_code = int(raw_val)
                except ValueError:
                    parsed_code = 1
                continue
            kept_lines.append(line)

        new_text = "".join(kept_lines)
        if new_text:
            if isinstance(block, TextContent):
                cleaned.append(TextContent(type="text", text=new_text))
            elif isinstance(block, dict):
                cleaned.append({**block, "text": new_text})
            else:
                cleaned.append(new_text)

    if parsed_code is not None:
        return cleaned, parsed_code
    if has_error:
        return cleaned, int(CLIExitCode.EXECUTION_ERROR)
    return cleaned, 0


def build_runtime_browser_url(
    endpoint: str,
    host: str = "https://colab.research.google.com",
) -> str:
    """Build a Colab browser attach URL pointing directly to an existing runtime endpoint."""
    host_clean = host.strip().rstrip("/") or "https://colab.research.google.com"
    backend_path = f"/tun/m/{endpoint}"
    dbu_value = quote(backend_path, safe="")
    fragment_value = f"{host_clean}{backend_path}"
    return (
        f"{host_clean}/notebooks/empty.ipynb"
        f"?dbu={dbu_value}"
        f"#datalabBackendUrl={fragment_value}"
    )


__all__ = [
    "build_job_payload",
    "build_runtime_browser_url",
    "extract_job_result",
    "strip_shebang",
]
