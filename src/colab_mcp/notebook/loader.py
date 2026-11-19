"""
Notebook Loader & Workspace Path Security Guard (TECH_SPEC §6.1, §8.3).

Validates workspace path boundaries to prevent directory traversal and parses
``.ipynb`` files or raw JSON strings against the ``nbformat`` v4 schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import nbformat
import nbformat.validator


def validate_workspace_path(
    path_str: str,
    workspace_root: Optional[Path] = None,
) -> Path:
    """
    Validate that ``path_str`` resolves within ``workspace_root`` (default ``Path.cwd()``).

    Rejects empty/non-string paths, explicit ``..`` path segments, null bytes, and
    any target resolving outside the permitted workspace root.
    """
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError("Path must be a non-empty string.")
    if "\x00" in path_str:
        raise ValueError("Path must not contain null bytes.")

    raw_path = Path(path_str)
    if ".." in raw_path.parts:
        raise ValueError(f"Path traversal ('..') is not permitted: '{path_str}'.")

    root = (workspace_root if workspace_root is not None else Path.cwd()).resolve()
    candidate = (root / raw_path).resolve()

    if not candidate.is_relative_to(root):
        raise ValueError(
            f"Path '{path_str}' escapes permitted workspace boundary '{root}'."
        )

    return candidate


def load_and_validate_notebook(
    notebook_path: Optional[str] = None,
    notebook_content: Optional[str] = None,
    *,
    workspace_root: Optional[Path] = None,
) -> nbformat.NotebookNode:
    """
    Load and strictly validate a Jupyter notebook against the ``nbformat`` v4 schema.

    Exactly one of ``notebook_path`` or ``notebook_content`` must be provided.
    Rejects non-v4 notebooks (such as ``nbformat == 3``) instead of silently upgrading
    them, then parses and validates via ``nbformat.reads(..., as_version=4)`` and
    ``nbformat.validator.validate(nb)``.
    """
    if notebook_path is None and notebook_content is None:
        raise ValueError("Either notebook_path or notebook_content must be provided.")
    if notebook_path is not None and notebook_content is not None:
        raise ValueError("Provide either notebook_path OR notebook_content, not both.")

    if notebook_path is not None:
        if not isinstance(notebook_path, str) or not notebook_path.strip():
            raise ValueError("notebook_path must be a non-empty string.")
        if "\x00" in notebook_path:
            raise ValueError("notebook_path must not contain null bytes.")
        if ".." in Path(notebook_path).parts:
            raise ValueError(
                f"Path traversal ('..') is not permitted in notebook_path: '{notebook_path}'."
            )

        if workspace_root is not None:
            resolved_path = validate_workspace_path(
                notebook_path, workspace_root=workspace_root
            )
        else:
            resolved_path = Path(notebook_path).expanduser().resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"Notebook file not found: '{notebook_path}'.")
        if not resolved_path.is_file():
            raise ValueError(f"Notebook path is not a file: '{notebook_path}'.")

        raw_text = resolved_path.read_text(encoding="utf-8")
    else:
        if not isinstance(notebook_content, str):
            raise ValueError("notebook_content must be a string.")
        raw_text = notebook_content

    raw_payload = json.loads(raw_text)
    if not isinstance(raw_payload, dict):
        raise ValueError("Notebook JSON root must be a JSON object.")

    version = raw_payload.get("nbformat")
    if isinstance(version, bool) or not isinstance(version, int) or version != 4:
        raise ValueError(
            f"Unsupported nbformat version {version!r}; only nbformat v4 notebooks are supported."
        )

    nb = nbformat.reads(raw_text, as_version=4)
    nbformat.validator.validate(nb)
    return nb


load_notebook = load_and_validate_notebook


__all__ = [
    "load_and_validate_notebook",
    "load_notebook",
    "validate_workspace_path",
]
