"""Notebook hash (`nbh`) derivation utility for Google Colab TFE requests."""

from __future__ import annotations

import uuid
from typing import Optional


def generate_nbh(session_id: Optional[uuid.UUID] = None) -> str:
    """
    Generate the 44-character web-safe base64 notebook hash (`nbh`) required by Colab TFE.

    Algorithm (TECH_SPEC §4.2):
    1. Use the provided UUID v4 or generate a fresh ``uuid.uuid4()``.
    2. Replace hyphens (``-``) with underscores (``_``).
    3. Right-pad with ``.`` to reach exactly 44 characters (8 trailing dots).

    Raises:
        TypeError: If ``session_id`` is not ``None`` and not a ``uuid.UUID`` instance.
    """
    if session_id is not None and not isinstance(session_id, uuid.UUID):
        raise TypeError(
            f"session_id must be a uuid.UUID or None, got {type(session_id).__name__}"
        )

    resolved_id = session_id if session_id is not None else uuid.uuid4()
    raw_str = str(resolved_id).replace("-", "_")
    padding = "." * (44 - len(raw_str))
    nbh = raw_str + padding
    if len(nbh) != 44:
        raise ValueError(f"Derived nbh has unexpected length {len(nbh)}: {nbh!r}")
    return nbh


generate_notebook_hash = generate_nbh

__all__ = ["generate_nbh", "generate_notebook_hash"]
