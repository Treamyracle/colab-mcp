"""Shared WebSocket helper utilities."""

from __future__ import annotations

import inspect
from typing import Any

import websockets


def select_ws_headers_param(connect_fn: Any = None) -> str:
    """
    Determine whether the active ``websockets.connect`` expects
    ``additional_headers`` (v13+/v16+) or ``extra_headers`` (<v13).
    """
    target = connect_fn if connect_fn is not None else websockets.connect
    try:
        params = inspect.signature(target).parameters
        if "additional_headers" in params:
            return "additional_headers"
        if "extra_headers" in params:
            return "extra_headers"
    except (TypeError, ValueError):
        pass

    try:
        import websockets.asyncio.client as ws_asyncio_client  # type: ignore[import-untyped]

        async_params = inspect.signature(ws_asyncio_client.connect).parameters
        if "additional_headers" in async_params:
            return "additional_headers"
        if "extra_headers" in async_params:
            return "extra_headers"
    except Exception:
        pass

    return "additional_headers"


__all__ = ["select_ws_headers_param"]
