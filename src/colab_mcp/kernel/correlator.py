"""
Asynchronous Jupyter Message Correlator & Multiplexer (TECH_SPEC §5.4).

Routes incoming multiplexed WebSocket frames (iopub, shell, stdin, control)
to the originating execution request queue keyed by ``parent_header.msg_id``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional


class KernelMessageCorrelator:
    """
    Multiplexes concurrent Jupyter kernel executions over a single WebSocket channel.

    Each active execution registers an ``asyncio.Queue`` keyed by its unique
    ``msg_id``. Incoming frames are dispatched strictly by ``parent_header.msg_id``.
    """

    def __init__(self) -> None:
        self.active_executions: Dict[str, asyncio.Queue[Dict[str, Any]]] = {}

    def register_execution(self, msg_id: str) -> asyncio.Queue[Dict[str, Any]]:
        """Register a new execution ``msg_id`` and return its dedicated message queue."""
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.active_executions[msg_id] = queue
        return queue

    def route_incoming_message(self, message: Dict[str, Any]) -> None:
        """
        Route an incoming Jupyter wire message to its matching execution queue.

        Messages with missing, malformed, or unregistered ``parent_header.msg_id``
        (e.g., unsolicited kernel beacons) are safely ignored.
        """
        if not isinstance(message, dict):
            return
        parent_header = message.get("parent_header")
        if not isinstance(parent_header, dict):
            return
        parent_id = parent_header.get("msg_id")
        if not isinstance(parent_id, str) or not parent_id:
            return
        queue = self.active_executions.get(parent_id)
        if queue is not None:
            queue.put_nowait(message)

    def complete_execution(self, msg_id: str) -> None:
        """Unregister an execution queue once execution completes or aborts (idempotent)."""
        self.active_executions.pop(msg_id, None)

    def is_registered(self, msg_id: str) -> bool:
        """Return True if ``msg_id`` currently has an active registered queue."""
        return msg_id in self.active_executions

    @property
    def active_count(self) -> int:
        """Return the number of currently registered active executions."""
        return len(self.active_executions)

    def flush_queue(self, msg_id: str) -> int:
        """Drain all pending messages from a single execution queue without removing it."""
        queue = self.active_executions.get(msg_id)
        if queue is None:
            return 0
        drained = 0
        while not queue.empty():
            try:
                queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        return drained

    def flush_all(
        self, terminal_message: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Drain all active queues and clear registrations.

        If ``terminal_message`` is provided, enqueues it into each drained queue
        before clearing so blocked consumers wake up cleanly on disconnect/restart.
        """
        flushed_ids = list(self.active_executions.keys())
        for msg_id in flushed_ids:
            queue = self.active_executions.get(msg_id)
            if queue is None:
                continue
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            if terminal_message is not None:
                queue.put_nowait(dict(terminal_message))
        self.active_executions.clear()
        return flushed_ids


MessageCorrelator = KernelMessageCorrelator

__all__ = [
    "KernelMessageCorrelator",
    "MessageCorrelator",
]
