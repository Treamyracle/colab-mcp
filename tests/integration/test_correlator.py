"""
Integration tests for Jupyter Kernel Message Correlator & Protocol Multiplexing.

Requirements:
- TECH_SPEC §5.3, §5.4
- Feature 36 (Jupyter Messaging v5.3 Multiplexing)
- Feature 37 (Message Correlator - msg_id routing)
- Authoritative contract:
    class KernelMessageCorrelator:
        def register_execution(self, msg_id: str) -> asyncio.Queue
        def route_incoming_message(self, message: dict) -> None
        def complete_execution(self, msg_id: str) -> None

Adheres to Python 3.13 and progressive testability across milestones.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
import pytest

# Progressive testability: check implementation availability
try:
    import colab_mcp.kernel.correlator as correlator_mod
    CORRELATOR_AVAILABLE = True
except ImportError:
    correlator_mod = None
    CORRELATOR_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_correlator_implementation():
    """Progressive testability: cleanly skip if colab_mcp.kernel.correlator is not yet implemented."""
    if not CORRELATOR_AVAILABLE:
        pytest.skip("colab_mcp.kernel.correlator is scheduled for Milestone 3 implementation")


def _get_correlator_class():
    return getattr(
        correlator_mod,
        "KernelMessageCorrelator",
        getattr(correlator_mod, "MessageCorrelator", None),
    )


# Helper to build Jupyter v5.3 wire protocol message frames
def _make_frame(
    msg_type: str,
    parent_msg_id: str | None = None,
    channel: str = "iopub",
    content: dict | None = None,
) -> dict:
    return {
        "header": {
            "msg_id": f"hdr-{parent_msg_id or 'none'}-{msg_type}",
            "msg_type": msg_type,
            "version": "5.3",
        },
        "parent_header": {"msg_id": parent_msg_id} if parent_msg_id else {},
        "metadata": {},
        "content": content or {},
        "channel": channel,
    }


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests (>= 5 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_register_execution_creates_and_returns_async_queue():
    """Verify register_execution creates an active asyncio.Queue for a given msg_id."""
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    msg_id = "exec-test-1122"

    queue = correlator.register_execution(msg_id)
    assert isinstance(queue, asyncio.Queue)
    assert queue.empty()


@pytest.mark.asyncio
async def test_route_iopub_stream_message_to_registered_queue():
    """Verify incoming iopub stream frame matching parent_header.msg_id is queued."""
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    msg_id = "exec-test-3344"
    queue = correlator.register_execution(msg_id)

    stream_frame = _make_frame("stream", parent_msg_id=msg_id, channel="iopub", content={"name": "stdout", "text": "OK"})
    correlator.route_incoming_message(stream_frame)

    assert not queue.empty()
    received = await queue.get()
    assert received["header"]["msg_type"] == "stream"
    assert received["content"]["text"] == "OK"


@pytest.mark.asyncio
async def test_route_shell_execute_reply_to_registered_queue():
    """Verify incoming shell execute_reply frame matching parent_header.msg_id is queued."""
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    msg_id = "exec-test-5566"
    queue = correlator.register_execution(msg_id)

    reply_frame = _make_frame("execute_reply", parent_msg_id=msg_id, channel="shell", content={"status": "ok", "execution_count": 5})
    correlator.route_incoming_message(reply_frame)

    assert not queue.empty()
    received = await queue.get()
    assert received["header"]["msg_type"] == "execute_reply"
    assert received["content"]["status"] == "ok"


@pytest.mark.asyncio
async def test_complete_execution_removes_registration():
    """Verify complete_execution unregisters the execution queue cleanly."""
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    msg_id = "exec-test-7788"
    queue = correlator.register_execution(msg_id)

    correlator.complete_execution(msg_id)

    late_frame = _make_frame("stream", parent_msg_id=msg_id)
    correlator.route_incoming_message(late_frame)

    assert queue.empty()


@pytest.mark.asyncio
async def test_full_execution_lifecycle_sequence_routed():
    """
    Verify complete sequence for a single execution:
    status: busy -> stream (stdout) -> display_data -> execute_reply -> status: idle.
    All frames routed in sequence to the consumer queue.
    """
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    msg_id = "exec-lifecycle-001"
    queue = correlator.register_execution(msg_id)

    frames = [
        _make_frame("status", parent_msg_id=msg_id, channel="iopub", content={"execution_state": "busy"}),
        _make_frame("stream", parent_msg_id=msg_id, channel="iopub", content={"name": "stdout", "text": "Epoch 1/1"}),
        _make_frame("display_data", parent_msg_id=msg_id, channel="iopub", content={"data": {"text/plain": "Plot"}}),
        _make_frame("execute_reply", parent_msg_id=msg_id, channel="shell", content={"status": "ok"}),
        _make_frame("status", parent_msg_id=msg_id, channel="iopub", content={"execution_state": "idle"}),
    ]

    for frame in frames:
        correlator.route_incoming_message(frame)

    received_types = []
    while not queue.empty():
        msg = await queue.get()
        received_types.append(msg["header"]["msg_type"])

    assert received_types == ["status", "stream", "display_data", "execute_reply", "status"]


# ============================================================================
# Tier 2: Concurrency, Out-of-Order Delivery, Timeout Tests (>= 5 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_executions_isolation_no_crosstalk():
    """
    Verify multiple concurrent executions (A, B, C) receive strictly their own frames.
    No cross-talk occurs across queues.
    """
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    msg_a = "exec-A"
    msg_b = "exec-B"
    msg_c = "exec-C"

    queue_a = correlator.register_execution(msg_a)
    queue_b = correlator.register_execution(msg_b)
    queue_c = correlator.register_execution(msg_c)

    correlator.route_incoming_message(_make_frame("stream", parent_msg_id=msg_b, content={"text": "B1"}))
    correlator.route_incoming_message(_make_frame("stream", parent_msg_id=msg_a, content={"text": "A1"}))
    correlator.route_incoming_message(_make_frame("stream", parent_msg_id=msg_c, content={"text": "C1"}))
    correlator.route_incoming_message(_make_frame("stream", parent_msg_id=msg_a, content={"text": "A2"}))
    correlator.route_incoming_message(_make_frame("stream", parent_msg_id=msg_b, content={"text": "B2"}))

    item_a1 = await queue_a.get()
    item_a2 = await queue_a.get()
    assert item_a1["content"]["text"] == "A1"
    assert item_a2["content"]["text"] == "A2"
    assert queue_a.empty()

    item_b1 = await queue_b.get()
    item_b2 = await queue_b.get()
    assert item_b1["content"]["text"] == "B1"
    assert item_b2["content"]["text"] == "B2"
    assert queue_b.empty()

    item_c1 = await queue_c.get()
    assert item_c1["content"]["text"] == "C1"
    assert queue_c.empty()


@pytest.mark.asyncio
async def test_route_message_with_unregistered_parent_id_ignored():
    """Verify frames with unknown/unregistered parent_header.msg_id are safely discarded without error."""
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    orphan_frame = _make_frame("stream", parent_msg_id="non-existent-msg-id-1234")
    correlator.route_incoming_message(orphan_frame)


@pytest.mark.asyncio
async def test_route_message_missing_parent_header_ignored():
    """Verify frames without parent_header (e.g. unsolicited kernel heartbeat) are handled safely."""
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    frame_no_parent = {
        "header": {"msg_id": "kernel-beacon-1", "msg_type": "status"},
        "parent_header": {},
        "channel": "iopub",
        "content": {"execution_state": "idle"},
    }
    correlator.route_incoming_message(frame_no_parent)


@pytest.mark.asyncio
async def test_out_of_order_frame_arrival():
    """
    Verify out-of-order message delivery (e.g. stream chunk arrives before status busy)
    is buffered into the queue without dropping data.
    """
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    msg_id = "exec-ooo-123"
    queue = correlator.register_execution(msg_id)

    correlator.route_incoming_message(_make_frame("stream", parent_msg_id=msg_id, content={"text": "output before busy"}))
    correlator.route_incoming_message(_make_frame("status", parent_msg_id=msg_id, content={"execution_state": "busy"}))

    f1 = await queue.get()
    f2 = await queue.get()
    assert f1["header"]["msg_type"] == "stream"
    assert f2["header"]["msg_type"] == "status"


@pytest.mark.asyncio
async def test_queue_timeout_on_unacknowledged_message():
    """Verify awaiting queue when kernel stops sending frames times out after expected duration."""
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    msg_id = "exec-timeout-test"
    queue = correlator.register_execution(msg_id)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.get(), timeout=0.05)


@pytest.mark.asyncio
async def test_high_volume_message_burst():
    """Stress test: burst of 100 messages routed into correlator queue without loss."""
    CorrelatorClass = _get_correlator_class()
    if CorrelatorClass is None:
        pytest.skip("KernelMessageCorrelator class not available")

    correlator = CorrelatorClass()
    msg_id = "exec-burst-test"
    queue = correlator.register_execution(msg_id)

    num_messages = 100
    for i in range(num_messages):
        correlator.route_incoming_message(_make_frame("stream", parent_msg_id=msg_id, content={"text": f"chunk_{i}"}))

    assert queue.qsize() == num_messages
    for i in range(num_messages):
        msg = await queue.get()
        assert msg["content"]["text"] == f"chunk_{i}"
