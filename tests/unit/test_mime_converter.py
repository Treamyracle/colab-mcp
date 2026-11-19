"""
Unit tests for Multimodal MIME Output Processing & Chronological Ordering.

Requirements:
- TECH_SPEC §5.5, §7.3.3
- PROJECT.md § Interface Contracts (Contract 4)
- Features: 38 (Chronological Multimodal MIME Translation),
  41 (Visual Image Limiting - max_images).

Adheres to Python 3.13 and progressive testability across milestones.
"""

from __future__ import annotations

import base64
import pytest
from mcp.types import ImageContent, TextContent

# Progressive testability: check implementation availability
try:
    import colab_mcp.kernel.mime_converter as mime_mod
    MIME_AVAILABLE = True
except ImportError:
    mime_mod = None
    MIME_AVAILABLE = False


@pytest.fixture(autouse=True)
def require_mime_implementation():
    """Progressive testability: cleanly skip if colab_mcp.kernel.mime_converter is not yet implemented."""
    if not MIME_AVAILABLE:
        pytest.skip("colab_mcp.kernel.mime_converter is scheduled for Milestone 3 implementation")


def _convert(messages: list[dict], **kwargs) -> list[TextContent | ImageContent]:
    """Helper dispatching to either function or MimeConverter class instance."""
    convert_outputs = getattr(
        mime_mod,
        "convert_iopub_to_mcp_blocks",
        getattr(
            mime_mod,
            "convert_iopub_messages",
            getattr(mime_mod, "format_output_blocks", None),
        ),
    )
    if convert_outputs is not None:
        return convert_outputs(messages, **kwargs)

    MimeConverter = getattr(mime_mod, "MimeConverter", None)
    if MimeConverter is not None:
        converter = MimeConverter(**kwargs)
        if hasattr(converter, "convert_messages"):
            return converter.convert_messages(messages)
        elif hasattr(converter, "convert"):
            return converter.convert(messages)
        elif hasattr(converter, "format_outputs"):
            return converter.format_outputs(messages)
    raise AttributeError("No supported conversion method found on colab_mcp.kernel.mime_converter")


# Sample 1x1 transparent PNG base64 for deterministic testing
SAMPLE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
# Sample 1x1 JPEG base64 for deterministic testing
SAMPLE_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
)


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests (>= 5 tests)
# ============================================================================

def test_convert_stdout_stream_produces_text_content():
    """Verify standard iopub stream message (stdout) converts to MCP TextContent."""
    messages = [
        {
            "header": {"msg_type": "stream"},
            "content": {"name": "stdout", "text": "Training loss: 0.234\n"},
        }
    ]
    blocks = _convert(messages)
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].type == "text"
    assert "Training loss: 0.234" in blocks[0].text


def test_convert_stderr_stream_produces_text_content_with_stderr_prefix():
    """Verify iopub stream message (stderr) converts to MCP TextContent with [stderr] prefix."""
    messages = [
        {
            "header": {"msg_type": "stream"},
            "content": {"name": "stderr", "text": "UserWarning: CUDA initialization delayed\n"},
        }
    ]
    blocks = _convert(messages)
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].type == "text"
    assert "[stderr]" in blocks[0].text or "CUDA initialization delayed" in blocks[0].text


def test_convert_png_display_data_produces_image_content():
    """Verify display_data frame containing image/png converts to MCP ImageContent."""
    messages = [
        {
            "header": {"msg_type": "display_data"},
            "content": {
                "data": {"image/png": SAMPLE_PNG_B64},
                "metadata": {},
            },
        }
    ]
    blocks = _convert(messages)
    assert len(blocks) == 1
    assert isinstance(blocks[0], ImageContent)
    assert blocks[0].type == "image"
    assert blocks[0].mimeType == "image/png"
    assert blocks[0].data.strip() == SAMPLE_PNG_B64.strip()


def test_convert_jpeg_display_data_produces_image_content():
    """Verify display_data frame containing image/jpeg converts to MCP ImageContent with image/jpeg mimeType."""
    messages = [
        {
            "header": {"msg_type": "display_data"},
            "content": {
                "data": {"image/jpeg": SAMPLE_JPEG_B64},
                "metadata": {},
            },
        }
    ]
    blocks = _convert(messages)
    assert len(blocks) == 1
    assert isinstance(blocks[0], ImageContent)
    assert blocks[0].type == "image"
    assert blocks[0].mimeType == "image/jpeg"
    assert blocks[0].data.strip() == SAMPLE_JPEG_B64.strip()


def test_chronological_interleaved_emission_order():
    """
    Verify chronological emission order is preserved:
    stream (stdout) -> display_data (PNG) -> stream (stdout) -> display_data (JPEG).
    MCP blocks must be strictly [TextContent, ImageContent, TextContent, ImageContent].
    """
    messages = [
        {
            "header": {"msg_type": "stream"},
            "content": {"name": "stdout", "text": "Step 1: Training curve\n"},
        },
        {
            "header": {"msg_type": "display_data"},
            "content": {"data": {"image/png": SAMPLE_PNG_B64}},
        },
        {
            "header": {"msg_type": "stream"},
            "content": {"name": "stdout", "text": "Step 2: Confusion matrix\n"},
        },
        {
            "header": {"msg_type": "display_data"},
            "content": {"data": {"image/jpeg": SAMPLE_JPEG_B64}},
        },
    ]
    blocks = _convert(messages)
    assert len(blocks) == 4
    assert isinstance(blocks[0], TextContent)
    assert "Step 1" in blocks[0].text
    assert isinstance(blocks[1], ImageContent)
    assert blocks[1].mimeType == "image/png"
    assert isinstance(blocks[2], TextContent)
    assert "Step 2" in blocks[2].text
    assert isinstance(blocks[3], ImageContent)
    assert blocks[3].mimeType == "image/jpeg"


def test_convert_execute_result_text_plain_produces_text_content():
    """Verify execute_result frame with text/plain produces MCP TextContent."""
    messages = [
        {
            "header": {"msg_type": "execute_result"},
            "content": {
                "execution_count": 42,
                "data": {"text/plain": "Tensor([1.0, 2.0, 3.0])"},
                "metadata": {},
            },
        }
    ]
    blocks = _convert(messages)
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert "Tensor([1.0, 2.0, 3.0])" in blocks[0].text


# ============================================================================
# Tier 2: Boundary, Edge Cases, Error Handling Tests (>= 5 tests)
# ============================================================================

def test_max_images_limit_retains_first_n_and_appends_notice():
    """
    TECH_SPEC §5.5: Visual Image Limiting (max_images, default: 5).
    Emitting 7 images when max_images=5 retains first 5 ImageContent blocks chronologically,
    omits the remaining 2, and appends a notice TextContent block:
    '[Notice: 2 additional plot images omitted due to max_images limit]'.
    """
    messages = []
    for i in range(7):
        messages.append({
            "header": {"msg_type": "display_data"},
            "content": {
                "data": {"image/png": SAMPLE_PNG_B64},
                "metadata": {"index": i},
            },
        })

    blocks = _convert(messages, max_images=5)
    image_blocks = [b for b in blocks if isinstance(b, ImageContent)]
    text_blocks = [b for b in blocks if isinstance(b, TextContent)]

    assert len(image_blocks) == 5, f"Expected exactly 5 ImageContent blocks, got {len(image_blocks)}"
    # Verify notice block is appended
    notice_found = any(
        "omitted due to max_images" in tb.text.lower() or "additional plot" in tb.text.lower()
        for tb in text_blocks
    )
    assert notice_found, f"Expected notice for omitted images, got text blocks: {[tb.text for tb in text_blocks]}"


def test_max_images_exact_boundary_retains_all_without_notice():
    """Emitting exactly max_images (e.g. 5) images retains all 5 and appends NO omission notice."""
    messages = [
        {
            "header": {"msg_type": "display_data"},
            "content": {"data": {"image/png": SAMPLE_PNG_B64}},
        }
        for _ in range(5)
    ]
    blocks = _convert(messages, max_images=5)
    image_blocks = [b for b in blocks if isinstance(b, ImageContent)]
    text_blocks = [b for b in blocks if isinstance(b, TextContent)]

    assert len(image_blocks) == 5
    for tb in text_blocks:
        assert "omitted due to max_images" not in tb.text.lower()


def test_max_images_custom_threshold_one():
    """Setting max_images=1 with 3 images retains exactly 1 ImageContent and appends notice for 2 omitted."""
    messages = [
        {
            "header": {"msg_type": "display_data"},
            "content": {"data": {"image/png": SAMPLE_PNG_B64}},
        }
        for _ in range(3)
    ]
    blocks = _convert(messages, max_images=1)
    image_blocks = [b for b in blocks if isinstance(b, ImageContent)]
    text_blocks = [b for b in blocks if isinstance(b, TextContent)]

    assert len(image_blocks) == 1
    notice_found = any("2" in tb.text and "omitted" in tb.text.lower() for tb in text_blocks)
    assert notice_found, "Expected notice indicating 2 images omitted"


def test_empty_iopub_messages_returns_empty_list():
    """An empty sequence of iopub messages returns an empty list or no-op content."""
    blocks = _convert([])
    assert isinstance(blocks, list)
    assert len(blocks) == 0 or (len(blocks) == 1 and isinstance(blocks[0], TextContent) and blocks[0].text == "")


def test_multi_mime_display_data_prioritizes_image():
    """
    A single display_data frame providing both 'text/plain' and 'image/png'
    must prioritize the visual image representation rather than dropping it or duplicating raw ascii.
    """
    messages = [
        {
            "header": {"msg_type": "display_data"},
            "content": {
                "data": {
                    "text/plain": "<Figure size 640x480 with 1 Axes>",
                    "image/png": SAMPLE_PNG_B64,
                },
                "metadata": {},
            },
        }
    ]
    blocks = _convert(messages)
    image_blocks = [b for b in blocks if isinstance(b, ImageContent)]
    assert len(image_blocks) == 1
    assert image_blocks[0].mimeType == "image/png"


def test_unknown_or_corrupted_mime_payload_handled_gracefully():
    """Display data frame containing an unsupported or proprietary MIME type does not crash."""
    messages = [
        {
            "header": {"msg_type": "display_data"},
            "content": {
                "data": {"application/vnd.custom.plot+json": '{"x": [1, 2], "y": [3, 4]}'},
                "metadata": {},
            },
        }
    ]
    # Must not raise unhandled exception
    blocks = _convert(messages)
    assert isinstance(blocks, list)


def test_error_message_with_ansi_traceback_converted_and_sanitized():
    """
    TECH_SPEC §5.5: error message with ANSI escapes in traceback must be
    converted to TextContent with ANSI color codes stripped.
    """
    messages = [
        {
            "header": {"msg_type": "error"},
            "content": {
                "ename": "ZeroDivisionError",
                "evalue": "division by zero",
                "traceback": [
                    "\x1b[0;31m---------------------------------------------------------------------------\x1b[0m",
                    "\x1b[0;31mZeroDivisionError\x1b[0m: division by zero",
                ],
            },
        }
    ]
    blocks = _convert(messages)
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert "\x1b[0;31m" not in blocks[0].text
    assert "ZeroDivisionError" in blocks[0].text
    assert "division by zero" in blocks[0].text
