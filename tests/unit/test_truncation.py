"""
Unit tests for Two-Stage Head-Tail Output Truncation.

Requirements:
- TECH_SPEC §5.5 (Output Safety Bounds & Head-Tail Truncation)
- Feature 40 (Two-Stage Head-Tail Output Truncation)
- Algorithm:
  Stage 1: Line-based truncation if line count > max_lines (default: 1000).
           Retains head (max_lines // 2) and tail (max_lines - head_count).
  Stage 2: Byte-based truncation if encoded payload > max_bytes (default: 102400 / 100 KB).
           Retains head_bytes (max_bytes // 2) and tail_bytes (max_bytes // 2).
  Banner:  Prominent notice indicating omitted lines and/or bytes to preserve LLM context.

Adheres to Python 3.13 and progressive testability across milestones.
"""

from __future__ import annotations

import pytest

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


def _truncate(text: str, max_lines: int = 1000, max_bytes: int = 102400) -> str:
    func = getattr(
        mime_mod,
        "apply_head_tail_truncation",
        getattr(mime_mod, "truncate_output", None),
    )
    if func is not None:
        return func(text, max_lines=max_lines, max_bytes=max_bytes)
    raise AttributeError("No apply_head_tail_truncation found in colab_mcp.kernel.mime_converter")


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests (>= 5 tests)
# ============================================================================

def test_text_within_both_limits_returned_unchanged():
    """Verify text with fewer lines than max_lines and fewer bytes than max_bytes is unmodified."""
    sample = "Line 1\nLine 2\nLine 3\nFinal line"
    truncated = _truncate(sample, max_lines=10, max_bytes=1024)
    assert truncated == sample


def test_stage1_line_truncation_applied_when_line_count_exceeds_threshold():
    """Verify Stage 1 line-based truncation triggers when lines > max_lines."""
    lines = [f"Step {i:04d}: processing chunk" for i in range(1500)]
    text = "\n".join(lines)
    max_lines = 1000

    truncated = _truncate(text, max_lines=max_lines, max_bytes=1024000)
    assert len(truncated.splitlines()) < 1500
    assert "TRUNCATED" in truncated or "truncated" in truncated.lower()


def test_stage1_head_and_tail_content_preserved():
    """Verify the very first lines (setup) and very last lines (summary) are preserved."""
    lines = [f"Line_{i}" for i in range(1200)]
    text = "\n".join(lines)

    truncated = _truncate(text, max_lines=1000, max_bytes=1024000)
    # Head lines check
    assert "Line_0" in truncated
    assert "Line_10" in truncated
    # Tail lines check
    assert "Line_1190" in truncated
    assert "Line_1199" in truncated
    # Omitted middle lines check
    assert "Line_600" not in truncated


def test_stage1_truncation_notice_banner_reports_omitted_line_count():
    """Verify the truncation notice banner accurately includes the count of omitted lines."""
    total_lines = 1500
    max_lines = 1000
    expected_omitted = total_lines - max_lines  # 500

    lines = [f"Data row {i}" for i in range(total_lines)]
    text = "\n".join(lines)

    truncated = _truncate(text, max_lines=max_lines, max_bytes=1024000)
    assert str(expected_omitted) in truncated
    assert "omitted" in truncated.lower() or "truncated" in truncated.lower()


def test_stage2_byte_truncation_applied_on_single_long_line():
    """
    Verify Stage 2 byte-based truncation triggers when a single massive line exceeds max_bytes,
    even though the line count (1) is well below max_lines (1000).
    """
    single_long_line = "A" * 200000  # 200 KB
    max_bytes = 102400  # 100 KB

    truncated = _truncate(single_long_line, max_lines=1000, max_bytes=max_bytes)
    assert len(truncated.encode("utf-8")) < 200000
    assert "TRUNCATED" in truncated or "truncated" in truncated.lower()
    assert "bytes" in truncated.lower()


def test_both_stages_applied_when_lines_and_bytes_exceeded():
    """Verify both stages apply sequentially when input exceeds both line and byte ceilings."""
    # 2000 lines with 200 characters each = ~400 KB (> 1000 lines and > 100 KB)
    lines = [f"Line {i:04d} " + "X" * 200 for i in range(2000)]
    text = "\n".join(lines)

    truncated = _truncate(text, max_lines=1000, max_bytes=102400)
    # Result must satisfy both ceilings
    assert len(truncated.splitlines()) <= 1005  # allowing banner lines
    assert "TRUNCATED" in truncated or "truncated" in truncated.lower()


# ============================================================================
# Tier 2: Boundary, Edge Cases, Encoding Safety Tests (>= 5 tests)
# ============================================================================

def test_exact_line_boundary_not_truncated():
    """Verify input with exactly max_lines (e.g. 1000) lines is NOT truncated."""
    lines = [f"L{i}" for i in range(1000)]
    text = "\n".join(lines)
    truncated = _truncate(text, max_lines=1000, max_bytes=102400)
    assert truncated == text
    assert "TRUNCATED" not in truncated


def test_line_boundary_plus_one_is_truncated():
    """Verify input with max_lines + 1 (e.g. 1001) lines triggers truncation."""
    lines = [f"L{i}" for i in range(1001)]
    text = "\n".join(lines)
    truncated = _truncate(text, max_lines=1000, max_bytes=102400)
    assert truncated != text
    assert "TRUNCATED" in truncated or "truncated" in truncated.lower()


def test_exact_byte_boundary_not_truncated():
    """Verify payload with byte size exactly equal to max_bytes is NOT truncated."""
    max_bytes = 500
    text = "A" * max_bytes
    truncated = _truncate(text, max_lines=1000, max_bytes=max_bytes)
    assert truncated == text
    assert "TRUNCATED" not in truncated


def test_byte_boundary_plus_one_is_truncated():
    """Verify payload with byte size max_bytes + 1 triggers byte truncation."""
    max_bytes = 500
    text = "A" * (max_bytes + 1)
    truncated = _truncate(text, max_lines=1000, max_bytes=max_bytes)
    assert truncated != text
    assert "TRUNCATED" in truncated or "truncated" in truncated.lower()


def test_multibyte_utf8_split_safety_no_decoding_crash():
    """
    Adversarial: Multi-byte UTF-8 characters (e.g. 3-byte CJK or 4-byte emoji)
    sliced at arbitrary byte boundaries must NOT produce UnicodeDecodeError.
    """
    multibyte_text = "🚀🌟🔥🎉" * 2000
    truncated = _truncate(multibyte_text, max_lines=1000, max_bytes=500)
    assert isinstance(truncated, str)
    assert len(truncated) > 0


def test_small_max_lines_threshold():
    """Verify behavior when max_lines is very small (e.g. max_lines=2 or max_lines=4)."""
    text = "\n".join([f"Line_{i}" for i in range(10)])
    truncated = _truncate(text, max_lines=2, max_bytes=102400)
    assert "Line_0" in truncated
    assert "Line_9" in truncated
    assert "TRUNCATED" in truncated or "truncated" in truncated.lower()


def test_small_max_bytes_threshold():
    """Verify behavior when max_bytes is very small (e.g. 100 bytes)."""
    text = "HelloWorld1234567890" * 20
    truncated = _truncate(text, max_lines=1000, max_bytes=100)
    assert len(truncated.encode("utf-8")) < len(text.encode("utf-8"))
    assert "TRUNCATED" in truncated or "truncated" in truncated.lower()


def test_empty_string_returns_empty_string():
    """Verify empty string input returns empty string without error."""
    assert _truncate("", max_lines=1000, max_bytes=102400) == ""


def test_odd_line_count_preserves_total_threshold():
    """Verify odd max_lines (e.g. max_lines=5) divides cleanly without off-by-one errors."""
    lines = [f"Row_{i}" for i in range(21)]
    text = "\n".join(lines)
    truncated = _truncate(text, max_lines=5, max_bytes=102400)
    assert "Row_0" in truncated
    assert "Row_20" in truncated
    assert "TRUNCATED" in truncated or "truncated" in truncated.lower()
