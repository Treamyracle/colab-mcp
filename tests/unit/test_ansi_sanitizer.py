r"""
Unit tests for ECMA-48 Compliant ANSI Sanitizer Regex & Traceback Sanitization.

Requirements:
- TECH_SPEC §5.5 (ECMA-48 Compliant ANSI Escape Sanitization Regex)
- Feature 39 (ECMA-48 Compliant ANSI Sanitizer)
- Authoritative pattern:
    r'\x1b(?:'
    r'\[[0-?]*[ -/]*[@-~]'           # CSI sequences (colors, cursor modes, line clears)
    r'|(?s:\].*?(?:\x1b\\|\x07))'    # OSC sequences (hyperlinks, window titles)
    r'|[\(\)][A-Z0-9]'               # VT100 character set designations
    r'|[@-Z\\-_]'                    # 2-byte Fe control sequences
    r')'

Adheres to Python 3.13 and progressive testability across milestones.
"""

from __future__ import annotations

import re
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


def _get_pattern() -> re.Pattern | None:
    return getattr(mime_mod, "ANSI_ESCAPE_PATTERN", None)


def _sanitize_lines(lines: list[str]) -> str:
    """Helper to sanitize a list of traceback lines."""
    sanitize_ansi_traceback = getattr(mime_mod, "sanitize_ansi_traceback", None)
    if sanitize_ansi_traceback is not None:
        return sanitize_ansi_traceback(lines)
    pattern = _get_pattern()
    if pattern is not None:
        return "\n".join(pattern.sub("", line) for line in lines)
    raise AttributeError("No sanitize_ansi_traceback or ANSI_ESCAPE_PATTERN found")


def _sanitize_string(text: str) -> str:
    """Helper to sanitize a single text string."""
    sanitize_ansi_text = getattr(
        mime_mod,
        "sanitize_ansi_text",
        getattr(mime_mod, "clean_ansi", None),
    )
    if sanitize_ansi_text is not None:
        return sanitize_ansi_text(text)
    pattern = _get_pattern()
    if pattern is not None:
        return pattern.sub("", text)
    sanitize_ansi_traceback = getattr(mime_mod, "sanitize_ansi_traceback", None)
    if sanitize_ansi_traceback is not None:
        return sanitize_ansi_traceback([text])
    raise AttributeError("No sanitization function or pattern found")


# ============================================================================
# Tier 1: Primary Behavior (Happy Path) Tests (>= 5 tests)
# ============================================================================

def test_strip_standard_csi_color_codes():
    """Verify standard 16-color ANSI escapes (red, green, bold, reset) are cleanly stripped."""
    colored_text = "\x1b[31mRed Alert\x1b[0m \x1b[1;32mBold Green\x1b[0m"
    cleaned = _sanitize_string(colored_text)
    assert cleaned == "Red Alert Bold Green"
    assert "\x1b" not in cleaned


def test_strip_csi_24bit_truecolor_sequences():
    """Verify 24-bit RGB truecolor escape sequences (foreground 38;2 and background 48;2) are stripped."""
    truecolor_text = "\x1b[38;2;255;128;64mCustom Orange\x1b[0m \x1b[48;2;0;0;128mDark Blue BG\x1b[0m"
    cleaned = _sanitize_string(truecolor_text)
    assert cleaned == "Custom Orange Dark Blue BG"
    assert "\x1b" not in cleaned


def test_strip_csi_cursor_and_screen_clear_sequences():
    """Verify CSI cursor navigation, screen clear, and line clear sequences are stripped."""
    cursor_text = "\x1b[2J\x1b[HLoading...\x1b[10;20H\x1b[KDone"
    cleaned = _sanitize_string(cursor_text)
    assert cleaned == "Loading...Done"


def test_preserve_plain_text_newlines_and_indentation():
    """Verify ordinary text with spaces, Python indentation (4 spaces), and newlines is preserved intact."""
    raw_snippet = "def calculate_loss(pred, target):\n    loss = (pred - target) ** 2\n    return loss\n"
    cleaned = _sanitize_string(raw_snippet)
    assert cleaned == raw_snippet


def test_preserve_tabs_and_unicode_characters():
    """Verify tab characters, mathematical symbols, Greek letters, and emojis are untouched."""
    unicode_text = "Metric\tValue\nalpha (α)\t0.001\nAccuracy 🎯\t98.5%\nΔ_loss\t-0.04"
    cleaned = _sanitize_string(unicode_text)
    assert cleaned == unicode_text


def test_sanitize_ansi_traceback_joins_lines_with_newlines():
    """Verify sanitize_ansi_traceback cleans multiple lines and joins them with standard newlines."""
    traceback_lines = [
        "\x1b[0;31mTraceback (most recent call last):\x1b[0m",
        "  File \x1b[0;32m\"main.py\"\x1b[0m, line 15, in \x1b[0;34m<module>\x1b[0m",
        "    x = 1 / 0",
        "\x1b[0;31mZeroDivisionError\x1b[0m: division by zero",
    ]
    cleaned = _sanitize_lines(traceback_lines)
    expected_lines = [
        "Traceback (most recent call last):",
        "  File \"main.py\", line 15, in <module>",
        "    x = 1 / 0",
        "ZeroDivisionError: division by zero",
    ]
    assert cleaned == "\n".join(expected_lines)


# ============================================================================
# Tier 2: Boundary, Edge Cases, Adversarial Tests (>= 5 tests)
# ============================================================================

def test_strip_multiline_osc_sequence():
    r"""
    TECH_SPEC §5.5: Strips multiline OSC sequences matching (?s:\].*?(?:\x1b\\|\x07)).
    Verifies terminal title or payload spanning multiple lines is completely removed.
    """
    osc_payload = "Before\x1b]0;Title\nMultiLine\nWindow\x07After"
    cleaned = _sanitize_string(osc_payload)
    assert cleaned == "BeforeAfter"
    assert "Title" not in cleaned
    assert "MultiLine" not in cleaned


def test_strip_osc_hyperlinks_with_st_terminator():
    """Verify OSC 8 terminal hyperlinks terminated with String Terminator (\\x1b\\\\) are stripped."""
    hyperlink_text = "Documentation: \x1b]8;;https://colab.research.google.com\x1b\\Google Colab Portal\x1b]8;;\x1b\\ (Click here)"
    cleaned = _sanitize_string(hyperlink_text)
    assert cleaned == "Documentation: Google Colab Portal (Click here)"
    assert "https://colab.research.google.com" not in cleaned


def test_strip_vt100_character_set_designations():
    """Verify VT100 character set designations such as \\x1b(B and \\x1b)0 are stripped."""
    vt100_text = "\x1b(BNormal Text\x1b)0Special Set"
    cleaned = _sanitize_string(vt100_text)
    assert cleaned == "Normal TextSpecial Set"


def test_strip_fe_control_sequences():
    """Verify 2-byte Fe control sequences such as \\x1bN, \\x1bO, \\x1bM are stripped."""
    fe_text = "Line1\x1bMLine2\x1bNLine3"
    cleaned = _sanitize_string(fe_text)
    assert cleaned == "Line1Line2Line3"


def test_realistic_ipython_syntax_highlighted_traceback():
    """
    Verify complete realistic IPython traceback containing complex ANSI 24-bit codes,
    reverse video, line markers (--->), and colored frame borders is fully sanitized.
    """
    raw_tb = [
        "\x1b[0;31m---------------------------------------------------------------------------\x1b[0m",
        "\x1b[0;31mRuntimeError\x1b[0m                              Traceback (most recent call last)",
        "\x1b[0;32m<ipython-input-4-7a3b4c12d5e6>\x1b[0m in \x1b[0;35m<cell line: 3>\x1b[0m\x1b[0;34m()\x1b[0m",
        "      \x1b[0;32m1\x1b[0m \x1b[0;34mimport torch\x1b[0m\n",
        "      \x1b[0;32m2\x1b[0m \x1b[0;34m# Attempt GPU allocation\x1b[0m\n",
        "\x1b[0;31m----> 3\x1b[0m \x1b[0;34mtorch.zeros(1000000000000, device='cuda')\x1b[0m\n",
        "\x1b[0;31mRuntimeError\x1b[0m: CUDA out of memory. Tried to allocate 3.73 TiB",
    ]
    cleaned = _sanitize_lines(raw_tb)
    assert "\x1b[" not in cleaned
    assert "----> 3" in cleaned
    assert "torch.zeros(1000000000000, device='cuda')" in cleaned
    assert "CUDA out of memory" in cleaned


def test_empty_traceback_lines_returns_empty_string():
    """An empty list of traceback lines returns an empty string without raising errors."""
    cleaned = _sanitize_lines([])
    assert cleaned == ""


def test_lone_escape_byte_without_sequence_does_not_crash():
    """
    Adversarial: Lone escape character '\\x1b' without following control characters
    does not cause infinite loop, catastrophic backtracking, or crash.
    """
    adversarial_input = "Hello \x1b world \x1b"
    # Should safely process without unhandled exceptions
    cleaned = _sanitize_string(adversarial_input)
    assert isinstance(cleaned, str)
    assert "Hello" in cleaned
    assert "world" in cleaned


def test_compiled_regex_pattern_attributes():
    """Verify ANSI_ESCAPE_PATTERN is a compiled regex with correct flags and functionality."""
    pattern = _get_pattern()
    if pattern is not None:
        assert isinstance(pattern, re.Pattern)
        assert pattern.search("\x1b[31m") is not None
        assert pattern.search("\x1b]0;Title\x07") is not None
        assert pattern.search("Normal text") is None
