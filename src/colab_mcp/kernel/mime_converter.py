"""
Multimodal MIME Output Processor, Chronological Interleaving, ECMA-48 ANSI Sanitizer,
and Two-Stage Head-Tail Truncation (TECH_SPEC §5.5 & §8.4).
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Sequence, Union

from mcp.types import ImageContent, TextContent

from colab_mcp.utils.redaction import redact_credentials

# ECMA-48 compliant ANSI escape sequence pattern (TECH_SPEC §5.5 & §8.4)
ANSI_ESCAPE_PATTERN: re.Pattern[str] = re.compile(
    r'\x1b(?:'
    r'\[[0-?]*[ -/]*[@-~]'           # CSI sequences (colors, cursor modes, line clears)
    r'|(?s:\].*?(?:\x1b\\|\x07))'    # OSC sequences with DOTALL (hyperlinks, window titles)
    r'|[\(\)][A-Z0-9]'               # VT100 character set designations
    r'|[@-Z\\-_]'                    # 2-byte Fe control sequences
    r')'
)

_HTML_STYLE_SCRIPT_PATTERN = re.compile(
    r'<(script|style)\b[^>]*>.*?</\1>',
    re.IGNORECASE | re.DOTALL,
)
_HTML_BLOCK_BREAK_PATTERN = re.compile(
    r'</?(?:br|p|div|tr|li|h[1-6]|pre|blockquote)\b[^>]*>',
    re.IGNORECASE,
)
_HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

MCPContentBlock = Union[TextContent, ImageContent]


def sanitize_ansi_text(text: str) -> str:
    """Strip ECMA-48 ANSI escape codes, OSC sequences, VT100 charsets, and Fe sequences."""
    if not text:
        return ""
    return ANSI_ESCAPE_PATTERN.sub("", text)


clean_ansi = sanitize_ansi_text


def sanitize_ansi_traceback(raw_traceback_lines: Sequence[str] | str) -> str:
    """Strip ANSI escape codes from traceback line lists (or a single string)."""
    if isinstance(raw_traceback_lines, str):
        return sanitize_ansi_text(raw_traceback_lines)
    if not raw_traceback_lines:
        return ""
    cleaned_lines = [ANSI_ESCAPE_PATTERN.sub("", str(line)) for line in raw_traceback_lines]
    return "\n".join(cleaned_lines)


def _html_to_readable_text(raw_html: str) -> str:
    """Convert simple Jupyter text/html payloads to clean readable plain text."""
    if not raw_html:
        return ""
    cleaned = _HTML_STYLE_SCRIPT_PATTERN.sub("", raw_html)
    cleaned = _HTML_BLOCK_BREAK_PATTERN.sub("\n", cleaned)
    cleaned = _HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in cleaned.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


def apply_head_tail_truncation(
    text: str,
    max_lines: int = 1000,
    max_bytes: int = 102400,
) -> str:
    """
    Two-stage head-tail output truncation preserving start and end of output.

    Stage 1: Line-based truncation if line count > ``max_lines``.
    Stage 2: UTF-8 byte-based truncation if encoded size > ``max_bytes``.
    """
    if not text:
        return ""

    safe_max_lines = max(0, int(max_lines))
    safe_max_bytes = max(0, int(max_bytes))

    encoded = text.encode("utf-8")
    lines = text.splitlines()

    if len(lines) <= safe_max_lines and len(encoded) <= safe_max_bytes:
        return text

    # Stage 1: Line-based truncation
    omitted_lines = 0
    if len(lines) > safe_max_lines:
        head_line_count = safe_max_lines // 2
        tail_line_count = safe_max_lines - head_line_count
        head_lines = lines[:head_line_count]
        tail_lines = lines[-tail_line_count:] if tail_line_count > 0 else []
        omitted_lines = len(lines) - (len(head_lines) + len(tail_lines))

        line_banner = (
            f"\n\n... [OUTPUT TRUNCATED: {omitted_lines} lines omitted "
            f"to preserve LLM context window] ...\n\n"
        )
        stage1_text = "\n".join(head_lines) + line_banner + "\n".join(tail_lines)
        encoded = stage1_text.encode("utf-8")
        if len(encoded) <= safe_max_bytes:
            return stage1_text

    # Stage 2: Byte-based truncation (UTF-8 boundary safe)
    half_bytes = safe_max_bytes // 2
    head_str = encoded[:half_bytes].decode("utf-8", errors="ignore") if half_bytes > 0 else ""
    tail_str = encoded[-half_bytes:].decode("utf-8", errors="ignore") if half_bytes > 0 else ""
    omitted_bytes = max(
        0,
        len(encoded) - (len(head_str.encode("utf-8")) + len(tail_str.encode("utf-8"))),
    )

    banner = (
        "\n\n... [OUTPUT TRUNCATED: "
        + (f"{omitted_lines} lines / " if omitted_lines > 0 else "")
        + f"{omitted_bytes} bytes omitted to preserve LLM context window] ...\n\n"
    )
    return head_str + banner + tail_str


truncate_output = apply_head_tail_truncation


class MimeConverter:
    """
    Translates raw Jupyter ``iopub`` messages into chronological MCP content blocks
    while enforcing ANSI sanitization, credential redaction, image limits, and
    head-tail text truncation.
    """

    def __init__(
        self,
        include_images: bool = True,
        max_text_lines: int = 1000,
        max_output_bytes: int = 102400,
        max_images: int = 5,
    ) -> None:
        self.include_images = include_images
        self.max_text_lines = max_text_lines
        self.max_output_bytes = max_output_bytes
        self.max_images = max_images

    def convert_messages(self, messages: Sequence[Dict[str, Any]]) -> List[MCPContentBlock]:
        """Convert a sequence of Jupyter ``iopub`` frames into chronological MCP blocks."""
        if not messages:
            return []

        blocks: List[MCPContentBlock] = []
        pending_text_parts: List[str] = []
        images_emitted = 0
        omitted_images = 0
        used_lines = 0
        used_bytes = 0

        def _append_text_segment(segment: str, ensure_newline_before: bool = False) -> None:
            if not segment:
                return
            if ensure_newline_before and pending_text_parts:
                last = pending_text_parts[-1]
                if last and not last.endswith("\n"):
                    pending_text_parts.append("\n")
            pending_text_parts.append(segment)

        def _flush_pending_text() -> None:
            nonlocal used_lines, used_bytes
            if not pending_text_parts:
                return
            combined = "".join(pending_text_parts)
            pending_text_parts.clear()
            if not combined:
                return

            remaining_lines = max(2, self.max_text_lines - used_lines)
            remaining_bytes = max(64, self.max_output_bytes - used_bytes)
            bounded = apply_head_tail_truncation(
                combined,
                max_lines=remaining_lines,
                max_bytes=remaining_bytes,
            )
            bounded = redact_credentials(bounded)
            used_lines += len(bounded.splitlines()) or 1
            used_bytes += len(bounded.encode("utf-8"))
            blocks.append(TextContent(type="text", text=bounded))

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            header = msg.get("header") if isinstance(msg.get("header"), dict) else {}
            msg_type = header.get("msg_type") or msg.get("msg_type")
            content = msg.get("content") if isinstance(msg.get("content"), dict) else {}

            if msg_type == "stream":
                stream_name = str(content.get("name", "stdout")).lower()
                raw_text = str(content.get("text", ""))
                cleaned = redact_credentials(sanitize_ansi_text(raw_text))
                if not cleaned:
                    continue
                if stream_name == "stderr":
                    _append_text_segment(f"[stderr] {cleaned}", ensure_newline_before=True)
                else:
                    _append_text_segment(cleaned, ensure_newline_before=False)

            elif msg_type in ("display_data", "execute_result"):
                data = content.get("data") if isinstance(content.get("data"), dict) else {}

                # Image MIME types take priority over plain/html fallback in multi-MIME frames
                selected_image_mime: str | None = None
                selected_image_b64: str | None = None
                for mime in ("image/png", "image/jpeg"):
                    val = data.get(mime)
                    if isinstance(val, str) and val.strip():
                        selected_image_mime = mime
                        selected_image_b64 = val.strip()
                        break

                if selected_image_mime is not None and selected_image_b64 is not None:
                    if self.include_images:
                        _flush_pending_text()
                        if images_emitted < self.max_images:
                            blocks.append(
                                ImageContent(
                                    type="image",
                                    data=selected_image_b64,
                                    mimeType=selected_image_mime,
                                )
                            )
                            images_emitted += 1
                        else:
                            omitted_images += 1
                    continue

                if "text/plain" in data and data["text/plain"] is not None:
                    plain = redact_credentials(sanitize_ansi_text(str(data["text/plain"])))
                    _append_text_segment(plain, ensure_newline_before=True)
                elif "text/html" in data and data["text/html"] is not None:
                    html_text = _html_to_readable_text(str(data["text/html"]))
                    cleaned_html = redact_credentials(sanitize_ansi_text(html_text))
                    _append_text_segment(cleaned_html, ensure_newline_before=True)

            elif msg_type == "error":
                tb_lines = content.get("traceback")
                if isinstance(tb_lines, list) and tb_lines:
                    tb_text = sanitize_ansi_traceback([str(line) for line in tb_lines])
                else:
                    ename = str(content.get("ename", "Error"))
                    evalue = str(content.get("evalue", ""))
                    tb_text = f"{ename}: {evalue}" if evalue else ename
                tb_text = redact_credentials(sanitize_ansi_text(tb_text))
                _append_text_segment(tb_text, ensure_newline_before=True)

        _flush_pending_text()

        if self.include_images and omitted_images > 0:
            blocks.append(
                TextContent(
                    type="text",
                    text=f"[Notice: {omitted_images} additional plot images omitted due to max_images limit]",
                )
            )

        return blocks

    convert = convert_messages
    format_outputs = convert_messages


def convert_iopub_to_mcp_blocks(
    messages: Sequence[Dict[str, Any]],
    *,
    include_images: bool = True,
    max_text_lines: int = 1000,
    max_output_bytes: int = 102400,
    max_images: int = 5,
) -> List[MCPContentBlock]:
    """Functional entrypoint translating ``iopub`` frames into chronological MCP blocks."""
    converter = MimeConverter(
        include_images=include_images,
        max_text_lines=max_text_lines,
        max_output_bytes=max_output_bytes,
        max_images=max_images,
    )
    return converter.convert_messages(messages)


convert_iopub_messages = convert_iopub_to_mcp_blocks
format_output_blocks = convert_iopub_to_mcp_blocks

__all__ = [
    "ANSI_ESCAPE_PATTERN",
    "MCPContentBlock",
    "MimeConverter",
    "apply_head_tail_truncation",
    "clean_ansi",
    "convert_iopub_messages",
    "convert_iopub_to_mcp_blocks",
    "format_output_blocks",
    "sanitize_ansi_text",
    "sanitize_ansi_traceback",
    "truncate_output",
]
