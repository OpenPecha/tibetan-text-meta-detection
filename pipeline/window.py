"""Windowed text extraction for begin/end regions."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.tokenize import SyllableToken, tokenize_tibetan


@dataclass(frozen=True)
class TextWindow:
    name: str  # "full", "begin", or "end"
    text: str
    tokens: list[SyllableToken]
    doc_char_start: int
    doc_char_end: int


def build_windows(
    content: str,
    window_size: int,
) -> list[TextWindow]:
    """Build begin/end syllable windows, or full text when short.

    When len(syllables) <= 2 * window_size, returns a single full window.
    Otherwise returns separate begin and end windows.
    """
    tokens = tokenize_tibetan(content)
    if not tokens:
        return [
            TextWindow(
                name="full",
                text="",
                tokens=[],
                doc_char_start=0,
                doc_char_end=0,
            )
        ]

    if len(tokens) <= 2 * window_size:
        return [
            TextWindow(
                name="full",
                text=content,
                tokens=tokens,
                doc_char_start=tokens[0].start,
                doc_char_end=tokens[-1].end,
            )
        ]

    begin_tokens = tokens[:window_size]
    end_tokens = tokens[-window_size:]
    windows = [
        TextWindow(
            name="begin",
            text="".join(t.text for t in begin_tokens),
            tokens=begin_tokens,
            doc_char_start=begin_tokens[0].start,
            doc_char_end=begin_tokens[-1].end,
        ),
        TextWindow(
            name="end",
            text="".join(t.text for t in end_tokens),
            tokens=end_tokens,
            doc_char_start=end_tokens[0].start,
            doc_char_end=end_tokens[-1].end,
        ),
    ]
    return windows


def filter_annotations_for_window(
    annotations: list[dict],
    window: TextWindow,
) -> list[dict]:
    """Keep annotations fully contained in the window's document char range."""
    filtered: list[dict] = []
    for ann in annotations:
        start = ann["span_start"]
        end = ann["span_end"]
        if start >= window.doc_char_start and end <= window.doc_char_end:
            filtered.append(
                {
                    **ann,
                    "span_start": start - window.doc_char_start,
                    "span_end": end - window.doc_char_start,
                }
            )
    return filtered


def annotation_fully_captured(
    annotation: dict,
    windows: list[TextWindow],
) -> bool:
    """Return True if annotation lies entirely inside any window."""
    start = annotation["span_start"]
    end = annotation["span_end"]
    for window in windows:
        if start >= window.doc_char_start and end <= window.doc_char_end:
            return True
    return False
