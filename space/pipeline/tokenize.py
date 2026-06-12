"""Tokenize Tibetan text at tsheg and shad boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Tsheg (syllable separator) and shad (phrase/sentence marker)
_DELIMITERS = "\u0f0b\u0f0d"
_TOKEN_PATTERN = re.compile(rf"[^{_DELIMITERS}]+[{_DELIMITERS}]?")


@dataclass(frozen=True)
class SyllableToken:
    text: str
    start: int
    end: int


def tokenize_tibetan(text: str) -> list[SyllableToken]:
    """Split text into syllable-level tokens with document character offsets."""
    if not text:
        return []

    tokens: list[SyllableToken] = []
    for match in _TOKEN_PATTERN.finditer(text):
        piece = match.group()
        if not piece.strip():
            continue
        tokens.append(SyllableToken(text=piece, start=match.start(), end=match.end()))
    return tokens
