"""Token-aware randomized span-centered cropping for LLM SFT."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from llm_sft.config import SFTConfig


@dataclass(frozen=True)
class TokenizedSegment:
    text: str
    offsets: list[tuple[int, int]]
    n_tokens: int


@dataclass(frozen=True)
class CropResult:
    text: str
    char_start: int
    char_end: int
    kind: str  # full | positive | negative
    preset_index: int | None = None


@dataclass(frozen=True)
class SpanOutput:
    text: str
    start: int
    end: int


def tokenize_segment(tokenizer: Any, text: str) -> TokenizedSegment:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets: list[tuple[int, int]] = []
    for start, end in enc["offset_mapping"]:
        if end > start:
            offsets.append((start, end))
    return TokenizedSegment(text=text, offsets=offsets, n_tokens=len(offsets))


def _span_overlaps_token(cs: int, ce: int, span_start: int, span_end: int) -> bool:
    return ce > span_start and cs <= span_end


def normalize_annotation(text: str, ann: dict) -> dict | None:
    """Return annotation with segment-valid offsets, or None if unusable."""
    s, e = ann.get("span_start"), ann.get("span_end")
    if s is None or e is None:
        return None
    n = len(text)
    if 0 <= s <= e < n:
        return ann
    needle = (ann.get("text") or "").strip()
    if not needle:
        return None
    pos = text.find(needle)
    if pos < 0:
        return None
    fixed = dict(ann)
    fixed["span_start"] = pos
    fixed["span_end"] = pos + len(needle) - 1
    return fixed


def normalize_annotations(text: str, annotations: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ann in annotations:
        fixed = normalize_annotation(text, ann)
        if fixed is not None:
            out.append(fixed)
    return out


def char_span_to_token_range(
    offsets: list[tuple[int, int]],
    span_start: int,
    span_end: int,
) -> tuple[int, int] | None:
    """Inclusive char span -> inclusive token index range, or None if unmappable."""
    if not offsets:
        return None
    tok_lo: int | None = None
    tok_hi: int | None = None
    for i, (cs, ce) in enumerate(offsets):
        if not _span_overlaps_token(cs, ce, span_start, span_end):
            if cs > span_end:
                break
            continue
        if tok_lo is None:
            tok_lo = i
        tok_hi = i
    if tok_lo is not None and tok_hi is not None:
        return tok_lo, tok_hi
    # Span may fall in whitespace between tokens — attach nearest token.
    best_i = 0
    best_dist = float("inf")
    for i, (cs, ce) in enumerate(offsets):
        if span_start <= ce and span_end >= cs:
            return i, i
        mid = (cs + ce) / 2
        dist = min(abs(mid - span_start), abs(mid - span_end))
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i, best_i


def crop_from_token_window(
    text: str,
    offsets: list[tuple[int, int]],
    tok_start: int,
    tok_end_exclusive: int,
) -> tuple[str, int, int]:
    if tok_end_exclusive <= tok_start:
        raise ValueError("empty token window")
    char_start = offsets[tok_start][0]
    char_end = offsets[tok_end_exclusive - 1][1]
    return text[char_start:char_end], char_start, char_end


def remap_spans(
    crop_text: str,
    annotations: list[dict],
    task: str,
    crop_char_start: int,
    crop_char_end: int,
) -> list[SpanOutput]:
    """Map gold spans to crop-relative coordinates; drop spans outside crop."""
    out: list[SpanOutput] = []
    for ann in annotations:
        if ann.get("label") != task:
            continue
        s, e = ann["span_start"], ann["span_end"]
        if e < crop_char_start or s > crop_char_end:
            continue
        if s < crop_char_start or e > crop_char_end:
            continue
        rel_s = s - crop_char_start
        rel_e = e - crop_char_start
        snippet = crop_text[rel_s : rel_e + 1]
        out.append(SpanOutput(text=snippet, start=rel_s, end=rel_e))
    return out


def spans_to_json(spans: list[SpanOutput]) -> str:
    payload = {
        "spans": [
            {"text": s.text, "start": s.start, "end": s.end} for s in spans
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


def validate_spans(crop_text: str, spans: list[SpanOutput]) -> None:
    for span in spans:
        got = crop_text[span.start : span.end + 1]
        if got != span.text:
            raise ValueError(
                f"Span mismatch: expected {span.text!r}, got {got!r} "
                f"at [{span.start}, {span.end}]"
            )


def _valid_token_starts(
    n_tokens: int,
    max_context_tokens: int,
    span_tok_lo: int,
    span_tok_hi: int,
) -> list[int]:
    span_width = span_tok_hi - span_tok_lo + 1
    if span_width > max_context_tokens:
        center = (span_tok_lo + span_tok_hi) // 2
        half = max_context_tokens // 2
        start = max(0, min(center - half, n_tokens - max_context_tokens))
        return [start]
    lo = max(0, span_tok_hi - max_context_tokens + 1)
    hi = min(span_tok_lo, n_tokens - max_context_tokens)
    if lo > hi:
        return [lo]
    return list(range(lo, hi + 1))


def _pick_token_start_from_preset(
    rng: random.Random,
    offsets: list[tuple[int, int]],
    span_start: int,
    span_end: int,
    span_tok_lo: int,
    span_tok_hi: int,
    n_tokens: int,
    max_context_tokens: int,
    before_hint: int,
    after_hint: int,
    random_slack: bool,
) -> tuple[int, int]:
    """Return (token_start, preset_index). preset_index is index or -1 for slack."""
    valid = _valid_token_starts(
        n_tokens, max_context_tokens, span_tok_lo, span_tok_hi
    )
    if not random_slack and before_hint == 0 and after_hint == 0:
        return rng.choice(valid), -1

    desired_char_start = max(0, span_start - before_hint)
    desired_char_end = min(
        len(offsets) and offsets[-1][1] or 0,
        span_end + after_hint + 1,
    )

    text_len = _char_end_from_offsets(offsets)
    best_start = valid[0]
    best_dist = float("inf")
    for tok_start in valid:
        tok_end = min(tok_start + max_context_tokens, n_tokens)
        cs = offsets[tok_start][0]
        ce = offsets[tok_end - 1][1]
        desired_char_end = min(text_len, span_end + after_hint + 1)
        dist = abs(cs - desired_char_start) + abs(ce - desired_char_end)
        if dist < best_dist:
            best_dist = dist
            best_start = tok_start

    if random_slack and len(valid) > 1:
        if rng.random() < 0.5:
            return rng.choice(valid), -1
    return best_start, -1


def _char_end_from_offsets(offsets: list[tuple[int, int]]) -> int:
    return offsets[-1][1] if offsets else 0


def generate_crops_for_task(
    tokenized: TokenizedSegment,
    annotations: list[dict],
    task: str,
    config: SFTConfig,
    rng: random.Random,
) -> list[CropResult]:
    text = tokenized.text
    offsets = tokenized.offsets
    n_tokens = tokenized.n_tokens
    if not text.strip() or n_tokens == 0:
        return []

    annotations = normalize_annotations(text, annotations)
    gold = [a for a in annotations if a.get("label") == task]

    if n_tokens <= config.max_context_tokens:
        return [
            CropResult(
                text=text,
                char_start=0,
                char_end=len(text),
                kind="full",
            )
        ]

    if gold:
        crops: list[CropResult] = []
        span_start = min(a["span_start"] for a in gold)
        span_end = max(a["span_end"] for a in gold)
        tok_ranges = []
        for a in gold:
            tr = char_span_to_token_range(offsets, a["span_start"], a["span_end"])
            if tr is None:
                continue
            tok_ranges.append(tr)
        if not tok_ranges:
            return []
        span_tok_lo = min(lo for lo, _ in tok_ranges)
        span_tok_hi = max(hi for _, hi in tok_ranges)
        presets = list(config.crop_presets)
        for i in range(config.crops_per_positive):
            preset_idx = i % len(presets)
            before_hint, after_hint = presets[preset_idx]
            tok_start, _ = _pick_token_start_from_preset(
                rng,
                offsets,
                span_start,
                span_end,
                span_tok_lo,
                span_tok_hi,
                n_tokens,
                config.max_context_tokens,
                before_hint,
                after_hint,
                config.random_slack,
            )
            tok_end = min(tok_start + config.max_context_tokens, n_tokens)
            crop_text, cs, ce = crop_from_token_window(
                text, offsets, tok_start, tok_end
            )
            crops.append(
                CropResult(
                    text=crop_text,
                    char_start=cs,
                    char_end=ce,
                    kind="positive",
                    preset_index=preset_idx,
                )
            )
        return crops

    crops = []
    max_start = max(0, n_tokens - config.max_context_tokens)
    for _ in range(config.crops_per_negative):
        tok_start = rng.randint(0, max_start) if max_start > 0 else 0
        tok_end = min(tok_start + config.max_context_tokens, n_tokens)
        crop_text, cs, ce = crop_from_token_window(
            text, offsets, tok_start, tok_end
        )
        crops.append(
            CropResult(
                text=crop_text,
                char_start=cs,
                char_end=ce,
                kind="negative",
            )
        )
    return crops


def span_position_ratio(spans: list[SpanOutput], crop_len: int) -> float | None:
    """Relative center of first span in crop (0=start, 1=end)."""
    if not spans or crop_len <= 0:
        return None
    span = spans[0]
    center = (span.start + span.end) / 2.0
    return center / crop_len


def build_example_row(
    *,
    doc_id: str,
    segment_id: str,
    task: str,
    instruction: str,
    crop: CropResult,
    spans: list[SpanOutput],
    crop_index: int,
) -> dict:
    return {
        "id": f"{doc_id}:{segment_id}:{task}:{crop_index}",
        "doc_id": doc_id,
        "segment_id": segment_id,
        "task": task,
        "instruction": instruction,
        "input": crop.text,
        "output": spans_to_json(spans),
        "crop_kind": crop.kind,
        "crop_char_start": crop.char_start,
        "crop_char_end": crop.char_end,
        "preset_index": crop.preset_index,
        "span_position_ratio": span_position_ratio(spans, len(crop.text)),
    }
