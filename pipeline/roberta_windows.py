"""RoBERTa subword sliding-window pipeline for title/author NER."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from config import (
    ROBERTA_MAX_BEGIN_SLIDES,
    ROBERTA_MAX_END_SLIDES,
    ROBERTA_MEDIUM_TOKEN_THRESHOLD,
    ROBERTA_STRIDE,
    ROBERTA_WINDOW_SIZE,
)
from pipeline.bio import (
    BIO_LABELS,
    IGNORE_LABEL_ID,
    LABEL_TO_ID,
    subword_annotations_to_bio,
    subword_bio_to_spans,
)


@dataclass(frozen=True)
class SubwordResult:
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    n_tokens: int


@dataclass(frozen=True)
class WindowSpec:
    start_tok: int
    end_tok: int  # exclusive
    side: str  # "full", "begin", or "end"
    slide_index: int

    @property
    def name(self) -> str:
        if self.side == "full":
            return "full"
        return f"{self.side}_{self.slide_index:02d}"


@dataclass
class WindowPrediction:
    window_name: str
    side: str
    slide_index: int
    segment_char_start: int
    segment_char_end: int
    offsets: list[tuple[int, int]]
    label_ids: list[int]
    confidences: list[float] | None = None


def tokenize_segment(tokenizer: Any, text: str) -> SubwordResult:
    """Tokenize segment text and return subword ids with char offsets."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(s, e) for (s, e) in enc["offset_mapping"] if e > s]
    input_ids = enc["input_ids"][: len(offsets)]
    return SubwordResult(
        input_ids=input_ids,
        offsets=offsets,
        n_tokens=len(offsets),
    )


def _begin_slide_windows(
    n_tokens: int,
    window_size: int,
    n_slides: int,
    stride: int,
) -> list[tuple[int, int]]:
    if n_tokens <= window_size:
        return [(0, n_tokens)]
    out: list[tuple[int, int]] = []
    for i in range(n_slides):
        start = i * stride
        if start >= n_tokens:
            break
        end = min(start + window_size, n_tokens)
        if end - start < 1:
            break
        out.append((start, end))
    return out


def _end_slide_windows(
    n_tokens: int,
    window_size: int,
    n_slides: int,
    stride: int,
) -> list[tuple[int, int]]:
    if n_tokens <= window_size:
        return [(0, n_tokens)]
    out: list[tuple[int, int]] = []
    for i in range(n_slides):
        end = n_tokens - i * stride
        if end <= 0:
            break
        start = max(0, end - window_size)
        if end - start < 1:
            break
        out.append((start, end))
    return out


def _is_subset(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
    return inner[0] >= outer[0] and inner[1] <= outer[1]


def _deduplicate_windows(
    windows: list[tuple[int, int, str, int]],
) -> list[WindowSpec]:
    """Remove windows whose token range is fully covered by another window."""
    if not windows:
        return []

    ranges = [(w[0], w[1]) for w in windows]
    keep = [True] * len(windows)
    for i, inner in enumerate(ranges):
        for j, outer in enumerate(ranges):
            if i == j:
                continue
            if _is_subset(inner, outer):
                keep[i] = False
                break

    out: list[WindowSpec] = []
    for kept, (start, end, side, slide_index) in zip(keep, windows):
        if kept:
            out.append(WindowSpec(start, end, side, slide_index))
    return out


def segment_tier(n_tokens: int, window_size: int = ROBERTA_WINDOW_SIZE) -> str:
    if n_tokens <= window_size:
        return "short"
    if n_tokens <= ROBERTA_MEDIUM_TOKEN_THRESHOLD:
        return "medium"
    return "long"


def compute_windows(
    n_tokens: int,
    window_size: int = ROBERTA_WINDOW_SIZE,
    stride: int = ROBERTA_STRIDE,
    max_begin: int = ROBERTA_MAX_BEGIN_SLIDES,
    max_end: int = ROBERTA_MAX_END_SLIDES,
) -> list[WindowSpec]:
    """Generate overlap-aware begin/end sliding windows for a segment."""
    if n_tokens <= window_size:
        return [WindowSpec(0, n_tokens, "full", 0)]

    raw: list[tuple[int, int, str, int]] = []
    for i, (start, end) in enumerate(
        _begin_slide_windows(n_tokens, window_size, max_begin, stride)
    ):
        raw.append((start, end, "begin", i))
    for i, (start, end) in enumerate(
        _end_slide_windows(n_tokens, window_size, max_end, stride)
    ):
        raw.append((start, end, "end", i))

    return _deduplicate_windows(raw)


def _window_char_bounds(
    offsets: list[tuple[int, int]],
    window: WindowSpec,
) -> tuple[int, int]:
    if window.start_tok >= window.end_tok:
        return 0, 0
    return offsets[window.start_tok][0], offsets[window.end_tok - 1][1]


def filter_annotations_for_window(
    annotations: list[dict],
    window_char_start: int,
    window_char_end: int,
) -> list[dict]:
    """Keep annotations fully contained in the window char range."""
    filtered: list[dict] = []
    for ann in annotations:
        start = ann["span_start"]
        end = ann["span_end"]
        if start >= window_char_start and end <= window_char_end:
            filtered.append(
                {
                    **ann,
                    "span_start": start - window_char_start,
                    "span_end": end - window_char_start,
                }
            )
    return filtered


def label_window(
    offsets: list[tuple[int, int]],
    window: WindowSpec,
    annotations: list[dict],
) -> tuple[list[str], list[dict]]:
    """Assign BIO tags to subword tokens in a window."""
    window_offsets = offsets[window.start_tok : window.end_tok]
    win_char_start, win_char_end = _window_char_bounds(offsets, window)
    window_anns = filter_annotations_for_window(
        annotations,
        win_char_start,
        win_char_end,
    )
    rel_offsets = [
        (max(0, s - win_char_start), max(0, e - win_char_start))
        for s, e in window_offsets
    ]
    tags = subword_annotations_to_bio(rel_offsets, window_anns)
    return tags, window_anns


def _max_content_tokens(max_length: int) -> int:
    """Content tokens allowed before adding CLS/SEP."""
    return max_length - 2


def build_training_example(
    tokenizer: Any,
    window: WindowSpec,
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    bio_tags: list[str],
    metadata: dict,
    max_length: int = ROBERTA_WINDOW_SIZE,
) -> dict:
    """Build a HuggingFace-ready training example with CLS/SEP and padding."""
    content_limit = _max_content_tokens(max_length)
    win_input_ids = input_ids[window.start_tok : window.end_tok][:content_limit]
    win_offsets = offsets[window.start_tok : window.end_tok][:content_limit]
    win_tags = bio_tags[:content_limit]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    seq_ids = [cls_id] + win_input_ids + [sep_id]
    seq_offsets = [(0, 0)] + win_offsets + [(0, 0)]
    seq_labels = [IGNORE_LABEL_ID]
    seq_labels += [LABEL_TO_ID.get(tag, 0) for tag in win_tags]
    seq_labels.append(IGNORE_LABEL_ID)

    attention_mask = [1] * len(seq_ids)
    while len(seq_ids) < max_length:
        seq_ids.append(pad_id)
        seq_offsets.append((0, 0))
        seq_labels.append(IGNORE_LABEL_ID)
        attention_mask.append(0)

    return {
        **metadata,
        "window_name": window.name,
        "window_side": window.side,
        "slide_index": window.slide_index,
        "input_ids": seq_ids,
        "attention_mask": attention_mask,
        "labels": seq_labels,
        "offset_mapping": seq_offsets,
        "label_list": BIO_LABELS,
    }


def slide_segment(
    tokenizer: Any,
    text: str,
    annotations: list[dict],
    metadata: dict | None = None,
    window_size: int = ROBERTA_WINDOW_SIZE,
    stride: int = ROBERTA_STRIDE,
    max_begin: int = ROBERTA_MAX_BEGIN_SLIDES,
    max_end: int = ROBERTA_MAX_END_SLIDES,
) -> list[dict]:
    """Tokenize a segment, slide windows, label, and build training examples."""
    if not text.strip():
        return []

    meta = metadata or {}
    subword = tokenize_segment(tokenizer, text)
    if subword.n_tokens == 0:
        return []

    windows = compute_windows(
        subword.n_tokens,
        window_size=window_size,
        stride=stride,
        max_begin=max_begin,
        max_end=max_end,
    )

    has_title = any(a["label"] == "title" for a in annotations)
    has_author = any(a["label"] == "author" for a in annotations)
    tier = segment_tier(subword.n_tokens, window_size)

    examples: list[dict] = []
    for window in windows:
        bio_tags, window_anns = label_window(subword.offsets, window, annotations)
        example = build_training_example(
            tokenizer,
            window,
            subword.input_ids,
            subword.offsets,
            bio_tags,
            {
                **meta,
                "char_length": len(text),
                "token_length": subword.n_tokens,
                "segment_tier": tier,
                "has_title": has_title or any(a["label"] == "title" for a in window_anns),
                "has_author": has_author or any(a["label"] == "author" for a in window_anns),
                "window_annotations": window_anns,
            },
        )
        examples.append(example)
    return examples


def merge_predictions(
    predictions: list[WindowPrediction],
    id_to_label: dict[int, str] | None = None,
) -> list[dict]:
    """Merge overlapping window predictions back to segment-level spans."""
    if id_to_label is None:
        from pipeline.bio import ID_TO_LABEL

        id_to_label = ID_TO_LABEL

    token_votes: dict[int, list[tuple[str, float]]] = defaultdict(list)

    for pred in predictions:
        for local_idx, label_id in enumerate(pred.label_ids):
            if label_id < 0:
                continue
            if local_idx >= len(pred.offsets):
                continue
            char_start, char_end = pred.offsets[local_idx]
            if char_end <= char_start:
                continue

            seg_char_start = pred.segment_char_start + char_start
            confidence = 1.0
            if pred.confidences and local_idx < len(pred.confidences):
                confidence = pred.confidences[local_idx]

            token_votes[seg_char_start].append(
                (id_to_label.get(label_id, "O"), confidence)
            )

    if not token_votes:
        return []

    # Pick highest-confidence label per char position
    position_label: dict[int, str] = {}
    for pos, votes in token_votes.items():
        best = max(votes, key=lambda x: x[1])
        position_label[pos] = best[0]

    sorted_positions = sorted(position_label)
    spans: list[dict] = []
    idx = 0
    while idx < len(sorted_positions):
        pos = sorted_positions[idx]
        label = position_label[pos]
        if label == "O" or "-" not in label:
            idx += 1
            continue

        prefix, entity = label.split("-", 1)
        if prefix != "B":
            idx += 1
            continue

        start = pos
        end = pos
        idx += 1
        while idx < len(sorted_positions):
            next_pos = sorted_positions[idx]
            next_label = position_label[next_pos]
            if next_label == f"I-{entity}":
                end = next_pos
                idx += 1
            else:
                break

        spans.append(
            {
                "label": entity.lower(),
                "span_start": start,
                "span_end": end,
                "confidence": 1.0,
            }
        )

    return spans


def validate_window_roundtrip(
    offsets: list[tuple[int, int]],
    window: WindowSpec,
    annotations: list[dict],
    bio_tags: list[str],
) -> tuple[bool, list[str]]:
    """Verify BIO tags reconstruct token-aligned spans for window annotations."""
    window_offsets = offsets[window.start_tok : window.end_tok]
    win_char_start, win_char_end = _window_char_bounds(offsets, window)
    window_anns = filter_annotations_for_window(
        annotations,
        win_char_start,
        win_char_end,
    )
    rel_offsets = [
        (max(0, s - win_char_start), max(0, e - win_char_start))
        for s, e in window_offsets
    ]

    reconstructed = subword_bio_to_spans(rel_offsets, bio_tags)
    expected: list[dict] = []
    for ann in window_anns:
        overlapping = [
            (s, e)
            for s, e in rel_offsets
            if s < ann["span_end"] and e > ann["span_start"]
        ]
        if not overlapping:
            continue
        expected.append(
            {
                "label": ann["label"],
                "span_start": overlapping[0][0],
                "span_end": overlapping[-1][1],
            }
        )

    expected = sorted(
        expected,
        key=lambda x: (x["label"], x["span_start"], x["span_end"]),
    )
    actual = sorted(
        reconstructed,
        key=lambda x: (x["label"], x["span_start"], x["span_end"]),
    )
    errors: list[str] = []
    if expected != actual:
        errors.append(
            f"window {window.name}: expected {len(expected)} spans, "
            f"got {len(actual)}"
        )
    return len(errors) == 0, errors


def validate_segment_roundtrip(
    tokenizer: Any,
    text: str,
    annotations: list[dict],
    window_size: int = ROBERTA_WINDOW_SIZE,
    stride: int = ROBERTA_STRIDE,
    max_begin: int = ROBERTA_MAX_BEGIN_SLIDES,
    max_end: int = ROBERTA_MAX_END_SLIDES,
) -> tuple[int, list[str]]:
    """Validate BIO roundtrip for all windows in a segment. Returns failure count."""
    subword = tokenize_segment(tokenizer, text)
    if subword.n_tokens == 0:
        return 0, []

    windows = compute_windows(
        subword.n_tokens,
        window_size=window_size,
        stride=stride,
        max_begin=max_begin,
        max_end=max_end,
    )
    failures = 0
    errors: list[str] = []
    for window in windows:
        bio_tags, _ = label_window(subword.offsets, window, annotations)
        ok, window_errors = validate_window_roundtrip(
            subword.offsets,
            window,
            annotations,
            bio_tags,
        )
        if not ok:
            failures += 1
            errors.extend(window_errors)
    return failures, errors


@dataclass
class WindowStatsAccumulator:
    """Incremental stats while streaming RoBERTa training examples."""

    total_examples: int = 0
    tier_example_counts: Counter[str] = None  # type: ignore[assignment]
    windows_per_segment: Counter[str] = None  # type: ignore[assignment]
    segment_tiers: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tier_example_counts is None:
            self.tier_example_counts = Counter()
        if self.windows_per_segment is None:
            self.windows_per_segment = Counter()
        if self.segment_tiers is None:
            self.segment_tiers = {}

    def add(self, example: dict) -> None:
        tier = example.get("segment_tier", "unknown")
        self.tier_example_counts[tier] += 1
        self.total_examples += 1
        seg_key = f"{example.get('doc_id')}:{example.get('segment_id')}"
        self.windows_per_segment[seg_key] += 1
        if seg_key not in self.segment_tiers:
            self.segment_tiers[seg_key] = tier

    def to_dict(self) -> dict:
        tier_window_dist: dict[str, list[int]] = defaultdict(list)
        for seg_key, count in self.windows_per_segment.items():
            tier_window_dist[self.segment_tiers[seg_key]].append(count)

        tier_summary = {}
        for tier, counts in tier_window_dist.items():
            tier_summary[tier] = {
                "segments": len(counts),
                "min_windows": min(counts) if counts else 0,
                "max_windows": max(counts) if counts else 0,
                "mean_windows": sum(counts) / len(counts) if counts else 0,
            }

        return {
            "total_examples": self.total_examples,
            "tier_example_counts": dict(self.tier_example_counts),
            "tier_window_stats": tier_summary,
            "unique_segments": len(self.windows_per_segment),
        }


def summarize_window_stats(examples: list[dict]) -> dict:
    """Summarize tier and window counts from processed examples."""
    acc = WindowStatsAccumulator()
    for ex in examples:
        acc.add(ex)
    return acc.to_dict()
