"""Measure begin/end window capture rate for title/author spans.

For each annotated segment we tokenize the raw segment text with the target
model's subword tokenizer, then ask: if we keep only the first K + last K
tokens of the segment (begin/end windows), is each title/author span fully
captured? Segments with <= 2*K tokens are kept whole (always captured).

Also supports sliding windows of fixed size (default 512) from the begin
and/or end of each segment to find how many slides are needed for ~95%
capture.

Usage:
    python analyze_capture.py --num-docs 10
    python analyze_capture.py --num-docs 10 --slide-analysis
    python analyze_capture.py --num-docs 10 --model spsither/tibetan_RoBERTa_S_e3
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

from config import EXTRACTED_DIR

DEFAULT_MODEL = "spsither/tibetan_RoBERTa_S_e3"
K_VALUES = [64, 128, 256, 512]
MODEL_MAX = 512  # RoBERTa hard cap (max_position_embeddings - 2)
DEFAULT_WINDOW_SIZE = 512
DEFAULT_STRIDE = 256
MAX_SLIDES = 40
TARGET_CAPTURE = 0.95


@dataclass(frozen=True)
class SpanRecord:
    label: str
    lo: int  # first token index (inclusive)
    hi: int  # last token index (inclusive)
    n_tokens: int


def content_tokens(tokenizer, text: str) -> list[tuple[int, int]]:
    """Return (char_start, char_end) for each non-special subword token."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(s, e) for (s, e) in enc["offset_mapping"] if e > s]


def span_token_bounds(
    offsets: list[tuple[int, int]],
    span_start: int,
    span_end: int,
) -> tuple[int, int] | None:
    """Return (first_token_idx, last_token_idx) overlapping the char span."""
    lo = hi = None
    for idx, (s, e) in enumerate(offsets):
        if s < span_end and e > span_start:
            if lo is None:
                lo = idx
            hi = idx
    if lo is None:
        return None
    return lo, hi


def captured(n_tokens: int, lo: int, hi: int, k: int) -> bool:
    """True if span [lo, hi] survives a begin-K + end-K window."""
    if n_tokens <= 2 * k:
        return True  # whole segment kept
    in_begin = hi < k
    in_end = lo >= n_tokens - k
    return in_begin or in_end


def begin_slide_windows(
    n_tokens: int, window_size: int, n_slides: int, stride: int
) -> list[tuple[int, int]]:
    """Token-index windows sliding forward from the segment start."""
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


def end_slide_windows(
    n_tokens: int, window_size: int, n_slides: int, stride: int
) -> list[tuple[int, int]]:
    """Token-index windows sliding backward from the segment end."""
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


def span_in_window(lo: int, hi: int, ws: int, we: int) -> bool:
    """Span fully contained in token window [ws, we) — hi is inclusive."""
    return lo >= ws and hi < we


def capture_rate(spans: list[SpanRecord], windows: list[tuple[int, int]]) -> float:
    if not spans:
        return 1.0
    hit = sum(
        1
        for s in spans
        if any(span_in_window(s.lo, s.hi, ws, we) for ws, we in windows)
    )
    return hit / len(spans)


def capture_by_label(
    spans: list[SpanRecord], windows: list[tuple[int, int]]
) -> dict[str, float]:
    by_label: dict[str, list[SpanRecord]] = defaultdict(list)
    for s in spans:
        by_label[s.label].append(s)
    return {label: capture_rate(items, windows) for label, items in by_label.items()}


def min_slides_for_target(
    spans: list[SpanRecord],
    n_tokens_fn,
    window_size: int,
    stride: int,
    target: float,
    max_slides: int,
) -> int | None:
    """Return smallest n_slides where capture >= target, or None if not reached."""
    if not spans:
        return 0
    # Group spans by segment length for efficiency — spans carry n_tokens
    for n in range(1, max_slides + 1):
        # Build windows per unique segment length is wrong — each span has its own n_tokens
        hit = 0
        for s in spans:
            wins = n_tokens_fn(s.n_tokens, window_size, n, stride)
            if any(span_in_window(s.lo, s.hi, ws, we) for ws, we in wins):
                hit += 1
        if hit / len(spans) >= target:
            return n
    return None


def run_slide_analysis(
    spans: list[SpanRecord],
    window_size: int,
    stride: int,
    target: float,
    max_slides: int,
) -> None:
    print()
    print(
        f"=== Sliding window analysis (W={window_size}, stride={stride}, "
        f"target={target:.0%}) ==="
    )
    print(f"Spans analyzed: {len(spans)}")
    print()

    min_begin = min_slides_for_target(
        spans, begin_slide_windows, window_size, stride, target, max_slides
    )
    min_end = min_slides_for_target(
        spans, end_slide_windows, window_size, stride, target, max_slides
    )

    print("Minimum slides for target capture (single side):")
    if min_begin is not None:
        hit = sum(
            1
            for s in spans
            if capture_rate([s], begin_slide_windows(s.n_tokens, window_size, min_begin, stride)) == 1.0
        )
        print(
            f"  Begin-only: {min_begin} slide(s) -> {100*hit/len(spans):.1f}% overall"
        )
    else:
        print(f"  Begin-only: not reached within {max_slides} slides")

    if min_end is not None:
        hit = sum(
            1
            for s in spans
            if capture_rate([s], end_slide_windows(s.n_tokens, window_size, min_end, stride)) == 1.0
        )
        print(
            f"  End-only:   {min_end} slide(s) -> {100*hit/len(spans):.1f}% overall"
        )
    else:
        print(f"  End-only:   not reached within {max_slides} slides")

    # Combined: n begin + m end slides (union of windows)
    print()
    print("Combined begin + end slides (union of all windows):")
    best_pair: tuple[int, int] | None = None
    best_rate = 0.0
    for nb in range(1, max_slides + 1):
        for ne in range(1, max_slides + 1):
            hit = 0
            for s in spans:
                wins = begin_slide_windows(s.n_tokens, window_size, nb, stride)
                wins += end_slide_windows(s.n_tokens, window_size, ne, stride)
                if any(span_in_window(s.lo, s.hi, ws, we) for ws, we in wins):
                    hit += 1
            rate = hit / len(spans) if spans else 1.0
            if rate >= target:
                if best_pair is None or (nb + ne, nb, ne) < (best_pair[0] + best_pair[1], best_pair[0], best_pair[1]):
                    best_pair = (nb, ne)
                    best_rate = rate

    if best_pair:
        nb, ne = best_pair
        print(
            f"  Minimum for >={target:.0%}: {nb} begin + {ne} end = "
            f"{nb + ne} total forward passes -> {100*best_rate:.1f}% capture"
        )
    else:
        print(f"  >={target:.0%} not reached with up to {max_slides} slides per side")

    # Capture curve table
    print()
    print("Capture rate vs number of slides (per side, used alone):")
    header = "slides".ljust(8) + "begin-all".rjust(12) + "end-all".rjust(12)
    for label in sorted({s.label for s in spans}):
        header += f"b-{label[:5]}".rjust(10) + f"e-{label[:5]}".rjust(10)
    print(header)
    print("-" * len(header))
    slide_counts = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
    for n in slide_counts:
        if n > max_slides:
            continue
        begin_hit = sum(
            1
            for s in spans
            if capture_rate(
                [s], begin_slide_windows(s.n_tokens, window_size, n, stride)
            )
            == 1.0
        )
        end_hit = sum(
            1
            for s in spans
            if capture_rate(
                [s], end_slide_windows(s.n_tokens, window_size, n, stride)
            )
            == 1.0
        )
        row = f"{n}".ljust(8)
        row += f"{100*begin_hit/len(spans):>10.1f}%".rjust(12)
        row += f"{100*end_hit/len(spans):>10.1f}%".rjust(12)
        for label in sorted({s.label for s in spans}):
            sub = [s for s in spans if s.label == label]
            bh = sum(
                1
                for s in sub
                if capture_rate(
                    [s], begin_slide_windows(s.n_tokens, window_size, n, stride)
                )
                == 1.0
            )
            eh = sum(
                1
                for s in sub
                if capture_rate(
                    [s], end_slide_windows(s.n_tokens, window_size, n, stride)
                )
                == 1.0
            )
            row += f"{100*bh/len(sub):>8.1f}%".rjust(10)
            row += f"{100*eh/len(sub):>8.1f}%".rjust(10)
        print(row)

    # Combined curve: equal slides on both sides
    print()
    print("Combined (N begin + N end slides, same N):")
    print(
        "N".ljust(6)
        + "overall".rjust(10)
        + "author".rjust(10)
        + "title".rjust(10)
        + "total_passes".rjust(14)
    )
    for n in slide_counts:
        if n > max_slides:
            continue
        hit_by_label: Counter[str] = Counter()
        tot_by_label: Counter[str] = Counter()
        for s in spans:
            tot_by_label[s.label] += 1
            wins = begin_slide_windows(s.n_tokens, window_size, n, stride)
            wins += end_slide_windows(s.n_tokens, window_size, n, stride)
            if any(span_in_window(s.lo, s.hi, ws, we) for ws, we in wins):
                hit_by_label[s.label] += 1
        overall = sum(hit_by_label.values()) / len(spans) if spans else 1.0
        row = f"{n}".ljust(6)
        row += f"{100*overall:>8.1f}%".rjust(10)
        for label in sorted(tot_by_label):
            t = tot_by_label[label]
            row += f"{100*hit_by_label[label]/t:>8.1f}%".rjust(10)
        row += f"{2*n:>12}".rjust(14)
        print(row)
        if overall >= target:
            print(f"  ^ first N where combined capture >= {target:.0%}")
            break

    print()
    print(
        "Note: each slide is one RoBERTa-sized window (512 tokens). "
        "Begin slides move forward from start; end slides move backward from end."
    )


def content_tokens(tokenizer, text: str) -> list[tuple[int, int]]:
    """Return (char_start, char_end) for each non-special subword token."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(s, e) for (s, e) in enc["offset_mapping"] if e > s]


def span_token_bounds(
    offsets: list[tuple[int, int]],
    span_start: int,
    span_end: int,
) -> tuple[int, int] | None:
    """Return (first_token_idx, last_token_idx) overlapping the char span."""
    lo = hi = None
    for idx, (s, e) in enumerate(offsets):
        if s < span_end and e > span_start:
            if lo is None:
                lo = idx
            hi = idx
    if lo is None:
        return None
    return lo, hi


def captured(n_tokens: int, lo: int, hi: int, k: int) -> bool:
    """True if span [lo, hi] survives a begin-K + end-K window."""
    if n_tokens <= 2 * k:
        return True  # whole segment kept
    in_begin = hi < k
    in_end = lo >= n_tokens - k
    return in_begin or in_end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-docs", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--extracted-dir", type=Path, default=EXTRACTED_DIR)
    parser.add_argument(
        "--slide-analysis",
        action="store_true",
        help="Report sliding begin/end window capture (512-token windows)",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--target", type=float, default=TARGET_CAPTURE)
    parser.add_argument("--max-slides", type=int, default=MAX_SLIDES)
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

    index_path = args.extracted_dir / "index.jsonl"
    entries: list[dict] = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
            if len(entries) >= args.num_docs:
                break

    seg_token_lens: list[int] = []
    total = Counter()              # spans per label
    cap = defaultdict(Counter)     # cap[k][label] = captured count
    missing_tokens = Counter()     # spans with no tokens (alignment failure)
    span_records: list[SpanRecord] = []

    for entry in entries:
        ann_path = args.extracted_dir / entry["annotations_path"]
        with ann_path.open(encoding="utf-8") as af:
            payload = json.load(af)

        for seg in payload.get("segments", []):
            text = seg["text"]
            offsets = content_tokens(tokenizer, text)
            n = len(offsets)
            seg_token_lens.append(n)

            for ann in seg.get("annotations", []):
                label = ann["label"]
                total[label] += 1
                bounds = span_token_bounds(offsets, ann["span_start"], ann["span_end"])
                if bounds is None:
                    missing_tokens[label] += 1
                    continue
                lo, hi = bounds
                span_records.append(SpanRecord(label=label, lo=lo, hi=hi, n_tokens=n))
                for k in K_VALUES:
                    if captured(n, lo, hi, k):
                        cap[k][label] += 1

    print()
    print(f"=== Capture-rate report ({len(entries)} docs, model={args.model}) ===")
    print(f"Segments analyzed: {len(seg_token_lens)}")
    if seg_token_lens:
        s = sorted(seg_token_lens)
        p = lambda q: s[min(len(s) - 1, int(len(s) * q))]
        print(
            "Segment token length: min=%d median=%d mean=%.0f p90=%d p99=%d max=%d"
            % (s[0], statistics.median(s), statistics.mean(s), p(0.9), p(0.99), s[-1])
        )
        over = sum(1 for n in s if n > MODEL_MAX)
        print(
            f"Segments > {MODEL_MAX} tokens (need windowing): "
            f"{over}/{len(s)} ({100*over/len(s):.1f}%)"
        )
    print()
    labels = sorted(total)
    print("Total spans:", dict(total))
    if any(missing_tokens.values()):
        print("Spans with no token alignment:", dict(missing_tokens))
    print()
    header = "K/side".ljust(8) + "".join(f"{l:>14}" for l in labels) + f"{'overall':>14}"
    print(header)
    print("-" * len(header))
    for k in K_VALUES:
        row = f"{k}".ljust(8)
        tot_all = cap_all = 0
        for l in labels:
            c, t = cap[k][l], total[l]
            tot_all += t
            cap_all += c
            row += f"{(100*c/t if t else 0):>12.1f}%".rjust(14)
        row += f"{(100*cap_all/tot_all if tot_all else 0):>12.1f}%".rjust(14)
        print(row)
    print()
    print("Note: K is tokens per side; segments <= 2*K tokens are kept whole.")

    if args.slide_analysis:
        run_slide_analysis(
            span_records,
            window_size=args.window_size,
            stride=args.stride,
            target=args.target,
            max_slides=args.max_slides,
        )


if __name__ == "__main__":
    main()
