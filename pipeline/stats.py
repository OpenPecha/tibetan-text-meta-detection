"""Statistics reports for export and window coverage."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pipeline.window import annotation_fully_captured, build_windows


def _length_bucket(char_length: int) -> str:
    if char_length < 50_000:
        return "short"
    if char_length < 500_000:
        return "medium"
    return "long"


def analyze_export(extracted_dir: Path) -> dict:
    """Story 1.1: label distribution, span lengths, text lengths."""
    index_path = extracted_dir / "index.jsonl"
    annotations_dir = extracted_dir / "annotations"

    label_counts: Counter[str] = Counter()
    span_lengths: Counter[str] = Counter()
    text_lengths: list[int] = []
    segment_count = 0
    docs = 0

    with index_path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            docs += 1
            text_lengths.append(entry["content_length"])
            ann_path = extracted_dir / entry["annotations_path"]
            with ann_path.open(encoding="utf-8") as af:
                payload = json.load(af)

            segments = payload.get("segments", [])
            segment_count += len(segments)
            annotations = payload.get("annotations", [])
            if not annotations and segments:
                for seg in segments:
                    annotations.extend(seg.get("annotations", []))

            for ann in annotations:
                label_counts[ann["label"]] += 1
                span_lengths[ann["label"]] += ann["span_end"] - ann["span_start"]

    span_avg = {
        label: (span_lengths[label] / label_counts[label] if label_counts[label] else 0)
        for label in label_counts
    }

    text_lengths.sort()
    mid = len(text_lengths) // 2

    return {
        "documents": docs,
        "segments": segment_count,
        "label_distribution": dict(label_counts),
        "avg_span_length_by_label": span_avg,
        "text_length": {
            "min": text_lengths[0] if text_lengths else 0,
            "max": text_lengths[-1] if text_lengths else 0,
            "median": text_lengths[mid] if text_lengths else 0,
            "mean": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        },
        "length_buckets": dict(Counter(_length_bucket(n) for n in text_lengths)),
    }


def analyze_window_coverage(
    extracted_dir: Path,
    window_sizes: list[int],
) -> dict:
    """Story 1.2: percentage of spans captured at various window sizes."""
    index_path = extracted_dir / "index.jsonl"
    texts_dir = extracted_dir / "texts"

    results: dict[str, dict] = {}

    for window_size in window_sizes:
        total = 0
        captured = 0
        by_label: Counter[str] = Counter()
        captured_by_label: Counter[str] = Counter()

        with index_path.open(encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                text_path = texts_dir / f"{entry['doc_id']}.txt"
                content = text_path.read_text(encoding="utf-8")
                ann_path = extracted_dir / entry["annotations_path"]
                with ann_path.open(encoding="utf-8") as af:
                    annotations = json.load(af)["annotations"]

                windows = build_windows(content, window_size)
                for ann in annotations:
                    total += 1
                    by_label[ann["label"]] += 1
                    if annotation_fully_captured(ann, windows):
                        captured += 1
                        captured_by_label[ann["label"]] += 1

        results[str(window_size)] = {
            "total_annotations": total,
            "captured_annotations": captured,
            "capture_rate": captured / total if total else 0.0,
            "by_label": {
                label: {
                    "total": by_label[label],
                    "captured": captured_by_label[label],
                    "capture_rate": (
                        captured_by_label[label] / by_label[label]
                        if by_label[label]
                        else 0.0
                    ),
                }
                for label in sorted(by_label)
            },
        }

    return {"window_sizes": window_sizes, "coverage": results}


def write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
