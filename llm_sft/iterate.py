"""Iterate extracted documents and segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_index(extracted_dir: Path) -> list[dict]:
    index_path = extracted_dir / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")
    entries: list[dict] = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_annotations(extracted_dir: Path, entry: dict) -> dict:
    ann_path = extracted_dir / entry["annotations_path"]
    with ann_path.open(encoding="utf-8") as f:
        return json.load(f)


def iter_segments(extracted_dir: Path) -> Iterator[dict]:
    """Yield segment records with doc metadata and annotations."""
    for entry in load_index(extracted_dir):
        payload = load_annotations(extracted_dir, entry)
        doc_id = entry["doc_id"]
        for segment in payload.get("segments", []):
            yield {
                "doc_id": doc_id,
                "filename": entry.get("filename", ""),
                "segment_id": segment["segment_id"],
                "segment_index": segment.get("segment_index", 0),
                "segment_label": segment.get("segment_label", "TEXT"),
                "text": segment.get("text", ""),
                "annotations": segment.get("annotations", []),
                "has_title": any(
                    a.get("label") == "title" for a in segment.get("annotations", [])
                ),
                "has_author": any(
                    a.get("label") == "author" for a in segment.get("annotations", [])
                ),
                "char_length": len(segment.get("text", "")),
            }


def iter_doc_meta(extracted_dir: Path) -> Iterator[dict]:
    """One stratification record per document (for split.py)."""
    for entry in load_index(extracted_dir):
        yield {
            "doc_id": entry["doc_id"],
            "has_title": entry.get("has_title", False),
            "has_author": entry.get("has_author", False),
            "char_length": entry.get("content_length", 0),
        }
