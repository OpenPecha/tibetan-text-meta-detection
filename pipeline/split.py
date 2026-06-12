"""Stratified train/val/test splitting."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _length_bucket(char_length: int) -> str:
    if char_length < 50_000:
        return "short"
    if char_length < 500_000:
        return "medium"
    return "long"


def stratification_key(record: dict) -> str:
    """Build stratification bucket from label presence and text length."""
    has_title = record.get("has_title", False)
    has_author = record.get("has_author", False)
    if has_title and has_author:
        label_bucket = "both"
    elif has_title:
        label_bucket = "title_only"
    elif has_author:
        label_bucket = "author_only"
    else:
        label_bucket = "neither"

    length_bucket = _length_bucket(record.get("char_length", 0))
    return f"{label_bucket}|{length_bucket}"


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Split records by doc_id groups with stratification."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    by_doc: dict[str, list[dict]] = defaultdict(list)
    doc_meta: dict[str, dict] = {}

    for record in records:
        doc_id = record["doc_id"]
        by_doc[doc_id].append(record)
        if doc_id not in doc_meta:
            doc_meta[doc_id] = {
                "doc_id": doc_id,
                "has_title": record.get("has_title", False),
                "has_author": record.get("has_author", False),
                "char_length": record.get("char_length", 0),
            }

    buckets: dict[str, list[str]] = defaultdict(list)
    for doc_id, meta in doc_meta.items():
        buckets[stratification_key(meta)].append(doc_id)

    rng = random.Random(seed)
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    test_ids: set[str] = set()

    for _bucket, doc_ids in buckets.items():
        ids = doc_ids[:]
        rng.shuffle(ids)
        n = len(ids)
        if n == 1:
            train_ids.add(ids[0])
            continue
        if n == 2:
            train_ids.add(ids[0])
            val_ids.add(ids[1])
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = 1
            n_train = n - 2
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = max(1, n - n_val - n_test)

        train_ids.update(ids[:n_train])
        val_ids.update(ids[n_train : n_train + n_val])
        test_ids.update(ids[n_train + n_val :])

    def collect(selected: set[str]) -> list[dict]:
        out: list[dict] = []
        for doc_id in sorted(selected):
            out.extend(by_doc[doc_id])
        return out

    train = collect(train_ids)
    val = collect(val_ids)
    test = collect(test_ids)

    def split_stats(selected_doc_ids: set[str], split_records: list[dict]) -> dict:
        keys = Counter(stratification_key(doc_meta[doc_id]) for doc_id in selected_doc_ids)
        return {
            "examples": len(split_records),
            "documents": len(selected_doc_ids),
            "strata": dict(keys),
            "has_title": sum(1 for r in split_records if r.get("has_title")),
            "has_author": sum(1 for r in split_records if r.get("has_author")),
        }

    report = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "train": split_stats(train_ids, train),
        "val": split_stats(val_ids, val),
        "test": split_stats(test_ids, test),
    }
    return train, val, test, report


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
