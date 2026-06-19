#!/usr/bin/env python3
"""Download segment-eval test manifest from HF (same split as eval_segment.py / EXPERIMENT_REPORT)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HF_DATASET = "ganga4364/tibetan-metadata-detector"


def export_test_jsonl(output_path: Path) -> int:
    from datasets import load_dataset

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET, split="test")
    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            record = {
                "doc_id": row["doc_id"],
                "segment_id": row["segment_id"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/roberta_full/splits/test.jsonl"),
    )
    args = parser.parse_args()

    if args.output.is_file():
        lines = sum(1 for _ in args.output.open(encoding="utf-8"))
        print(f"OK: {args.output} already exists ({lines} window rows)")
        return

    print(f"Downloading test split from {HF_DATASET} ...")
    n = export_test_jsonl(args.output)
    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
