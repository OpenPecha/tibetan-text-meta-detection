"""Verify RoBERTa pipeline output format and window tier behavior."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from config import ROBERTA_WINDOW_SIZE


def main() -> None:
    jsonl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/sample_4doc_roberta/roberta_all_examples.jsonl"
    )
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    print(f"total examples: {len(lines)}")

    required_keys = {
        "doc_id",
        "segment_id",
        "window_name",
        "input_ids",
        "attention_mask",
        "labels",
        "offset_mapping",
        "label_list",
        "has_title",
        "has_author",
        "segment_tier",
    }

    seg_windows: dict[str, list[dict]] = defaultdict(list)
    errors: list[str] = []

    for i, line in enumerate(lines):
        ex = json.loads(line)
        missing = required_keys - set(ex.keys())
        if missing:
            errors.append(f"line {i}: missing keys {missing}")
        if len(ex["input_ids"]) != ROBERTA_WINDOW_SIZE:
            errors.append(
                f"line {i}: input_ids length {len(ex['input_ids'])} != {ROBERTA_WINDOW_SIZE}"
            )
        if len(ex["labels"]) != ROBERTA_WINDOW_SIZE:
            errors.append(f"line {i}: labels length mismatch")
        if ex["labels"][0] != -100:
            errors.append(f"line {i}: CLS label should be -100")
        attn = ex["attention_mask"]
        last_content = max(i for i, m in enumerate(attn) if m == 1)
        if ex["labels"][last_content] != -100:
            errors.append(f"line {i}: SEP label should be -100")

        seg_key = f"{ex['doc_id']}:{ex['segment_id']}"
        seg_windows[seg_key].append(ex)

    tier_seg_counts: Counter[str] = Counter()
    for seg_key, windows in seg_windows.items():
        tier = windows[0]["segment_tier"]
        tier_seg_counts[tier] += 1
        n = len(windows)
        token_len = windows[0]["token_length"]
        if tier == "short" and n != 1:
            errors.append(f"{seg_key}: short segment has {n} windows, expected 1")
        if tier == "medium" and n > 30:
            errors.append(f"{seg_key}: medium segment has {n} windows, expected <=30")
        if tier == "long" and n > 30:
            errors.append(f"{seg_key}: long segment has {n} windows, expected <=30")
        print(
            f"  {seg_key}: tier={tier} tokens={token_len} windows={n} "
            f"names={[w['window_name'] for w in windows[:3]]}..."
        )

    print(f"tier segment counts: {dict(tier_seg_counts)}")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
