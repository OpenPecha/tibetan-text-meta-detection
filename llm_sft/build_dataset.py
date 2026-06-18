"""Build LLM SFT JSONL datasets (title / author) from extracted annotations."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline.split import stratified_split, write_jsonl
from transformers import AutoTokenizer

from llm_sft.config import SFTConfig, TILAMB_MODEL
from llm_sft.crop import (
    build_example_row,
    generate_crops_for_task,
    normalize_annotations,
    remap_spans,
    tokenize_segment,
    validate_spans,
)
from llm_sft.iterate import iter_doc_meta, iter_segments
from llm_sft.prompts import INSTRUCTIONS


def _doc_split_map(config: SFTConfig) -> dict[str, str]:
    doc_records = list(iter_doc_meta(config.extracted_dir))
    if not doc_records:
        raise SystemExit(f"No documents under {config.extracted_dir}")

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for rec in doc_records:
        by_doc[rec["doc_id"]].append(rec)

    flat = [rec for recs in by_doc.values() for rec in recs]
    train, val, test, _report = stratified_split(
        flat,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    mapping: dict[str, str] = {}
    for split_name, records in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        for rec in records:
            mapping[rec["doc_id"]] = split_name
    return mapping


def _generate_rows(
    tokenizer: Any,
    config: SFTConfig,
    rng: random.Random,
) -> tuple[dict[str, dict[str, list[dict]]], dict]:
    doc_split = _doc_split_map(config)
    buckets: dict[str, dict[str, list[dict]]] = {
        "train": {"title": [], "author": []},
        "val": {"title": [], "author": []},
        "test": {"title": [], "author": []},
    }
    stats = {
        "crop_kind": Counter(),
        "task": Counter(),
        "split": Counter(),
        "span_position_ratio": [],
        "rows": 0,
        "segments": 0,
        "errors": 0,
    }

    for segment in iter_segments(config.extracted_dir):
        stats["segments"] += 1
        doc_id = segment["doc_id"]
        split_name = doc_split.get(doc_id, "train")
        text = segment["text"]
        if not text.strip():
            continue

        tokenized = tokenize_segment(tokenizer, text)
        segment_anns = normalize_annotations(text, segment["annotations"])

        for task in config.tasks:
            instruction = INSTRUCTIONS[task]
            crops = generate_crops_for_task(
                tokenized,
                segment_anns,
                task,
                config,
                rng,
            )
            for crop_index, crop in enumerate(crops):
                spans = remap_spans(
                    crop.text,
                    segment_anns,
                    task,
                    crop.char_start,
                    crop.char_end,
                )
                if crop.kind in ("full", "positive") and not spans:
                    stats["errors"] += 1
                    continue
                if crop.kind == "negative" and spans:
                    stats["errors"] += 1
                    continue
                try:
                    validate_spans(crop.text, spans)
                except ValueError:
                    stats["errors"] += 1
                    continue

                row = build_example_row(
                    doc_id=doc_id,
                    segment_id=segment["segment_id"],
                    task=task,
                    instruction=instruction,
                    crop=crop,
                    spans=spans,
                    crop_index=crop_index,
                )
                buckets[split_name][task].append(row)
                stats["crop_kind"][crop.kind] += 1
                stats["task"][task] += 1
                stats["split"][f"{split_name}:{task}"] += 1
                stats["rows"] += 1
                ratio = row.get("span_position_ratio")
                if ratio is not None:
                    stats["span_position_ratio"].append(ratio)

    return buckets, stats


def _write_task_splits(
    buckets: dict[str, dict[str, list[dict]]],
    config: SFTConfig,
) -> None:
    for task in config.tasks:
        task_dir = config.title_dir if task == "title" else config.author_dir
        task_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val", "test"):
            rows = buckets[split_name][task]
            alpaca_rows = [
                {
                    "instruction": r["instruction"],
                    "input": r["input"],
                    "output": r["output"],
                }
                for r in rows
            ]
            write_jsonl(task_dir / f"{split_name}.jsonl", alpaca_rows)
            write_jsonl(
                task_dir / f"{split_name}_meta.jsonl",
                rows,
            )


def _write_dataset_info(config: SFTConfig) -> None:
    columns = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
    }
    info = {}
    for task in config.tasks:
        prefix = f"tibetan_{task}_sft"
        info[prefix] = {
            "file_name": f"{prefix}/train.jsonl",
            "columns": columns,
        }
        info[f"{prefix}_val"] = {
            "file_name": f"{prefix}/val.jsonl",
            "columns": columns,
        }
    path = config.output_dir / "dataset_info.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def _write_crop_stats(stats: dict, config: SFTConfig) -> None:
    ratios = stats.get("span_position_ratio") or []
    ratio_hist = Counter()
    for r in ratios:
        bucket = min(9, int(r * 10))
        ratio_hist[f"{bucket * 0.1:.1f}-{bucket * 0.1 + 0.1:.1f}"] += 1

    report = {
        "rows": stats["rows"],
        "segments": stats["segments"],
        "errors": stats["errors"],
        "crop_kind": dict(stats["crop_kind"]),
        "task": dict(stats["task"]),
        "split": dict(stats["split"]),
        "span_position_ratio_histogram": dict(sorted(ratio_hist.items())),
        "span_position_ratio_median": (
            sorted(ratios)[len(ratios) // 2] if ratios else None
        ),
        "config": {
            "tokenizer": config.tokenizer_name,
            "max_context_tokens": config.max_context_tokens,
            "crops_per_positive": config.crops_per_positive,
            "crops_per_negative": config.crops_per_negative,
            "seed": config.seed,
        },
    }
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    out = config.reports_dir / "crop_stats.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def build(config: SFTConfig) -> dict:
    print(f"Loading tokenizer {config.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    rng = random.Random(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    buckets, stats = _generate_rows(tokenizer, config, rng)
    _write_task_splits(buckets, config)
    _write_dataset_info(config)
    _write_crop_stats(stats, config)

    for task in config.tasks:
        task_dir = config.title_dir if task == "title" else config.author_dir
        for split_name in ("train", "val", "test"):
            n = len(buckets[split_name][task])
            print(f"  {task}/{split_name}.jsonl: {n} rows")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("data/extracted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/llm_sft"),
    )
    parser.add_argument("--tokenizer", default=TILAMB_MODEL)
    parser.add_argument("--max-context-tokens", type=int, default=3584)
    parser.add_argument("--crops-per-positive", type=int, default=3)
    parser.add_argument("--crops-per-negative", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = SFTConfig(
        extracted_dir=args.extracted_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        max_context_tokens=args.max_context_tokens,
        crops_per_positive=args.crops_per_positive,
        crops_per_negative=args.crops_per_negative,
        seed=args.seed,
    )
    build(config)


if __name__ == "__main__":
    main()
