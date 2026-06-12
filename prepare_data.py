"""Epic 1 pipeline: export analysis, windowing, BIO tagging, and splits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from config import (
    AUTHOR_OVERSAMPLE,
    BALANCED_EXAMPLES_FILENAME,
    DEFAULT_WINDOW_SIZE,
    EXTRACTED_DIR,
    O_ONLY_CAP_RATIO,
    PROCESSED_DIR,
    RANDOM_SEED,
    ROBERTA_MAX_BEGIN_SLIDES,
    ROBERTA_MAX_END_SLIDES,
    ROBERTA_MODEL,
    ROBERTA_STRIDE,
    ROBERTA_WINDOW_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    WINDOW_SIZE_CANDIDATES,
)
from extract_data import extract_metadata_data
from pipeline.bio import BIO_LABELS, LABEL_TO_ID, annotations_to_bio, validate_bio_reconstruction
from pipeline.roberta_windows import (
    WindowStatsAccumulator,
    slide_segment,
    validate_segment_roundtrip,
)
from pipeline.split import stratified_split, write_jsonl
from pipeline.stats import analyze_export, analyze_window_coverage, write_json_report
from pipeline.tokenize import tokenize_tibetan
from pipeline.window import build_windows, filter_annotations_for_window


def _load_index(extracted_dir: Path) -> list[dict]:
    index_path = extracted_dir / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing {index_path}. Run: python extract_data.py --all"
        )
    entries: list[dict] = []
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def run_analyze(extracted_dir: Path, reports_dir: Path) -> dict:
    """Story 1.1: export statistics."""
    stats = analyze_export(extracted_dir)
    write_json_report(reports_dir / "export_stats.json", stats)
    print(f"Export stats: {reports_dir / 'export_stats.json'}")
    print(f"  Documents: {stats['documents']}")
    print(f"  Segments: {stats.get('segments', 'n/a')}")
    print(f"  Labels: {stats['label_distribution']}")
    return stats


def run_window_report(extracted_dir: Path, reports_dir: Path) -> dict:
    """Story 1.2: document-level window coverage at multiple sizes."""
    report = analyze_window_coverage(extracted_dir, WINDOW_SIZE_CANDIDATES)
    write_json_report(reports_dir / "window_coverage.json", report)
    print(f"Window coverage: {reports_dir / 'window_coverage.json'}")
    for size, data in report["coverage"].items():
        print(f"  window={size}: {data['capture_rate']:.1%} captured")
    return report


def _process_segment_example(
    doc_id: str,
    filename: str,
    segment: dict,
) -> dict | None:
    """Build one BIO-tagged training example from a segment record."""
    segment_text = segment["text"]
    tokens = tokenize_tibetan(segment_text)
    if not tokens:
        return None

    annotations = segment["annotations"]
    tags = annotations_to_bio(tokens, annotations)
    ok, _errors = validate_bio_reconstruction(tokens, annotations, tags)

    has_title = any(a["label"] == "title" for a in annotations)
    has_author = any(a["label"] == "author" for a in annotations)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "segment_id": segment["segment_id"],
        "segment_index": segment["segment_index"],
        "segment_label": segment["segment_label"],
        "window": "segment",
        "char_length": len(segment_text),
        "syllable_count": len(tokens),
        "has_title": has_title,
        "has_author": has_author,
        "tokens": [t.text for t in tokens],
        "ner_tags": tags,
        "label_list": BIO_LABELS,
        "annotations": annotations,
    }


def _process_document_windows(
    doc_id: str,
    filename: str,
    content: str,
    flat_annotations: list[dict],
    window_size: int,
) -> list[dict]:
    """Optional document begin/end windows when spans fall in those regions."""
    records: list[dict] = []
    for window in build_windows(content, window_size):
        if window.name == "full":
            continue

        window_anns = filter_annotations_for_window(flat_annotations, window)
        if not window_anns:
            continue

        tokens = tokenize_tibetan(window.text)
        if not tokens:
            continue

        tags = annotations_to_bio(tokens, window_anns)
        ok, _errors = validate_bio_reconstruction(tokens, window_anns, tags)
        if not ok:
            continue

        records.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "segment_id": None,
                "segment_index": None,
                "segment_label": None,
                "window": window.name,
                "window_size": window_size,
                "char_length": len(window.text),
                "syllable_count": len(tokens),
                "has_title": any(a["label"] == "title" for a in window_anns),
                "has_author": any(a["label"] == "author" for a in window_anns),
                "tokens": [t.text for t in tokens],
                "ner_tags": tags,
                "label_list": BIO_LABELS,
                "annotations": window_anns,
            }
        )
    return records


def run_process(
    extracted_dir: Path,
    processed_dir: Path,
    window_size: int,
    include_doc_windows: bool = False,
) -> dict:
    """Stories 1.2 + 1.3: segment BIO examples (+ optional doc windows)."""
    entries = _load_index(extracted_dir)
    windows_dir = processed_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    validation_failures = 0
    skipped_empty = 0

    for entry in entries:
        doc_id = entry["doc_id"]
        ann_path = extracted_dir / entry["annotations_path"]
        text_path = extracted_dir / entry["text_path"]
        content = text_path.read_text(encoding="utf-8")

        with ann_path.open(encoding="utf-8") as f:
            payload = json.load(f)

        segments = payload.get("segments", [])
        flat_annotations = payload.get("annotations", [])

        if not segments and flat_annotations:
            raise ValueError(
                f"Legacy export format in {ann_path}. Re-run extract_data.py."
            )

        for segment in segments:
            record = _process_segment_example(doc_id, entry["filename"], segment)
            if record is None:
                skipped_empty += 1
                continue

            tokens = tokenize_tibetan(segment["text"])
            tags = record["ner_tags"]
            ok, _ = validate_bio_reconstruction(tokens, segment["annotations"], tags)
            if not ok:
                validation_failures += 1

            records.append(record)
            window_file = windows_dir / f"{doc_id}_{segment['segment_id']}.json"
            window_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if include_doc_windows:
            doc_window_records = _process_document_windows(
                doc_id,
                entry["filename"],
                content,
                flat_annotations,
                window_size,
            )
            for record in doc_window_records:
                records.append(record)
                window_file = windows_dir / f"{doc_id}_{record['window']}.json"
                window_file.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        if not segments:
            skipped_empty += 1

    all_records_path = processed_dir / "all_examples.jsonl"
    write_jsonl(all_records_path, records)

    summary = {
        "documents": len(entries),
        "training_examples": len(records),
        "window_size": window_size,
        "include_doc_windows": include_doc_windows,
        "validation_failures": validation_failures,
        "skipped_empty_documents": skipped_empty,
        "bio_labels": BIO_LABELS,
        "all_examples_path": str(all_records_path),
    }
    write_json_report(processed_dir / "process_summary.json", summary)

    print(f"Processed {len(entries)} documents -> {len(records)} training examples")
    print(f"  Per-example files: {windows_dir}")
    print(f"  Combined: {all_records_path}")
    print(f"  Skipped/failed: {validation_failures}")
    return summary


def _resolve_examples_path(processed_dir: Path, examples_file: Path | None) -> Path:
    if examples_file is not None:
        if not examples_file.exists():
            raise FileNotFoundError(f"Missing {examples_file}")
        return examples_file
    balanced_path = processed_dir / BALANCED_EXAMPLES_FILENAME
    if balanced_path.exists():
        return balanced_path
    roberta_path = processed_dir / "roberta_all_examples.jsonl"
    if roberta_path.exists():
        return roberta_path
    legacy_path = processed_dir / "all_examples.jsonl"
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        f"Missing examples JSONL in {processed_dir}. "
        "Run: python prepare_data.py roberta-process"
    )


def _roberta_worker_shard_path(processed_dir: Path, worker_id: int) -> Path:
    return processed_dir / f"roberta_all_examples.worker{worker_id}.jsonl"


def _roberta_worker_summary_path(processed_dir: Path, worker_id: int) -> Path:
    return processed_dir / f"roberta_process_summary.worker{worker_id}.json"


def _acquire_roberta_process_lock(
    processed_dir: Path,
    worker_id: int | None,
    num_workers: int,
) -> Path:
    """Prevent two processes from using the same worker slot."""
    if num_workers > 1:
        if worker_id is None:
            raise ValueError("--worker-id is required when --num-workers > 1")
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(
                f"--worker-id must be between 0 and {num_workers - 1}, got {worker_id}"
            )
        lock_path = processed_dir / f".roberta_process.lock.worker{worker_id}"
    else:
        lock_path = processed_dir / ".roberta_process.lock"

    processed_dir.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        raise RuntimeError(f"Another roberta-process holds {lock_path}")
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return lock_path


def _process_roberta_document(
    entry: dict,
    extracted_dir: Path,
    tokenizer,
    stats_acc: WindowStatsAccumulator,
    out_f,
    *,
    window_size: int,
    stride: int,
    max_begin: int,
    max_end: int,
    write_per_window_files: bool,
    windows_dir: Path | None,
) -> tuple[int, int]:
    """Process one document; returns (skipped_empty, roundtrip_failures)."""
    skipped_empty = 0
    roundtrip_failures = 0
    doc_id = entry["doc_id"]
    ann_path = extracted_dir / entry["annotations_path"]
    with ann_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    segments = payload.get("segments", [])
    if not segments and payload.get("annotations"):
        raise ValueError(
            f"Legacy export format in {ann_path}. Re-run extract_data.py."
        )

    for segment in segments:
        text = segment["text"]
        if not text.strip():
            skipped_empty += 1
            continue

        annotations = segment.get("annotations", [])
        metadata = {
            "doc_id": doc_id,
            "filename": entry["filename"],
            "segment_id": segment["segment_id"],
            "segment_index": segment["segment_index"],
            "segment_label": segment.get("segment_label"),
        }

        examples = slide_segment(
            tokenizer,
            text,
            annotations,
            metadata=metadata,
            window_size=window_size,
            stride=stride,
            max_begin=max_begin,
            max_end=max_end,
        )
        if not examples:
            skipped_empty += 1
            continue

        failures, _ = validate_segment_roundtrip(
            tokenizer,
            text,
            annotations,
            window_size=window_size,
            stride=stride,
            max_begin=max_begin,
            max_end=max_end,
        )
        roundtrip_failures += failures

        for example in examples:
            stats_acc.add(example)
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            if write_per_window_files and windows_dir is not None:
                window_file = (
                    windows_dir
                    / f"{doc_id}_{segment['segment_id']}_{example['window_name']}.json"
                )
                window_file.write_text(
                    json.dumps(example, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    return skipped_empty, roundtrip_failures


def run_roberta_process(
    extracted_dir: Path,
    processed_dir: Path,
    model_name: str = ROBERTA_MODEL,
    window_size: int = ROBERTA_WINDOW_SIZE,
    stride: int = ROBERTA_STRIDE,
    max_begin: int = ROBERTA_MAX_BEGIN_SLIDES,
    max_end: int = ROBERTA_MAX_END_SLIDES,
    num_docs: int | None = None,
    write_per_window_files: bool = False,
    worker_id: int | None = None,
    num_workers: int = 1,
) -> dict:
    """RoBERTa subword sliding-window BIO examples for HuggingFace training."""
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if num_workers > 1 and worker_id is None:
        raise ValueError("--worker-id is required when --num-workers > 1")

    from transformers import AutoTokenizer

    lock_path = _acquire_roberta_process_lock(processed_dir, worker_id, num_workers)
    try:
        return _run_roberta_process_inner(
            extracted_dir=extracted_dir,
            processed_dir=processed_dir,
            model_name=model_name,
            window_size=window_size,
            stride=stride,
            max_begin=max_begin,
            max_end=max_end,
            num_docs=num_docs,
            write_per_window_files=write_per_window_files,
            worker_id=worker_id,
            num_workers=num_workers,
        )
    finally:
        lock_path.unlink(missing_ok=True)


def _run_roberta_process_inner(
    extracted_dir: Path,
    processed_dir: Path,
    model_name: str,
    window_size: int,
    stride: int,
    max_begin: int,
    max_end: int,
    num_docs: int | None,
    write_per_window_files: bool,
    worker_id: int | None,
    num_workers: int,
) -> dict:
    from transformers import AutoTokenizer

    worker_label = (
        f"worker {worker_id}/{num_workers}" if num_workers > 1 else "single worker"
    )
    print(f"Loading tokenizer: {model_name} ({worker_label})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    entries = _load_index(extracted_dir)
    if num_docs is not None:
        entries = entries[:num_docs]
    if num_workers > 1:
        entries = [
            entry
            for idx, entry in enumerate(entries)
            if idx % num_workers == worker_id
        ]
        print(f"  assigned {len(entries)} documents")

    processed_dir.mkdir(parents=True, exist_ok=True)
    if write_per_window_files:
        windows_dir = processed_dir / "roberta_windows"
        if num_workers > 1:
            windows_dir = processed_dir / f"roberta_windows.worker{worker_id}"
        windows_dir.mkdir(parents=True, exist_ok=True)
    else:
        windows_dir = None

    stats_acc = WindowStatsAccumulator()
    skipped_empty = 0
    roundtrip_failures = 0
    if num_workers > 1:
        all_records_path = _roberta_worker_shard_path(processed_dir, worker_id)
        summary_path = _roberta_worker_summary_path(processed_dir, worker_id)
    else:
        all_records_path = processed_dir / "roberta_all_examples.jsonl"
        summary_path = processed_dir / "roberta_process_summary.json"

    with all_records_path.open("w", encoding="utf-8") as out_f:
        for doc_idx, entry in enumerate(entries, start=1):
            doc_skipped, doc_failures = _process_roberta_document(
                entry,
                extracted_dir,
                tokenizer,
                stats_acc,
                out_f,
                window_size=window_size,
                stride=stride,
                max_begin=max_begin,
                max_end=max_end,
                write_per_window_files=write_per_window_files,
                windows_dir=windows_dir,
            )
            skipped_empty += doc_skipped
            roundtrip_failures += doc_failures

            if doc_idx % 50 == 0 or doc_idx == len(entries):
                print(
                    f"  [{doc_idx}/{len(entries)}] docs, "
                    f"{stats_acc.total_examples} examples ({worker_label})"
                )

    stats = stats_acc.to_dict()
    summary = {
        "documents": len(entries),
        "training_examples": stats_acc.total_examples,
        "model": model_name,
        "window_size": window_size,
        "stride": stride,
        "max_begin_slides": max_begin,
        "max_end_slides": max_end,
        "skipped_empty_segments": skipped_empty,
        "roundtrip_failures": roundtrip_failures,
        "write_per_window_files": write_per_window_files,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "bio_labels": BIO_LABELS,
        "all_examples_path": str(all_records_path),
        **stats,
    }
    write_json_report(summary_path, summary)

    print(
        f"Processed {len(entries)} documents -> {stats_acc.total_examples} "
        f"RoBERTa examples ({worker_label})"
    )
    print(f"  Output: {all_records_path}")
    if write_per_window_files:
        print(f"  Per-window files: {windows_dir}")
    print(f"  Tier stats: {stats.get('tier_window_stats', {})}")
    print(f"  Roundtrip failures: {roundtrip_failures}")
    if num_workers > 1:
        print(
            f"Worker {worker_id} finished. "
            "Run merge-roberta-shards after all workers complete."
        )
    return summary


def run_merge_roberta_shards(
    processed_dir: Path,
    num_workers: int | None = None,
    keep_shards: bool = False,
) -> dict:
    """Concatenate worker JSONL shards into roberta_all_examples.jsonl."""
    shard_paths = sorted(processed_dir.glob("roberta_all_examples.worker*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No worker shards under {processed_dir}. "
            "Run roberta-process with --num-workers > 1 first."
        )
    if num_workers is not None and len(shard_paths) != num_workers:
        raise RuntimeError(
            f"Expected {num_workers} shards, found {len(shard_paths)}: "
            f"{[p.name for p in shard_paths]}"
        )

    out_path = processed_dir / "roberta_all_examples.jsonl"
    total_examples = 0
    total_documents = 0
    skipped_empty = 0
    roundtrip_failures = 0
    merged_tier_counts: dict[str, int] = {}
    merged_tier_window_stats: dict[str, dict] = {}

    with out_path.open("w", encoding="utf-8") as out_f:
        for shard in shard_paths:
            with shard.open(encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    out_f.write(line + "\n")
                    total_examples += 1

            worker_suffix = shard.name.replace("roberta_all_examples.", "").replace(
                ".jsonl", ""
            )
            summary_path = processed_dir / f"roberta_process_summary.{worker_suffix}.json"
            if summary_path.is_file():
                worker_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                total_documents += worker_summary.get("documents", 0)
                skipped_empty += worker_summary.get("skipped_empty_segments", 0)
                roundtrip_failures += worker_summary.get("roundtrip_failures", 0)
                for tier, count in worker_summary.get("tier_example_counts", {}).items():
                    merged_tier_counts[tier] = merged_tier_counts.get(tier, 0) + count

    summary = {
        "documents": total_documents,
        "training_examples": total_examples,
        "skipped_empty_segments": skipped_empty,
        "roundtrip_failures": roundtrip_failures,
        "num_shards": len(shard_paths),
        "shard_files": [str(p) for p in shard_paths],
        "all_examples_path": str(out_path),
        "tier_example_counts": merged_tier_counts,
        "tier_window_stats": merged_tier_window_stats,
    }
    write_json_report(processed_dir / "roberta_process_summary.json", summary)

    if not keep_shards:
        for shard in shard_paths:
            shard.unlink()

    print(f"Merged {len(shard_paths)} shards -> {out_path}")
    print(f"  {total_documents} documents, {total_examples} examples")
    print(f"  Roundtrip failures (sum): {roundtrip_failures}")
    return summary


def run_split(
    processed_dir: Path,
    seed: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    examples_file: Path | None = None,
) -> dict:
    """Story 1.4: stratified train/val/test split."""
    splits_dir = processed_dir / "splits"
    all_path = _resolve_examples_path(processed_dir, examples_file)
    print(f"Splitting from: {all_path}")

    records: list[dict] = []
    with all_path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    train, val, test, report = stratified_split(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(splits_dir / "train.jsonl", train)
    write_jsonl(splits_dir / "val.jsonl", val)
    write_jsonl(splits_dir / "test.jsonl", test)
    write_json_report(splits_dir / "split_stats.json", report)

    print(f"Splits written to {splits_dir}")
    print(
        f"  train: {report['train']['examples']} examples / "
        f"{report['train']['documents']} docs"
    )
    print(
        f"  val:   {report['val']['examples']} examples / "
        f"{report['val']['documents']} docs"
    )
    print(
        f"  test:  {report['test']['examples']} examples / "
        f"{report['test']['documents']} docs"
    )
    return report


def _record_has_entity(record: dict) -> bool:
    o_id = LABEL_TO_ID["O"]
    for label_id in record.get("labels", []):
        if label_id not in (-100, o_id):
            return True
    return bool(record.get("window_annotations"))


def _record_has_author(record: dict) -> bool:
    author_ids = {LABEL_TO_ID["B-AUTHOR"], LABEL_TO_ID["I-AUTHOR"]}
    if any(label_id in author_ids for label_id in record.get("labels", [])):
        return True
    return bool(record.get("has_author")) and _record_has_entity(record)


def run_balance_windows(
    processed_dir: Path,
    seed: int = RANDOM_SEED,
    o_only_cap_ratio: float = O_ONLY_CAP_RATIO,
    author_oversample: int = AUTHOR_OVERSAMPLE,
    input_file: Path | None = None,
    output_file: Path | None = None,
) -> dict:
    """Subsample O-only windows and oversample author-bearing windows per segment."""
    import random
    from collections import defaultdict

    source = input_file or (processed_dir / "roberta_all_examples.jsonl")
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}. Run prepare_data.py roberta-process first.")

    dest = output_file or (processed_dir / BALANCED_EXAMPLES_FILENAME)
    rng = random.Random(seed)

    by_segment: dict[str, list[dict]] = defaultdict(list)
    with source.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            seg_key = f"{record.get('doc_id')}:{record.get('segment_id')}"
            by_segment[seg_key].append(record)

    kept: list[dict] = []
    total_in = 0
    total_o_only_dropped = 0
    total_author_dupes = 0

    for seg_key in sorted(by_segment):
        records = by_segment[seg_key]
        total_in += len(records)
        entity_records = [r for r in records if _record_has_entity(r)]
        o_only_records = [r for r in records if not _record_has_entity(r)]

        cap = max(0, int(round(len(entity_records) * o_only_cap_ratio)))
        original_o_only = len(o_only_records)
        if len(o_only_records) > cap:
            rng.shuffle(o_only_records)
            o_only_records = o_only_records[:cap]
            total_o_only_dropped += original_o_only - len(o_only_records)

        segment_out: list[dict] = entity_records + o_only_records
        if author_oversample > 1:
            for record in entity_records:
                if _record_has_author(record):
                    for _ in range(author_oversample - 1):
                        segment_out.append(record)
                        total_author_dupes += 1

        kept.extend(segment_out)

    dest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(dest, kept)

    report = {
        "input_file": str(source),
        "output_file": str(dest),
        "input_examples": total_in,
        "output_examples": len(kept),
        "o_only_dropped": total_o_only_dropped,
        "author_duplicates_added": total_author_dupes,
        "o_only_cap_ratio": o_only_cap_ratio,
        "author_oversample": author_oversample,
        "segments": len(by_segment),
    }
    write_json_report(processed_dir / "balance_windows_report.json", report)

    print(f"Balanced windows written to {dest}")
    print(
        f"  {total_in} -> {len(kept)} examples "
        f"(dropped {total_o_only_dropped} O-only, +{total_author_dupes} author dupes)"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Epic 1: prepare training data from extracted outliner exports",
    )
    parser.add_argument(
        "command",
        choices=[
            "extract",
            "analyze",
            "window-report",
            "process",
            "roberta-process",
            "merge-roberta-shards",
            "balance-windows",
            "split",
            "all",
        ],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=EXTRACTED_DIR,
        help="Raw export directory (index.jsonl, texts/, annotations/)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Processed output directory",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Syllables for optional doc begin/end windows (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--include-doc-windows",
        action="store_true",
        help="Also emit document begin/end window examples when spans match",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for stratified split",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit DB extraction to N documents (extract/all only)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-extract all documents (do not skip existing exports)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"Train split ratio (default: {TRAIN_RATIO})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Validation split ratio (default: {VAL_RATIO})",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help=f"Test split ratio (default: {TEST_RATIO})",
    )
    parser.add_argument(
        "--model",
        default=ROBERTA_MODEL,
        help=f"RoBERTa model for subword tokenization (default: {ROBERTA_MODEL})",
    )
    parser.add_argument(
        "--roberta-window-size",
        type=int,
        default=ROBERTA_WINDOW_SIZE,
        help=f"RoBERTa sliding window size in tokens (default: {ROBERTA_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--roberta-stride",
        type=int,
        default=ROBERTA_STRIDE,
        help=f"RoBERTa sliding window stride (default: {ROBERTA_STRIDE})",
    )
    parser.add_argument(
        "--max-begin-slides",
        type=int,
        default=ROBERTA_MAX_BEGIN_SLIDES,
        help=f"Max begin-side slides (default: {ROBERTA_MAX_BEGIN_SLIDES})",
    )
    parser.add_argument(
        "--max-end-slides",
        type=int,
        default=ROBERTA_MAX_END_SLIDES,
        help=f"Max end-side slides (default: {ROBERTA_MAX_END_SLIDES})",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Limit processing to first N documents from index.jsonl",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help="Worker index for parallel roberta-process (0 .. num-workers-1)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel roberta-process workers (default: 1)",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep worker JSONL shards after merge-roberta-shards",
    )
    parser.add_argument(
        "--write-per-window-files",
        action="store_true",
        help="Also write one JSON file per RoBERTa window (large disk use)",
    )
    parser.add_argument(
        "--examples-file",
        type=Path,
        default=None,
        help="JSONL file to split (default: roberta_balanced_examples.jsonl, roberta_all_examples.jsonl, or all_examples.jsonl)",
    )
    parser.add_argument(
        "--o-only-cap-ratio",
        type=float,
        default=O_ONLY_CAP_RATIO,
        help=f"Max O-only windows per segment as multiple of entity windows (default: {O_ONLY_CAP_RATIO})",
    )
    parser.add_argument(
        "--author-oversample",
        type=int,
        default=AUTHOR_OVERSAMPLE,
        help=f"Duplicate author-bearing entity windows this many times total (default: {AUTHOR_OVERSAMPLE})",
    )
    parser.add_argument(
        "--balance-input",
        type=Path,
        default=None,
        help="Input JSONL for balance-windows (default: roberta_all_examples.jsonl)",
    )
    parser.add_argument(
        "--balance-output",
        type=Path,
        default=None,
        help=f"Output JSONL for balance-windows (default: {BALANCED_EXAMPLES_FILENAME})",
    )
    args = parser.parse_args()

    reports_dir = args.processed_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.command in ("extract", "all"):
        print("=== Story 1.1: Extract from database ===")
        extract_metadata_data(
            num_samples=args.num_samples,
            output_dir=args.extracted_dir,
            resume=not getattr(args, "no_resume", False),
        )

    if args.command in ("analyze", "all"):
        print("\n=== Story 1.1: Export statistics ===")
        run_analyze(args.extracted_dir, reports_dir)

    if args.command in ("window-report", "all"):
        print("\n=== Story 1.2: Window coverage report ===")
        run_window_report(args.extracted_dir, reports_dir)

    if args.command in ("process", "all"):
        print("\n=== Stories 1.2 + 1.3: Segment window + BIO tagging ===")
        run_process(
            args.extracted_dir,
            args.processed_dir,
            args.window_size,
            include_doc_windows=args.include_doc_windows,
        )

    if args.command == "roberta-process":
        print("\n=== RoBERTa subword sliding-window BIO tagging ===")
        run_roberta_process(
            args.extracted_dir,
            args.processed_dir,
            model_name=args.model,
            window_size=args.roberta_window_size,
            stride=args.roberta_stride,
            max_begin=args.max_begin_slides,
            max_end=args.max_end_slides,
            num_docs=args.num_docs,
            write_per_window_files=args.write_per_window_files,
            worker_id=args.worker_id,
            num_workers=args.num_workers,
        )

    if args.command == "merge-roberta-shards":
        print("\n=== Merge RoBERTa worker shards ===")
        run_merge_roberta_shards(
            args.processed_dir,
            num_workers=args.num_workers if args.num_workers > 1 else None,
            keep_shards=args.keep_shards,
        )

    if args.command == "balance-windows":
        print("\n=== Balance O-only / author windows ===")
        run_balance_windows(
            args.processed_dir,
            seed=args.seed,
            o_only_cap_ratio=args.o_only_cap_ratio,
            author_oversample=args.author_oversample,
            input_file=args.balance_input,
            output_file=args.balance_output,
        )

    if args.command in ("split", "all"):
        print("\n=== Story 1.4: Stratified split ===")
        run_split(
            args.processed_dir,
            args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            examples_file=args.examples_file,
        )

    if args.command == "all":
        print("\n=== Epic 1 pipeline complete ===")
        print(f"  Raw export:     {args.extracted_dir}")
        print(f"  Reports:        {reports_dir}")
        print(f"  Training data:  {args.processed_dir / 'all_examples.jsonl'}")
        print(f"  Splits:         {args.processed_dir / 'splits'}")


if __name__ == "__main__":
    main()
