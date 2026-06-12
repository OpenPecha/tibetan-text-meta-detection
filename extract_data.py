"""Extract title and author span annotations from the outliner database."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psycopg2

from config import DB_CONFIG, EXTRACTED_DIR

DOCUMENT_IDS_BASE = """
SELECT d.id
FROM outliner_documents d
JOIN outliner_segments s
    ON s.document_id = d.id
    AND s.is_annotated = true
    AND (
        (s.title_span_start IS NOT NULL AND s.title_span_end IS NOT NULL)
        OR (s.author_span_start IS NOT NULL AND s.author_span_end IS NOT NULL)
    )
WHERE d.status = 'approved'
GROUP BY d.id
ORDER BY COUNT(s.id) DESC
"""

DOCUMENT_BY_ID_SQL = """
SELECT id, filename, content, LENGTH(content) AS content_length
FROM outliner_documents
WHERE id = %s;
"""

SEGMENT_ANNOTATION_SQL = """
SELECT
    id,
    segment_index,
    label,
    span_start,
    span_end,
    text,
    title,
    title_span_start,
    title_span_end,
    author,
    author_span_start,
    author_span_end
FROM outliner_segments
WHERE document_id = %s
    AND is_annotated = true
    AND (
        (title_span_start IS NOT NULL AND title_span_end IS NOT NULL)
        OR (author_span_start IS NOT NULL AND author_span_end IS NOT NULL)
    )
ORDER BY span_start ASC;
"""


def _require_db_config() -> None:
    for key in ("host", "user", "password"):
        if not DB_CONFIG[key]:
            raise ValueError(
                f"Missing BENCHMARK_DB_{key.upper()} environment variable. "
                "See .env.example for required variables."
            )


def _local_annotation(
    label: str,
    doc_start: int | None,
    doc_end: int | None,
    field_text: str | None,
    segment_text: str,
    seg_start: int,
    segment_id: str,
    segment_index: int | None,
) -> tuple[dict, dict] | None:
    """Return (segment-local ann, document-level ann) or None."""
    if doc_start is None or doc_end is None:
        return None

    local_start = doc_start - seg_start
    local_end = doc_end - seg_start
    text = segment_text[local_start:local_end]
    if not text.strip() and field_text:
        text = field_text

    local = {
        "label": label,
        "span_start": local_start,
        "span_end": local_end,
        "text": text,
    }
    doc_level = {
        "label": label,
        "span_start": doc_start,
        "span_end": doc_end,
        "text": text,
        "segment_id": segment_id,
        "segment_index": segment_index,
    }
    return local, doc_level


def _build_segment_records(content: str, rows: list[tuple]) -> tuple[list[dict], list[dict]]:
    """Build segment-level records and flat document-level annotation list."""
    segments: list[dict] = []
    flat_annotations: list[dict] = []

    for row in rows:
        (
            segment_id,
            segment_index,
            segment_label,
            seg_start,
            seg_end,
            segment_text,
            title,
            title_start,
            title_end,
            author,
            author_start,
            author_end,
        ) = row

        text = segment_text or content[seg_start:seg_end]
        local_anns: list[dict] = []

        for label, start, end, field in (
            ("title", title_start, title_end, title),
            ("author", author_start, author_end, author),
        ):
            result = _local_annotation(
                label,
                start,
                end,
                field,
                text,
                seg_start,
                segment_id,
                segment_index,
            )
            if result is None:
                continue
            local, doc_level = result
            local_anns.append(local)
            flat_annotations.append(doc_level)

        if not local_anns:
            continue

        segments.append(
            {
                "segment_id": segment_id,
                "segment_index": segment_index,
                "segment_label": segment_label,
                "span_start": seg_start,
                "span_end": seg_end,
                "text": text,
                "annotations": local_anns,
            }
        )

    flat_annotations.sort(key=lambda a: (a["span_start"], a["label"]))
    return segments, flat_annotations


def _prepare_output_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    texts_dir = output_dir / "texts"
    annotations_dir = output_dir / "annotations"
    texts_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    return texts_dir, annotations_dir, output_dir / "index.jsonl"


def _write_document_files(
    output_dir: Path,
    doc_id: str,
    filename: str,
    content: str,
    content_length: int,
    segments: list[dict],
    flat_annotations: list[dict],
) -> dict:
    """Write one document's text and segment annotations to separate files."""
    texts_dir, annotations_dir, _ = _prepare_output_dirs(output_dir)

    text_rel = Path("texts") / f"{doc_id}.txt"
    ann_rel = Path("annotations") / f"{doc_id}.json"

    text_path = output_dir / text_rel
    ann_path = output_dir / ann_rel

    text_path.write_text(content, encoding="utf-8")
    ann_path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "filename": filename,
                "segments": segments,
                "annotations": flat_annotations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    has_title = any(a["label"] == "title" for a in flat_annotations)
    has_author = any(a["label"] == "author" for a in flat_annotations)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "content_length": content_length,
        "text_path": text_rel.as_posix(),
        "annotations_path": ann_rel.as_posix(),
        "segment_count": len(segments),
        "annotation_count": len(flat_annotations),
        "has_title": has_title,
        "has_author": has_author,
    }


def _connect():
    _require_db_config()
    return psycopg2.connect(
        **DB_CONFIG,
        connect_timeout=30,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )


def _load_completed_doc_ids(output_dir: Path) -> set[str]:
    """Return doc IDs with valid text + annotation files on disk."""
    texts_dir = output_dir / "texts"
    anns_dir = output_dir / "annotations"
    if not texts_dir.exists():
        return set()

    completed: set[str] = set()
    for text_path in texts_dir.glob("*.txt"):
        doc_id = text_path.stem
        ann_path = anns_dir / f"{doc_id}.json"
        if (
            ann_path.exists()
            and text_path.stat().st_size > 0
            and ann_path.stat().st_size > 0
        ):
            try:
                json.loads(ann_path.read_text(encoding="utf-8"))
                completed.add(doc_id)
            except json.JSONDecodeError:
                continue
    return completed


def _rebuild_index(output_dir: Path) -> int:
    """Rebuild a deduplicated index.jsonl from files on disk."""
    anns_dir = output_dir / "annotations"
    index_path = output_dir / "index.jsonl"
    entries: list[dict] = []

    for ann_path in sorted(anns_dir.glob("*.json")):
        doc_id = ann_path.stem
        text_path = output_dir / "texts" / f"{doc_id}.txt"
        if not text_path.exists() or text_path.stat().st_size == 0:
            continue
        try:
            payload = json.loads(ann_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        segments = payload.get("segments", [])
        flat = payload.get("annotations", [])
        content = text_path.read_text(encoding="utf-8")
        entries.append(
            {
                "doc_id": doc_id,
                "filename": payload.get("filename", doc_id),
                "content_length": len(content),
                "text_path": f"texts/{doc_id}.txt",
                "annotations_path": f"annotations/{doc_id}.json",
                "segment_count": len(segments),
                "annotation_count": len(flat),
                "has_title": any(a["label"] == "title" for a in flat),
                "has_author": any(a["label"] == "author" for a in flat),
            }
        )

    with index_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(entries)


def _acquire_lock(
    output_dir: Path,
    worker_id: int | None = None,
    num_workers: int = 1,
) -> Path:
    """Prevent multiple extract processes from using the same worker slot."""
    if num_workers > 1:
        if worker_id is None:
            raise ValueError("--worker-id is required when --num-workers > 1")
        lock_path = output_dir / f".extract.lock.worker{worker_id}"
    else:
        lock_path = output_dir / ".extract.lock"
    if lock_path.exists():
        raise RuntimeError(
            f"Lock file exists at {lock_path}. "
            "Another extraction may be running — stop it first or delete the lock file."
        )
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return lock_path


def _db_call_with_retry(label: str, fn, max_retries: int = 5):
    """Run a DB callable with retries on transient connection errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except psycopg2.OperationalError as exc:
            if attempt >= max_retries - 1:
                raise
            wait = min(60, 2 ** attempt)
            print(f"  [retry {attempt + 1}/{max_retries}] DB error during {label}. Waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed DB operation: {label}")


def count_matching_documents() -> int:
    """Return how many documents match the extraction filters."""
    def _count():
        conn = _connect()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM ({DOCUMENT_IDS_BASE}) sub;")
            return int(cur.fetchone()[0])
        finally:
            conn.close()

    return _db_call_with_retry("count documents", _count)


def _document_ids_cache_path(output_dir: Path) -> Path:
    return output_dir / "document_ids.json"


def _load_or_fetch_document_ids(
    output_dir: Path,
    limit: int | None = None,
) -> list[str]:
    """Load cached document IDs, or fetch once from DB and cache locally."""
    cache_path = _document_ids_cache_path(output_dir)
    if cache_path.exists():
        all_ids = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Using cached document IDs: {len(all_ids)}")
        return all_ids[:limit] if limit is not None else all_ids

    def _fetch():
        conn = _connect()
        try:
            cur = conn.cursor()
            if limit is not None:
                cur.execute(DOCUMENT_IDS_BASE + " LIMIT %s;", (limit,))
            else:
                cur.execute(DOCUMENT_IDS_BASE + ";")
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    all_ids = _db_call_with_retry("fetch document IDs", _fetch)
    cache_path.write_text(json.dumps(all_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Cached {len(all_ids)} document IDs to {cache_path}")
    return all_ids


def _extract_one_document(
    doc_id: str,
    max_retries: int = 5,
) -> tuple[dict, list[dict], list[dict]] | None:
    """Fetch and parse one document with retries on transient DB errors."""
    for attempt in range(max_retries):
        try:
            conn = _connect()
            try:
                cur = conn.cursor()
                cur.execute(DOCUMENT_BY_ID_SQL, (doc_id,))
                row = cur.fetchone()
                if not row:
                    return None

                doc = {
                    "id": row[0],
                    "filename": row[1],
                    "content": row[2] or "",
                    "content_length": row[3],
                }

                cur.execute(SEGMENT_ANNOTATION_SQL, (doc_id,))
                rows = cur.fetchall()
                segments, flat_annotations = _build_segment_records(doc["content"], rows)
                return doc, segments, flat_annotations
            finally:
                conn.close()
        except psycopg2.OperationalError as exc:
            if attempt >= max_retries - 1:
                raise
            wait = min(60, 2 ** attempt)
            print(
                f"  [retry {attempt + 1}/{max_retries}] DB error for {doc_id}: "
                f"{exc}. Waiting {wait}s..."
            )
            time.sleep(wait)
    return None


def extract_metadata_data(
    num_samples: int | None = None,
    output_dir: Path | None = None,
    resume: bool = True,
    batch_size: int = 50,
    worker_id: int | None = None,
    num_workers: int = 1,
) -> dict:
    """Extract documents to modular files under output_dir.

    Uses per-document DB fetches and supports resume after connection failures.
    With num_workers > 1, each worker processes every Nth document ID.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if num_workers > 1:
        if worker_id is None:
            raise ValueError("--worker-id is required when --num-workers > 1")
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError(
                f"--worker-id must be between 0 and {num_workers - 1}, got {worker_id}"
            )

    output_dir = output_dir or EXTRACTED_DIR
    texts_dir, annotations_dir, index_path = _prepare_output_dirs(output_dir)

    lock_path = _acquire_lock(output_dir, worker_id, num_workers)
    try:
        return _extract_metadata_data_inner(
            num_samples,
            output_dir,
            texts_dir,
            annotations_dir,
            index_path,
            resume,
            worker_id=worker_id,
            num_workers=num_workers,
        )
    finally:
        lock_path.unlink(missing_ok=True)
        if num_workers <= 1:
            rebuilt = _rebuild_index(output_dir)
            print(f"Rebuilt index: {rebuilt} unique documents")
        else:
            print(
                f"Worker {worker_id} finished. "
                "Run --rebuild-index after all workers complete."
            )


def _extract_metadata_data_inner(
    num_samples: int | None,
    output_dir: Path,
    texts_dir: Path,
    annotations_dir: Path,
    index_path: Path,
    resume: bool,
    worker_id: int | None = None,
    num_workers: int = 1,
) -> dict:
    completed = _load_completed_doc_ids(output_dir) if resume else set()
    if completed:
        print(f"Resuming: {len(completed)} documents already exported")

    all_ids = _load_or_fetch_document_ids(output_dir, limit=num_samples)
    if num_workers > 1:
        all_ids = [
            doc_id
            for idx, doc_id in enumerate(all_ids)
            if idx % num_workers == worker_id
        ]
        print(
            f"Worker {worker_id}/{num_workers}: "
            f"assigned {len(all_ids)} documents"
        )

    total_target = len(all_ids)
    if num_samples is not None:
        total_target = min(total_target, num_samples)

    if num_workers > 1:
        stats_path = output_dir / f"stats.worker{worker_id}.json"
    else:
        stats_path = output_dir / "stats.json"
    if resume and stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["documents"] = len(completed)
        stats["skipped_resume"] = len(completed)
    else:
        stats = {
            "documents": len(completed),
            "annotations": 0,
            "title_annotations": 0,
            "author_annotations": 0,
            "documents_missing_title": 0,
            "documents_missing_author": 0,
            "output_dir": str(output_dir),
            "skipped_resume": len(completed),
        }

    print(f"Target documents: {total_target}")
    print(f"Output directory: {output_dir}")

    pending_ids = [doc_id for doc_id in all_ids if doc_id not in completed]
    if num_samples is not None:
        remaining_slots = max(0, num_samples - len(completed))
        pending_ids = pending_ids[:remaining_slots]

    print(f"Pending documents: {len(pending_ids)}")

    extracted_this_run = 0

    for idx, doc_id in enumerate(pending_ids, start=1):
        if doc_id in completed:
            continue

        result = _extract_one_document(doc_id)
        if result is None:
            print(f"  [skip] document not found: {doc_id}")
            continue

        doc, segments, flat_annotations = result
        index_entry = _write_document_files(
            output_dir,
            doc["id"],
            doc["filename"],
            doc["content"],
            doc["content_length"],
            segments,
            flat_annotations,
        )
        completed.add(doc_id)
        extracted_this_run += 1

        if not index_entry["has_title"]:
            stats["documents_missing_title"] += 1
        if not index_entry["has_author"]:
            stats["documents_missing_author"] += 1

        stats["documents"] += 1
        stats["annotations"] += len(flat_annotations)
        stats["title_annotations"] += sum(
            1 for a in flat_annotations if a["label"] == "title"
        )
        stats["author_annotations"] += sum(
            1 for a in flat_annotations if a["label"] == "author"
        )

        print(
            f"  [{idx}/{len(pending_ids)}] {doc['filename']}: "
            f"{doc['content_length']:,} chars, {len(segments)} segments, "
            f"{len(flat_annotations)} annotations"
        )

    stats["extracted_this_run"] = extracted_this_run
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved {stats['documents']} documents under {output_dir}")
    print(f"  New this run: {extracted_this_run}")
    print(f"  Index:       {index_path}")
    print(f"  Texts:       {texts_dir}/")
    print(f"  Annotations: {annotations_dir}/")
    print(f"  Stats:       {stats_path}")
    print(
        f"Annotations: {stats['annotations']} total "
        f"({stats['title_annotations']} title, {stats['author_annotations']} author)"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract title/author span annotations from outliner PostgreSQL DB",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Extract every matching document (no LIMIT)",
    )
    group.add_argument(
        "--num-samples",
        type=int,
        metavar="N",
        help="Extract at most N documents (ordered by annotated segment count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching document count only; do not write files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXTRACTED_DIR,
        help=f"Output directory for modular files (default: {EXTRACTED_DIR})",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-extract all documents (do not skip existing index entries)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Document ID page size for batched extraction (default: 50)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild index.jsonl from files on disk (deduplicate) and exit",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help="Worker index (0-based) for parallel extraction",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel extraction workers (default: 1)",
    )
    args = parser.parse_args()

    if args.rebuild_index:
        n = _rebuild_index(args.output_dir)
        print(f"Rebuilt index with {n} unique documents")
        return

    if args.dry_run:
        n = count_matching_documents()
        print(f"Matching documents: {n}")
        print(
            "Filters: status=approved, is_annotated=true, "
            "title_span or author_span present"
        )
        return

    num_samples = args.num_samples if args.num_samples is not None else None
    extract_metadata_data(
        num_samples=num_samples,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        batch_size=args.batch_size,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
