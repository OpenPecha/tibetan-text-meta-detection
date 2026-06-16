#!/usr/bin/env python3
"""Download ganga4364/tibetan-metadata-extracted into local data/extracted layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DEFAULT_REPO = "ganga4364/tibetan-metadata-extracted"


def pull(repo_id: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts_dir = output_dir / "texts"
    annotations_dir = output_dir / "annotations"
    texts_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo_id, split="train")
    index_path = output_dir / "index.jsonl"
    doc_count = 0

    with index_path.open("w", encoding="utf-8") as index_f:
        for row in ds:
            doc_id = row["doc_id"]
            filename = row["filename"]
            text = row["text"]
            annotations_json = row["annotations_json"]
            if isinstance(annotations_json, dict):
                payload = annotations_json
            else:
                payload = json.loads(annotations_json)

            text_path = texts_dir / f"{doc_id}.txt"
            ann_path = annotations_dir / f"{doc_id}.json"
            text_path.write_text(text, encoding="utf-8")
            ann_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            segments = payload.get("segments", [])
            flat = payload.get("annotations", [])
            has_title = any(a.get("label") == "title" for a in flat)
            has_author = any(a.get("label") == "author" for a in flat)
            index_f.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "filename": filename,
                        "content_length": len(text),
                        "text_path": f"texts/{doc_id}.txt",
                        "annotations_path": f"annotations/{doc_id}.json",
                        "segment_count": len(segments),
                        "annotation_count": len(flat),
                        "has_title": has_title,
                        "has_author": has_author,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            doc_count += 1
            if doc_count % 200 == 0:
                print(f"  wrote {doc_count} documents...")

    stats = {"documents": doc_count, "repo_id": repo_id}
    (output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/extracted"),
        help="Target extracted directory (default: data/extracted)",
    )
    args = parser.parse_args()

    print(f"Pulling {args.repo_id} -> {args.output_dir}")
    stats = pull(args.repo_id, args.output_dir)
    print(f"Done: {stats['documents']} documents in {args.output_dir}")


if __name__ == "__main__":
    main()
