#!/usr/bin/env python3
"""Push extracted documents to a separate HF dataset repo as Parquet (streaming shards)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_dataset_parquet import (  # noqa: E402
    DEFAULT_EXTRACTED_DIR,
    _iter_extracted_records,
)

DEFAULT_REPO = "ganga4364/tibetan-metadata-extracted"
DEFAULT_README = Path(__file__).resolve().parents[1] / "hub" / "extracted_dataset_README.md"
DOCS_PER_SHARD = 75


def _count_index(extracted_dir: Path) -> int:
    index_path = extracted_dir / "index.jsonl"
    if not index_path.is_file():
        return 0
    with index_path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def push_streaming(
    repo_id: str,
    extracted_dir: Path,
    token: str,
    private: bool,
    docs_per_shard: int,
) -> int:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    total = _count_index(extracted_dir)
    if total == 0:
        raise SystemExit(f"No documents under {extracted_dir}")

    n_shards = (total + docs_per_shard - 1) // docs_per_shard
    print(f"Uploading {total} documents in {n_shards} Parquet shards to {repo_id}")

    shard_idx = 0
    batch: list[dict] = []
    uploaded = 0

    for record in _iter_extracted_records(extracted_dir):
        batch.append(record)
        if len(batch) < docs_per_shard:
            continue

        uploaded += _upload_shard(
            api, repo_id, batch, shard_idx, n_shards, token
        )
        shard_idx += 1
        batch = []
        print(f"  uploaded {uploaded}/{total}")

    if batch:
        uploaded += _upload_shard(
            api, repo_id, batch, shard_idx, n_shards, token
        )
        print(f"  uploaded {uploaded}/{total}")

    return uploaded


def _upload_shard(
    api: HfApi,
    repo_id: str,
    batch: list[dict],
    shard_idx: int,
    n_shards: int,
    token: str,
) -> int:
    table = pa.Table.from_pylist(
        batch,
        schema=pa.schema(
            [
                ("doc_id", pa.string()),
                ("filename", pa.string()),
                ("text", pa.string()),
                ("annotations_json", pa.string()),
            ]
        ),
    )
    path_in_repo = f"train-{shard_idx:05d}-of-{n_shards:05d}.parquet"
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pq.write_table(table, tmp_path, compression="snappy")
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {path_in_repo}",
            token=token,
        )
    finally:
        os.unlink(tmp_path)
    return len(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--docs-per-shard", type=int, default=DOCS_PER_SHARD)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN")

    n = push_streaming(
        repo_id=args.repo_id,
        extracted_dir=args.extracted_dir,
        token=token,
        private=args.private,
        docs_per_shard=args.docs_per_shard,
    )

    if args.readme.is_file():
        HfApi(token=token).upload_file(
            path_or_fileobj=str(args.readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )

    print(f"Done: {n} documents -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
