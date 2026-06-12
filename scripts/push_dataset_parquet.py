#!/usr/bin/env python3
"""Convert JSONL splits (and optional extracted docs) to HF Dataset and push as Parquet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset

DEFAULT_REPO = "ganga4364/tibetan-metadata-detector"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data" / "roberta_full" / "splits"
DEFAULT_EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
DEFAULT_README = PROJECT_ROOT / "hub" / "dataset_README.md"


def load_windows_dataset(splits_dir: Path) -> DatasetDict:
    """Memory-map RoBERTa window JSONL and return a DatasetDict."""
    paths = {
        "train": str(splits_dir / "train.jsonl"),
        "validation": str(splits_dir / "val.jsonl"),
        "test": str(splits_dir / "test.jsonl"),
    }
    for split, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {split} split: {path}")

    return load_dataset("json", data_files=paths)


DOCUMENT_FEATURES = Features(
    {
        "doc_id": Value("string"),
        "filename": Value("string"),
        "text": Value("string"),
        "annotations_json": Value("string"),
    }
)


def _iter_extracted_records(extracted_dir: Path):
    """Yield one document record at a time from index.jsonl."""
    index_path = extracted_dir / "index.jsonl"
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text_path = extracted_dir / entry["text_path"]
            ann_path = extracted_dir / entry["annotations_path"]
            if not text_path.is_file() or not ann_path.is_file():
                continue
            yield {
                "doc_id": entry["doc_id"],
                "filename": entry.get("filename", ""),
                "text": text_path.read_text(encoding="utf-8"),
                "annotations_json": ann_path.read_text(encoding="utf-8"),
            }


def load_extracted_dataset(extracted_dir: Path) -> Dataset | None:
    """Build a documents dataset from index.jsonl + text/annotation files."""
    index_path = extracted_dir / "index.jsonl"
    if not index_path.is_file():
        return None

    return Dataset.from_generator(
        _iter_extracted_records,
        gen_kwargs={"extracted_dir": extracted_dir},
        features=DOCUMENT_FEATURES,
    )


def push_to_hub(
    repo_id: str,
    windows: DatasetDict | None,
    documents: Dataset | None,
    max_shard_size: str,
    private: bool,
) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN")

    kwargs = {
        "repo_id": repo_id,
        "private": private,
        "token": token,
        "max_shard_size": max_shard_size,
    }

    if windows is not None:
        print(f"Pushing windows ({', '.join(f'{k}={len(v)}' for k, v in windows.items())})…")
        windows.push_to_hub(
            config_name="windows",
            commit_message="Upload windows config as Parquet",
            **kwargs,
        )

    if documents is not None:
        print(f"Pushing documents ({len(documents)} rows)…")
        documents.push_to_hub(
            config_name="documents",
            commit_message="Upload documents config as Parquet",
            **kwargs,
        )
    elif windows is None:
        raise RuntimeError("No dataset splits selected for upload.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--windows-only",
        action="store_true",
        help="Upload RoBERTa window splits only",
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="Upload extracted documents only (requires index.jsonl)",
    )
    args = parser.parse_args()

    if args.windows_only and args.documents_only:
        parser.error("Choose only one of --windows-only or --documents-only")

    windows = None
    documents = None

    if not args.documents_only:
        windows = load_windows_dataset(args.splits_dir)

    if not args.windows_only:
        documents = load_extracted_dataset(args.extracted_dir)
        if documents is None:
            print(f"No documents to upload under {args.extracted_dir}")
            if args.documents_only:
                raise SystemExit(1)

    push_to_hub(
        repo_id=args.repo_id,
        windows=windows,
        documents=documents,
        max_shard_size=args.max_shard_size,
        private=args.private,
    )

    if args.readme.is_file():
        from huggingface_hub import HfApi

        print(f"Uploading dataset card from {args.readme}…")
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=str(args.readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Update dataset card for Parquet configs",
        )

    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
