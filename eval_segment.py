"""Segment-level evaluation: merge window predictions like inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import EXTRACTED_DIR
from pipeline.inference import predict_segment


def _load_doc_entries(extracted_dir: Path) -> dict[str, dict]:
    index_path = extracted_dir / "index.jsonl"
    by_doc: dict[str, dict] = {}
    with index_path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            by_doc[entry["doc_id"]] = entry
    return by_doc


def _load_segment_payload(extracted_dir: Path, entry: dict, segment_id: str) -> dict | None:
    ann_path = extracted_dir / entry["annotations_path"]
    with ann_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    for segment in payload.get("segments", []):
        if segment["segment_id"] == segment_id:
            return segment
    return None


def _gold_span_sequences(annotations: list[dict]) -> list[list[str]]:
    sequences: list[list[str]] = []
    for ann in annotations:
        prefix = ann["label"].upper()
        n_tokens = max(1, (ann["span_end"] - ann["span_start"] + 3) // 4)
        tags = [f"B-{prefix}"] + [f"I-{prefix}"] * (n_tokens - 1)
        sequences.append(tags)
    return sequences


def _pred_span_sequences(spans: list[dict]) -> list[list[str]]:
    sequences: list[list[str]] = []
    for span in spans:
        prefix = span["label"].upper()
        n_tokens = max(1, (span["span_end"] - span["span_start"] + 3) // 4)
        tags = [f"B-{prefix}"] + [f"I-{prefix}"] * (n_tokens - 1)
        sequences.append(tags)
    return sequences


def _exact_span_match(gold: list[dict], pred: list[dict]) -> tuple[int, int, int]:
    gold_set = {
        (a["label"].lower(), a["span_start"], a["span_end"]) for a in gold
    }
    pred_set = {
        (s["label"].lower(), s["span_start"], s["span_end"]) for s in pred
    }
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def collect_test_segments(
    splits_dir: Path,
    extracted_dir: Path,
) -> list[dict]:
    test_path = splits_dir / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing {test_path}")

    seen: set[tuple[str, str]] = set()
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            seen.add((row["doc_id"], row["segment_id"]))

    doc_entries = _load_doc_entries(extracted_dir)
    segments: list[dict] = []

    for doc_id, segment_id in sorted(seen):
        entry = doc_entries.get(doc_id)
        if entry is None:
            continue
        segment = _load_segment_payload(extracted_dir, entry, segment_id)
        if segment is None:
            continue
        annotations = segment.get("annotations", [])
        if not annotations:
            continue
        segments.append(
            {
                "doc_id": doc_id,
                "segment_id": segment_id,
                "text": segment["text"],
                "annotations": annotations,
            }
        )
    return segments


def evaluate_segments(
    model,
    tokenizer,
    segments: list[dict],
    device: torch.device,
) -> dict:
    total_tp = total_fp = total_fn = 0
    title_tp = title_fp = title_fn = 0
    author_tp = author_fp = author_fn = 0
    all_gold_tags: list[list[str]] = []
    all_pred_tags: list[list[str]] = []

    for seg in segments:
        gold = [
            {
                "label": a["label"],
                "span_start": a["span_start"],
                "span_end": a["span_end"],
            }
            for a in seg["annotations"]
        ]
        pred = predict_segment(model, tokenizer, seg["text"], device=device)
        pred = [
            {
                "label": s["label"],
                "span_start": s["span_start"],
                "span_end": s["span_end"],
            }
            for s in pred
        ]

        tp, fp, fn = _exact_span_match(gold, pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        g_title = [s for s in gold if s["label"] == "title"]
        p_title = [s for s in pred if s["label"] == "title"]
        ltp, lfp, lfn = _exact_span_match(g_title, p_title)
        title_tp += ltp
        title_fp += lfp
        title_fn += lfn

        g_author = [s for s in gold if s["label"] == "author"]
        p_author = [s for s in pred if s["label"] == "author"]
        ltp, lfp, lfn = _exact_span_match(g_author, p_author)
        author_tp += ltp
        author_fp += lfp
        author_fn += lfn

        all_gold_tags.extend(_gold_span_sequences(gold))
        all_pred_tags.extend(_pred_span_sequences(pred))

    def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    span_metrics = prf(total_tp, total_fp, total_fn)
    return {
        "segments_evaluated": len(segments),
        "exact_span_match": {**span_metrics, "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "exact_title": {**prf(title_tp, title_fp, title_fn), "tp": title_tp},
        "exact_author": {**prf(author_tp, author_fp, author_fn), "tp": author_tp},
        "seqeval_span_f1": float(f1_score(all_gold_tags, all_pred_tags))
        if all_gold_tags
        else 0.0,
        "seqeval_span_precision": float(precision_score(all_gold_tags, all_pred_tags))
        if all_gold_tags
        else 0.0,
        "seqeval_span_recall": float(recall_score(all_gold_tags, all_pred_tags))
        if all_gold_tags
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment-level merged NER evaluation")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/roberta_full/splits"),
    )
    parser.add_argument("--extracted-dir", type=Path, default=EXTRACTED_DIR)
    parser.add_argument(
        "--model",
        default="data/roberta_full/model/best",
        help="Local model dir or HuggingFace model id",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/roberta_full/model/segment_test_metrics.json"),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    segments = collect_test_segments(args.splits_dir, args.extracted_dir)
    print(f"Evaluating {len(segments)} annotated test segments (merged inference)...")
    metrics = evaluate_segments(model, tokenizer, segments, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
