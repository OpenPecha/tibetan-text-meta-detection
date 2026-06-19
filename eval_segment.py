"""Segment-level evaluation: merge window predictions like inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import EXTRACTED_DIR
from eval_common import (
    collect_test_segments,
    filter_segments_by_keys,
    gold_from_annotations,
    load_segment_keys_from_predictions,
    span_eval_metrics,
)
from pipeline.inference import predict_segment


def _accumulate_metrics(totals: dict[str, dict], block: dict[str, dict]) -> None:
    for name, metrics in block.items():
        bucket = totals.setdefault(name, {"tp": 0, "fp": 0, "fn": 0})
        bucket["tp"] += metrics["tp"]
        bucket["fp"] += metrics["fp"]
        bucket["fn"] += metrics["fn"]


def _finalize_metrics(totals: dict[str, dict]) -> dict[str, dict]:
    from eval_common import prf

    out: dict[str, dict] = {}
    for name, counts in totals.items():
        out[name] = {
            **prf(counts["tp"], counts["fp"], counts["fn"]),
            **counts,
        }
    return out


def evaluate_segments(
    model,
    tokenizer,
    segments: list[dict],
    device: torch.device,
    *,
    offset_tolerances: tuple[int, ...] = (10, 50),
) -> dict:
    span_totals: dict[str, dict] = {}
    title_totals: dict[str, dict] = {}
    author_totals: dict[str, dict] = {}

    for seg in segments:
        gold = gold_from_annotations(seg["annotations"])
        pred = predict_segment(model, tokenizer, seg["text"], device=device)
        pred = [
            {
                "label": s["label"],
                "span_start": s["span_start"],
                "span_end": s["span_end"],
            }
            for s in pred
        ]

        _accumulate_metrics(
            span_totals,
            span_eval_metrics(
                gold,
                pred,
                seg["text"],
                offset_tolerances=offset_tolerances,
            ),
        )

        g_title = gold_from_annotations(seg["annotations"], label="title")
        p_title = [s for s in pred if s["label"] == "title"]
        _accumulate_metrics(
            title_totals,
            span_eval_metrics(
                g_title,
                p_title,
                seg["text"],
                offset_tolerances=offset_tolerances,
            ),
        )

        g_author = gold_from_annotations(seg["annotations"], label="author")
        p_author = [s for s in pred if s["label"] == "author"]
        _accumulate_metrics(
            author_totals,
            span_eval_metrics(
                g_author,
                p_author,
                seg["text"],
                offset_tolerances=offset_tolerances,
            ),
        )

    span_block = _finalize_metrics(span_totals)
    title_block = _finalize_metrics(title_totals)
    author_block = _finalize_metrics(author_totals)

    def _offset_exports(block: dict[str, dict], prefix: str) -> dict:
        out = {
            f"{prefix}_10": block.get("offset_relaxed_10", {}),
            f"{prefix}_50": block.get("offset_relaxed_50", {}),
        }
        out[prefix] = out[f"{prefix}_10"]
        return out

    return {
        "offset_tolerances": list(offset_tolerances),
        "exact_span_match": span_block["exact"],
        "overlap_span_iou50": span_block["overlap_iou50"],
        "overlap_span_iou80": span_block["overlap_iou80"],
        "text_equal_span": span_block["text_equal"],
        **_offset_exports(span_block, "offset_relaxed_span"),
        "exact_title": title_block["exact"],
        "overlap_title_iou50": title_block["overlap_iou50"],
        "text_equal_title": title_block["text_equal"],
        **_offset_exports(title_block, "offset_relaxed_title"),
        "exact_author": author_block["exact"],
        "overlap_author_iou50": author_block["overlap_iou50"],
        "text_equal_author": author_block["text_equal"],
        **_offset_exports(author_block, "offset_relaxed_author"),
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
    parser.add_argument(
        "--offset-tols",
        type=int,
        nargs="+",
        default=[10, 50],
        help="Start/end tolerance(s) in chars for offset-relaxed match (default: 10 50)",
    )
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=None,
        help="Evaluate only segments listed in this JSONL (same order, e.g. TiLamb interim)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="With --predictions-jsonl, evaluate at most this many segments from the file",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    segments = collect_test_segments(args.splits_dir, args.extracted_dir)
    if args.predictions_jsonl is not None:
        keys = load_segment_keys_from_predictions(args.predictions_jsonl)
        if args.limit is not None:
            keys = keys[: args.limit]
        segments = filter_segments_by_keys(segments, keys)
        print(
            f"Filtered to {len(segments)} segments from {args.predictions_jsonl}"
            + (f" (limit={args.limit})" if args.limit else "")
        )
    print(f"Evaluating {len(segments)} annotated test segments (merged inference)...")
    metrics = evaluate_segments(
        model,
        tokenizer,
        segments,
        device,
        offset_tolerances=tuple(args.offset_tols),
    )

    payload = {
        "eval_type": "segment_multi_metric",
        "segments_evaluated": len(segments),
        "segment_metrics": metrics,
    }
    if args.predictions_jsonl is not None:
        payload["subset_source"] = str(args.predictions_jsonl)
        payload["subset_limit"] = args.limit

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
