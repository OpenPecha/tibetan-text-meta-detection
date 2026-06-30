"""Row-level benchmark evaluation on pilot SFT test JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from eval_common import prf, span_eval_metrics
from llm_sft.prompts import INSTRUCTIONS, TITLE_INSTRUCTION
from pipeline.inference import predict_segment

GENERATIVE_KINDS = frozenset({
    "tilamb",
    "tilamb_lora",
    "alpaca",
    "qwen",
    "gemma4",
    "qwen36_27b",
    "deepseek_r1_14b",
})
ROBERTA_KINDS = frozenset({"koichi"})


def load_test_rows(
    test_jsonl: Path,
    meta_jsonl: Path | None,
    *,
    default_instruction: str = TITLE_INSTRUCTION,
) -> list[dict[str, Any]]:
    meta_by_line: list[dict[str, Any]] = []
    if meta_jsonl and meta_jsonl.is_file():
        with meta_jsonl.open(encoding="utf-8") as f:
            meta_by_line = [json.loads(line) for line in f if line.strip()]

    rows: list[dict[str, Any]] = []
    with test_jsonl.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = meta_by_line[line_idx] if line_idx < len(meta_by_line) else {}
            row_id = meta.get("id") or f"line_{line_idx}"
            rows.append(
                {
                    "row_id": row_id,
                    "line_idx": line_idx,
                    "doc_id": meta.get("doc_id", ""),
                    "segment_id": meta.get("segment_id", ""),
                    "instruction": row.get("instruction", default_instruction),
                    "input": row["input"],
                    "output": row["output"],
                    "meta": meta,
                }
            )
    return rows


def gold_spans_from_output(output: Any, label: str = "title") -> list[dict[str, Any]]:
    if isinstance(output, str):
        output = json.loads(output)
    spans = output.get("spans", []) if isinstance(output, dict) else []
    return [
        {
            "label": label,
            "span_start": int(s["start"]),
            "span_end": int(s["end"]),
            "text": s.get("text", ""),
        }
        for s in spans
        if "start" in s and "end" in s
    ]


def load_completed_row_ids(predictions_path: Path) -> set[str]:
    if not predictions_path.is_file():
        return set()
    done: set[str] = set()
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(row["row_id"])
    return done


def _accumulate_metrics(totals: dict[str, dict], block: dict[str, dict]) -> None:
    for name, metrics in block.items():
        bucket = totals.setdefault(name, {"tp": 0, "fp": 0, "fn": 0})
        bucket["tp"] += metrics["tp"]
        bucket["fp"] += metrics["fp"]
        bucket["fn"] += metrics["fn"]


def _finalize_metrics(totals: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, counts in totals.items():
        out[name] = {
            **prf(counts["tp"], counts["fp"], counts["fn"]),
            **counts,
        }
    return out


def aggregate_row_metrics(
    predictions_path: Path,
    *,
    offset_tolerances: tuple[int, ...] = (10, 50),
) -> dict[str, Any]:
    totals: dict[str, dict] = {}
    inference_times_ms: list[float] = []
    parse_failures = 0
    rows_evaluated = 0

    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows_evaluated += 1
            if rec.get("parse_ok") is False:
                parse_failures += 1
            if rec.get("inference_ms") is not None:
                inference_times_ms.append(float(rec["inference_ms"]))

            gold = rec.get("gold_spans", [])
            pred = rec.get("pred_spans", [])
            text = rec.get("input", "")
            _accumulate_metrics(
                totals,
                span_eval_metrics(
                    gold,
                    pred,
                    text,
                    offset_tolerances=offset_tolerances,
                ),
            )

    metrics = _finalize_metrics(totals)
    return {
        "rows_evaluated": rows_evaluated,
        "parse_failures": parse_failures,
        "parse_fail_rate": parse_failures / rows_evaluated if rows_evaluated else 0.0,
        "exact_title": metrics.get("exact", {}),
        "overlap_title_iou50": metrics.get("overlap_iou50", {}),
        "overlap_title_iou80": metrics.get("overlap_iou80", {}),
        "text_equal_title": metrics.get("text_equal", {}),
        "offset_relaxed_title_10": metrics.get("offset_relaxed_10", {}),
        "offset_relaxed_title_50": metrics.get("offset_relaxed_50", {}),
        "_inference_times_ms": inference_times_ms,
    }


def aggregate_segment_dedup_metrics(
    predictions_path: Path,
    *,
    offset_tolerances: tuple[int, ...] = (10, 50),
) -> dict[str, Any]:
    """Best row per (doc_id, segment_id) by exact-title TP on that row."""
    by_segment: dict[tuple[str, str], list[dict]] = {}
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec.get("doc_id", ""), rec.get("segment_id", ""))
            if not key[0]:
                continue
            by_segment.setdefault(key, []).append(rec)

    totals: dict[str, dict] = {}
    for recs in by_segment.values():
        best = recs[0]
        for rec in recs[1:]:
            gold = rec.get("gold_spans", [])
            pred = rec.get("pred_spans", [])
            text = rec.get("input", "")
            block = span_eval_metrics(gold, pred, text, offset_tolerances=offset_tolerances)
            if block["exact"]["tp"] > 0:
                best = rec
        gold = best.get("gold_spans", [])
        pred = best.get("pred_spans", [])
        text = best.get("input", "")
        _accumulate_metrics(
            totals,
            span_eval_metrics(gold, pred, text, offset_tolerances=offset_tolerances),
        )

    metrics = _finalize_metrics(totals)
    return {
        "segments_evaluated": len(by_segment),
        "exact_title": metrics.get("exact", {}),
        "overlap_title_iou50": metrics.get("overlap_iou50", {}),
        "text_equal_title": metrics.get("text_equal", {}),
        "offset_relaxed_title_10": metrics.get("offset_relaxed_10", {}),
        "offset_relaxed_title_50": metrics.get("offset_relaxed_50", {}),
    }


def load_roberta_model(checkpoint: str, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_koichi_titles(
    model: Any,
    tokenizer: Any,
    input_text: str,
    device: torch.device,
    label: str = "title",
) -> list[dict[str, Any]]:
    pred = predict_segment(model, tokenizer, input_text, device=device)
    return [
        {
            "label": label,
            "span_start": int(s["span_start"]),
            "span_end": int(s["span_end"]),
        }
        for s in pred
        if s.get("label") == label
    ]


def evaluate_rows(
    *,
    model_kind: str,
    rows: list[dict[str, Any]],
    predictions_path: Path,
    resume: bool,
    limit: int | None,
    offset_tolerances: tuple[int, ...],
    checkpoint: str,
    base_model: str | None,
    adapter: str | None,
    load_in_4bit: bool,
    max_new_tokens: int,
    label: str = "title",
) -> dict[str, Any]:
    done = load_completed_row_ids(predictions_path) if resume else set()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_file = predictions_path.open("a", encoding="utf-8")

    load_ms = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_spec = None
    roberta_model = None
    roberta_tokenizer = None
    gen_model = None
    gen_tokenizer = None

    t0 = time.perf_counter()
    if model_kind in ROBERTA_KINDS:
        roberta_model, roberta_tokenizer = load_roberta_model(checkpoint, device)
    elif model_kind in GENERATIVE_KINDS:
        from llm_sft.model_backends import load_generative_model, predict_input_text, spec_for_kind

        gen_spec = spec_for_kind(
            model_kind,
            base_model=base_model,
            adapter_path=adapter or None,
            load_in_4bit=load_in_4bit,
        )
        gen_model, gen_tokenizer = load_generative_model(gen_spec)
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    load_ms = (time.perf_counter() - t0) * 1000.0

    evaluated = 0
    skipped = 0
    try:
        for row in rows:
            if row["row_id"] in done:
                skipped += 1
                continue
            if limit is not None and evaluated >= limit:
                break

            input_text = row["input"]
            gold = gold_spans_from_output(row["output"], label=label)
            t1 = time.perf_counter()

            parse_ok = True
            raw_response = None
            truncated = False
            if model_kind in ROBERTA_KINDS:
                pred = predict_koichi_titles(
                    roberta_model,
                    roberta_tokenizer,
                    input_text,
                    device,
                    label=label,
                )
            else:
                assert gen_model is not None and gen_tokenizer is not None and gen_spec
                pred, gen_meta = predict_input_text(
                    gen_model,
                    gen_tokenizer,
                    row["instruction"],
                    input_text,
                    family=gen_spec.family,
                    max_new_tokens=max_new_tokens,
                    label=label,
                )
                parse_ok = gen_meta["parse_ok"]
                raw_response = gen_meta.get("raw_response")
                truncated = gen_meta.get("truncated", False)

            inference_ms = (time.perf_counter() - t1) * 1000.0
            record = {
                "row_id": row["row_id"],
                "line_idx": row["line_idx"],
                "doc_id": row["doc_id"],
                "segment_id": row["segment_id"],
                "input": input_text,
                "gold_spans": gold,
                "pred_spans": pred,
                "inference_ms": round(inference_ms, 2),
                "parse_ok": parse_ok,
                "truncated": truncated,
            }
            if raw_response is not None:
                record["raw_response"] = raw_response
            pred_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            pred_file.flush()
            evaluated += 1
    finally:
        pred_file.close()

    agg = aggregate_row_metrics(
        predictions_path,
        offset_tolerances=offset_tolerances,
    )
    seg_agg = aggregate_segment_dedup_metrics(
        predictions_path,
        offset_tolerances=offset_tolerances,
    )
    inference_times_ms = agg.pop("_inference_times_ms", [])
    total_s = sum(inference_times_ms) / 1000.0 if inference_times_ms else 0.0

    return {
        "rows_evaluated": agg["rows_evaluated"],
        "rows_in_this_run": evaluated,
        "rows_skipped_resume": skipped,
        "parse_failures": agg["parse_failures"],
        "parse_fail_rate": agg["parse_fail_rate"],
        "exact_title": agg["exact_title"],
        "overlap_title_iou50": agg["overlap_title_iou50"],
        "overlap_title_iou80": agg["overlap_title_iou80"],
        "text_equal_title": agg["text_equal_title"],
        "offset_relaxed_title_10": agg["offset_relaxed_title_10"],
        "offset_relaxed_title_50": agg["offset_relaxed_title_50"],
        "segment_dedup": seg_agg,
        "timing": {
            "load_ms": round(load_ms, 2),
            "total_inference_s": round(total_s, 2),
            "mean_ms_per_row": round(statistics.mean(inference_times_ms), 2)
            if inference_times_ms
            else 0.0,
            "median_ms_per_row": round(statistics.median(inference_times_ms), 2)
            if inference_times_ms
            else 0.0,
        },
        "max_new_tokens": max_new_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-kind",
        required=True,
        choices=sorted(ROBERTA_KINDS | GENERATIVE_KINDS),
    )
    parser.add_argument(
        "--task",
        choices=("title", "author"),
        default="title",
        help="Detection task; selects instruction, span label, and default paths.",
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=None,
        help="Defaults to data/llm_sft_pilot_10pct/<task>/test.jsonl",
    )
    parser.add_argument(
        "--meta-jsonl",
        type=Path,
        default=None,
        help="Defaults to data/llm_sft_pilot_10pct/<task>/test_meta.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/koichi-ner",
        help="Koichi RoBERTa checkpoint directory",
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter path (tilamb_lora only)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Defaults to logs/benchmark_<kind>_predictions.jsonl",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Defaults to logs/benchmark_<kind>_metrics.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--offset-tols",
        type=int,
        nargs="+",
        default=[10, 50],
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generative decode limit (default: llm_sft.model_backends.DEFAULT_MAX_NEW_TOKENS)",
    )
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        from llm_sft.model_backends import DEFAULT_MAX_NEW_TOKENS

        max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    task = args.task
    instruction = INSTRUCTIONS[task]
    # Title artifacts keep their historical (taskless) filenames; author artifacts
    # get a "_author" suffix so the two tasks never overwrite each other.
    suffix = "" if task == "title" else f"_{task}"
    test_jsonl = args.test_jsonl or Path(f"data/llm_sft_pilot_10pct/{task}/test.jsonl")
    meta_jsonl = args.meta_jsonl or Path(f"data/llm_sft_pilot_10pct/{task}/test_meta.jsonl")
    predictions = args.predictions or Path(
        f"logs/benchmark_{args.model_kind}{suffix}_predictions.jsonl"
    )
    metrics_out = args.metrics_out or Path(
        f"logs/benchmark_{args.model_kind}{suffix}_metrics.json"
    )

    rows = load_test_rows(test_jsonl, meta_jsonl, default_instruction=instruction)
    print(f"Loaded {len(rows)} test rows from {test_jsonl}")

    base_model = args.base_model
    if base_model is None and args.model_kind in GENERATIVE_KINDS:
        from llm_sft.model_backends import spec_for_kind

        base_model = spec_for_kind(args.model_kind).base_model

    metrics_body = evaluate_rows(
        model_kind=args.model_kind,
        rows=rows,
        predictions_path=predictions,
        resume=args.resume,
        limit=args.limit,
        offset_tolerances=tuple(args.offset_tols),
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        adapter=args.adapter,
        load_in_4bit=not args.no_4bit,
        max_new_tokens=max_new_tokens,
        label=task,
    )

    result = {
        "run_id": f"benchmark_{args.model_kind}_pilot_{task}",
        "eval_type": "row_multi_metric",
        "task": task,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_kind": args.model_kind,
        "max_new_tokens": max_new_tokens,
        "test_jsonl": str(test_jsonl),
        "meta_jsonl": str(meta_jsonl),
        "checkpoint": args.checkpoint if args.model_kind in ROBERTA_KINDS else None,
        "base_model": base_model,
        "adapter": args.adapter if args.model_kind == "tilamb_lora" else None,
        "row_metrics": metrics_body,
        "predictions_path": str(predictions),
    }

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {metrics_out}")
    print(f"Predictions: {predictions}")


if __name__ == "__main__":
    main()
