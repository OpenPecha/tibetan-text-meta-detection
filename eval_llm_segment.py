"""Segment-level evaluation for TiLamb title/author LoRA models."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from config import EXTRACTED_DIR
from eval_common import (
    collect_test_segments,
    compare_title_spans,
    gold_from_annotations,
    load_completed_keys,
    prf,
    span_eval_counts,
    span_eval_metrics,
)
from llm_sft.inference import (
    CUTOFF_LEN,
    load_model_and_tokenizer,
    llm_spans_to_eval,
    predict_segment_detailed,
)
from llm_sft.prompts import INSTRUCTIONS

SEP = "=" * 80
SEGMENT_TOKEN_BUDGET = 3584
MAX_NEW_TOKENS = 256


def _safe_name(doc_id: str, segment_id: str) -> str:
    return f"{doc_id}__{segment_id}".replace("/", "_").replace("\\", "_")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def build_token_stats(
    *,
    instruction: str,
    segment_chars: int,
    segment_tokens: int,
    windows: list,
    model_max_position_embeddings: int | None,
) -> dict:
    prompt_tokens = sum(w.prompt_tokens for w in windows)
    output_tokens = sum(w.output_tokens for w in windows)
    prompt_chars = sum(len(w.prompt) for w in windows)
    effective_max = min(CUTOFF_LEN, model_max_position_embeddings or CUTOFF_LEN)
    return {
        "training_cutoff_len": CUTOFF_LEN,
        "segment_token_budget": SEGMENT_TOKEN_BUDGET,
        "max_new_tokens_configured": MAX_NEW_TOKENS,
        "model_max_position_embeddings": model_max_position_embeddings,
        "effective_max_context_length": effective_max,
        "instruction_chars": len(instruction),
        "segment_chars_before_crop": segment_chars,
        "segment_chars_after_crop": segment_chars,
        "segment_tokens_before_crop": segment_tokens,
        "segment_tokens_after_crop": segment_tokens,
        "segment_was_cropped": any(len(w.crop_text) < segment_chars for w in windows),
        "num_windows": len(windows),
        "prompt_chars_total": prompt_chars,
        "prompt_tokens_input_total": prompt_tokens,
        "output_chars_total": sum(len(w.raw_response) for w in windows),
        "output_tokens_total": output_tokens,
        "total_sequence_tokens_max_window": max(
            (w.prompt_tokens + w.output_tokens for w in windows),
            default=0,
        ),
        "remaining_context_after_prompt_min": min(
            (effective_max - w.prompt_tokens for w in windows),
            default=effective_max,
        ),
    }


def write_segment_detail(
    detail_dir: Path,
    *,
    meta: dict,
    token_stats: dict,
    instruction: str,
    segment_text: str,
    pred_detail,
    gold_title: list[dict],
    pred_eval: list[dict],
    comparison: dict,
    inference_ms: float,
) -> None:
    detail_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(meta["doc_id"], meta["segment_id"])
    log_path = detail_dir / f"{stem}.log"
    json_path = detail_dir / f"{stem}.json"

    lines = [
        SEP,
        "TIBETAN TITLE LoRA SEGMENT EVAL — DETAIL LOG",
        SEP,
        "",
        "--- RUN METADATA ---",
        json.dumps(meta, ensure_ascii=False, indent=2),
        "",
        "--- TOKEN & CONTEXT STATS ---",
        json.dumps({**token_stats, "inference_ms": inference_ms}, ensure_ascii=False, indent=2),
        "",
        "--- INSTRUCTION ---",
        instruction,
        "",
        f"--- SEGMENT TEXT ({len(segment_text)} chars) ---",
        segment_text,
        "",
    ]
    for i, window in enumerate(pred_detail.windows):
        lines.extend(
            [
                f"--- WINDOW {i} (char_offset={window.char_offset}, "
                f"crop_chars={len(window.crop_text)}, "
                f"prompt_tokens={window.prompt_tokens}, "
                f"output_tokens={window.output_tokens}) ---",
                f"--- PROMPT ---",
                window.prompt,
                "",
                f"--- RAW RESPONSE ---",
                window.raw_response,
                "",
            ]
        )
    lines.extend(
        [
            "--- MERGED PREDICTION ---",
            json.dumps({"spans": pred_detail.spans}, ensure_ascii=False, indent=2),
            "",
            "--- GOLD TITLE ---",
            json.dumps(gold_title, ensure_ascii=False, indent=2),
            "",
            "--- PRED TITLE (eval format) ---",
            json.dumps(pred_eval, ensure_ascii=False, indent=2),
            "",
            "--- SPAN COMPARISON ---",
            json.dumps(comparison, ensure_ascii=False, indent=2),
            "",
            SEP,
            "END OF LOG",
            SEP,
        ]
    )
    log_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "meta": meta,
        "token_stats": {**token_stats, "inference_ms": inference_ms},
        "instruction": instruction,
        "segment_text": segment_text,
        "windows": [
            {
                "char_offset": w.char_offset,
                "crop_text": w.crop_text,
                "prompt": w.prompt,
                "prompt_tokens": w.prompt_tokens,
                "output_tokens": w.output_tokens,
                "raw_response": w.raw_response,
                "parsed": w.parsed,
            }
            for w in pred_detail.windows
        ],
        "merged_spans": pred_detail.spans,
        "gold_title": gold_title,
        "pred_title": pred_eval,
        "comparison": comparison,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_segment_log(
    *,
    seg: dict,
    row: dict,
    token_stats: dict,
    pred_detail,
    detail_path: Path | None,
) -> str:
    lines = [
        SEP,
        f"SEGMENT {row['doc_id']}:{row['segment_id']}  "
        f"(#{row.get('index', '?')}, inference_ms={row['inference_ms']})",
        SEP,
        f"segment_chars={row['segment_chars']}  segment_tokens={row['segment_tokens']}  "
        f"windows={row['num_windows']} (first-window-only)",
        f"prompt_tokens={row['prompt_tokens']}  output_tokens={row['output_tokens']}",
        "",
        "--- GOLD TITLE ---",
        json.dumps(row["gold_title"], ensure_ascii=False, indent=2),
        "",
        "--- PRED TITLE ---",
        json.dumps(row["pred_title"], ensure_ascii=False, indent=2),
        "",
        "--- RAW RESPONSE (per window) ---",
    ]
    for i, w in enumerate(pred_detail.windows):
        lines.append(f"window {i}: {w.raw_response!r}")
    lines.extend(
        [
            "",
            "--- COMPARISON ---",
            json.dumps(row["comparison"], ensure_ascii=False, indent=2),
            f"tp={row['tp']} fp={row['fp']} fn={row['fn']}",
        ]
    )
    if detail_path:
        lines.append(f"detail_log={detail_path}")
    lines.append("")
    return "\n".join(lines)


def append_run_log(run_log: Path | None, text: str) -> None:
    if run_log is None:
        return
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _title_metric_exports(metrics: dict[str, dict]) -> dict:
    out = {
        "exact_title": metrics.get("exact", {}),
        "overlap_title_iou50": metrics.get("overlap_iou50", {}),
        "overlap_title_iou80": metrics.get("overlap_iou80", {}),
        "text_equal_title": metrics.get("text_equal", {}),
        "offset_relaxed_title_10": metrics.get("offset_relaxed_10", {}),
        "offset_relaxed_title_50": metrics.get("offset_relaxed_50", {}),
    }
    out["offset_relaxed_title"] = out["offset_relaxed_title_10"]
    return out


def aggregate_title_metrics_from_predictions(
    predictions_path: Path,
    *,
    offset_tolerances: tuple[int, ...] = (10, 50),
    extracted_dir: Path | None = None,
    splits_dir: Path | None = None,
) -> dict:
    """Sum tp/fp/fn from prediction rows; recompute overlap metrics when missing."""
    segment_texts: dict[tuple[str, str], str] = {}
    if extracted_dir is not None and splits_dir is not None:
        for seg in collect_test_segments(splits_dir, extracted_dir):
            segment_texts[(seg["doc_id"], seg["segment_id"])] = seg["text"]

    totals: dict[str, dict] = {}
    inference_times_ms: list[float] = []
    n = 0
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            if "inference_ms" in row:
                inference_times_ms.append(float(row["inference_ms"]))

            if "title_eval" in row:
                for name, counts in row["title_eval"].items():
                    bucket = totals.setdefault(name, {"tp": 0, "fp": 0, "fn": 0})
                    bucket["tp"] += int(counts["tp"])
                    bucket["fp"] += int(counts["fp"])
                    bucket["fn"] += int(counts["fn"])
                continue

            gold = row.get("gold_title", [])
            pred = row.get("pred_title", [])
            text = row.get("segment_text") or segment_texts.get(
                (row["doc_id"], row["segment_id"])
            )
            counts = span_eval_counts(
                gold,
                pred,
                text,
                offset_tolerances=offset_tolerances,
            )
            for name, (tp, fp, fn) in counts.items():
                bucket = totals.setdefault(name, {"tp": 0, "fp": 0, "fn": 0})
                bucket["tp"] += tp
                bucket["fp"] += fp
                bucket["fn"] += fn

    metrics = {
        name: {**prf(v["tp"], v["fp"], v["fn"]), **v} for name, v in totals.items()
    }
    return {
        "segments_evaluated": n,
        "offset_tolerances": list(offset_tolerances),
        **_title_metric_exports(metrics),
        "timing": {},
        "_inference_times_ms": inference_times_ms,
    }


def evaluate_llm_segments(
    *,
    model,
    tokenizer,
    segments: list[dict],
    task: str,
    adapter: str,
    base_model: str,
    predictions_path: Path,
    detail_dir: Path | None,
    resume: bool,
    limit: int | None,
    max_input_tokens: int,
    stride_tokens: int,
    offset_tolerances: tuple[int, ...] = (10, 50),
    extracted_dir: Path | None = None,
    splits_dir: Path | None = None,
    run_log: Path | None = None,
) -> dict:
    done = load_completed_keys(predictions_path) if resume else set()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pred_file = predictions_path.open("a", encoding="utf-8")

    load_ms = 0.0
    evaluated = 0
    skipped = 0

    model_max_pos = getattr(model.config, "max_position_embeddings", None)
    instruction = INSTRUCTIONS[task]
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    try:
        for seg in segments:
            key = (seg["doc_id"], seg["segment_id"])
            if key in done:
                skipped += 1
                continue
            if limit is not None and evaluated >= limit:
                break

            gold_title = gold_from_annotations(seg["annotations"], label=task)
            t0 = time.perf_counter()
            pred_detail = predict_segment_detailed(
                model,
                tokenizer,
                task=task,
                segment=seg["text"],
                max_input_tokens=max_input_tokens,
                stride_tokens=stride_tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
            )
            inference_ms = (time.perf_counter() - t0) * 1000.0
            inference_times_ms.append(inference_ms)

            pred_eval = llm_spans_to_eval(pred_detail.spans, label=task)
            title_eval = span_eval_counts(
                gold_title,
                pred_eval,
                seg["text"],
                offset_tolerances=offset_tolerances,
            )
            tp, fp, fn = title_eval["exact"]
            comparison = compare_title_spans(
                gold_title,
                pred_detail.spans,
                text=seg["text"],
            )

            token_stats = build_token_stats(
                instruction=instruction,
                segment_chars=pred_detail.segment_chars,
                segment_tokens=pred_detail.segment_tokens,
                windows=pred_detail.windows,
                model_max_position_embeddings=model_max_pos,
            )

            row = {
                "doc_id": seg["doc_id"],
                "segment_id": seg["segment_id"],
                "index": evaluated + 1,
                "task": task,
                "inference_ms": round(inference_ms, 2),
                "segment_chars": pred_detail.segment_chars,
                "segment_tokens": pred_detail.segment_tokens,
                "num_windows": pred_detail.num_windows,
                "prompt_tokens": token_stats["prompt_tokens_input_total"],
                "output_tokens": token_stats["output_tokens_total"],
                "gold_title": gold_title,
                "pred_title": pred_eval,
                "pred_spans": pred_detail.spans,
                "title_eval": {
                    name: {"tp": a, "fp": b, "fn": c}
                    for name, (a, b, c) in title_eval.items()
                },
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "comparison": comparison,
            }
            pred_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            pred_file.flush()
            done.add(key)
            evaluated += 1

            detail_path = None
            if detail_dir is not None:
                write_segment_detail(
                    detail_dir,
                    meta={
                        "doc_id": seg["doc_id"],
                        "segment_id": seg["segment_id"],
                        "task": task,
                        "base_model": base_model,
                        "adapter": adapter,
                        "gpu_name": gpu_name,
                        "cuda_available": torch.cuda.is_available(),
                    },
                    token_stats=token_stats,
                    instruction=instruction,
                    segment_text=seg["text"],
                    pred_detail=pred_detail,
                    gold_title=gold_title,
                    pred_eval=pred_eval,
                    comparison=comparison,
                    inference_ms=inference_ms,
                )
                detail_path = detail_dir / f"{_safe_name(seg['doc_id'], seg['segment_id'])}.log"

            segment_log = format_segment_log(
                seg=seg,
                row=row,
                token_stats=token_stats,
                pred_detail=pred_detail,
                detail_path=detail_path,
            )
            print(segment_log, flush=True)
            append_run_log(run_log, segment_log)
    finally:
        pred_file.close()

    # Aggregate from full predictions file (supports resume + legacy rows)
    agg = aggregate_title_metrics_from_predictions(
        predictions_path,
        offset_tolerances=offset_tolerances,
        extracted_dir=extracted_dir,
        splits_dir=splits_dir,
    )
    total_evaluated = agg["segments_evaluated"]
    inference_times_ms = agg.pop("_inference_times_ms", [])

    total_s = sum(inference_times_ms) / 1000.0 if inference_times_ms else 0.0
    timing = {
        "load_ms": round(load_ms, 2),
        "total_inference_s": round(total_s, 2),
        "segments_in_this_run": evaluated,
        "segments_skipped_resume": skipped,
        "mean_ms_per_segment": round(statistics.mean(inference_times_ms), 2)
        if inference_times_ms
        else 0.0,
        "median_ms_per_segment": round(statistics.median(inference_times_ms), 2)
        if inference_times_ms
        else 0.0,
        "p95_ms_per_segment": round(_percentile(inference_times_ms, 95), 2)
        if inference_times_ms
        else 0.0,
    }

    agg["timing"] = timing
    return {
        "segments_evaluated": total_evaluated,
        "offset_tolerances": list(offset_tolerances),
        **{k: v for k, v in agg.items() if k not in ("segments_evaluated", "timing")},
        "timing": timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment-level TiLamb LoRA evaluation")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/roberta_full/splits"),
    )
    parser.add_argument("--extracted-dir", type=Path, default=EXTRACTED_DIR)
    parser.add_argument("--base-model", default="YoLo2000/TiLamb-7B")
    parser.add_argument("--adapter", default="/root/lora/tibetan-title-pilot")
    parser.add_argument("--task", choices=["title", "author"], default="title")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/llm_title_segment_metrics.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("logs/llm_title_segment_predictions.jsonl"),
    )
    parser.add_argument(
        "--detail-dir",
        type=Path,
        default=Path("logs/llm_title_segment_details"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=SEGMENT_TOKEN_BUDGET)
    parser.add_argument("--stride-tokens", type=int, default=3000)
    parser.add_argument(
        "--offset-tols",
        type=int,
        nargs="+",
        default=[10, 50],
    )
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--no-detail", action="store_true")
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path("logs/llm_segment_eval_run.log"),
        help="Append human-readable log after each segment",
    )
    args = parser.parse_args()

    segments = collect_test_segments(args.splits_dir, args.extracted_dir)
    print(f"Found {len(segments)} annotated test segments")
    if args.limit:
        print(f"Limiting to {args.limit} new segments")

    print(f"Loading model base={args.base_model} adapter={args.adapter}")
    t0 = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
    )
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Model loaded in {load_ms:.0f} ms, cuda={torch.cuda.is_available()}")

    metrics_body = evaluate_llm_segments(
        model=model,
        tokenizer=tokenizer,
        segments=segments,
        task=args.task,
        adapter=args.adapter,
        base_model=args.base_model,
        predictions_path=args.predictions,
        detail_dir=None if args.no_detail else args.detail_dir,
        resume=args.resume,
        limit=args.limit,
        max_input_tokens=args.max_input_tokens,
        stride_tokens=args.stride_tokens,
        offset_tolerances=tuple(args.offset_tols),
        extracted_dir=args.extracted_dir,
        splits_dir=args.splits_dir,
        run_log=args.run_log,
    )
    metrics_body["timing"]["load_ms"] = round(load_ms, 2)

    result = {
        "run_id": "tilamb_title_lora_pilot",
        "eval_type": "segment_multi_metric",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter": args.adapter,
        "task": args.task,
        "splits_dir": str(args.splits_dir),
        "extracted_dir": str(args.extracted_dir),
        "segment_metrics": metrics_body,
        "predictions_path": str(args.predictions),
        "detail_dir": str(args.detail_dir) if not args.no_detail else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")
    print(f"Predictions: {args.predictions}")
    print(f"Run log: {args.run_log}")


if __name__ == "__main__":
    main()
