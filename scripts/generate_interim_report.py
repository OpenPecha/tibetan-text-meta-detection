#!/usr/bin/env python3
"""Aggregate partial predictions JSONL and generate comparison report (non-destructive)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EXTRACTED_DIR
from eval_llm_segment import aggregate_title_metrics_from_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("logs/llm_title_segment_predictions.jsonl"),
    )
    parser.add_argument(
        "--interim-metrics",
        type=Path,
        default=Path("logs/llm_title_segment_metrics_interim.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("logs/llm_segment_eval_report_interim.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/llm_segment_eval_report_interim.json"),
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=EXTRACTED_DIR,
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/roberta_full/splits"),
    )
    parser.add_argument(
        "--total-segments",
        type=int,
        default=6492,
        help="Expected total test segments for progress note",
    )
    parser.add_argument("--offset-tols", type=int, nargs="+", default=[10, 50])
    args = parser.parse_args()

    if not args.predictions.is_file():
        print(f"ERROR: missing {args.predictions}", file=sys.stderr)
        sys.exit(1)

    body = aggregate_title_metrics_from_predictions(
        args.predictions,
        offset_tolerances=tuple(args.offset_tols),
        extracted_dir=args.extracted_dir,
        splits_dir=args.splits_dir,
    )
    inference_times_ms = body.pop("_inference_times_ms", [])
    if inference_times_ms:
        import statistics

        body["timing"] = {
            "total_inference_s": round(sum(inference_times_ms) / 1000.0, 2),
            "mean_ms_per_segment": round(statistics.mean(inference_times_ms), 2),
            "median_ms_per_segment": round(statistics.median(inference_times_ms), 2),
        }

    n = body["segments_evaluated"]
    payload = {
        "run_id": "tilamb_title_lora_pilot_interim",
        "eval_type": "segment_multi_metric",
        "status": "interim_partial",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": "YoLo2000/TiLamb-7B",
        "adapter": "/root/lora/tibetan-title-pilot",
        "splits_dir": str(args.splits_dir),
        "progress": {
            "completed": n,
            "total_expected": args.total_segments,
            "percent": round(100.0 * n / args.total_segments, 2) if args.total_segments else None,
        },
        "segment_metrics": body,
        "predictions_path": str(args.predictions),
        "note": "Partial run — full eval may still be in progress. Do not stop tmux.",
    }

    args.interim_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.interim_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Aggregated {n} segments -> {args.interim_metrics}")

    compare = Path(__file__).resolve().parent / "compare_segment_eval.py"
    subprocess.run(
        [
            sys.executable,
            str(compare),
            "--current",
            str(args.interim_metrics),
            "--output-md",
            str(args.output_md),
            "--output-json",
            str(args.output_json),
        ],
        check=True,
    )

    md = args.output_md.read_text(encoding="utf-8")
    banner = (
        f"> **INTERIM REPORT** — {n}/{args.total_segments} segments "
        f"({payload['progress']['percent']}%) — eval still running in tmux\n\n"
    )
    args.output_md.write_text(banner + md, encoding="utf-8")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
