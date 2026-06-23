#!/usr/bin/env python3
"""Recompute benchmark metrics from saved predictions (no inference).

Focuses on offset-first diagnostics: start-only, end-only, both-boundary hit rates,
and boundary error distributions. Reads benchmark_*_predictions.jsonl files.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_benchmark_rows import aggregate_row_metrics, aggregate_segment_dedup_metrics
from eval_common import prf, row_offset_diagnostics, span_eval_counts

DEFAULT_TOLS = (10, 50)


def _pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.2f}%"


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f)


def summarize_offset_diagnostics(
    predictions_path: Path,
    *,
    tolerances: tuple[int, ...] = DEFAULT_TOLS,
) -> dict:
    """Row-level offset hit rates and boundary error stats (best IoU pair per row)."""
    rows_total = 0
    rows_with_gold = 0
    rows_with_pred = 0
    rows_paired = 0
    start_errs: list[float] = []
    end_errs: list[float] = []
    ious: list[float] = []
    hit: dict[str, int] = {}
    for tol in tolerances:
        for key in (
            f"start_within_{tol}",
            f"end_within_{tol}",
            f"both_within_{tol}",
            f"both_within_{tol}_overlap",
        ):
            hit[key] = 0

    micro: dict[str, tuple[int, int, int]] = {}
    for tol in tolerances:
        for key in (
            f"offset_start_{tol}",
            f"offset_end_{tol}",
            f"offset_both_{tol}",
            f"offset_relaxed_{tol}",
        ):
            micro[key] = (0, 0, 0)

    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows_total += 1
            gold = rec.get("gold_spans", [])
            pred = rec.get("pred_spans", [])
            if gold:
                rows_with_gold += 1
            if pred:
                rows_with_pred += 1

            diag = row_offset_diagnostics(gold, pred, tolerances=tolerances)
            if diag["paired"]:
                rows_paired += 1
                start_errs.append(float(diag["start_abs_err"]))
                end_errs.append(float(diag["end_abs_err"]))
                ious.append(float(diag["char_iou"]))
                for tol in tolerances:
                    for key in hit:
                        if key.endswith(f"_{tol}") or key.endswith(f"_{tol}_overlap"):
                            if diag.get(key):
                                hit[key] += 1

            for name, counts in span_eval_counts(
                gold,
                pred,
                rec.get("input", ""),
                offset_tolerances=tolerances,
            ).items():
                if name.startswith("offset_"):
                    tp, fp, fn = counts
                    cur = micro.setdefault(name, (0, 0, 0))
                    micro[name] = (cur[0] + tp, cur[1] + fp, cur[2] + fn)

    def hit_rate(key: str, denom: int) -> float | None:
        if denom == 0:
            return None
        return hit[key] / denom

    tol_summary: dict[str, dict] = {}
    for tol in tolerances:
        tol_summary[str(tol)] = {
            "row_hit_rate_start": hit_rate(f"start_within_{tol}", rows_paired),
            "row_hit_rate_end": hit_rate(f"end_within_{tol}", rows_paired),
            "row_hit_rate_both": hit_rate(f"both_within_{tol}", rows_paired),
            "row_hit_rate_both_overlap": hit_rate(
                f"both_within_{tol}_overlap", rows_paired
            ),
            "micro_f1_start_only": prf(*micro[f"offset_start_{tol}"])["f1"],
            "micro_f1_end_only": prf(*micro[f"offset_end_{tol}"])["f1"],
            "micro_f1_both_no_overlap_req": prf(*micro[f"offset_both_{tol}"])["f1"],
            "micro_f1_both_overlap_req": prf(*micro[f"offset_relaxed_{tol}"])["f1"],
        }

    return {
        "predictions_path": str(predictions_path),
        "rows_total": rows_total,
        "rows_with_gold": rows_with_gold,
        "rows_with_pred": rows_with_pred,
        "rows_paired": rows_paired,
        "char_iou_mean": _mean(ious),
        "char_iou_median": _median(ious),
        "char_iou_p90": _percentile(ious, 90),
        "start_abs_err_mean": _mean(start_errs),
        "start_abs_err_median": _median(start_errs),
        "start_abs_err_p90": _percentile(start_errs, 90),
        "end_abs_err_mean": _mean(end_errs),
        "end_abs_err_median": _median(end_errs),
        "end_abs_err_p90": _percentile(end_errs, 90),
        "tolerances": tol_summary,
    }


def _model_name_from_path(path: Path) -> str:
    stem = path.name
    if stem.startswith("benchmark_") and stem.endswith("_predictions.jsonl"):
        return stem[len("benchmark_") : -len("_predictions.jsonl")]
    return stem


def render_markdown(summaries: list[dict], *, tolerances: tuple[int, ...]) -> str:
    lines = [
        "# Benchmark offset diagnostics (recomputed from predictions)",
        "",
        "No inference — metrics derived from saved `gold_spans` / `pred_spans`.",
        "",
        "**Row hit rate** = share of paired rows (best IoU same-label match) where the boundary test passes.",
        "**Micro F1** = greedy span matching aggregated over all rows (legacy view).",
        "",
    ]
    for tol in tolerances:
        lines.append(f"## Tolerance ±{tol} chars")
        lines.append("")
        lines.append(
            "| Model | Rows | Start hit | End hit | Both hit | Both+overlap | "
            "MAE start | MAE end | Median IoU |"
        )
        lines.append("|-------|------|-----------|---------|----------|--------------|"
                     "----------|---------|------------|")
        for s in summaries:
            t = s["offset"]["tolerances"][str(tol)]
            o = s["offset"]
            mae_s = o["start_abs_err_mean"]
            mae_e = o["end_abs_err_mean"]
            med_iou = o["char_iou_median"]
            lines.append(
                f"| {s['model']} | {o['rows_paired']} | "
                f"{_pct(t['row_hit_rate_start'])} | {_pct(t['row_hit_rate_end'])} | "
                f"{_pct(t['row_hit_rate_both'])} | {_pct(t['row_hit_rate_both_overlap'])} | "
                f"{mae_s:.1f} | {mae_e:.1f} | {_pct(med_iou)} |"
                if o["rows_paired"] and mae_s is not None and mae_e is not None
                else f"| {s['model']} | {o['rows_paired']} | "
                f"{_pct(t['row_hit_rate_start'])} | {_pct(t['row_hit_rate_end'])} | "
                f"{_pct(t['row_hit_rate_both'])} | {_pct(t['row_hit_rate_both_overlap'])} | "
                f"— | — | — |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=ROOT / "logs",
        help="Directory containing benchmark_*_predictions.jsonl",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="*",
        help="Explicit prediction files (default: glob in metrics-dir)",
    )
    parser.add_argument(
        "--offset-tols",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOLS),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "docs/metrics/benchmark_offset_diagnostics.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "docs/metrics/benchmark_offset_diagnostics.md",
    )
    args = parser.parse_args()
    tolerances = tuple(args.offset_tols)

    if args.predictions:
        pred_paths = list(args.predictions)
    else:
        pred_paths = sorted(args.metrics_dir.glob("benchmark_*_predictions.jsonl"))

    if not pred_paths:
        raise SystemExit(f"No prediction files under {args.metrics_dir}")

    summaries: list[dict] = []
    for pred_path in pred_paths:
        if pred_path.stat().st_size == 0:
            continue
        model = _model_name_from_path(pred_path)
        offset = summarize_offset_diagnostics(pred_path, tolerances=tolerances)
        standard = aggregate_row_metrics(pred_path, offset_tolerances=tolerances)
        seg = aggregate_segment_dedup_metrics(pred_path, offset_tolerances=tolerances)
        summaries.append(
            {
                "model": model,
                "offset": offset,
                "standard_row_metrics": {
                    k: v for k, v in standard.items() if not k.startswith("_")
                },
                "segment_dedup": seg,
            }
        )
        print(f"{model}: paired={offset['rows_paired']}/{offset['rows_total']} rows")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"tolerances": list(tolerances), "models": summaries}, indent=2),
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(summaries, tolerances=tolerances), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
