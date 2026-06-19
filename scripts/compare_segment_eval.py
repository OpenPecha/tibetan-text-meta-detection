#!/usr/bin/env python3
"""Generate LLM segment eval report vs saved RoBERTa baselines (no re-run of baselines)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_BLOCKS = [
    ("exact_title", "Exact offset match"),
    ("overlap_title_iou50", "Text overlap (char IoU ≥ 50%)"),
    ("text_equal_title", "Exact title text match"),
    ("offset_relaxed_title_10", "Offset relaxed (±10 chars)"),
    ("offset_relaxed_title_50", "Offset relaxed (±50 chars)"),
]


def _metric_block(inner: dict, key: str) -> dict | None:
    block = inner.get(key)
    if block and block.get("f1") is not None:
        return block
    return None


def load_llm_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    inner = data.get("segment_metrics", data)
    exact = inner.get("exact_title")
    if exact is None:
        raise KeyError(f"No exact_title in {path}")
    return {
        "run_id": data.get("run_id", "tilamb_title_lora"),
        "base_model": data.get("base_model", ""),
        "adapter": data.get("adapter", ""),
        "splits_dir": data.get("splits_dir", ""),
        "segments_evaluated": inner.get("segments_evaluated"),
        "metrics": inner,
        "timing": inner.get("timing"),
        "eval_type": data.get("eval_type", "segment_multi_metric"),
    }


def load_roberta_baseline(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sm = data.get("segment_metrics", data)
    return {
        "run_id": data.get("run_id", path.stem),
        "base_model": data.get("base_model", ""),
        "segments_evaluated": data.get("segments_evaluated"),
        "metrics": sm,
        "hf_model": data.get("hf_model", ""),
    }


def pct(n: float | None) -> str:
    if n is None:
        return "n/a"
    return f"{n * 100:.2f}%"


def delta_pp(new: float | None, old: float | None) -> str:
    if new is None or old is None:
        return "n/a"
    d = (new - old) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f} pp"


def _row(label: str, block: dict | None) -> str:
    if not block:
        return f"| {label} | n/a | n/a | n/a | n/a |"
    return (
        f"| {label} | {pct(block.get('precision'))} | {pct(block.get('recall'))} | "
        f"**{pct(block.get('f1'))}** | "
        f"{block.get('tp', '?')} / {block.get('fp', '?')} / {block.get('fn', '?')} |"
    )


def build_report(llm: dict, baselines: list[dict]) -> str:
    lm = llm["metrics"]
    lines = [
        "# TiLamb Title LoRA — Segment Eval Report",
        "",
        "Test segments = unique `(doc_id, segment_id)` from `test.jsonl` with gold annotations.",
        "",
        "**Metrics (title spans only):**",
        "- **Exact** — `(label, span_start, span_end)` must equal gold",
        "- **Overlap IoU50** — same label, character IoU ≥ 0.5 (greedy one-to-one match)",
        "- **Text equal** — same label, extracted title string matches exactly",
        "- **Offset relaxed** — same label, start/end within ±N chars and spans overlap",
        "",
        "RoBERTa baselines are from saved `docs/metrics/*_segment.json` "
        "(re-run `eval_segment.py` to refresh overlap metrics).",
        "",
        "## TiLamb LoRA run",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Base model | `{llm.get('base_model', '')}` |",
        f"| Adapter | `{llm.get('adapter', '')}` |",
        f"| Segments evaluated | {llm.get('segments_evaluated', '?')} |",
        "",
        "| Metric | Precision | Recall | F1 | TP / FP / FN |",
        "|--------|-----------|--------|-----|--------------|",
    ]
    for key, label in METRIC_BLOCKS:
        lines.append(_row(label, _metric_block(lm, key)))

    lines.extend(
        [
            "",
            "## Comparison table (title F1)",
            "",
            "| Model | Segments | Exact F1 | Overlap IoU50 F1 | Text equal F1 | Offset ±10 F1 | Offset ±50 F1 |",
            "|-------|----------|----------|------------------|---------------|---------------|---------------|",
        ]
    )
    for b in baselines:
        bm = b["metrics"]
        lines.append(
            f"| {b['run_id']} | {b.get('segments_evaluated', '?')} | "
            f"{pct(_metric_block(bm, 'exact_title') and bm['exact_title'].get('f1'))} | "
            f"{pct(_metric_block(bm, 'overlap_title_iou50') and bm['overlap_title_iou50'].get('f1'))} | "
            f"{pct(_metric_block(bm, 'text_equal_title') and bm['text_equal_title'].get('f1'))} | "
            f"{pct(_metric_block(bm, 'offset_relaxed_title_10') and bm['offset_relaxed_title_10'].get('f1'))} | "
            f"{pct(_metric_block(bm, 'offset_relaxed_title_50') and bm['offset_relaxed_title_50'].get('f1'))} |"
        )
    lines.append(
        f"| **TiLamb title LoRA** | {llm.get('segments_evaluated', '?')} | "
        f"**{pct(_metric_block(lm, 'exact_title') and lm['exact_title'].get('f1'))}** | "
        f"**{pct(_metric_block(lm, 'overlap_title_iou50') and lm['overlap_title_iou50'].get('f1'))}** | "
        f"**{pct(_metric_block(lm, 'text_equal_title') and lm['text_equal_title'].get('f1'))}** | "
        f"**{pct(_metric_block(lm, 'offset_relaxed_title_10') and lm['offset_relaxed_title_10'].get('f1'))}** | "
        f"**{pct(_metric_block(lm, 'offset_relaxed_title_50') and lm['offset_relaxed_title_50'].get('f1'))}** |"
    )

    if llm.get("timing"):
        t = llm["timing"]
        lines.extend(
            [
                "",
                "## Inference timing (TiLamb only)",
                "",
                f"- Model load: {t.get('load_ms', '?')} ms",
                f"- Total inference: {t.get('total_inference_s', '?')} s",
                f"- Mean per segment: {t.get('mean_ms_per_segment', '?')} ms",
                f"- Median per segment: {t.get('median_ms_per_segment', '?')} ms",
                f"- P95 per segment: {t.get('p95_ms_per_segment', '?')} ms",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("logs/llm_title_segment_metrics.json"),
    )
    parser.add_argument(
        "--spsither-baseline",
        type=Path,
        default=Path("docs/metrics/spsither_balanced_segment.json"),
    )
    parser.add_argument(
        "--koichi-baseline",
        type=Path,
        default=Path("docs/metrics/koichi_balanced_segment.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/metrics/tilamb_title_lora_segment.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("logs/llm_segment_eval_report.md"),
    )
    args = parser.parse_args()

    llm = load_llm_metrics(args.current)
    baselines = []
    for path in (args.spsither_baseline, args.koichi_baseline):
        if path.is_file():
            baselines.append(load_roberta_baseline(path))

    out_json = {
        "run_id": llm["run_id"],
        "base_model": llm.get("base_model"),
        "adapter": llm.get("adapter"),
        "eval_type": llm.get("eval_type"),
        "segments_evaluated": llm.get("segments_evaluated"),
        "segment_metrics": llm["metrics"],
        "comparison_baselines": {
            b["run_id"]: {
                "segments_evaluated": b.get("segments_evaluated"),
                "exact_title_f1": b["metrics"].get("exact_title", {}).get("f1"),
                "overlap_title_iou50_f1": b["metrics"].get("overlap_title_iou50", {}).get("f1"),
                "source": str(
                    args.spsither_baseline
                    if "spsither" in b["run_id"]
                    else args.koichi_baseline
                ),
            }
            for b in baselines
        },
    }

    report = build_report(llm, baselines)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
