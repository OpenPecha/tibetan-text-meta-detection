#!/usr/bin/env python3
"""Aggregate pilot benchmark metrics into a leaderboard report."""

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

MODEL_ORDER = [
    "koichi",
    "tilamb",
    "tilamb_lora",
    "alpaca",
    "qwen",
    "gemma4",
    "qwen36_27b",
    "deepseek_r1_14b",
]


def pct(n: float | None) -> str:
    if n is None:
        return "n/a"
    return f"{n * 100:.2f}%"


def _metric_block(inner: dict, key: str) -> dict | None:
    block = inner.get(key)
    if block and block.get("f1") is not None:
        return block
    return None


def load_benchmark_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    inner = data.get("row_metrics", data)
    return {
        "run_id": data.get("run_id", path.stem),
        "model_kind": data.get("model_kind", path.stem.replace("benchmark_", "").replace("_metrics", "")),
        "base_model": data.get("base_model", ""),
        "adapter": data.get("adapter", ""),
        "checkpoint": data.get("checkpoint", ""),
        "rows_evaluated": inner.get("rows_evaluated"),
        "metrics": inner,
        "timing": inner.get("timing"),
        "segment_dedup": inner.get("segment_dedup"),
        "source": str(path),
    }


def _row(label: str, block: dict | None) -> str:
    if not block:
        return f"| {label} | n/a | n/a | n/a | n/a |"
    return (
        f"| {label} | {pct(block.get('precision'))} | {pct(block.get('recall'))} | "
        f"**{pct(block.get('f1'))}** | "
        f"{block.get('tp', '?')} / {block.get('fp', '?')} / {block.get('fn', '?')} |"
    )


def model_label(entry: dict) -> str:
    kind = entry.get("model_kind", entry["run_id"])
    if kind == "koichi":
        return "Koichi RoBERTa NER"
    if kind == "tilamb":
        return "TiLamb-7B (zero-shot)"
    if kind == "tilamb_lora":
        return "TiLamb pilot LoRA"
    if kind == "alpaca":
        return "Tibetan Alpaca 7B"
    if kind == "qwen":
        return "Qwen2.5-7B-Instruct"
    if kind == "gemma4":
        return "Gemma 4 E4B-it"
    if kind == "qwen36_27b":
        return "Qwen3.6-27B"
    if kind == "deepseek_r1_14b":
        return "DeepSeek-R1-Distill-Qwen-14B"
    return kind


def build_report(entries: list[dict], task: str = "title") -> str:
    task_title = task.capitalize()
    lines = [
        f"# Tibetan Metadata Benchmark — Pilot {task_title} Test",
        "",
        "Primary metric scope: **row-level** evaluation on "
        f"`ganga4364/tibetan-metadata-llm-sft` → `{task}/test.jsonl`.",
        "",
        "Each model sees the same cropped `input` text; gold spans are crop-relative "
        "from the SFT `output` JSON.",
        "",
        f"## Leaderboard ({task} F1)",
        "",
        "| Model | Rows | Exact F1 | Overlap IoU50 F1 | Text equal F1 | Offset ±10 F1 | Offset ±50 F1 | Parse fail | Mean ms/row |",
        "|-------|------|----------|------------------|---------------|---------------|---------------|------------|-------------|",
    ]
    for entry in entries:
        m = entry["metrics"]
        timing = entry.get("timing") or {}
        lines.append(
            f"| {model_label(entry)} | {entry.get('rows_evaluated', '?')} | "
            f"{pct(_metric_block(m, 'exact_title') and m['exact_title'].get('f1'))} | "
            f"{pct(_metric_block(m, 'overlap_title_iou50') and m['overlap_title_iou50'].get('f1'))} | "
            f"{pct(_metric_block(m, 'text_equal_title') and m['text_equal_title'].get('f1'))} | "
            f"{pct(_metric_block(m, 'offset_relaxed_title_10') and m['offset_relaxed_title_10'].get('f1'))} | "
            f"{pct(_metric_block(m, 'offset_relaxed_title_50') and m['offset_relaxed_title_50'].get('f1'))} | "
            f"{pct(m.get('parse_fail_rate'))} | "
            f"{timing.get('mean_ms_per_row', 'n/a')} |"
        )

    lines.extend(["", "## Per-model detail", ""])
    for entry in entries:
        m = entry["metrics"]
        lines.extend(
            [
                f"### {model_label(entry)}",
                "",
                f"- Run ID: `{entry['run_id']}`",
                f"- Rows evaluated: {entry.get('rows_evaluated', '?')}",
            ]
        )
        if entry.get("base_model"):
            lines.append(f"- Base model: `{entry['base_model']}`")
        if entry.get("adapter"):
            lines.append(f"- Adapter: `{entry['adapter']}`")
        if entry.get("checkpoint"):
            lines.append(f"- Checkpoint: `{entry['checkpoint']}`")
        lines.extend(
            [
                "",
                "| Metric | Precision | Recall | F1 | TP / FP / FN |",
                "|--------|-----------|--------|-----|--------------|",
            ]
        )
        for key, label in METRIC_BLOCKS:
            lines.append(_row(label, _metric_block(m, key)))

        seg = entry.get("segment_dedup")
        if seg:
            lines.extend(
                [
                    "",
                    f"**Segment-dedup view** ({seg.get('segments_evaluated', '?')} unique segments, "
                    "best row per segment by exact match):",
                    "",
                    f"- Overlap IoU50 F1: {pct(_metric_block(seg, 'overlap_title_iou50') and seg['overlap_title_iou50'].get('f1'))}",
                    f"- Offset ±50 F1: {pct(_metric_block(seg, 'offset_relaxed_title_50') and seg['offset_relaxed_title_50'].get('f1'))}",
                ]
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("title", "author"),
        default="title",
        help="Detection task; selects metric filenames, wording, and default outputs.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("logs"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to docs/metrics/benchmark_pilot_<task>.md",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to docs/metrics/benchmark_pilot_<task>.json",
    )
    args = parser.parse_args()

    task = args.task
    output_md = args.output_md or Path(f"docs/metrics/benchmark_pilot_{task}.md")
    output_json = args.output_json or Path(f"docs/metrics/benchmark_pilot_{task}.json")

    by_kind: dict[str, dict] = {}
    for path in sorted(args.metrics_dir.glob("benchmark_*_metrics.json")):
        is_author = path.name.endswith("_author_metrics.json")
        if (task == "author") != is_author:
            continue
        entry = load_benchmark_metrics(path)
        by_kind[entry["model_kind"]] = entry

    entries = [by_kind[k] for k in MODEL_ORDER if k in by_kind]
    for kind, entry in by_kind.items():
        if kind not in MODEL_ORDER:
            entries.append(entry)

    if not entries:
        raise SystemExit(
            f"No {task} benchmark_*_metrics.json files in {args.metrics_dir}"
        )

    report = build_report(entries, task=task)
    out_json = {
        "benchmark": f"pilot_{task}_test",
        "models": [
            {
                "model_kind": e["model_kind"],
                "run_id": e["run_id"],
                "rows_evaluated": e.get("rows_evaluated"),
                "row_metrics": e["metrics"],
                "segment_dedup": e.get("segment_dedup"),
                "source": e["source"],
            }
            for e in entries
        ],
    }

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    output_json.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(report)
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
