#!/usr/bin/env python3
"""Compare TiLamb interim vs Koichi RoBERTa on the same segment subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(n: float | None) -> str:
    if n is None:
        return "n/a"
    return f"{n * 100:.2f}%"


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sm = data.get("segment_metrics", data)
    return {
        "run_id": data.get("run_id", path.stem),
        "segments_evaluated": data.get("segments_evaluated", sm.get("segments_evaluated")),
        "subset_source": data.get("subset_source"),
        "metrics": sm,
    }


def row(label: str, block: dict | None) -> str:
    if not block:
        return f"| {label} | n/a | n/a | n/a |"
    return (
        f"| {label} | {pct(block.get('precision'))} | {pct(block.get('recall'))} | "
        f"**{pct(block.get('f1'))}** |"
    )


def build_report(tilamb: dict, koichi: dict) -> str:
    tm = tilamb["metrics"]
    km = koichi["metrics"]
    n = tilamb.get("segments_evaluated")
    lines = [
        "# TiLamb vs Koichi — Same Segment Subset",
        "",
        f"Segments: **{n}** (first rows from `{tilamb.get('subset_source', 'TiLamb predictions JSONL')}`)",
        "",
        "## Side-by-side (title metrics)",
        "",
        "| Metric | Koichi RoBERTa P | R | F1 | TiLamb LoRA P | R | F1 |",
        "|--------|------------------|---|---|---------------|---|---|",
    ]
    blocks = [
        ("exact_title", "Exact offset"),
        ("overlap_title_iou50", "Overlap IoU50"),
        ("text_equal_title", "Text equal"),
        ("offset_relaxed_title_10", "Offset ±10"),
        ("offset_relaxed_title_50", "Offset ±50"),
    ]
    for key, label in blocks:
        t = tm.get(key, {})
        k = km.get(key, {})
        lines.append(
            f"| {label} | {pct(k.get('precision'))} | {pct(k.get('recall'))} | "
            f"**{pct(k.get('f1'))}** | {pct(t.get('precision'))} | {pct(t.get('recall'))} | "
            f"**{pct(t.get('f1'))}** |"
        )
    lines.extend(["", "## Koichi detail", ""])
    for key, label in blocks:
        lines.append(row(label, km.get(key)))
    lines.extend(["", "## TiLamb detail", ""])
    for key, label in blocks:
        lines.append(row(label, tm.get(key)))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tilamb",
        type=Path,
        default=Path("logs/llm_title_segment_metrics_interim.json"),
    )
    parser.add_argument(
        "--koichi",
        type=Path,
        default=Path("logs/koichi_subset_717_segment_metrics.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("logs/tilamb_vs_koichi_subset_717.md"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/tilamb_vs_koichi_subset_717.json"),
    )
    args = parser.parse_args()

    tilamb_raw = json.loads(args.tilamb.read_text(encoding="utf-8"))
    tilamb = load_metrics(args.tilamb)
    tilamb["subset_source"] = tilamb_raw.get("predictions_path")
    koichi = load_metrics(args.koichi)

    report = build_report(tilamb, koichi)
    payload = {
        "segments_evaluated": tilamb.get("segments_evaluated"),
        "tilamb_source": str(args.tilamb),
        "koichi_source": str(args.koichi),
        "tilamb_title": {k: tilamb["metrics"].get(k) for k in (
            "exact_title", "overlap_title_iou50", "text_equal_title",
            "offset_relaxed_title_10", "offset_relaxed_title_50",
        )},
        "koichi_title": {k: koichi["metrics"].get(k) for k in (
            "exact_title", "overlap_title_iou50", "text_equal_title",
            "offset_relaxed_title_10", "offset_relaxed_title_50",
        )},
    }
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report, encoding="utf-8")
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report)
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
