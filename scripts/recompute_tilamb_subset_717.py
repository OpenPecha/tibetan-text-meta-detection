#!/usr/bin/env python3
"""Recompute TiLamb subset metrics with offset tolerances 10 and 50."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import EXTRACTED_DIR
from eval_llm_segment import aggregate_title_metrics_from_predictions

pred = Path("logs/llm_title_segment_predictions_717.jsonl")
body = aggregate_title_metrics_from_predictions(
    pred,
    offset_tolerances=(10, 50),
    extracted_dir=EXTRACTED_DIR,
    splits_dir=Path("data/roberta_full/splits"),
)
payload = {
    "run_id": "tilamb_title_lora_pilot_subset_717",
    "eval_type": "segment_multi_metric",
    "segments_evaluated": body["segments_evaluated"],
    "predictions_path": str(pred),
    "segment_metrics": body,
}
out = Path("logs/llm_title_segment_metrics_subset_717.json")
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
for k in (
    "exact_title",
    "overlap_title_iou50",
    "offset_relaxed_title_10",
    "offset_relaxed_title_50",
):
    b = body[k]
    print(
        f"{k}: F1={b.get('f1', 0) * 100:.2f}% "
        f"P={b.get('precision', 0) * 100:.2f}% R={b.get('recall', 0) * 100:.2f}%"
    )
print(f"Wrote {out}")
