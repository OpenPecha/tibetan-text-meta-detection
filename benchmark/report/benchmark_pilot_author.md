# Tibetan Metadata Benchmark — Pilot Author Test

Primary metric scope: **row-level** evaluation on `ganga4364/tibetan-metadata-llm-sft` → `author/test.jsonl`.

Each model sees the same cropped `input` text; gold spans are crop-relative from the SFT `output` JSON.

## Leaderboard (author F1)

| Model | Rows | Exact F1 | Overlap IoU50 F1 | Text equal F1 | Offset ±10 F1 | Offset ±50 F1 | Parse fail | Mean ms/row |
|-------|------|----------|------------------|---------------|---------------|---------------|------------|-------------|
| TiLamb pilot LoRA | 688 | 0.00% | 1.39% | 0.00% | 0.93% | 2.94% | 0.00% | 4033.97 |

## Per-model detail

### TiLamb pilot LoRA

- Run ID: `benchmark_tilamb_lora_pilot_author`
- Rows evaluated: 688
- Base model: `YoLo2000/TiLamb-7B`
- Adapter: `ganga4364/tibetan-metadata-author-tilamb-lora-pilot`

| Metric | Precision | Recall | F1 | TP / FP / FN |
|--------|-----------|--------|-----|--------------|
| Exact offset match | 0.00% | 0.00% | **0.00%** | 0 / 685 / 609 |
| Text overlap (char IoU ≥ 50%) | 1.31% | 1.48% | **1.39%** | 9 / 676 / 600 |
| Exact title text match | 0.00% | 0.00% | **0.00%** | 0 / 685 / 609 |
| Offset relaxed (±10 chars) | 0.88% | 0.99% | **0.93%** | 6 / 679 / 603 |
| Offset relaxed (±50 chars) | 2.77% | 3.12% | **2.94%** | 19 / 666 / 590 |

**Segment-dedup view** (642 unique segments, best row per segment by exact match):

- Overlap IoU50 F1: 1.49%
- Offset ±50 F1: 3.14%

