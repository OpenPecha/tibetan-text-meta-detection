# Tibetan Metadata Benchmark — Pilot Title Test

Primary metric scope: **row-level** evaluation on `ganga4364/tibetan-metadata-llm-sft` → `title/test.jsonl`.

Each model sees the same cropped `input` text; gold spans are crop-relative from the SFT `output` JSON.

## Leaderboard (title F1)

| Model | Rows | Exact F1 | Overlap IoU50 F1 | Text equal F1 | Offset ±10 F1 | Offset ±50 F1 | Parse fail | Mean ms/row |
|-------|------|----------|------------------|---------------|---------------|---------------|------------|-------------|
| Koichi RoBERTa NER | 769 | 10.08% | 35.21% | 10.08% | 30.73% | 37.66% | 0.00% | 138.18 |
| TiLamb-7B (zero-shot) | 769 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 8436.7 |
| TiLamb pilot LoRA | 769 | 1.83% | 58.89% | 2.11% | 40.90% | 62.97% | 0.00% | 37440.34 |
| Qwen2.5-7B-Instruct | 769 | 0.00% | 15.04% | 0.00% | 8.01% | 26.56% | 3.38% | 2126.88 |
| Gemma 4 E4B-it | 769 | 0.00% | 20.70% | 0.00% | 7.30% | 46.58% | 3.12% | 9228.66 |

## Per-model detail

### Koichi RoBERTa NER

- Run ID: `benchmark_koichi_pilot_title`
- Rows evaluated: 769
- Checkpoint: `models/koichi-ner`

| Metric | Precision | Recall | F1 | TP / FP / FN |
|--------|-----------|--------|-----|--------------|
| Exact offset match | 6.64% | 20.93% | **10.08%** | 144 / 2025 / 544 |
| Text overlap (char IoU ≥ 50%) | 23.19% | 73.11% | **35.21%** | 503 / 1666 / 185 |
| Exact title text match | 6.64% | 20.93% | **10.08%** | 144 / 2025 / 544 |
| Offset relaxed (±10 chars) | 20.24% | 63.81% | **30.73%** | 439 / 1730 / 249 |
| Offset relaxed (±50 chars) | 24.80% | 78.20% | **37.66%** | 538 / 1631 / 150 |

**Segment-dedup view** (720 unique segments, best row per segment by exact match):

- Overlap IoU50 F1: 35.33%
- Offset ±50 F1: 37.81%

### TiLamb-7B (zero-shot)

- Run ID: `benchmark_tilamb_pilot_title`
- Rows evaluated: 769
- Base model: `YoLo2000/TiLamb-7B`

| Metric | Precision | Recall | F1 | TP / FP / FN |
|--------|-----------|--------|-----|--------------|
| Exact offset match | 0.00% | 0.00% | **0.00%** | 0 / 0 / 688 |
| Text overlap (char IoU ≥ 50%) | 0.00% | 0.00% | **0.00%** | 0 / 0 / 688 |
| Exact title text match | 0.00% | 0.00% | **0.00%** | 0 / 0 / 688 |
| Offset relaxed (±10 chars) | 0.00% | 0.00% | **0.00%** | 0 / 0 / 688 |
| Offset relaxed (±50 chars) | 0.00% | 0.00% | **0.00%** | 0 / 0 / 688 |

**Segment-dedup view** (720 unique segments, best row per segment by exact match):

- Overlap IoU50 F1: 0.00%
- Offset ±50 F1: 0.00%

### TiLamb pilot LoRA

- Run ID: `benchmark_tilamb_lora_pilot_title`
- Rows evaluated: 769
- Base model: `YoLo2000/TiLamb-7B`
- Adapter: `/root/lora/tibetan-title-pilot`

| Metric | Precision | Recall | F1 | TP / FP / FN |
|--------|-----------|--------|-----|--------------|
| Exact offset match | 1.77% | 1.89% | **1.83%** | 13 / 722 / 675 |
| Text overlap (char IoU ≥ 50%) | 57.01% | 60.90% | **58.89%** | 419 / 316 / 269 |
| Exact title text match | 2.04% | 2.18% | **2.11%** | 15 / 720 / 673 |
| Offset relaxed (±10 chars) | 39.59% | 42.30% | **40.90%** | 291 / 444 / 397 |
| Offset relaxed (±50 chars) | 60.95% | 65.12% | **62.97%** | 448 / 287 / 240 |

**Segment-dedup view** (720 unique segments, best row per segment by exact match):

- Overlap IoU50 F1: 59.29%
- Offset ±50 F1: 63.66%

### Qwen2.5-7B-Instruct

- Run ID: `benchmark_qwen_pilot_title`
- Rows evaluated: 769
- Base model: `Qwen/Qwen2.5-7B-Instruct`

| Metric | Precision | Recall | F1 | TP / FP / FN |
|--------|-----------|--------|-----|--------------|
| Exact offset match | 0.00% | 0.00% | **0.00%** | 0 / 336 / 688 |
| Text overlap (char IoU ≥ 50%) | 22.92% | 11.19% | **15.04%** | 77 / 259 / 611 |
| Exact title text match | 0.00% | 0.00% | **0.00%** | 0 / 336 / 688 |
| Offset relaxed (±10 chars) | 12.20% | 5.96% | **8.01%** | 41 / 295 / 647 |
| Offset relaxed (±50 chars) | 40.48% | 19.77% | **26.56%** | 136 / 200 / 552 |

**Segment-dedup view** (720 unique segments, best row per segment by exact match):

- Overlap IoU50 F1: 15.06%
- Offset ±50 F1: 26.78%

### Gemma 4 E4B-it

- Run ID: `benchmark_gemma4_pilot_title`
- Rows evaluated: 769
- Base model: `google/gemma-4-E4B-it`

| Metric | Precision | Recall | F1 | TP / FP / FN |
|--------|-----------|--------|-----|--------------|
| Exact offset match | 0.00% | 0.00% | **0.00%** | 0 / 816 / 688 |
| Text overlap (char IoU ≥ 50%) | 19.05% | 22.67% | **20.70%** | 156 / 663 / 532 |
| Exact title text match | 0.00% | 0.00% | **0.00%** | 0 / 819 / 688 |
| Offset relaxed (±10 chars) | 6.72% | 7.99% | **7.30%** | 55 / 764 / 633 |
| Offset relaxed (±50 chars) | 42.86% | 51.02% | **46.58%** | 351 / 468 / 337 |

**Segment-dedup view** (720 unique segments, best row per segment by exact match):

- Overlap IoU50 F1: 21.41%
- Offset ±50 F1: 47.13%

