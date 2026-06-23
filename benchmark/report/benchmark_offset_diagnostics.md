# Benchmark offset diagnostics (recomputed from predictions)

No inference — metrics derived from saved `gold_spans` / `pred_spans`.

**Row hit rate** = share of paired rows (best IoU same-label match) where the boundary test passes.
**Micro F1** = greedy span matching aggregated over all rows (legacy view).

## Tolerance ±10 chars

| Model | Rows | Start hit | End hit | Both hit | Both+overlap | MAE start | MAE end | Median IoU |
|-------|------|-----------|---------|----------|--------------|----------|---------|------------|
| deepseek_r1_14b | 4 | 100.00% | 50.00% | 50.00% | 50.00% | 2.5 | 10.8 | 82.71% |
| gemma4 | 551 | 63.16% | 12.70% | 9.98% | 9.98% | 478.5 | 500.5 | 38.10% |
| koichi | 667 | 78.86% | 68.22% | 65.82% | 65.82% | 497.4 | 502.4 | 89.33% |
| qwen36_27b | 50 | 56.00% | 36.00% | 34.00% | 34.00% | 1037.2 | 1045.9 | 58.69% |
| qwen | 308 | 40.91% | 16.56% | 13.31% | 13.31% | 586.6 | 597.0 | 2.39% |
| tilamb_lora | 679 | 63.18% | 44.04% | 42.86% | 42.86% | 544.5 | 553.6 | 69.05% |
| tilamb | 0 | — | — | — | — | — | — | — |

## Tolerance ±50 chars

| Model | Rows | Start hit | End hit | Both hit | Both+overlap | MAE start | MAE end | Median IoU |
|-------|------|-----------|---------|----------|--------------|----------|---------|------------|
| deepseek_r1_14b | 4 | 100.00% | 100.00% | 100.00% | 100.00% | 2.5 | 10.8 | 82.71% |
| gemma4 | 551 | 74.77% | 65.52% | 64.97% | 63.16% | 478.5 | 500.5 | 38.10% |
| koichi | 667 | 82.76% | 81.11% | 80.66% | 80.51% | 497.4 | 502.4 | 89.33% |
| qwen36_27b | 50 | 66.00% | 66.00% | 64.00% | 64.00% | 1037.2 | 1045.9 | 58.69% |
| qwen | 308 | 55.52% | 48.38% | 47.08% | 44.16% | 586.6 | 597.0 | 2.39% |
| tilamb_lora | 679 | 70.54% | 68.92% | 67.60% | 65.98% | 544.5 | 553.6 | 69.05% |
| tilamb | 0 | — | — | — | — | — | — | — |
