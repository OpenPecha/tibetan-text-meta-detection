# Benchmark offset diagnostics (recomputed from predictions)

No inference — metrics derived from saved `gold_spans` / `pred_spans`.

**Row hit rate** = share of paired rows (best IoU same-label match) where the boundary test passes.
**Micro F1** = greedy span matching aggregated over all rows (legacy view).

## Tolerance ±10 chars

| Model | Rows | Start hit | End hit | Both hit | Both+overlap | MAE start | MAE end | Median IoU |
|-------|------|-----------|---------|----------|--------------|----------|---------|------------|
| tilamb_lora_author | 607 | 2.31% | 1.65% | 0.99% | 0.99% | 980.7 | 994.8 | 0.00% |

## Tolerance ±50 chars

| Model | Rows | Start hit | End hit | Both hit | Both+overlap | MAE start | MAE end | Median IoU |
|-------|------|-----------|---------|----------|--------------|----------|---------|------------|
| tilamb_lora_author | 607 | 6.75% | 6.59% | 5.93% | 3.13% | 980.7 | 994.8 | 0.00% |
