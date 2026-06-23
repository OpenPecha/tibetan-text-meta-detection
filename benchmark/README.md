# Tibetan metadata pilot title benchmark

Row-level benchmark comparing span extractors on the **10% pilot LLM SFT title test split** (769 rows).

## Contents

| Path | Description |
|------|-------------|
| [report/PILOT_TITLE_BENCHMARK_REPORT.md](report/PILOT_TITLE_BENCHMARK_REPORT.md) | **Main report** — corpus, models, metrics, conclusions |
| [report/METRICS_EXPLAINED.md](report/METRICS_EXPLAINED.md) | **Metrics guide** — IoU, offset tolerances, examples & diagrams |
| [report/benchmark_pilot_title.json](report/benchmark_pilot_title.json) | Machine-readable leaderboard (standard metrics) |
| [report/benchmark_offset_diagnostics.json](report/benchmark_offset_diagnostics.json) | Offset-first diagnostics (start/end/both hit rates) |
| [report/benchmark_pilot_title.md](report/benchmark_pilot_title.md) | Auto-generated standard metrics tables |
| [report/benchmark_offset_diagnostics.md](report/benchmark_offset_diagnostics.md) | Auto-generated offset diagnostics tables |
| [logs/](logs/) | Per-model `*_predictions.jsonl` and `*_metrics.json` |

## Reproduce metrics from predictions (no GPU)

```bash
python scripts/compare_benchmark.py \
  --metrics-dir benchmark/logs \
  --output-md benchmark/report/benchmark_pilot_title.md \
  --output-json benchmark/report/benchmark_pilot_title.json

python scripts/recompute_benchmark_offset_metrics.py \
  --metrics-dir benchmark/logs \
  --output-json benchmark/report/benchmark_offset_diagnostics.json \
  --output-md benchmark/report/benchmark_offset_diagnostics.md
```

## Re-run inference

See [docs/TILAMB_TITLE_LORA_INFERENCE.md](../docs/TILAMB_TITLE_LORA_INFERENCE.md) for the **TiLamb title LoRA** adapter. Other models: [docs/BENCHMARK.md](../docs/BENCHMARK.md) and `scripts/run_benchmark_suite.sh`.
