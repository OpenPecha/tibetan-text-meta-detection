# Tibetan metadata pilot benchmarks

Row-level benchmarks of span extractors on the **10% pilot LLM SFT test splits**.

- **Title** — multi-model comparison on the title test split (769 rows).
- **Author** — TiLamb author LoRA on the author test split (688 rows).

## Title contents

| Path | Description |
|------|-------------|
| [report/PILOT_TITLE_BENCHMARK_REPORT.md](report/PILOT_TITLE_BENCHMARK_REPORT.md) | **Main report** — corpus, models, metrics, conclusions |
| [report/METRICS_EXPLAINED.md](report/METRICS_EXPLAINED.md) | **Metrics guide** — IoU, offset tolerances, examples & diagrams |
| [report/benchmark_pilot_title.json](report/benchmark_pilot_title.json) | Machine-readable leaderboard (standard metrics) |
| [report/benchmark_offset_diagnostics.json](report/benchmark_offset_diagnostics.json) | Offset-first diagnostics (start/end/both hit rates) |
| [report/benchmark_pilot_title.md](report/benchmark_pilot_title.md) | Auto-generated standard metrics tables |
| [report/benchmark_offset_diagnostics.md](report/benchmark_offset_diagnostics.md) | Auto-generated offset diagnostics tables |
| [logs/](logs/) | Per-model `*_predictions.jsonl` and `*_metrics.json` |

## Author contents

| Path | Description |
|------|-------------|
| [report/PILOT_AUTHOR_BENCHMARK_REPORT.md](report/PILOT_AUTHOR_BENCHMARK_REPORT.md) | **Main report** — TiLamb author LoRA; offset-vs-text finding |
| [report/benchmark_pilot_author.json](report/benchmark_pilot_author.json) | Machine-readable leaderboard (standard metrics) |
| [report/benchmark_author_offset_diagnostics.json](report/benchmark_author_offset_diagnostics.json) | Offset-first diagnostics (start/end/both hit rates, MAE) |
| [report/benchmark_author_text_match.json](report/benchmark_author_text_match.json) | Offset-independent emitted-text match rates |
| [report/benchmark_pilot_author.md](report/benchmark_pilot_author.md) / [..._offset_diagnostics.md](report/benchmark_author_offset_diagnostics.md) | Auto-generated tables |
| [logs/benchmark_tilamb_lora_author_predictions.jsonl](logs/benchmark_tilamb_lora_author_predictions.jsonl) | Per-row predictions |

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

For the author benchmark:

```bash
python scripts/compare_benchmark.py --task author \
  --metrics-dir benchmark/logs \
  --output-md benchmark/report/benchmark_pilot_author.md \
  --output-json benchmark/report/benchmark_pilot_author.json

python scripts/recompute_benchmark_offset_metrics.py \
  --predictions benchmark/logs/benchmark_tilamb_lora_author_predictions.jsonl \
  --output-json benchmark/report/benchmark_author_offset_diagnostics.json \
  --output-md benchmark/report/benchmark_author_offset_diagnostics.md

python scripts/analyze_emitted_text_match.py \
  benchmark/logs/benchmark_tilamb_lora_author_predictions.jsonl \
  --output-json benchmark/report/benchmark_author_text_match.json
```

## Re-run inference

See [docs/TILAMB_TITLE_LORA_INFERENCE.md](../docs/TILAMB_TITLE_LORA_INFERENCE.md) for the **TiLamb title LoRA** and [docs/TILAMB_AUTHOR_LORA_INFERENCE.md](../docs/TILAMB_AUTHOR_LORA_INFERENCE.md) for the **TiLamb author LoRA**. Other models: [docs/BENCHMARK.md](../docs/BENCHMARK.md) and `scripts/run_benchmark_suite.sh`.
