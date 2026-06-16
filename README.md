# Tibetan Metadata Detector

Detect **title** and **author** spans in digitized Tibetan text using a RoBERTa token-classifier with a sliding-window training and inference pipeline.

**GitHub:** [OpenPecha/tibetan-text-meta-detection](https://github.com/OpenPecha/tibetan-text-meta-detection)

## Hugging Face

| Asset | Link |
|-------|------|
| Extracted documents | [ganga4364/tibetan-metadata-extracted](https://huggingface.co/datasets/ganga4364/tibetan-metadata-extracted) |
| Window splits (balanced) | [ganga4364/tibetan-metadata-detector](https://huggingface.co/datasets/ganga4364/tibetan-metadata-detector) |
| Fine-tuned model | [ganga4364/tibetan-metadata-roberta-ner](https://huggingface.co/ganga4364/tibetan-metadata-roberta-ner) |
| Gradio demo | [ganga4364/tibetan-metadata-highlight](https://huggingface.co/spaces/ganga4364/tibetan-metadata-highlight) |

## Quick start (GPU instance, no database)

Full step-by-step: **[docs/GPU_INSTANCE.md](docs/GPU_INSTANCE.md)**

```bash
git clone https://github.com/OpenPecha/tibetan-text-meta-detection.git
cd tibetan-text-meta-detection
pip install -r requirements.txt

# HF token at /root/.hf_token, then:
bash scripts/run_pull_extracted.sh
NUM_WORKERS=6 bash scripts/start_roberta_process_multiworker.sh
NUM_WORKERS=6 bash scripts/wait_roberta_merge_balance_push.sh
BATCH_SIZE=64 EPOCHS=3 bash scripts/start_train_tmux.sh
```

Train from HF without local JSONL:

```bash
python train_roberta.py \
  --hf-dataset ganga4364/tibetan-metadata-detector \
  --hf-config default \
  --output-dir data/roberta_full/model \
  --batch-size 64 \
  --epochs 3 \
  --entity-weight 10
```

> **Note:** Parquet splits on HF use the `default` dataset config (not `windows`). Use `--hf-config default` or omit (default in `train_roberta.py`).

## Local setup (with PostgreSQL extract)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # BENCHMARK_DB_* credentials
```

```bash
python extract_data.py --all
python prepare_data.py roberta-process --processed-dir data/roberta_full
python prepare_data.py balance-windows --processed-dir data/roberta_full
python prepare_data.py split --processed-dir data/roberta_full
```

## RoBERTa sliding-window pipeline

- Tokenizer: [`spsither/tibetan_RoBERTa_S_e3`](https://huggingface.co/spsither/tibetan_RoBERTa_S_e3)
- Window size **512**, stride **256**, up to **15 begin + 15 end** slides (overlap-aware dedup)
- Short segments (≤512 tokens): single window
- Training labels: BIO (`O`, `B-TITLE`, `I-TITLE`, `B-AUTHOR`, `I-AUTHOR`)

### Commands

```bash
# Pull HF extracted docs → data/extracted/
python scripts/pull_extracted_from_hf.py --output-dir data/extracted

# Windowing (multi-worker)
NUM_WORKERS=6 bash scripts/start_roberta_process_multiworker.sh
python prepare_data.py merge-roberta-shards --processed-dir data/roberta_full --num-workers 6

# Balance + split + HF upload
python prepare_data.py balance-windows --processed-dir data/roberta_full
python prepare_data.py split --processed-dir data/roberta_full
bash scripts/push_balanced_splits.sh

# Or combined waiter / resume scripts — see scripts/
```

### Balancing (before split)

| Setting | Default | Effect |
|---------|---------|--------|
| `O_ONLY_CAP_RATIO` | 2.0 | Max O-only windows = 2× entity windows per segment |
| `AUTHOR_OVERSAMPLE` | 2 | Duplicate author-bearing windows 2× |
| Split ratios | 89% / 1% / 10% | Document-stratified |

Balanced run (Jun 2026): **1,061,770 → 274,422** windows → **241,377** train / **2,688** val / **30,357** test.

## Evaluation

| Level | Command / source | Metric file |
|-------|------------------|-------------|
| Window | End of `train_roberta.py` | `model/test_metrics.json` |
| Segment (deploy-like) | `eval_segment.py` | `model/segment_test_metrics.json` |

```bash
python eval_segment.py \
  --model data/roberta_full/model/best \
  --extracted-dir data/extracted \
  --splits-dir data/roberta_full/splits
```

Segment eval uses `pipeline/inference.py` → `predict_segment()` (same as Gradio Space).

## Inference

```python
from pipeline.inference import load_model_and_tokenizer, predict_segment, highlight_spans

model, tokenizer, device = load_model_and_tokenizer("ganga4364/tibetan-metadata-roberta-ner")
spans = predict_segment(model, tokenizer, your_tibetan_segment_text, device=device)
print(highlight_spans(your_tibetan_segment_text, spans))
```

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/pull_extracted_from_hf.py` | HF extracted dataset → `data/extracted/` |
| `scripts/run_pull_extracted.sh` | Run pull with HF token |
| `scripts/setup_hf_token.sh` | Configure `/root/.hf_token` on instance |
| `scripts/start_roberta_process_multiworker.sh` | Parallel `roberta-process` workers |
| `scripts/wait_roberta_merge_balance_push.sh` | Wait → merge → balance → split → HF push |
| `scripts/resume_balance_split_push.sh` | Resume after merge if balance was interrupted |
| `scripts/push_balanced_splits.sh` | Upload splits as Parquet |
| `scripts/start_train_tmux.sh` | Train in tmux from HF dataset |
| `scripts/start_gradio_gpu.sh` | Gradio app on GPU (port 7860) |

## Tests

```bash
pytest tests/test_label_window.py -v -m "not slow"
```

## Model metrics (latest balanced run)

| Eval | Span F1 | Title F1 | Author F1 |
|------|---------|----------|-----------|
| Window test | 3.1% | 7.4% | 1.0% |
| Segment exact match | 8.0% | 12.7% | 0.7% |

Trained on fixed window-relative BIO labels with class weight 10. Author detection remains the main weakness; see improvement ideas below.

## Improvement ideas

1. Lower `entity_weight` (5) and looser O-only cap (3–4×) to improve precision.
2. Train 5–10 epochs; early-stop on segment-level val F1.
3. Confidence threshold in `merge_predictions` at inference.
4. Try `KoichiYasuoka/roberta-base-tibetan` (requires re-windowing with new tokenizer).
5. Per-entity loss weights or CRF head for BIO consistency.

## Scope

- **In scope:** title, author (`title_span_*` / `author_span_*` on `outliner_segments`)
- **Out of scope:** translator
