# GPU Instance Playbook

End-to-end workflow for training on a VastAI (or similar) GPU instance **without** a local PostgreSQL database. Extracted documents and trained artifacts live on Hugging Face.

## Hugging Face assets

| Repo | URL | Contents |
|------|-----|----------|
| Extracted docs | [ganga4364/tibetan-metadata-extracted](https://huggingface.co/datasets/ganga4364/tibetan-metadata-extracted) | 3,794 documents |
| Window splits | [ganga4364/tibetan-metadata-detector](https://huggingface.co/datasets/ganga4364/tibetan-metadata-detector) | Balanced train/val/test Parquet |
| Model | [ganga4364/tibetan-metadata-roberta-ner](https://huggingface.co/ganga4364/tibetan-metadata-roberta-ner) | Fine-tuned weights + metrics |
| Demo Space | [ganga4364/tibetan-metadata-highlight](https://huggingface.co/spaces/ganga4364/tibetan-metadata-highlight) | Gradio segment highlight |

## 1. Bootstrap instance

```bash
cd /root
git clone https://github.com/OpenPecha/tibetan-text-meta-detection.git
ln -sfn /root/tibetan-text-meta-detection /root/tibetan-metadata-detector
cd /root/tibetan-metadata-detector
pip install -r requirements.txt
pip install -U datasets huggingface_hub pyarrow transformers accelerate torch seqeval gradio
```

Set Hugging Face token (never commit this file):

```bash
echo "hf_..." > /root/.hf_token
chmod 600 /root/.hf_token
bash scripts/setup_hf_token.sh
```

## 2. Pull extracted data from HF (no DB)

```bash
bash scripts/run_pull_extracted.sh
# expect: wc -l data/extracted/index.jsonl → 3794
```

## 3. Multi-worker windowing → balance → split → HF push

```bash
NUM_WORKERS=6 bash scripts/start_roberta_process_multiworker.sh

# In a second tmux session (or after workers finish):
NUM_WORKERS=6 bash scripts/wait_roberta_merge_balance_push.sh
# Or if balance OOM'd mid-run, resume from merge:
bash scripts/resume_balance_split_push.sh
```

**Balancing** (1.06M → 274k windows): caps O-only at 2× entity windows per segment; 2× author oversample. Uses SQLite-backed streaming to avoid RAM OOM on ~1M rows.

**Split:** 89% / 1% / 10% document-stratified.

Verify on HF:

```python
from datasets import load_dataset
ds = load_dataset("ganga4364/tibetan-metadata-detector")  # config: default
print(ds)  # train ~241k, validation ~2.7k, test ~30k
```

## 4. Train on GPU (from HF)

```bash
BATCH_SIZE=64 EPOCHS=3 bash scripts/start_train_tmux.sh
tmux attach -t train
```

Or directly:

```bash
python train_roberta.py \
  --hf-dataset ganga4364/tibetan-metadata-detector \
  --hf-config default \
  --output-dir data/roberta_full/model \
  --batch-size 64 \
  --epochs 3 \
  --entity-weight 10
```

## 5. Evaluate

**Window-level** (automatic at end of training): `data/roberta_full/model/test_metrics.json`

**Segment-level** (deployment-like, sliding-window merge):

```bash
python eval_segment.py \
  --model data/roberta_full/model/best \
  --extracted-dir data/extracted \
  --splits-dir data/roberta_full/splits \
  --output data/roberta_full/model/segment_test_metrics.json
```

## 6. Upload model to HF

```bash
export HF_TOKEN="$(cat /root/.hf_token)"
hf upload ganga4364/tibetan-metadata-roberta-ner data/roberta_full/model/best . \
  --commit-message 'Retrain on balanced fixed-label windows'
hf upload ganga4364/tibetan-metadata-roberta-ner \
  data/roberta_full/model/test_metrics.json test_metrics.json \
  --commit-message 'Window test metrics'
hf upload ganga4364/tibetan-metadata-roberta-ner \
  data/roberta_full/model/segment_test_metrics.json segment_test_metrics.json \
  --commit-message 'Segment test metrics'
```

## 7. Gradio on GPU (optional)

```bash
bash scripts/start_gradio_gpu.sh   # tmux session: gradio, port 7860
```

From your laptop:

```bash
ssh -L 7860:localhost:7860 vastai
# open http://localhost:7860
```

## 8. Deploy / update Space

```bash
hf upload ganga4364/tibetan-metadata-highlight space/ --type space \
  --commit-message 'Sync inference pipeline'
```

## Latest run metrics (Jun 2026, balanced + fixed labels)

| Eval | Span F1 | Title F1 | Author F1 |
|------|---------|----------|-----------|
| Window (30,357 test windows) | 3.1% | 7.4% | 1.0% |
| Segment exact match (6,492 segments) | 8.0% | 12.7% | 0.7% |

Base model: `spsither/tibetan_RoBERTa_S_e3`. See [docs/EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for full experiment history including Koichi comparison.

## Monitoring

```bash
tail -f data/roberta_process_logs/roberta_worker0.log
tail -f data/roberta_merge_balance_push.log
tail -f data/train.log
tmux ls
```

## Teardown

After pushing model + metrics to HF and updating the Space, the instance can be destroyed. All reproducible artifacts are on GitHub + Hugging Face; no need to keep local `data/roberta_full/` on the instance.
