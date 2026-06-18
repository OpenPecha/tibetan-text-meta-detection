# LLM SFT Dataset + VastAI-3 Playbook

Build **TiLamb-7B** supervised fine-tuning data for **title** and **author** span extraction from BDRC outliner segments, with **randomized span-centered crops** (avoids fixed begin/end position bias).

## Outputs

```
data/llm_sft/
  title/{train,val,test}.jsonl       # LLaMA-Factory Alpaca format
  title/{train,val,test}_meta.jsonl  # ids, crop_kind, span_position_ratio, …
  author/{train,val,test}.jsonl
  author/{train,val,test}_meta.jsonl
  dataset_info.json                  # LLaMA-Factory registry snippet
  reports/crop_stats.json
```

Each row: `instruction` + `input` (cropped segment) + `output` (JSON `{"spans":[...]}`).

## Local smoke test (sample_4doc)

```bash
cd tibetan-metadata-detector
pip install -r requirements.txt
pytest tests/test_llm_crop.py -v

python -m llm_sft.build_dataset \
  --extracted-dir data/sample_4doc \
  --output-dir data/llm_sft_sample \
  --max-context-tokens 3584 \
  --crops-per-positive 3 \
  --seed 42
```

## VastAI-3 instance

| Item | Value |
|------|-------|
| Host | `82.141.118.42` |
| Port | `10252` |
| SSH | `ssh -p 10252 root@82.141.118.42` |

Add to `~/.ssh/config`:

```
Host vastai3
  HostName 82.141.118.42
  Port 10252
  User root
  IdentityFile <your-key-from-remote-explorer>
  LocalForward 8080 localhost:8080
```

### 1. Bootstrap

```bash
cd /root
git clone https://github.com/OpenPecha/tibetan-text-meta-detection.git
ln -sfn /root/tibetan-text-meta-detection /root/tibetan-metadata-detector
cd /root/tibetan-metadata-detector
pip install -r requirements.txt sentencepiece
echo "hf_..." > /root/.hf_token && chmod 600 /root/.hf_token
bash scripts/setup_hf_token.sh
```

### 2. Pull extracted documents (streaming, no full-RAM load)

```bash
bash scripts/run_pull_extracted.sh
wc -l data/extracted/index.jsonl   # expect 3794
```

### 3. Build LLM SFT JSONL

```bash
tmux new -s llm_sft
bash scripts/build_llm_sft.sh
# or:
EXTRACTED_DIR=data/extracted OUTPUT_DIR=data/llm_sft bash scripts/build_llm_sft.sh
```

Review `data/llm_sft/reports/crop_stats.json` — `span_position_ratio_histogram` should not be pinned at `0.9-1.0` only.

### 4. LLaMA-Factory (TiLamb LoRA)

```bash
cd /root
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

Copy or symlink dataset registry:

```bash
mkdir -p data
cp /root/tibetan-metadata-detector/data/llm_sft/dataset_info.json data/
ln -sfn /root/tibetan-metadata-detector/data/llm_sft/title data/tibetan_title_sft
ln -sfn /root/tibetan-metadata-detector/data/llm_sft/author data/tibetan_author_sft
```

Train **title** and **author** as separate LoRA runs (`dataset: tibetan_title_sft` / `tibetan_author_sft`). Base model: `YoLo2000/TiLamb-7B` (accept Meta Llama 2 license on Hugging Face).

```bash
# One-shot setup + train (title then author) in tmux
bash scripts/setup_llama_factory.sh
bash scripts/start_llm_sft_train_tmux.sh
tmux attach -t llm_sft_train
```

Configs: `configs/llama_factory/title_lora_sft.yaml`, `author_lora_sft.yaml`.

Optional: push JSONL to `ganga4364/tibetan-metadata-llm-sft` on Hugging Face.

## Cropping behavior

| Case | Behavior |
|------|----------|
| Segment fits token budget | One **full** row |
| Has gold span, segment too long | `crops_per_positive` random windows **containing** the span; presets `(500,1000)`, `(100,1400)`, `(750,750)`, … + uniform slack |
| No gold span for task | Uniform **random** window (not begin/end anchored) |

Tokenizer: **TiLamb** (`YoLo2000/TiLamb-7B`) — budgets enforced in **tokens**, not characters.

## Related

- [GPU_INSTANCE.md](GPU_INSTANCE.md) — RoBERTa / Koichi pipeline on VastAI
- [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) — RoBERTa baseline metrics
