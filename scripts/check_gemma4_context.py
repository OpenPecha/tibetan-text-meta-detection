#!/usr/bin/env python3
import json
import statistics
from pathlib import Path

from transformers import AutoConfig, AutoProcessor

MODEL = "google/gemma-4-E4B-it"
cfg = AutoConfig.from_pretrained(MODEL)
print("model_type", cfg.model_type)
print("max_position_embeddings", getattr(cfg, "max_position_embeddings", None))
if getattr(cfg, "text_config", None):
    print("text_config.max_position_embeddings", cfg.text_config.max_position_embeddings)

processor = AutoProcessor.from_pretrained(MODEL)
tok = processor.tokenizer
test_path = Path("data/llm_sft_pilot_10pct/title/test.jsonl")
rows = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]
tokens = []
for row in rows[:20]:
    user = f"{row['instruction']}\n\n{row['input']}"
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    n = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
    tokens.append(n)
print("pilot test first20 prompt_tokens: mean", round(statistics.mean(tokens)), "max", max(tokens))
print("TiLamb crop budget was 3584 tokens (TiLamb tokenizer, not Gemma)")
