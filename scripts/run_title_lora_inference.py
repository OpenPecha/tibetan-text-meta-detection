#!/usr/bin/env python3
"""Title LoRA inference via PEFT + transformers (no Unsloth)."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "YoLo2000/TiLamb-7B"
JSON_RE = re.compile(r"\{[\s\S]*\}")
SEP = "=" * 80

# Training / inference limits (LLaMA-Factory pilot config)
CUTOFF_LEN = 4096
SEGMENT_TOKEN_BUDGET = 3584
MAX_NEW_TOKENS = 256


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    m = JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON in output: {text[:400]!r}")
    return json.loads(m.group(0))


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def crop_to_tokens(tokenizer, text: str, max_tokens: int) -> tuple[str, int]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(s, e) for s, e in enc["offset_mapping"] if e > s]
    if len(offsets) <= max_tokens:
        return text, len(offsets)
    crop = text[offsets[0][0] : offsets[max_tokens - 1][1]]
    return crop, max_tokens


def format_prompt(tokenizer, instruction: str, segment: str) -> str:
    user = f"{instruction}\n\n{segment}"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<s>[INST] {user} [/INST]"


def compare_spans(gold: dict, pred: dict) -> dict:
    g_spans = gold.get("spans") or []
    p_spans = pred.get("spans") or []
    if not g_spans or not p_spans:
        return {"has_gold": bool(g_spans), "has_pred": bool(p_spans)}
    g, p = g_spans[0], p_spans[0]
    return {
        "text_match": g.get("text") == p.get("text"),
        "start_match": g.get("start") == p.get("start"),
        "end_match": g.get("end") == p.get("end"),
        "start_delta": (p.get("start") or 0) - (g.get("start") or 0),
        "end_delta": (p.get("end") or 0) - (g.get("end") or 0),
        "gold_span": g,
        "pred_span": p,
    }


def build_token_stats(
    *,
    instruction: str,
    segment_original: str,
    segment_sent: str,
    segment_cropped: bool,
    prompt: str,
    prompt_tokens: int,
    output_tokens: int,
    raw_response: str,
    model_max_position_embeddings: int | None,
    tokenizer,
) -> dict:
    instruction_chars = len(instruction)
    instruction_tokens = count_tokens(tokenizer, instruction)
    segment_chars_original = len(segment_original)
    segment_chars_sent = len(segment_sent)
    segment_tokens_original = count_tokens(tokenizer, segment_original)
    segment_tokens_sent = count_tokens(tokenizer, segment_sent)
    prompt_chars = len(prompt)
    output_chars = len(raw_response)
    total_sequence_tokens = prompt_tokens + output_tokens
    effective_max_context = min(CUTOFF_LEN, model_max_position_embeddings or CUTOFF_LEN)
    remaining_context_after_prompt = effective_max_context - prompt_tokens
    remaining_context_after_generation = effective_max_context - total_sequence_tokens

    return {
        "training_cutoff_len": CUTOFF_LEN,
        "segment_token_budget": SEGMENT_TOKEN_BUDGET,
        "max_new_tokens_configured": MAX_NEW_TOKENS,
        "model_max_position_embeddings": model_max_position_embeddings,
        "effective_max_context_length": effective_max_context,
        "instruction_chars": instruction_chars,
        "instruction_tokens": instruction_tokens,
        "segment_chars_before_crop": segment_chars_original,
        "segment_chars_after_crop": segment_chars_sent,
        "segment_chars_removed_by_crop": segment_chars_original - segment_chars_sent,
        "segment_tokens_before_crop": segment_tokens_original,
        "segment_tokens_after_crop": segment_tokens_sent,
        "segment_tokens_removed_by_crop": segment_tokens_original - segment_tokens_sent,
        "segment_was_cropped": segment_cropped,
        "prompt_chars": prompt_chars,
        "prompt_tokens_input": prompt_tokens,
        "output_chars": output_chars,
        "output_tokens": output_tokens,
        "total_sequence_tokens": total_sequence_tokens,
        "remaining_context_after_prompt": remaining_context_after_prompt,
        "remaining_context_after_generation": remaining_context_after_generation,
        "prompt_fits_in_context": prompt_tokens <= effective_max_context,
        "generation_fits_in_context": total_sequence_tokens <= effective_max_context,
        "output_within_max_new_tokens": output_tokens <= MAX_NEW_TOKENS,
    }


def print_token_stats(stats: dict) -> None:
    print("\n=== TOKEN & CONTEXT STATS ===")
    for key, value in stats.items():
        print(f"{key}={value}")


def write_detail_log(
    log_path: Path,
    *,
    meta: dict,
    token_stats: dict,
    instruction: str,
    segment_original: str,
    segment_sent: str,
    prompt: str,
    generation_config: dict,
    raw_response: str,
    gold: dict,
    pred: dict,
    comparison: dict,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        SEP,
        "TIBETAN TITLE LoRA INFERENCE — DETAILED LOG",
        SEP,
        "",
        "--- RUN METADATA ---",
        json.dumps(meta, ensure_ascii=False, indent=2),
        "",
        "--- TOKEN & CONTEXT STATS ---",
        json.dumps(token_stats, ensure_ascii=False, indent=2),
        "",
        "--- INSTRUCTION (sent to model) ---",
        instruction,
        "",
        f"--- INPUT SEGMENT (before crop: {token_stats['segment_chars_before_crop']} chars, "
        f"{token_stats['segment_tokens_before_crop']} tokens) ---",
        segment_original,
        "",
        f"--- INPUT SEGMENT (after crop: {token_stats['segment_chars_after_crop']} chars, "
        f"{token_stats['segment_tokens_after_crop']} tokens"
        + (", CROPPED" if token_stats["segment_was_cropped"] else ", full segment")
        + ") ---",
        segment_sent,
        "",
        f"--- FULL PROMPT ({token_stats['prompt_chars']} chars, "
        f"{token_stats['prompt_tokens_input']} input tokens) ---",
        prompt,
        "",
        "--- GENERATION CONFIG ---",
        json.dumps(generation_config, ensure_ascii=False, indent=2),
        "",
        f"--- RAW MODEL RESPONSE ({token_stats['output_chars']} chars, "
        f"{token_stats['output_tokens']} output tokens) ---",
        raw_response,
        "",
        "--- PARSED PREDICTION ---",
        json.dumps(pred, ensure_ascii=False, indent=2),
        "",
        "--- GOLD LABEL ---",
        json.dumps(gold, ensure_ascii=False, indent=2),
        "",
        "--- SPAN COMPARISON ---",
        json.dumps(comparison, ensure_ascii=False, indent=2),
        "",
        SEP,
        "END OF LOG",
        SEP,
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = log_path.with_suffix(".json")
    payload = {
        "meta": meta,
        "token_stats": token_stats,
        "instruction": instruction,
        "input": {
            "segment_original": segment_original,
            "segment_sent": segment_sent,
            "segment_chars_before_crop": token_stats["segment_chars_before_crop"],
            "segment_chars_after_crop": token_stats["segment_chars_after_crop"],
            "segment_tokens_before_crop": token_stats["segment_tokens_before_crop"],
            "segment_tokens_after_crop": token_stats["segment_tokens_after_crop"],
            "segment_was_cropped": token_stats["segment_was_cropped"],
        },
        "prompt": {
            "text": prompt,
            "chars": token_stats["prompt_chars"],
            "tokens": token_stats["prompt_tokens_input"],
        },
        "output": {
            "raw_text": raw_response,
            "chars": token_stats["output_chars"],
            "tokens": token_stats["output_tokens"],
        },
        "generation_config": generation_config,
        "gold": gold,
        "pred": pred,
        "comparison": comparison,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    jsonl = Path(sys.argv[1] if len(sys.argv) > 1 else "data/llm_sft_sample/title/test.jsonl")
    row_idx = int(sys.argv[2] if len(sys.argv) > 2 else 0)
    adapter = sys.argv[3] if len(sys.argv) > 3 else "ganga4364/tibetan-metadata-title-tilamb-lora-pilot"
    detail_log = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    if detail_log is None and (env_log := __import__("os").environ.get("DETAIL_LOG")):
        detail_log = Path(env_log)

    row = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[row_idx])
    instruction = row["instruction"]
    segment = row["input"]
    gold = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]

    started_at = datetime.now(timezone.utc).isoformat()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    print(f"jsonl={jsonl} row={row_idx} adapter={adapter}")
    print(f"segment_chars_before_crop={len(segment)}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if detail_log:
        print(f"detail_log={detail_log}")

    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    model_max_pos = getattr(model.config, "max_position_embeddings", None)

    crop_text, _ = crop_to_tokens(tokenizer, segment, max_tokens=SEGMENT_TOKEN_BUDGET)
    segment_cropped = crop_text != segment

    prompt = format_prompt(tokenizer, instruction, crop_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    generation_config = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "pad_token_id": int(tokenizer.eos_token_id),
    }

    with torch.inference_mode():
        out = model.generate(**inputs, **generation_config)

    output_ids = out[0, inputs["input_ids"].shape[1] :]
    output_tokens = int(output_ids.shape[0])
    raw_response = tokenizer.decode(output_ids, skip_special_tokens=True)
    pred = parse_json(raw_response)
    comparison = compare_spans(gold, pred)
    finished_at = datetime.now(timezone.utc).isoformat()

    token_stats = build_token_stats(
        instruction=instruction,
        segment_original=segment,
        segment_sent=crop_text,
        segment_cropped=segment_cropped,
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        raw_response=raw_response,
        model_max_position_embeddings=model_max_pos,
        tokenizer=tokenizer,
    )
    print_token_stats(token_stats)

    print("\n=== GOLD ===")
    print(json.dumps(gold, ensure_ascii=False, indent=2))
    print("\n=== PRED ===")
    print(json.dumps(pred, ensure_ascii=False, indent=2))

    if detail_log:
        meta = {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "jsonl": str(jsonl),
            "row": row_idx,
            "base_model": BASE,
            "adapter": adapter,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "tokenizer": adapter,
        }
        write_detail_log(
            detail_log,
            meta=meta,
            token_stats=token_stats,
            instruction=instruction,
            segment_original=segment,
            segment_sent=crop_text,
            prompt=prompt,
            generation_config=generation_config,
            raw_response=raw_response,
            gold=gold,
            pred=pred,
            comparison=comparison,
        )
        print(f"\nDETAIL_LOG={detail_log}")
        print(f"DETAIL_JSON={detail_log.with_suffix('.json')}")


if __name__ == "__main__":
    main()
