#!/usr/bin/env python3
"""Run TiLamb title/author LoRA inference on LLM SFT JSONL rows."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_sft.crop import crop_from_token_window, tokenize_segment
from llm_sft.prompts import INSTRUCTIONS

BASE_MODEL = "YoLo2000/TiLamb-7B"
CUTOFF_LEN = 4096
JSON_RE = re.compile(r"\{[\s\S]*\}")


def build_user_message(instruction: str, segment: str) -> str:
    return f"{instruction}\n\n{segment}"


def format_prompt(tokenizer: Any, instruction: str, segment: str) -> str:
    user = build_user_message(instruction, segment)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<s>[INST] {user} [/INST]"


def parse_spans_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    match = JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"spans": []}


def iter_segment_windows(
    tokenizer: Any,
    segment: str,
    max_input_tokens: int,
    stride_tokens: int,
    *,
    first_window_only: bool = True,
) -> list[tuple[str, int]]:
    """Return segment crop(s) for inference.

    When first_window_only=True (default), use only the first max_input_tokens
    of the segment. Longer text is truncated; no sliding windows.
    """
    tokenized = tokenize_segment(tokenizer, segment)
    if tokenized.n_tokens <= max_input_tokens:
        return [(segment, 0)]
    crop_text, cs, _ = crop_from_token_window(
        segment, tokenized.offsets, 0, max_input_tokens
    )
    if first_window_only:
        return [(crop_text, cs)]
    windows: list[tuple[str, int]] = [(crop_text, cs)]
    start = stride_tokens
    while start < tokenized.n_tokens:
        end = min(start + max_input_tokens, tokenized.n_tokens)
        crop_text, cs, _ = crop_from_token_window(
            segment, tokenized.offsets, start, end
        )
        windows.append((crop_text, cs))
        if end >= tokenized.n_tokens:
            break
        start += stride_tokens
    return windows


def dedupe_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    out: list[dict[str, Any]] = []
    for span in spans:
        key = (span["start"], span["end"], span["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def load_model_and_tokenizer(
    *,
    base_model: str,
    adapter_path: str | None,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tok_source = adapter_path or base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@dataclass
class WindowResult:
    crop_text: str
    char_offset: int
    prompt: str
    prompt_tokens: int
    output_tokens: int
    raw_response: str
    parsed: dict[str, Any]


@dataclass
class SegmentPrediction:
    spans: list[dict[str, Any]]
    windows: list[WindowResult] = field(default_factory=list)
    segment_chars: int = 0
    segment_tokens: int = 0
    num_windows: int = 0


@torch.inference_mode()
def generate_json(
    model: Any,
    tokenizer: Any,
    instruction: str,
    segment: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, int, int, str]:
    prompt = format_prompt(tokenizer, instruction, segment)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
    output_tokens = int(new_tokens.shape[0])
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw, prompt_tokens, output_tokens, prompt


def predict_segment(
    model: Any,
    tokenizer: Any,
    *,
    task: str,
    segment: str,
    max_input_tokens: int = 3584,
    stride_tokens: int = 3000,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    return_details: bool = False,
) -> dict[str, Any] | SegmentPrediction:
    result = predict_segment_detailed(
        model,
        tokenizer,
        task=task,
        segment=segment,
        max_input_tokens=max_input_tokens,
        stride_tokens=stride_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    if return_details:
        return result
    return {"spans": result.spans}


def predict_segment_detailed(
    model: Any,
    tokenizer: Any,
    *,
    task: str,
    segment: str,
    max_input_tokens: int = 3584,
    stride_tokens: int = 3000,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    first_window_only: bool = True,
) -> SegmentPrediction:
    instruction = INSTRUCTIONS[task]
    tokenized = tokenize_segment(tokenizer, segment)
    windows = iter_segment_windows(
        tokenizer,
        segment,
        max_input_tokens,
        stride_tokens,
        first_window_only=first_window_only,
    )
    all_spans: list[dict[str, Any]] = []
    window_results: list[WindowResult] = []

    for crop_text, char_offset in windows:
        raw, prompt_tokens, output_tokens, prompt = generate_json(
            model,
            tokenizer,
            instruction,
            crop_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        parsed = parse_spans_json(raw)
        window_results.append(
            WindowResult(
                crop_text=crop_text,
                char_offset=char_offset,
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                raw_response=raw,
                parsed=parsed,
            )
        )
        for span in parsed.get("spans", []):
            all_spans.append(
                {
                    "text": span["text"],
                    "start": char_offset + int(span["start"]),
                    "end": char_offset + int(span["end"]),
                }
            )

    merged = dedupe_spans(all_spans)
    return SegmentPrediction(
        spans=merged,
        windows=window_results,
        segment_chars=len(segment),
        segment_tokens=tokenized.n_tokens,
        num_windows=len(windows),
    )


def llm_spans_to_eval(spans: list[dict[str, Any]], label: str = "title") -> list[dict]:
    return [
        {
            "label": label,
            "span_start": int(s["start"]),
            "span_end": int(s["end"]),
        }
        for s in spans
    ]


def load_jsonl_row(path: Path, index: int = 0) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"Row {index} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--task", choices=["title", "author"], default="title")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default="/root/lora/tibetan-title-pilot")
    parser.add_argument("--max-input-tokens", type=int, default=3584)
    parser.add_argument("--stride-tokens", type=int, default=3000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    row = load_jsonl_row(args.jsonl, args.row)
    segment = row["input"]
    gold = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]

    print(f"jsonl: {args.jsonl}")
    print(f"row: {args.row}")
    print(f"task: {args.task}")
    print(f"segment_chars: {len(segment)}")

    model, tokenizer = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter,
        load_in_4bit=not args.no_4bit,
    )
    pred = predict_segment_detailed(
        model,
        tokenizer,
        task=args.task,
        segment=segment,
        max_input_tokens=args.max_input_tokens,
        stride_tokens=args.stride_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"segment_tokens: {pred.segment_tokens}")
    print(f"windows: {pred.num_windows}")

    print("\n=== GOLD ===")
    print(json.dumps(gold, ensure_ascii=False, indent=2))
    print("\n=== PRED ===")
    print(json.dumps({"spans": pred.spans}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
