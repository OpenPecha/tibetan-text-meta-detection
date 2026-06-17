"""Sliding-window inference for Tibetan title/author NER."""

from __future__ import annotations

import argparse
import html
import json
from typing import Any

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import (
    ROBERTA_MAX_BEGIN_SLIDES,
    ROBERTA_MAX_END_SLIDES,
    ROBERTA_STRIDE,
    ROBERTA_WINDOW_SIZE,
)
from pipeline.bio import ID_TO_LABEL, IGNORE_LABEL_ID
from pipeline.roberta_windows import (
    WindowPrediction,
    WindowSpec,
    _max_content_tokens,
    _window_char_bounds,
    compute_windows,
    merge_predictions,
    tokenize_segment,
)

DEFAULT_MODEL_ID = "ganga4364/tibetan-metadata-koichi-ner"


def _build_inference_batch(
    tokenizer: Any,
    window: WindowSpec,
    input_ids: list[int],
    offsets: list[tuple[int, int]],
    max_length: int = ROBERTA_WINDOW_SIZE,
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """Build padded input_ids, attention_mask, and offset_mapping for one window."""
    content_limit = _max_content_tokens(max_length)
    win_input_ids = input_ids[window.start_tok : window.end_tok][:content_limit]
    win_offsets = offsets[window.start_tok : window.end_tok][:content_limit]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    seq_ids = [cls_id] + win_input_ids + [sep_id]
    seq_offsets = [(0, 0)] + win_offsets + [(0, 0)]
    attention_mask = [1] * len(seq_ids)

    while len(seq_ids) < max_length:
        seq_ids.append(pad_id)
        seq_offsets.append((0, 0))
        attention_mask.append(0)

    return seq_ids, attention_mask, seq_offsets


def prepare_inference_windows(
    tokenizer: Any,
    text: str,
    window_size: int = ROBERTA_WINDOW_SIZE,
    stride: int = ROBERTA_STRIDE,
    max_begin: int = ROBERTA_MAX_BEGIN_SLIDES,
    max_end: int = ROBERTA_MAX_END_SLIDES,
) -> tuple[Any, list[WindowSpec], list[tuple[list[int], list[int], list[tuple[int, int]], int, int]]]:
    """Tokenize text and prepare tensors for each sliding window.

    Returns (subword_result, window_specs, list of
    (input_ids, attention_mask, offset_mapping, seg_char_start, seg_char_end)).
    """
    if not text.strip():
        return None, [], []

    subword = tokenize_segment(tokenizer, text)
    if subword.n_tokens == 0:
        return subword, [], []

    windows = compute_windows(
        subword.n_tokens,
        window_size=window_size,
        stride=stride,
        max_begin=max_begin,
        max_end=max_end,
    )

    prepared: list[tuple[list[int], list[int], list[tuple[int, int]], int, int]] = []
    for window in windows:
        seg_char_start, seg_char_end = _window_char_bounds(subword.offsets, window)
        seq_ids, attention_mask, seq_offsets = _build_inference_batch(
            tokenizer,
            window,
            subword.input_ids,
            subword.offsets,
            max_length=window_size,
        )
        prepared.append(
            (seq_ids, attention_mask, seq_offsets, seg_char_start, seg_char_end)
        )

    return subword, windows, prepared


def _enrich_span_char_ends(
    offsets: list[tuple[int, int]],
    spans: list[dict],
) -> list[dict]:
    """Expand span_end to the last overlapping subword char end."""
    enriched: list[dict] = []
    for span in spans:
        start = span["span_start"]
        rough_end = span["span_end"]
        char_end = rough_end
        for tok_start, tok_end in offsets:
            if tok_end <= tok_start:
                continue
            if tok_start < rough_end + 1 and tok_end > start:
                char_end = max(char_end, tok_end)
        enriched.append({**span, "span_end": char_end})
    return enriched


def predict_segment(
    model: Any,
    tokenizer: Any,
    text: str,
    device: torch.device | None = None,
) -> list[dict]:
    """Run sliding-window inference and return title/author spans with text."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subword, windows, prepared = prepare_inference_windows(tokenizer, text)
    if not prepared:
        return []

    model.eval()
    predictions: list[WindowPrediction] = []

    with torch.no_grad():
        for window, (seq_ids, attention_mask, seq_offsets, seg_char_start, seg_char_end) in zip(
            windows, prepared
        ):
            input_ids = torch.tensor([seq_ids], device=device)
            attn = torch.tensor([attention_mask], device=device)
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            label_ids = logits.argmax(dim=-1).tolist()
            confidences = probs.max(dim=-1).values.tolist()

            predictions.append(
                WindowPrediction(
                    window_name=window.name,
                    side=window.side,
                    slide_index=window.slide_index,
                    segment_char_start=seg_char_start,
                    segment_char_end=seg_char_end,
                    offsets=seq_offsets,
                    label_ids=label_ids,
                    confidences=confidences,
                )
            )

    spans = merge_predictions(predictions, id_to_label=ID_TO_LABEL)
    spans = _enrich_span_char_ends(subword.offsets, spans)

    for span in spans:
        start = max(0, span["span_start"])
        end = min(len(text), span["span_end"])
        span["text"] = text[start:end]

    return spans


def highlight_spans(text: str, spans: list[dict]) -> str:
    """Return HTML with title (gold) and author (blue) highlights."""
    if not text:
        return ""
    if not spans:
        return f"<pre style='white-space:pre-wrap;font-size:1.1em'>{html.escape(text)}</pre>"

    sorted_spans = sorted(spans, key=lambda s: (s["span_start"], s["span_end"]))
    parts: list[str] = []
    cursor = 0

    for span in sorted_spans:
        start = max(0, int(span["span_start"]))
        end = min(len(text), int(span["span_end"]))
        if start < cursor:
            continue
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))

        label = span.get("label", "title").lower()
        css = "background:#fde68a" if label == "title" else "background:#bfdbfe"
        title_attr = label.capitalize()
        parts.append(
            f"<mark style='{css};padding:0 2px;border-radius:2px' "
            f"title='{html.escape(title_attr)}'>"
            f"{html.escape(text[start:end])}</mark>"
        )
        cursor = end

    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))

    body = "".join(parts)
    return (
        "<div style='font-size:1.15em;line-height:1.8;"
        "font-family:\"Noto Sans Tibetan\",sans-serif'>"
        f"{body}</div>"
    )


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    device: torch.device | None = None,
) -> tuple[Any, Any, torch.device]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.to(device)
    return model, tokenizer, device


def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window Tibetan NER inference")
    parser.add_argument("--text", required=True, help="Tibetan segment text")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--json", action="store_true", help="Print spans as JSON")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model, device=None)
    spans = predict_segment(model, tokenizer, args.text, device=device)

    if args.json:
        print(json.dumps(spans, ensure_ascii=False, indent=2))
    else:
        print(highlight_spans(args.text, spans))


if __name__ == "__main__":
    main()
