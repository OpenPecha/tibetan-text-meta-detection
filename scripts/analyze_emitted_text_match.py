#!/usr/bin/env python3
"""Text-level match analysis from saved predictions (offset-independent).

Complements the offset-based benchmark: many generative models emit the correct
span *text* but cannot place an accurate character offset in long inputs. This
script measures how often the model's emitted span text matches the gold text,
ignoring offsets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_TRAILING = re.compile(r"[\s།་]+$")


def norm(s: str | None) -> str:
    return _TRAILING.sub("", (s or "").strip())


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _exact_counts(gold: list[str], pred: list[str]) -> tuple[int, int, int]:
    """Multiset intersection (order-independent one-to-one matching)."""
    gc, pc = Counter(gold), Counter(pred)
    tp = sum((gc & pc).values())
    return tp, sum(pc.values()) - tp, sum(gc.values()) - tp


def _contained_counts(gold: list[str], pred: list[str]) -> tuple[int, int, int]:
    """Greedy one-to-one match where pred ⊆ gold or gold ⊆ pred."""
    used = [False] * len(pred)
    tp = 0
    for g in gold:
        for i, p in enumerate(pred):
            if used[i] or not p or not g:
                continue
            if p in g or g in p:
                used[i] = True
                tp += 1
                break
    return tp, len(pred) - tp, len(gold) - tp


def analyze(predictions_path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in predictions_path.open(encoding="utf-8")
        if line.strip()
    ]
    gold_rows = [r for r in rows if r.get("gold_spans")]

    # Row-level "did the model get it at all" rates (over rows with a gold span).
    exact_row = norm_row = contained_row = any_pred = 0
    # Span-level micro P/R/F1 (over ALL rows, so false positives on no-gold rows count).
    raw = [0, 0, 0]
    normd = [0, 0, 0]
    cont = [0, 0, 0]
    for r in rows:
        gold_texts = [g.get("text", "") for g in r.get("gold_spans", [])]
        pred_texts = [p.get("text", "") for p in r.get("pred_spans", [])]
        gold_norm = [norm(t) for t in gold_texts]
        pred_norm = [norm(t) for t in pred_texts]

        for acc, gset, pset in (
            (raw, gold_texts, pred_texts),
            (normd, gold_norm, pred_norm),
        ):
            tp, fp, fn = _exact_counts(gset, pset)
            acc[0] += tp
            acc[1] += fp
            acc[2] += fn
        tp, fp, fn = _contained_counts(gold_norm, pred_norm)
        cont[0] += tp
        cont[1] += fp
        cont[2] += fn

        if r.get("gold_spans"):
            golds = set(gold_texts)
            goldsn = set(gold_norm)
            if pred_texts:
                any_pred += 1
            if any(p in golds for p in pred_texts):
                exact_row += 1
            if any(pn in goldsn for pn in pred_norm):
                norm_row += 1
            if any(
                pn and gn and (pn in gn or gn in pn)
                for pn in pred_norm
                for gn in goldsn
            ):
                contained_row += 1

    n = len(gold_rows) or 1
    return {
        "predictions_path": str(predictions_path),
        "rows_total": len(rows),
        "rows_with_gold": len(gold_rows),
        "rows_with_pred_text": any_pred,
        # Primary leaderboard metric for author: offset-independent text-equal F1.
        "text_equal_f1": _prf(*raw),
        "text_equal_norm_f1": _prf(*normd),
        "text_contained_f1": _prf(*cont),
        # Row-level "got it at least once" rates (over rows with a gold span).
        "emitted_text_exact": exact_row,
        "emitted_text_exact_rate": exact_row / n,
        "emitted_text_norm": norm_row,
        "emitted_text_norm_rate": norm_row / n,
        "emitted_text_contained": contained_row,
        "emitted_text_contained_rate": contained_row / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = analyze(args.predictions)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output_json}")
    print(text)


if __name__ == "__main__":
    main()
