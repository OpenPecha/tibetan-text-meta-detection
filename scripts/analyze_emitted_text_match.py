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
from pathlib import Path

_TRAILING = re.compile(r"[\s།་]+$")


def norm(s: str | None) -> str:
    return _TRAILING.sub("", (s or "").strip())


def analyze(predictions_path: Path) -> dict:
    rows = [
        json.loads(line)
        for line in predictions_path.open(encoding="utf-8")
        if line.strip()
    ]
    gold_rows = [r for r in rows if r.get("gold_spans")]
    exact = norm_match = contained = any_pred = 0
    for r in gold_rows:
        golds = {g.get("text", "") for g in r["gold_spans"]}
        goldsn = {norm(g.get("text", "")) for g in r["gold_spans"]}
        preds = [p.get("text", "") for p in r.get("pred_spans", [])]
        if preds:
            any_pred += 1
        if any(p in golds for p in preds):
            exact += 1
        if any(norm(p) in goldsn for p in preds):
            norm_match += 1
        hit = False
        for p in preds:
            pn = norm(p)
            for gn in goldsn:
                if pn and gn and (pn in gn or gn in pn):
                    hit = True
        if hit:
            contained += 1
    n = len(gold_rows) or 1
    return {
        "predictions_path": str(predictions_path),
        "rows_total": len(rows),
        "rows_with_gold": len(gold_rows),
        "rows_with_pred_text": any_pred,
        "emitted_text_exact": exact,
        "emitted_text_exact_rate": exact / n,
        "emitted_text_norm": norm_match,
        "emitted_text_norm_rate": norm_match / n,
        "emitted_text_contained": contained,
        "emitted_text_contained_rate": contained / n,
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
