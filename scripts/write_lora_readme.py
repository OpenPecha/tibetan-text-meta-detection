#!/usr/bin/env python3
"""Write Hugging Face model card for TiLamb LoRA adapters."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = {
    "title": ROOT / "hub" / "title_tilamb_lora_pilot_README.md",
    "author": ROOT / "hub" / "author_tilamb_lora_pilot_README.md",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["title", "author"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()

    card = CARDS[args.task]
    if not card.is_file():
        raise SystemExit(f"Missing {card}")
    text = card.read_text(encoding="utf-8")

    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
