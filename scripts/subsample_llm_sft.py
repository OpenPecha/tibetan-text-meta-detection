"""Subsample LLM SFT JSONL splits (train/val/test) for pilot runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _subsample_file(src: Path, dst: Path, fraction: float, seed: int) -> int:
    lines = src.read_text(encoding="utf-8").splitlines()
    if not lines:
        dst.write_text("", encoding="utf-8")
        return 0
    rng = random.Random(seed)
    indices = list(range(len(lines)))
    rng.shuffle(indices)
    keep = max(1, int(len(lines) * fraction))
    chosen = sorted(indices[:keep])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines[i] for i in chosen) + "\n", encoding="utf-8")
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/llm_sft"),
        help="Full LLM SFT output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/llm_sft_pilot_10pct"),
        help="Pilot output directory",
    )
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        raise SystemExit("--fraction must be in (0, 1]")

    stats: dict[str, int] = {}
    for task in ("title", "author"):
        for split in ("train", "val", "test"):
            for suffix in ("", "_meta"):
                name = f"{split}{suffix}.jsonl"
                src = args.source_dir / task / name
                if not src.exists():
                    continue
                dst = args.output_dir / task / name
                n = _subsample_file(
                    src,
                    dst,
                    args.fraction,
                    args.seed + hash(f"{task}:{split}{suffix}") % 10_000,
                )
                stats[f"{task}/{name}"] = n

    print(json.dumps({"output_dir": str(args.output_dir), "counts": stats}, indent=2))


if __name__ == "__main__":
    main()
