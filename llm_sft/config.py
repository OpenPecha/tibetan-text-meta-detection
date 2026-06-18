"""Defaults for TiLamb-oriented LLM SFT dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config import RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO

TILAMB_MODEL = "YoLo2000/TiLamb-7B"

# Char hints for span-centered crops (converted to token budget per example).
DEFAULT_CROP_PRESETS: tuple[tuple[int, int], ...] = (
    (500, 1000),
    (100, 1400),
    (750, 750),
    (200, 200),
    (0, 0),  # minimal padding; random slack fills the rest
)


@dataclass
class SFTConfig:
    extracted_dir: Path
    output_dir: Path
    tokenizer_name: str = TILAMB_MODEL
    max_context_tokens: int = 3584
    crops_per_positive: int = 3
    crops_per_negative: int = 1
    crop_presets: tuple[tuple[int, int], ...] = DEFAULT_CROP_PRESETS
    random_slack: bool = True
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO
    seed: int = RANDOM_SEED
    tasks: tuple[str, ...] = ("title", "author")

    @property
    def title_dir(self) -> Path:
        return self.output_dir / "title"

    @property
    def author_dir(self) -> Path:
        return self.output_dir / "author"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"
