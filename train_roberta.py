"""Fine-tune Tibetan RoBERTa for title/author token classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from config import PROCESSED_DIR, ROBERTA_MODEL, SPLITS_DIR
from pipeline.bio import BIO_LABELS, ID_TO_LABEL, LABEL_TO_ID

FEATURE_COLUMNS = ["input_ids", "attention_mask", "labels"]
DEFAULT_ENTITY_WEIGHT = 10.0


def load_hf_split_datasets(
    repo_id: str,
    config_name: str = "windows",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    seed: int = 42,
):
    """Load train/validation/test from a Hugging Face dataset repo."""
    ds = load_dataset(repo_id, config_name)

    def trim_columns(d):
        drop = [c for c in d.column_names if c not in FEATURE_COLUMNS]
        return d.remove_columns(drop) if drop else d

    train_ds = trim_columns(ds["train"])
    val_full = trim_columns(ds["validation"])
    test_ds = trim_columns(ds["test"])

    if max_train_samples is not None and max_train_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=seed).select(range(max_train_samples))

    if max_val_samples is not None and max_val_samples < len(val_full):
        val_ds = val_full.shuffle(seed=seed).select(range(max_val_samples))
    else:
        val_ds = val_full

    return train_ds, val_ds, val_full, test_ds


def load_split_dataset(
    path: Path,
    max_samples: int | None = None,
    seed: int = 42,
):
    """Memory-map JSONL via HuggingFace datasets (no full Python list load)."""
    ds = load_dataset("json", data_files=str(path), split="train")
    drop_cols = [c for c in ds.column_names if c not in FEATURE_COLUMNS]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    return ds


def _decode_eval_sequences(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert label-id matrices to BIO tag sequences (ignore -100 padding)."""
    true_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []
    for pred_row, label_row in zip(predictions, labels):
        true_seq: list[str] = []
        pred_seq: list[str] = []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == -100:
                continue
            true_seq.append(ID_TO_LABEL[int(label_id)])
            pred_seq.append(ID_TO_LABEL[int(pred_id)])
        if true_seq:
            true_sequences.append(true_seq)
            pred_sequences.append(pred_seq)
    return true_sequences, pred_sequences


def _entity_f1_from_report(report: dict, entity: str) -> float:
    block = report.get(entity)
    if not isinstance(block, dict):
        return 0.0
    value = block.get("f1-score", 0.0)
    return float(value)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_sequences, pred_sequences = _decode_eval_sequences(predictions, labels)

    if not true_sequences:
        return {
            "span_f1": 0.0,
            "span_precision": 0.0,
            "span_recall": 0.0,
            "title_f1": 0.0,
            "author_f1": 0.0,
        }

    report = classification_report(
        true_sequences,
        pred_sequences,
        output_dict=True,
        zero_division=0,
    )
    return {
        "span_f1": float(f1_score(true_sequences, pred_sequences)),
        "span_precision": float(precision_score(true_sequences, pred_sequences)),
        "span_recall": float(recall_score(true_sequences, pred_sequences)),
        "title_f1": _entity_f1_from_report(report, "TITLE"),
        "author_f1": _entity_f1_from_report(report, "AUTHOR"),
    }


def build_class_weights(entity_weight: float, device: torch.device) -> torch.Tensor:
    weights = [1.0 if label == "O" else entity_weight for label in BIO_LABELS]
    return torch.tensor(weights, dtype=torch.float32, device=device)


class WeightedTrainer(Trainer):
    """Token-classification trainer with optional class-weighted cross-entropy."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for Tibetan NER")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory with train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument("--model", default=ROBERTA_MODEL)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Eval/test batch size per GPU (defaults to --batch-size). "
        "Can be set higher than train since inference uses less VRAM.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=10_000,
        help="Validation subset for live eval during training (~2 min at 10k). "
        "Full test.jsonl is used for final metrics.",
    )
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument(
        "--entity-weight",
        type=float,
        default=DEFAULT_ENTITY_WEIGHT,
        help="Cross-entropy weight for B-/I- entity labels (O stays 1.0)",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Hugging Face dataset repo id (e.g. ganga4364/tibetan-metadata-detector). "
        "When set, load splits from HF instead of local JSONL.",
    )
    parser.add_argument(
        "--hf-config",
        default="windows",
        help="HF dataset config name (default: windows)",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class-weighted loss (plain cross-entropy)",
    )
    args = parser.parse_args()
    eval_batch_size = args.eval_batch_size or args.batch_size

    train_path = args.splits_dir / "train.jsonl"
    val_path = args.splits_dir / "val.jsonl"
    test_path = args.splits_dir / "test.jsonl"
    if not args.hf_dataset:
        for path in (train_path, val_path, test_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Run prepare_data.py split first.")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(BIO_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    print("Loading datasets...")
    if args.hf_dataset:
        train_ds, val_ds, val_full, test_ds = load_hf_split_datasets(
            args.hf_dataset,
            config_name=args.hf_config,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            seed=args.eval_seed,
        )
        print(f"  Source: {args.hf_dataset} (config={args.hf_config})")
    else:
        train_ds = load_split_dataset(train_path, max_samples=args.max_train_samples)
        val_full = load_split_dataset(val_path)
        val_n = min(args.max_val_samples, len(val_full))
        val_ds = (
            val_full.shuffle(seed=args.eval_seed).select(range(val_n))
            if val_n < len(val_full)
            else val_full
        )
        test_ds = load_split_dataset(test_path)

    print(
        f"Dataset sizes: train={len(train_ds)}, "
        f"val={len(val_ds)}/{len(val_full)} (live subset/full), "
        f"test={len(test_ds)} (final eval)"
    )
    print(
        f"Batch sizes: train={args.batch_size}, "
        f"eval/test={eval_batch_size} (per GPU)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = None
    if not args.no_class_weights:
        class_weights = build_class_weights(args.entity_weight, device)
        print(f"Class weights: O=1.0, entity={args.entity_weight}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="span_f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer_cls = WeightedTrainer if class_weights is not None else Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "processing_class": tokenizer,
        "compute_metrics": compute_metrics,
    }
    if class_weights is not None:
        trainer_kwargs["class_weights"] = class_weights

    trainer = trainer_cls(**trainer_kwargs)

    print("Starting training...")
    trainer.train()

    print(f"Evaluating on full test set ({len(test_ds)} examples)...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    metrics_path = args.output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(f"Test metrics: {test_metrics}")

    trainer.save_model(str(args.output_dir / "best"))
    tokenizer.save_pretrained(str(args.output_dir / "best"))
    print(f"Model saved to {args.output_dir / 'best'}")


if __name__ == "__main__":
    main()
