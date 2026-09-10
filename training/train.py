"""Fine-tune a sequence-classification model on the DVC-tracked dataset.

Every run writes a self-contained, ready-to-upload model directory to
``models/model_<timestamp>/`` and records its name in ``models/latest.txt``
so that ``push_model.py`` always knows which run to publish.

Configuration comes from environment variables so the GitHub Actions workflow
can tune a run without editing code.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "raw.csv"
MODELS_DIR = REPO_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / ".checkpoints"

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased")
NUM_EPOCHS = float(os.getenv("NUM_EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "5e-5"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
SEED = int(os.getenv("SEED", "42"))


def load_split_dataset():
    if not DATA_FILE.exists():
        raise SystemExit(
            f"Dataset not found at {DATA_FILE}.\n"
            "It is tracked by DVC and deliberately kept out of git. Run:\n"
            "    dvc pull\n"
            "to fetch it from the configured remote before training."
        )

    dataset = load_dataset("csv", data_files=str(DATA_FILE))["train"]

    missing = {"text", "label"} - set(dataset.column_names)
    if missing:
        raise SystemExit(
            f"{DATA_FILE} is missing required column(s): {sorted(missing)}. "
            f"Found columns: {dataset.column_names}"
        )

    # train_test_split rounds the eval side up; both sides must end up non-empty.
    n_eval = int(np.ceil(len(dataset) * TEST_SIZE))
    if n_eval < 1 or len(dataset) - n_eval < 1:
        raise SystemExit(
            f"A {TEST_SIZE:.0%} eval split of {len(dataset)} rows leaves "
            f"{len(dataset) - n_eval} training and {n_eval} eval rows. "
            f"Add more rows to {DATA_FILE} or lower TEST_SIZE."
        )

    return dataset, dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)


def resolve_label_names(num_labels: int) -> list[str]:
    configured = os.getenv("LABEL_NAMES", "negative,positive" if num_labels == 2 else "")
    names = [name.strip() for name in configured.split(",") if name.strip()]
    if len(names) == num_labels:
        return names
    return [f"LABEL_{i}" for i in range(num_labels)]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def write_model_card(run_dir: Path, metrics: dict, num_rows: int) -> None:
    """Write the README that becomes the model card on the Hugging Face Hub."""
    scored = "\n".join(
        f"- **{key}**: {value:.4f}"
        for key, value in sorted(metrics.items())
        if isinstance(value, (int, float))
    )
    run_dir.joinpath("README.md").write_text(
        f"""---
license: apache-2.0
base_model: {MODEL_NAME}
pipeline_tag: text-classification
tags:
- text-classification
- generated-by-automatic-ml-pipeline
---

# {run_dir.name}

Fine-tuned from [`{MODEL_NAME}`](https://huggingface.co/{MODEL_NAME}) by the
[Automatic ML Pipeline](https://github.com/KCP17/Automatic-Machine-Learning-Pipeline),
which retrains and republishes automatically whenever the dataset or training
code changes.

## Training run

| | |
|---|---|
| Trained at (UTC) | {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} |
| Dataset rows | {num_rows} |
| Epochs | {NUM_EPOCHS:g} |
| Batch size | {BATCH_SIZE} |
| Learning rate | {LEARNING_RATE:g} |
| Seed | {SEED} |

## Evaluation

{scored}

The dataset is versioned with DVC; see the repository for the exact revision.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="Kiernan1410/auto-ml-model")
clf("I love this product")
```
""",
        encoding="utf-8",
    )


def main() -> None:
    set_seed(SEED)

    raw_dataset, dataset = load_split_dataset()
    num_labels = len(set(raw_dataset["label"]))
    label_names = resolve_label_names(num_labels)
    print(f"Loaded {len(raw_dataset)} rows, {num_labels} labels: {label_names}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    # Pad per batch via the collator instead of to MAX_LENGTH for every row.
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = MODELS_DIR / f"model_{timestamp}"
    checkpoint_dir = CHECKPOINTS_DIR / f"model_{timestamp}"

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        # The final model is exported explicitly below (trainer.save_model +
        # tokenizer), so per-epoch checkpoints would only cost disk and CI time.
        save_strategy="no",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    raw_metrics = trainer.evaluate()
    print(f"Evaluation: {raw_metrics}")

    # Keep the quality metrics; drop Trainer's timing/throughput/epoch noise.
    noise = ("_runtime", "_samples_per_second", "_steps_per_second")
    metrics = {
        key.removeprefix("eval_"): value
        for key, value in raw_metrics.items()
        if key != "epoch" and not key.endswith(noise)
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(run_dir))
    # Without the tokenizer the uploaded model cannot be used for inference.
    tokenizer.save_pretrained(str(run_dir))

    run_dir.joinpath("metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_model_card(run_dir, metrics, len(raw_dataset))

    # push_model.py reads this so publishing never depends on directory ordering.
    MODELS_DIR.joinpath("latest.txt").write_text(run_dir.name, encoding="utf-8")

    # Trainer may still leave scratch state behind; run_dir supersedes all of it.
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    print(f"Saved model to {run_dir}")


if __name__ == "__main__":
    main()
