from datasets import load_dataset
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "distilbert-base-uncased"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"models/model_{timestamp}"

# Load dataset
dataset = load_dataset("csv", data_files="data/raw.csv")

# Train/validation split
dataset = dataset["train"].train_test_split(test_size=0.2)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    logging_dir="logs",
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
trainer.save_model(OUTPUT_DIR)