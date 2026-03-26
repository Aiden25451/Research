import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import accuracy_score

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )

# 2. Load dataset
dataset = load_dataset("imdb")

# 3. Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# 4. Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


dataset = dataset.map(tokenize, batched=True)

# Use a small subset to keep it quick
train_dataset = dataset["train"].shuffle(seed=42).select(range(10000))
test_dataset = dataset["test"].shuffle(seed=42).select(range(2500))

# 5. Load model (2 labels: positive/negative) and move to GPU
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
).to(device)


# 6. Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./mymodels/distilbert-imdb",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # Mixed precision on GPU for speed
    dataloader_pin_memory=torch.cuda.is_available(),  # Faster CPU->GPU transfers
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 9. Train!
trainer.train()

# 10. Evaluate
results = trainer.evaluate()
print(results)

# 11. Save model
model.save_pretrained("./mymodels/my-distilbert-imdb")
tokenizer.save_pretrained("./mymodels/my-distilbert-imdb")
