"""Using the huggin face fine tuning tutorial https://huggingface.co/docs/transformers/training"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Getting tokenizer and model
MODEL_NAME = "distilbert-base-uncased"
print(f"Getting instance of tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)


# Preparing the dataset
def tokenize(batch):
    """Tokenize the input text."""
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
    )


dataset = load_dataset("imdb", split="train")
print(f"First sample: \n{dataset[0]}\n\n")
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=1000, train_size=10000, seed=1)
print(f"Training structure: \n{dataset.items()}\n\n")

# Training following HuggingFace Instructions
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=64,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)


print(f"Initial Results Before Fine Tuning: {trainer.evaluate()}")
print("BEGIN TRAINING: ")
trainer.train()

print("Saving pretrained models")
model.save_pretrained("./mymodels/huggingface-finetuning")
tokenizer.save_pretrained("./mymodels/huggingface-finetuning")
