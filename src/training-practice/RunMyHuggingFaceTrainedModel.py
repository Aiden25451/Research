"""Running my model from Hugging Face Tutorial. Source is the medium article Documents/School/Year4/Comp4960"""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

MODEL_NAME = "./mymodels/huggingface-finetuning"
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

for text in [
    "BERT is amazing!",
    "The movie was terrible",
    "I thought it would be terrible but it was solid",
    "It was great but it fell off hard",
]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    print(f"{text}: {"positive" if predicted_class == 1 else "negative"}")
