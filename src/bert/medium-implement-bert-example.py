from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "textattack/bert-base-uncased-imdb"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

for text in [
    "BERT is amazing!",
    "BERT is not amazing!",
    "Potatos taste gross sometimes",
    "This movie could've been better",
    "I thought this movie would be terrible but it was great",
]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    print("Predicted Sentiment Class:", "Good" if predicted_class == 1 else "Bad")
