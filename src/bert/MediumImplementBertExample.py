"""Example following basic tutorial. Using pretrained pre fine tuned Bert model for sentiment analysis"""

from transformers import BertTokenizer, BertForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "textattack/bert-base-uncased-imdb"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.to(device)

print("Device:", device)
print(torch.cuda.is_available())

for text in [
    "I really enjoyed this movie! The plot was engaging and the acting was superb. I would highly recommend it to anyone looking for a great film.",
    "This was a terrible movie. The story was boring and the acting was awful. I would not recommend it to anyone.",
]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    print("Predicted Sentiment Class:", "Good" if predicted_class == 1 else "Bad")

print(model)
