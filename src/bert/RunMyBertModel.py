from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load your saved model
model = DistilBertForSequenceClassification.from_pretrained(
    "./mymodels/my-distilbert-imdb"
)
tokenizer = DistilBertTokenizer.from_pretrained("./mymodels/my-distilbert-imdb")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][label].item()
    return {
        "label": "POSITIVE" if label == 1 else "NEGATIVE",
        "confidence": round(confidence * 100, 2),
    }


# Try it
print(predict("This movie was absolutely fantastic, I loved every second!"))
print(predict("Terrible film, complete waste of time."))
print(predict("It was okay, nothing special but not bad either."))
print(predict("I thought it would be terrible but it was not bad at all"))
