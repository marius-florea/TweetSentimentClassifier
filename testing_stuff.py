from transformers import DistilBertTokenizer,BertTokenizer, BertForSequenceClassification,DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("bert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
print(outputs)