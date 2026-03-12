import torch
from transformers import AutoModelForSequenceClassification

def build_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    return model