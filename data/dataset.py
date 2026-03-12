from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

class IMDBDataset(torch.utils.data.Dataset):

    def __init__(self, split="train"):
        dataset = load_dataset("imdb", split=split)
        self.texts = dataset["text"]
        self.labels = dataset["label"]

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])

        return item

if __name__ == '__main__':
    dataset = IMDBDataset()
    dataload = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )
    print(next(iter(dataload)))
