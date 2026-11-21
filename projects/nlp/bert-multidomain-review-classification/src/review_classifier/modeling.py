from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_tokenizer_and_model(
    model_name: str,
    categories: list[str],
    local_files_only: bool = False,
):
    label2id = {label: index for index, label in enumerate(categories)}
    id2label = {index: label for index, label in enumerate(categories)}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, local_files_only=local_files_only
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(categories),
        label2id=label2id,
        id2label=id2label,
        local_files_only=local_files_only,
    )
    return tokenizer, model


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {"text": self.texts[index], "labels": self.labels[index]}


class DynamicPaddingCollator:
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        encodings = self.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings["labels"] = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return encodings
