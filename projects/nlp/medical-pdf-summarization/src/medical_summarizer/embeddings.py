"""BioBERT sentence embedding implementation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BioBERTEmbedder:
    """Generate attention-mask-aware mean-pooled BioBERT embeddings."""

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        device: str | None = None,
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return L2-normalized embedding vectors with shape (N, hidden_size)."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not sentences:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        vectors = []
        for start in range(0, len(sentences), batch_size):
            batch = list(sentences[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self.model(**encoded)

            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            pooled = F.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))

        return np.concatenate(vectors, axis=0)
