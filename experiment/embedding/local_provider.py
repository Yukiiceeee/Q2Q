"""Local embedding provider using SentenceTransformers."""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from experiment.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)


class LocalEmbedding(BaseEmbedding):

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        dimension: int = 0,
        max_seq_length: int = 512,
        batch_size: int = 32,
    ):
        super().__init__(model_name, dimension, max_seq_length)
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading local embedding model: {self.model_name} on {self.device}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        if self.max_seq_length:
            self._model.max_seq_length = self.max_seq_length
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded: dim={self.dimension}")

    async def embed_text(self, text: str) -> np.ndarray:
        self._load_model()

        def _encode():
            return self._model.encode(
                text, normalize_embeddings=True, show_progress_bar=False
            )

        result = await asyncio.to_thread(_encode)
        return np.array(result, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        self._load_model()

        def _encode_batch():
            return self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )

        results = await asyncio.to_thread(_encode_batch)
        return [np.array(r, dtype=np.float32) for r in results]

    def _count_tokens(self, text: str) -> int:
        self._load_model()
        return len(self._model.tokenizer.tokenize(text))
