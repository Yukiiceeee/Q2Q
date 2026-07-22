from __future__ import annotations
import asyncio
import numpy as np
import logging

from src.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(self, model_path: str, max_seq_length: int = 512, device: str = "cpu"):
        super().__init__(model_name=model_path, max_seq_length=max_seq_length)
        self.device = device
        self._model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_path, device=self.device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        if self._model.max_seq_length:
            self.max_seq_length = self._model.max_seq_length
        logger.info(f"Loaded local embedding model: {model_path} (dim={self.dimension})")

    async def embed_text(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(
            self._model.encode, text, normalize_embeddings=True
        )

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True, batch_size=32
        )
        return [embeddings[i] for i in range(len(texts))]
