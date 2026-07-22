from __future__ import annotations
import os
import numpy as np
import logging

from src.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class APIEmbeddingProvider(BaseEmbeddingProvider):

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        dimension: int = 1536,
        max_seq_length: int = 8191,
    ):
        super().__init__(model_name=model_name, max_seq_length=max_seq_length, dimension=dimension)
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        logger.info(f"Initialized API embedding provider: {model_name} (dim={dimension})")

    async def embed_text(self, text: str) -> np.ndarray:
        response = await self._client.embeddings.create(input=[text], model=self.model_name)
        return np.array(response.data[0].embedding, dtype=np.float32)

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        response = await self._client.embeddings.create(input=texts, model=self.model_name)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [np.array(d.embedding, dtype=np.float32) for d in sorted_data]
