"""API-based embedding provider (OpenAI compatible)."""
from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

from experiment.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 5
RETRY_DELAY = 5.0


class APIEmbedding(BaseEmbedding):

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str = "",
        dimension: int = 0,
        max_seq_length: int = 8191,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        super().__init__(model_name, dimension, max_seq_length)
        self.batch_size = batch_size
        self._client = None
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        self._tiktoken_enc = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url or None,
        )
        return self._client

    async def embed_text(self, text: str) -> np.ndarray:
        results = await self._call_api([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = await self._call_api(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _call_api(self, texts: list[str]) -> list[np.ndarray]:
        client = self._get_client()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.embeddings.create(
                    input=texts,
                    model=self.model_name,
                )
                embeddings = []
                for item in sorted(response.data, key=lambda x: x.index):
                    emb = np.array(item.embedding, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    embeddings.append(emb)

                if self.dimension == 0 and embeddings:
                    self.dimension = len(embeddings[0])

                return embeddings
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"Embedding API call failed (attempt {attempt}): {e}, retrying..."
                    )
                    await asyncio.sleep(RETRY_DELAY * attempt)
                else:
                    raise

    def _count_tokens(self, text: str) -> int:
        if self._tiktoken_enc is None:
            import tiktoken
            try:
                self._tiktoken_enc = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        return len(self._tiktoken_enc.encode(text))
