from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):

    def __init__(self, model_name: str, max_seq_length: int = 512, dimension: int = 0):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dimension = dimension

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        ...

    def embed_long_text(
        self,
        text: str,
        window_size: int = 0,
        stride: int = 0,
    ) -> list[np.ndarray]:
        """Embed long text with sliding window, returning a list of chunk embeddings."""
        ws = window_size or self.max_seq_length
        st = stride or (ws // 2)
        chunks = self._sliding_window_split(text, ws, st)
        if not chunks:
            return [self.embed_text(text)]
        return self.embed_batch(chunks)

    def _sliding_window_split(self, text: str, window_size: int, stride: int) -> list[str]:
        """Split text into overlapping chunks by character count."""
        if len(text) <= window_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + window_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += stride
        return chunks

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
