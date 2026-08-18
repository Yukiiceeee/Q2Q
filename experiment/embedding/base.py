"""Abstract base class for experiment embedding providers."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


def _format_turn(turn: dict) -> str:
    speaker = turn.get("speaker") or turn.get("role", "")
    text = turn.get("text") or turn.get("content", "")
    if not text:
        return ""
    return f"{speaker}: {text}"


class BaseEmbedding(ABC):

    def __init__(self, model_name: str, dimension: int = 0, max_seq_length: int = 512):
        self.model_name = model_name
        self.dimension = dimension
        self.max_seq_length = max_seq_length

    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        ...

    @abstractmethod
    def _count_tokens(self, text: str) -> int:
        ...

    async def embed_long_text(
        self,
        text: str,
        window_size: int = 0,
        stride: int = 0,
    ) -> list[np.ndarray]:
        ws = window_size or self.max_seq_length
        st = stride or (ws // 2)
        chunks = self._sliding_window_split(text, ws, st)
        if not chunks:
            return [await self.embed_text(text)]
        return await self.embed_batch(chunks)

    def _sliding_window_split(self, text: str, window_size: int, stride: int) -> list[str]:
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

    async def embed_session_turns(
        self,
        turns: list[dict],
        max_tokens: int = 480,
    ) -> tuple[list[np.ndarray], list[dict]]:
        formatted = [_format_turn(t) for t in turns]
        turn_indices = [(i, line) for i, line in enumerate(formatted) if line]

        if not turn_indices:
            emb = await self.embed_text("")
            return [emb], [{"chunk_index": 0, "turn_start": 0, "turn_end": 0, "text": ""}]

        newline_tokens = self._count_tokens("\n")
        chunks: list[str] = []
        meta: list[dict] = []
        current_lines: list[str] = []
        current_token_count = 0
        chunk_turn_start = turn_indices[0][0]

        for orig_idx, line in turn_indices:
            line_tokens = self._count_tokens(line)
            sep_tokens = newline_tokens if current_lines else 0

            if current_lines and current_token_count + sep_tokens + line_tokens > max_tokens:
                chunk_text = "\n".join(current_lines)
                chunks.append(chunk_text)
                meta.append({
                    "chunk_index": len(chunks) - 1,
                    "turn_start": chunk_turn_start,
                    "turn_end": prev_orig_idx,
                    "text": chunk_text,
                })
                current_lines = []
                current_token_count = 0
                chunk_turn_start = orig_idx

            current_lines.append(line)
            current_token_count += (sep_tokens + line_tokens) if len(current_lines) > 1 else line_tokens
            prev_orig_idx = orig_idx

            if line_tokens > self.max_seq_length:
                logger.warning(
                    f"Turn {orig_idx} has {line_tokens} tokens, "
                    f"exceeds model max_seq_length {self.max_seq_length}"
                )

        if current_lines:
            chunk_text = "\n".join(current_lines)
            chunks.append(chunk_text)
            meta.append({
                "chunk_index": len(chunks) - 1,
                "turn_start": chunk_turn_start,
                "turn_end": prev_orig_idx,
                "text": chunk_text,
            })

        embeddings = await self.embed_batch(chunks)
        return embeddings, meta

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def batch_cosine_similarity(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between two sets of vectors.

        Args:
            queries: (N, D) array
            targets: (M, D) array

        Returns:
            (N, M) similarity matrix
        """
        q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
        t_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-10)
        return q_norm @ t_norm.T
