"""Embedding storage using numpy .npy files organized in directory structure.

Structure:
  {base_dir}/
    sessions/{session_id}/
      content_chunks.npy    (N, D) float32
      paragraphs.npy        (M, D) float32
    fake_queries/{session_id}/
      embeddings.npy        (K, D) float32
      texts.json            list[str]
    true_queries/{query_id}.npy   (D,) float32
    paraphrases/{query_id}/
      embeddings.npy        (5, D) float32
      styles.json           list[str]

This avoids h5py dependency while maintaining efficient numpy I/O.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingStore:

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # --- Save methods ---

    def save_session_content(
        self, session_id: str, content_chunks: list[np.ndarray]
    ) -> None:
        if not content_chunks:
            return
        path = self.base_dir / "sessions" / session_id
        path.mkdir(parents=True, exist_ok=True)
        data = np.stack(content_chunks, axis=0).astype(np.float32)
        np.save(path / "content_chunks.npy", data)

    def save_chunk_metadata(
        self, session_id: str, metadata: list[dict]
    ) -> None:
        path = self.base_dir / "sessions" / session_id
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "chunk_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def save_session_paragraphs(
        self, session_id: str, paragraph_embeddings: list[np.ndarray]
    ) -> None:
        if not paragraph_embeddings:
            return
        path = self.base_dir / "sessions" / session_id
        path.mkdir(parents=True, exist_ok=True)
        data = np.stack(paragraph_embeddings, axis=0).astype(np.float32)
        np.save(path / "paragraphs.npy", data)

    def save_fake_queries(
        self,
        session_id: str,
        embeddings: list[np.ndarray],
        texts: list[str],
    ) -> None:
        if not embeddings:
            return
        path = self.base_dir / "fake_queries" / session_id
        path.mkdir(parents=True, exist_ok=True)
        data = np.stack(embeddings, axis=0).astype(np.float32)
        np.save(path / "embeddings.npy", data)
        with open(path / "texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)

    def save_true_query(self, query_id: str, embedding: np.ndarray) -> None:
        path = self.base_dir / "true_queries"
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / f"{query_id}.npy", embedding.astype(np.float32))

    def save_paraphrases(
        self,
        query_id: str,
        embeddings: list[np.ndarray],
        styles: list[str],
    ) -> None:
        if not embeddings:
            return
        path = self.base_dir / "paraphrases" / query_id
        path.mkdir(parents=True, exist_ok=True)
        data = np.stack(embeddings, axis=0).astype(np.float32)
        np.save(path / "embeddings.npy", data)
        with open(path / "styles.json", "w", encoding="utf-8") as f:
            json.dump(styles, f, ensure_ascii=False)

    # --- Read methods ---

    def get_session_content(self, session_id: str) -> np.ndarray | None:
        path = self.base_dir / "sessions" / session_id / "content_chunks.npy"
        if path.exists():
            return np.load(path)
        return None

    def get_chunk_metadata(self, session_id: str) -> list[dict] | None:
        path = self.base_dir / "sessions" / session_id / "chunk_meta.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_session_paragraphs(self, session_id: str) -> np.ndarray | None:
        path = self.base_dir / "sessions" / session_id / "paragraphs.npy"
        if path.exists():
            return np.load(path)
        return None

    def get_fake_query_embeddings(self, session_id: str) -> np.ndarray | None:
        path = self.base_dir / "fake_queries" / session_id / "embeddings.npy"
        if path.exists():
            return np.load(path)
        return None

    def get_fake_query_texts(self, session_id: str) -> list[str] | None:
        path = self.base_dir / "fake_queries" / session_id / "texts.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_true_query_embedding(self, query_id: str) -> np.ndarray | None:
        path = self.base_dir / "true_queries" / f"{query_id}.npy"
        if path.exists():
            return np.load(path)
        return None

    def get_paraphrase_embeddings(self, query_id: str) -> np.ndarray | None:
        path = self.base_dir / "paraphrases" / query_id / "embeddings.npy"
        if path.exists():
            return np.load(path)
        return None

    def get_paraphrase_styles(self, query_id: str) -> list[str] | None:
        path = self.base_dir / "paraphrases" / query_id / "styles.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # --- Check methods ---

    def has_session(self, session_id: str) -> bool:
        return (self.base_dir / "sessions" / session_id / "content_chunks.npy").exists()

    def has_fake_queries(self, session_id: str) -> bool:
        return (self.base_dir / "fake_queries" / session_id / "embeddings.npy").exists()

    def has_true_query(self, query_id: str) -> bool:
        return (self.base_dir / "true_queries" / f"{query_id}.npy").exists()

    def has_paraphrases(self, query_id: str) -> bool:
        return (self.base_dir / "paraphrases" / query_id / "embeddings.npy").exists()

    def list_sessions(self) -> list[str]:
        path = self.base_dir / "sessions"
        if path.exists():
            return [d.name for d in path.iterdir() if d.is_dir()]
        return []

    def list_fake_query_sessions(self) -> list[str]:
        path = self.base_dir / "fake_queries"
        if path.exists():
            return [d.name for d in path.iterdir() if d.is_dir()]
        return []

    # --- Variant methods (propositions / notes / reflections) ---

    def save_variant(
        self,
        variant: str,
        session_id: str,
        embeddings: list[np.ndarray],
        texts: list[str],
    ) -> None:
        if not embeddings:
            return
        path = self.base_dir / variant / session_id
        path.mkdir(parents=True, exist_ok=True)
        data = np.stack(embeddings, axis=0).astype(np.float32)
        np.save(path / "embeddings.npy", data)
        with open(path / "texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)

    def get_variant_embeddings(
        self, variant: str, session_id: str
    ) -> np.ndarray | None:
        path = self.base_dir / variant / session_id / "embeddings.npy"
        if path.exists():
            return np.load(path)
        return None

    def get_variant_texts(
        self, variant: str, session_id: str
    ) -> list[str] | None:
        path = self.base_dir / variant / session_id / "texts.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def has_variant(self, variant: str, session_id: str) -> bool:
        return (
            self.base_dir / variant / session_id / "embeddings.npy"
        ).exists()

    def close(self):
        pass
