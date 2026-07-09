from __future__ import annotations
import json
import os
import fcntl
import logging
from pathlib import Path

import numpy as np

from src.schemas.models import MemoryEntry
from src.embedding.base import BaseEmbeddingProvider
from src.storage.base import BaseMemoryStore

logger = logging.getLogger(__name__)


class JsonMemoryStore(BaseMemoryStore):

    def __init__(self, storage_path: str = "data/memories.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._write_raw([])

    def _read_raw(self) -> list[dict]:
        if not self.storage_path.exists():
            return []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_raw(self, data: list[dict]) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def save(self, entry: MemoryEntry) -> None:
        entries = self._read_raw()
        entries = [e for e in entries if e["memory_id"] != entry.memory_id]
        entries.append(entry.to_dict())
        self._write_raw(entries)
        logger.info(f"Saved memory entry: {entry.memory_id}")

    def load_all(self) -> list[MemoryEntry]:
        raw = self._read_raw()
        return [MemoryEntry.from_dict(d) for d in raw]

    def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        for d in self._read_raw():
            if d["memory_id"] == memory_id:
                return MemoryEntry.from_dict(d)
        return None

    def delete(self, memory_id: str) -> bool:
        entries = self._read_raw()
        new_entries = [e for e in entries if e["memory_id"] != memory_id]
        if len(new_entries) == len(entries):
            return False
        self._write_raw(new_entries)
        logger.info(f"Deleted memory entry: {memory_id}")
        return True

    def count(self) -> int:
        return len(self._read_raw())

    def clear(self) -> None:
        self._write_raw([])
        logger.info("Cleared all memory entries")

    def search_all_fake_queries(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.80,
        max_results: int = 50,
    ) -> list[dict]:
        """Brute-force search all fake queries above threshold."""
        entries = self.load_all()
        hits = []
        for entry in entries:
            for fq in entry.fake_queries:
                if fq.embedding is None:
                    continue
                score = float(BaseEmbeddingProvider.cosine_similarity(query_embedding, fq.embedding))
                if score >= threshold:
                    hits.append({
                        "query_id": fq.query_id,
                        "memory_id": fq.memory_id,
                        "score": score,
                        "text": fq.text,
                        "answer": fq.answer,
                        "supersedes": fq.supersedes,
                        "version_seq": fq.version_seq,
                        "chain_id": fq.chain_id,
                        "created_at": entry.created_at,
                    })
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:max_results]

    def get_chain_nodes(self, chain_id: str) -> list[dict]:
        """Get all fake query nodes in a version chain."""
        if not chain_id:
            return []
        entries = self.load_all()
        nodes = []
        for entry in entries:
            for fq in entry.fake_queries:
                if fq.chain_id == chain_id:
                    nodes.append({
                        "query_id": fq.query_id,
                        "text": fq.text,
                        "answer": fq.answer,
                        "memory_id": fq.memory_id,
                        "version_seq": fq.version_seq,
                        "chain_id": chain_id,
                        "created_at": entry.created_at,
                    })
        nodes.sort(key=lambda n: n["version_seq"])
        return nodes
