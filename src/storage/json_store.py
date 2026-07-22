from __future__ import annotations
import asyncio
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
            self._write_raw_sync([])

    def _read_raw_sync(self) -> list[dict]:
        if not self.storage_path.exists():
            return []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_raw_sync(self, data: list[dict]) -> None:
        with open(self.storage_path, "w", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, ensure_ascii=False, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    async def _read_raw(self) -> list[dict]:
        return await asyncio.to_thread(self._read_raw_sync)

    async def _write_raw(self, data: list[dict]) -> None:
        await asyncio.to_thread(self._write_raw_sync, data)

    async def save(self, entry: MemoryEntry) -> None:
        entries = await self._read_raw()
        entries = [e for e in entries if e["memory_id"] != entry.memory_id]
        entries.append(entry.to_dict())
        await self._write_raw(entries)
        logger.info(f"Saved memory entry: {entry.memory_id}")

    async def load_all(self) -> list[MemoryEntry]:
        raw = await self._read_raw()
        return [MemoryEntry.from_dict(d) for d in raw]

    async def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        for d in await self._read_raw():
            if d["memory_id"] == memory_id:
                return MemoryEntry.from_dict(d)
        return None

    async def delete(self, memory_id: str) -> bool:
        entries = await self._read_raw()
        new_entries = [e for e in entries if e["memory_id"] != memory_id]
        if len(new_entries) == len(entries):
            return False
        await self._write_raw(new_entries)
        logger.info(f"Deleted memory entry: {memory_id}")
        return True

    async def count(self) -> int:
        return len(await self._read_raw())

    async def clear(self) -> None:
        await self._write_raw([])
        logger.info("Cleared all memory entries")

    async def search_all_fake_queries(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.80,
        max_results: int = 50,
    ) -> list[dict]:
        entries = await self.load_all()

        def _compute():
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
                            "parent_ids": fq.parent_ids,
                            "depth": fq.depth,
                            "created_at": entry.created_at,
                        })
            hits.sort(key=lambda h: h["score"], reverse=True)
            return hits[:max_results]
        return await asyncio.to_thread(_compute)

    async def get_fake_query_by_id(self, query_id: str) -> dict | None:
        entries = await self.load_all()
        for entry in entries:
            for fq in entry.fake_queries:
                if fq.query_id == query_id:
                    return {
                        "query_id": fq.query_id,
                        "text": fq.text,
                        "answer": fq.answer,
                        "memory_id": fq.memory_id,
                        "parent_ids": fq.parent_ids,
                        "depth": fq.depth,
                        "created_at": entry.created_at,
                    }
        return None

    async def get_fake_queries_by_ids(self, query_ids: list[str]) -> list[dict]:
        if not query_ids:
            return []
        id_set = set(query_ids)
        entries = await self.load_all()
        nodes = []
        for entry in entries:
            for fq in entry.fake_queries:
                if fq.query_id in id_set:
                    nodes.append({
                        "query_id": fq.query_id,
                        "text": fq.text,
                        "answer": fq.answer,
                        "memory_id": fq.memory_id,
                        "parent_ids": fq.parent_ids,
                        "depth": fq.depth,
                        "created_at": entry.created_at,
                    })
        return nodes

    async def search_paragraphs_for_memory(
        self,
        query_embedding: np.ndarray,
        memory_id: str,
        top_k: int = 3,
    ) -> list[dict]:
        entry = await self.get_by_id(memory_id)
        if not entry or not entry.paragraphs or not entry.paragraph_embeddings:
            return []

        def _compute():
            hits = []
            for i, (para, emb) in enumerate(zip(entry.paragraphs, entry.paragraph_embeddings)):
                score = float(BaseEmbeddingProvider.cosine_similarity(query_embedding, emb))
                hits.append({"score": score, "text": para})
            hits.sort(key=lambda h: h["score"], reverse=True)
            return hits[:top_k]
        return await asyncio.to_thread(_compute)

    async def search_kps_for_memory(
        self,
        query_embedding: np.ndarray,
        memory_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        entry = await self.get_by_id(memory_id)
        if not entry or not entry.knowledge_points or not entry.kp_embeddings:
            return []

        def _compute():
            hits = []
            for i, (kp, emb) in enumerate(zip(entry.knowledge_points, entry.kp_embeddings)):
                score = float(BaseEmbeddingProvider.cosine_similarity(query_embedding, emb))
                kp_text = f"[{kp.time}] {kp.subject}: {kp.fact} ({kp.entities_or_values})"
                hits.append({"score": score, "kp_index": i, "text": kp_text})
            hits.sort(key=lambda h: h["score"], reverse=True)
            return hits[:top_k]
        return await asyncio.to_thread(_compute)
