from __future__ import annotations
import asyncio
import logging

import numpy as np

from src.storage.base import BaseMemoryStore
from src.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

MAX_PARENT_COUNT = 3


class VersionDetector:

    def __init__(
        self,
        memory_store: BaseMemoryStore,
        embedding_provider: BaseEmbeddingProvider,
        threshold: float = 0.80,
    ):
        self.memory_store = memory_store
        self.embedding_provider = embedding_provider
        self.threshold = threshold

    async def detect_versions(
        self,
        new_queries: list[str],
        new_embeddings: list[np.ndarray],
    ) -> list[dict]:
        """For each new fake query, find existing FQs above threshold and build DAG edges."""

        async def _detect_single(query_text: str, embedding: np.ndarray, idx: int) -> dict:
            hits = await self.memory_store.search_all_fake_queries(
                query_embedding=embedding,
                threshold=self.threshold,
                max_results=50,
            )
            if not hits:
                return {"parent_ids": [], "depth": 0, "related_history": []}

            parents = self._prune_to_leaves(hits)
            parents = sorted(parents, key=lambda h: h["score"], reverse=True)[:MAX_PARENT_COUNT]
            depth = max(h.get("depth", 0) for h in parents) + 1
            parent_ids = [h["query_id"] for h in parents]

            logger.debug(
                f"  VersionDetect[{idx}]: query=\"{query_text[:50]}\" "
                f"-> {len(hits)} hits, parents={parent_ids}, depth={depth}"
            )
            return {"parent_ids": parent_ids, "depth": depth, "related_history": hits}

        results = await asyncio.gather(*[
            _detect_single(qt, emb, i)
            for i, (qt, emb) in enumerate(zip(new_queries, new_embeddings))
        ])

        logger.info(
            f"VersionDetector: {len(new_queries)} queries processed, "
            f"{sum(1 for r in results if r['parent_ids'])} with parents"
        )
        return list(results)

    def _prune_to_leaves(self, hits: list[dict]) -> list[dict]:
        hit_ids = {h["query_id"] for h in hits}
        ancestor_ids: set[str] = set()
        for h in hits:
            for pid in h.get("parent_ids", []):
                if pid in hit_ids:
                    ancestor_ids.add(pid)
        return [h for h in hits if h["query_id"] not in ancestor_ids]
