from __future__ import annotations
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

    def detect_versions(
        self,
        new_queries: list[str],
        new_embeddings: list[np.ndarray],
    ) -> list[dict]:
        """For each new fake query, find existing FQs above threshold and build DAG edges.

        Returns a parallel list (one per query) of:
        {
            "parent_ids": list[str],   # direct parent query_ids (pruned to leaves)
            "depth": int,              # max parent depth + 1 (0 if no parents)
            "related_history": [...],  # full hit list for answer generation context
        }
        """
        results = []
        for i, (query_text, embedding) in enumerate(zip(new_queries, new_embeddings)):
            hits = self.memory_store.search_all_fake_queries(
                query_embedding=embedding,
                threshold=self.threshold,
                max_results=50,
            )

            if not hits:
                results.append({
                    "parent_ids": [],
                    "depth": 0,
                    "related_history": [],
                })
                continue

            parents = self._prune_to_leaves(hits)
            parents = sorted(parents, key=lambda h: h["score"], reverse=True)[:MAX_PARENT_COUNT]

            depth = max(h.get("depth", 0) for h in parents) + 1
            parent_ids = [h["query_id"] for h in parents]

            results.append({
                "parent_ids": parent_ids,
                "depth": depth,
                "related_history": hits,
            })

            logger.debug(
                f"  VersionDetect[{i}]: query=\"{query_text[:50]}\" "
                f"-> {len(hits)} hits, parents={parent_ids}, depth={depth}"
            )

        logger.info(
            f"VersionDetector: {len(new_queries)} queries processed, "
            f"{sum(1 for r in results if r['parent_ids'])} with parents"
        )
        return results

    def _prune_to_leaves(self, hits: list[dict]) -> list[dict]:
        """Keep only leaf nodes: if A is an ancestor of B (A.query_id in B.parent_ids), remove A."""
        hit_ids = {h["query_id"] for h in hits}
        ancestor_ids: set[str] = set()

        for h in hits:
            for pid in h.get("parent_ids", []):
                if pid in hit_ids:
                    ancestor_ids.add(pid)

        return [h for h in hits if h["query_id"] not in ancestor_ids]
