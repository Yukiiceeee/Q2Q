from __future__ import annotations
import uuid
import logging

import numpy as np

from src.storage.base import BaseMemoryStore
from src.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


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
        """For each new fake query, find all existing fake queries above threshold.

        Returns a parallel list (one per query) of:
        {
            "related_history": [{"query_id", "text", "answer", "score", "chain_id", "version_seq"}],
            "chain_id": str,
            "version_seq": int,
            "supersedes": str,  # pipe-delimited query_ids
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
                    "related_history": [],
                    "chain_id": uuid.uuid4().hex[:12],
                    "version_seq": 0,
                    "supersedes": "",
                })
            else:
                best_hit = max(hits, key=lambda h: h["score"])
                chain_id = best_hit.get("chain_id") or uuid.uuid4().hex[:12]
                max_seq = max(h.get("version_seq", 0) for h in hits)
                supersede_ids = [h["query_id"] for h in hits]

                results.append({
                    "related_history": hits,
                    "chain_id": chain_id,
                    "version_seq": max_seq + 1,
                    "supersedes": "|".join(supersede_ids),
                })

                logger.debug(
                    f"  VersionDetect[{i}]: query=\"{query_text[:50]}\" "
                    f"-> {len(hits)} hits, chain={chain_id}, seq={max_seq + 1}"
                )

        logger.info(
            f"VersionDetector: {len(new_queries)} queries processed, "
            f"{sum(1 for r in results if r['related_history'])} with history"
        )
        return results
