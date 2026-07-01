from __future__ import annotations
import logging
import numpy as np

from src.schemas.models import (
    MemoryEntry,
    SubQuery,
    RetrievalResult,
)
from src.embedding.base import BaseEmbeddingProvider
from src.storage.base import BaseMemoryStore
from src.storage.chromadb_store import ChromaDBStore

logger = logging.getLogger(__name__)


class DualRetriever:

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        memory_store: BaseMemoryStore,
        alpha: float = 0.7,
        top_k_per_sub: int = 5,
        top_n: int = 10,
    ):
        self.embedding_provider = embedding_provider
        self.memory_store = memory_store
        self.alpha = alpha
        self.top_k_per_sub = top_k_per_sub
        self.top_n = top_n
        self._use_chromadb = isinstance(memory_store, ChromaDBStore)

    def retrieve(self, sub_queries: list[SubQuery]) -> list[RetrievalResult]:
        if self._use_chromadb:
            return self._retrieve_chromadb(sub_queries)
        else:
            return self._retrieve_bruteforce(sub_queries)

    # ========== ChromaDB path: sequential Q2Q → Q2C ==========

    def _retrieve_chromadb(self, sub_queries: list[SubQuery]) -> list[RetrievalResult]:
        store: ChromaDBStore = self.memory_store

        # --- Phase 1: Q2Q search across all sub-queries ---
        # Collect top-k fake query hits per sub-query, aggregate by memory_id
        # memory_id -> {score_q2q, matched_fqs: [(fq_text, sq_text, score)]}
        candidate_map: dict[str, dict] = {}

        fq_top_k = max(self.top_k_per_sub * 4, 20)

        for sq in sub_queries:
            if sq.embedding is None:
                continue

            fq_hits = store.search_fake_queries(sq.embedding, top_k=fq_top_k)
            for hit in fq_hits:
                mid = hit["memory_id"]
                if mid not in candidate_map:
                    candidate_map[mid] = {
                        "score_q2q": 0.0,
                        "matched_fqs": [],
                    }
                fq_text = hit["text"]
                fq_score = hit["score"]

                if fq_score > candidate_map[mid]["score_q2q"]:
                    candidate_map[mid]["score_q2q"] = fq_score

                already = any(t == fq_text for t, _, _ in candidate_map[mid]["matched_fqs"])
                if not already:
                    candidate_map[mid]["matched_fqs"].append((fq_text, sq.text, fq_score))

        if not candidate_map:
            logger.info("[ChromaDB] Q2Q found no candidates")
            return []

        # --- Phase 2: Q2C scoring for each candidate memory ---
        # Compute best content similarity for each candidate memory_id
        # Use the sub_query with highest Q2Q score as representative query embedding
        best_sq_embedding = None
        best_sq_score = -1.0
        for sq in sub_queries:
            if sq.embedding is None:
                continue
            for mid, info in candidate_map.items():
                for _, sq_text, fq_score in info["matched_fqs"]:
                    if sq_text == sq.text and fq_score > best_sq_score:
                        best_sq_score = fq_score
                        best_sq_embedding = sq.embedding

        for mid in candidate_map:
            best_q2c = 0.0
            for sq in sub_queries:
                if sq.embedding is None:
                    continue
                q2c_score = store.search_content_for_memory(
                    sq.embedding, memory_id=mid, top_k=3,
                )
                if q2c_score > best_q2c:
                    best_q2c = q2c_score
            candidate_map[mid]["score_q2c"] = best_q2c

        # --- Phase 3: combine scores, build results ---
        results: list[RetrievalResult] = []
        for mid, info in candidate_map.items():
            final_score = self.alpha * info["score_q2q"] + (1 - self.alpha) * info["score_q2c"]
            memory = store.get_by_id(mid)
            if memory is None:
                continue

            fq_list = sorted(info["matched_fqs"], key=lambda x: x[2], reverse=True)
            matched_fqs = [t for t, _, _ in fq_list]
            matched_sqs = list(dict.fromkeys(s for _, s, _ in fq_list))

            results.append(RetrievalResult(
                memory=memory,
                score_q2q=info["score_q2q"],
                score_q2c=info.get("score_q2c", 0.0),
                final_score=final_score,
                matched_fake_queries=matched_fqs,
                matched_sub_queries=matched_sqs,
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)
        top_results = results[:self.top_n]

        logger.info(
            f"[ChromaDB] Retrieved {len(top_results)} memories from "
            f"{len(sub_queries)} sub-queries (Q2Q candidates: {len(candidate_map)})"
        )
        return top_results

    # ========== Brute-force path: sequential Q2Q → Q2C ==========

    def _retrieve_bruteforce(self, sub_queries: list[SubQuery]) -> list[RetrievalResult]:
        memories = self.memory_store.load_all()
        if not memories:
            logger.warning("Memory store is empty, no results to retrieve")
            return []

        # Phase 1: Q2Q scoring for all memories
        # memory_id -> {score_q2q, matched_fqs, memory}
        candidate_map: dict[str, dict] = {}

        for memory in memories:
            mid = memory.memory_id
            best_q2q = 0.0
            matched_fqs: list[tuple[str, str, float]] = []

            for sq in sub_queries:
                if sq.embedding is None:
                    continue
                score, fq_text = self._compute_q2q_score(sq, memory)
                if score > best_q2q:
                    best_q2q = score
                if score > 0.0 and fq_text:
                    already = any(t == fq_text for t, _, _ in matched_fqs)
                    if not already:
                        matched_fqs.append((fq_text, sq.text, score))

            candidate_map[mid] = {
                "score_q2q": best_q2q,
                "matched_fqs": matched_fqs,
                "memory": memory,
            }

        # Phase 2: Q2C scoring for all memories
        for mid, info in candidate_map.items():
            memory = info["memory"]
            best_q2c = 0.0
            for sq in sub_queries:
                if sq.embedding is None:
                    continue
                score = self._compute_q2c_score(sq, memory)
                if score > best_q2c:
                    best_q2c = score
            info["score_q2c"] = best_q2c

        # Phase 3: combine and sort
        results: list[RetrievalResult] = []
        for mid, info in candidate_map.items():
            final_score = self.alpha * info["score_q2q"] + (1 - self.alpha) * info["score_q2c"]

            fq_list = sorted(info["matched_fqs"], key=lambda x: x[2], reverse=True)
            matched_fqs = [t for t, _, _ in fq_list]
            matched_sqs = list(dict.fromkeys(s for _, s, _ in fq_list))

            results.append(RetrievalResult(
                memory=info["memory"],
                score_q2q=info["score_q2q"],
                score_q2c=info["score_q2c"],
                final_score=final_score,
                matched_fake_queries=matched_fqs,
                matched_sub_queries=matched_sqs,
            ))

        results.sort(key=lambda r: r.final_score, reverse=True)
        top_results = results[:self.top_n]

        logger.info(
            f"[BruteForce] Retrieved {len(top_results)} memories from "
            f"{len(sub_queries)} sub-queries (candidates: {len(candidate_map)})"
        )
        return top_results

    def _compute_q2q_score(self, sq: SubQuery, memory: MemoryEntry) -> tuple[float, str]:
        if not memory.fake_queries or sq.embedding is None:
            return 0.0, ""
        best_score = -1.0
        best_text = ""
        for fq in memory.fake_queries:
            if fq.embedding is None:
                continue
            sim = BaseEmbeddingProvider.cosine_similarity(sq.embedding, fq.embedding)
            if sim > best_score:
                best_score = sim
                best_text = fq.text
        return max(best_score, 0.0), best_text

    def _compute_q2c_score(self, sq: SubQuery, memory: MemoryEntry) -> float:
        if not memory.content_embeddings or sq.embedding is None:
            return 0.0
        best_score = -1.0
        for chunk_emb in memory.content_embeddings:
            sim = BaseEmbeddingProvider.cosine_similarity(sq.embedding, chunk_emb)
            if sim > best_score:
                best_score = sim
        return max(best_score, 0.0)
