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
    """Dual-path retriever: Q2Q (query-to-fake-query) + Q2C (query-to-content) reranking.

    Retrieval flow:
      Phase 1 (Q2Q): Search fake queries, get top-k hits per sub-query.
      Phase 2 (Merge): Group hits by session/memory, compute per-session Q2Q score.
      Phase 3 (Q2C Rerank): If candidates > top_n, do Q2C scoring to rerank; otherwise return directly.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        memory_store: BaseMemoryStore,
        alpha: float = 0.7,
        top_k_per_sub: int = 20,
        top_n: int = 5,
        top_k_q2c: int = 3,
    ):
        self.embedding_provider = embedding_provider
        self.memory_store = memory_store
        self.alpha = alpha
        self.top_k_per_sub = top_k_per_sub
        self.top_n = top_n
        self.top_k_q2c = top_k_q2c
        self._use_chromadb = isinstance(memory_store, ChromaDBStore)

    def retrieve(self, sub_queries: list[SubQuery]) -> list[RetrievalResult]:
        if self._use_chromadb:
            return self._retrieve_chromadb(sub_queries)
        else:
            return self._retrieve_bruteforce(sub_queries)

    # ========== ChromaDB path ==========

    def _retrieve_chromadb(self, sub_queries: list[SubQuery]) -> list[RetrievalResult]:
        store: ChromaDBStore = self.memory_store

        # --- Phase 1: Q2Q search ---
        # For each sub-query, search top_k_per_sub fake query hits
        # Aggregate all hits by memory_id (session-level)
        candidate_map: dict[str, dict] = {}

        for sq in sub_queries:
            if sq.embedding is None:
                continue

            fq_hits = store.search_fake_queries(sq.embedding, top_k=self.top_k_per_sub)
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

        # --- Phase 2: Sort candidates by Q2Q score ---
        sorted_candidates = sorted(
            candidate_map.items(),
            key=lambda x: x[1]["score_q2q"],
            reverse=True,
        )

        logger.info(f"[ChromaDB] Q2Q phase: {len(sorted_candidates)} candidate sessions")

        # --- Phase 3: Q2C rerank (only if candidates > top_n) ---
        if len(sorted_candidates) <= self.top_n:
            # No need for Q2C, return all candidates directly
            results = self._build_results_chromadb(store, sorted_candidates)
            logger.info(
                f"[ChromaDB] Candidates ({len(sorted_candidates)}) <= top_n ({self.top_n}), "
                f"skipping Q2C, returning all"
            )
            return results
        else:
            # Do Q2C reranking on all candidates, take top_n
            for mid, info in sorted_candidates:
                best_q2c = 0.0
                for sq in sub_queries:
                    if sq.embedding is None:
                        continue
                    q2c_score = store.search_content_for_memory(
                        sq.embedding, memory_id=mid, top_k=self.top_k_q2c,
                    )
                    if q2c_score > best_q2c:
                        best_q2c = q2c_score
                info["score_q2c"] = best_q2c

            # Recompute final scores and re-sort
            for mid, info in sorted_candidates:
                info["final_score"] = (
                    self.alpha * info["score_q2q"]
                    + (1 - self.alpha) * info.get("score_q2c", 0.0)
                )

            sorted_candidates.sort(key=lambda x: x[1]["final_score"], reverse=True)
            top_candidates = sorted_candidates[:self.top_n]

            results = self._build_results_chromadb(store, top_candidates)
            logger.info(
                f"[ChromaDB] Q2C rerank: {len(sorted_candidates)} candidates -> top {self.top_n}"
            )
            return results

    def _build_results_chromadb(
        self,
        store: ChromaDBStore,
        candidates: list[tuple[str, dict]],
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for mid, info in candidates:
            memory = store.get_by_id(mid)
            if memory is None:
                continue

            fq_list = sorted(info["matched_fqs"], key=lambda x: x[2], reverse=True)
            matched_fqs = [t for t, _, _ in fq_list]
            matched_sqs = list(dict.fromkeys(s for _, s, _ in fq_list))

            final_score = info.get("final_score", info["score_q2q"])

            results.append(RetrievalResult(
                memory=memory,
                score_q2q=info["score_q2q"],
                score_q2c=info.get("score_q2c", 0.0),
                final_score=final_score,
                matched_fake_queries=matched_fqs,
                matched_sub_queries=matched_sqs,
            ))
        return results

    # ========== Brute-force path ==========

    def _retrieve_bruteforce(self, sub_queries: list[SubQuery]) -> list[RetrievalResult]:
        memories = self.memory_store.load_all()
        if not memories:
            logger.warning("Memory store is empty, no results to retrieve")
            return []

        # --- Phase 1: Q2Q scoring ---
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

            if best_q2q > 0.0:
                candidate_map[mid] = {
                    "score_q2q": best_q2q,
                    "matched_fqs": matched_fqs,
                    "memory": memory,
                }

        if not candidate_map:
            logger.warning("Q2Q found no candidates")
            return []

        # --- Phase 2: Sort by Q2Q score ---
        sorted_candidates = sorted(
            candidate_map.items(),
            key=lambda x: x[1]["score_q2q"],
            reverse=True,
        )

        logger.info(f"[BruteForce] Q2Q phase: {len(sorted_candidates)} candidate sessions")

        # --- Phase 3: Q2C rerank (only if candidates > top_n) ---
        if len(sorted_candidates) <= self.top_n:
            results = self._build_results_bruteforce(sorted_candidates)
            logger.info(
                f"[BruteForce] Candidates ({len(sorted_candidates)}) <= top_n ({self.top_n}), "
                f"skipping Q2C, returning all"
            )
            return results
        else:
            for mid, info in sorted_candidates:
                memory = info["memory"]
                best_q2c = 0.0
                for sq in sub_queries:
                    if sq.embedding is None:
                        continue
                    score = self._compute_q2c_score(sq, memory)
                    if score > best_q2c:
                        best_q2c = score
                info["score_q2c"] = best_q2c
                info["final_score"] = (
                    self.alpha * info["score_q2q"]
                    + (1 - self.alpha) * best_q2c
                )

            sorted_candidates.sort(key=lambda x: x[1]["final_score"], reverse=True)
            top_candidates = sorted_candidates[:self.top_n]

            results = self._build_results_bruteforce(top_candidates)
            logger.info(
                f"[BruteForce] Q2C rerank: {len(sorted_candidates)} candidates -> top {self.top_n}"
            )
            return results

    def _build_results_bruteforce(
        self,
        candidates: list[tuple[str, dict]],
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for mid, info in candidates:
            fq_list = sorted(info["matched_fqs"], key=lambda x: x[2], reverse=True)
            matched_fqs = [t for t, _, _ in fq_list]
            matched_sqs = list(dict.fromkeys(s for _, s, _ in fq_list))

            final_score = info.get("final_score", info["score_q2q"])

            results.append(RetrievalResult(
                memory=info["memory"],
                score_q2q=info["score_q2q"],
                score_q2c=info.get("score_q2c", 0.0),
                final_score=final_score,
                matched_fake_queries=matched_fqs,
                matched_sub_queries=matched_sqs,
            ))
        return results

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
        scores = []
        for chunk_emb in memory.content_embeddings:
            sim = BaseEmbeddingProvider.cosine_similarity(sq.embedding, chunk_emb)
            scores.append(sim)
        scores.sort(reverse=True)
        # Take average of top_k_q2c chunks as Q2C score
        top_scores = scores[:self.top_k_q2c]
        return max(sum(top_scores) / len(top_scores), 0.0) if top_scores else 0.0
