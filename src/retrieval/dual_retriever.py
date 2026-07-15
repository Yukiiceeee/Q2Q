from __future__ import annotations
import logging
import numpy as np

from src.schemas.models import (
    MemoryEntry,
    SubQuery,
    RetrievalResult,
    VersionChainNode,
)
from src.embedding.base import BaseEmbeddingProvider
from src.storage.base import BaseMemoryStore
from src.storage.chromadb_store import ChromaDBStore

logger = logging.getLogger(__name__)


class DualRetriever:
    """Dual-path retriever: Q2Q + Q2C reranking with DAG-based version expansion.

    Retrieval flow:
      Phase 1 (Q2Q): Search fake queries, get top-k hits per sub-query.
      Phase 1.5 (DAG Expansion): BFS upward through parent_ids to gather version history.
      Phase 2 (Merge): Group hits by session/memory, compute per-session Q2Q score.
      Phase 3 (Q2C Rerank): If candidates > top_n, do Q2C scoring to rerank.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        memory_store: BaseMemoryStore,
        alpha: float = 0.7,
        top_k_per_sub: int = 20,
        top_n: int = 5,
        top_k_q2c: int = 3,
        version_chain_depth: int = 3,
        fq_confidence_threshold: float = 0.80,
        paragraph_top_k: int = 3,
    ):
        self.embedding_provider = embedding_provider
        self.memory_store = memory_store
        self.alpha = alpha
        self.top_k_per_sub = top_k_per_sub
        self.top_n = top_n
        self.top_k_q2c = top_k_q2c
        self.version_chain_depth = version_chain_depth
        self.fq_confidence_threshold = fq_confidence_threshold
        self.paragraph_top_k = paragraph_top_k
        self._use_chromadb = isinstance(memory_store, ChromaDBStore)

    def retrieve(self, sub_queries: list[SubQuery]) -> list[RetrievalResult]:
        if self._use_chromadb:
            results = self._retrieve_chromadb(sub_queries)
        else:
            results = self._retrieve_bruteforce(sub_queries)
        self._enrich_low_confidence_results(results, sub_queries)
        return results

    # ========== DAG expansion (shared by both paths) ==========

    def _expand_dag_context(self, hit_query_ids: list[str]) -> list[list[VersionChainNode]]:
        """BFS upward through parent_ids for each hit, collecting ancestor nodes."""
        dag_chains: list[list[VersionChainNode]] = []
        seen_roots: set[str] = set()

        for start_qid in hit_query_ids:
            if start_qid in seen_roots:
                continue
            seen_roots.add(start_qid)

            ancestors = self._bfs_ancestors(start_qid)
            if len(ancestors) <= 1:
                continue
            dag_chains.append(ancestors)

        return dag_chains

    def _bfs_ancestors(self, start_query_id: str) -> list[VersionChainNode]:
        """BFS upward from a node, collecting ancestors up to version_chain_depth levels."""
        visited: set[str] = set()
        queue = [start_query_id]
        context_nodes: list[dict] = []
        current_depth = 0

        while queue and current_depth < self.version_chain_depth:
            next_queue: list[str] = []
            batch_ids = [qid for qid in queue if qid not in visited]
            if not batch_ids:
                break

            nodes = self.memory_store.get_fake_queries_by_ids(batch_ids)
            node_map = {n["query_id"]: n for n in nodes}

            for qid in batch_ids:
                if qid in visited:
                    continue
                visited.add(qid)
                node = node_map.get(qid)
                if node:
                    context_nodes.append(node)
                    next_queue.extend(
                        pid for pid in node.get("parent_ids", []) if pid not in visited
                    )

            queue = next_queue
            current_depth += 1

        context_nodes.sort(key=lambda n: n.get("depth", 0))

        return [
            VersionChainNode(
                query_id=n["query_id"],
                text=n["text"],
                answer=n.get("answer", ""),
                memory_id=n.get("memory_id", ""),
                depth=n.get("depth", 0),
                parent_ids=n.get("parent_ids", []),
                created_at=n.get("created_at", ""),
            )
            for n in context_nodes
        ]

    # ========== ChromaDB path ==========

    def _retrieve_chromadb(self, sub_queries: list[SubQuery]) -> list[RetrievalResult]:
        store: ChromaDBStore = self.memory_store

        # --- Phase 1: Q2Q search ---
        candidate_map: dict[str, dict] = {}
        matched_fq_details: dict[str, list[dict]] = {}

        for sq in sub_queries:
            if sq.embedding is None:
                continue

            fq_hits = store.search_fake_queries(sq.embedding, top_k=self.top_k_per_sub)
            for hit in fq_hits:
                mid = hit["memory_id"]
                if mid not in candidate_map:
                    candidate_map[mid] = {"score_q2q": 0.0, "matched_fqs": []}
                    matched_fq_details[mid] = []

                fq_score = hit["score"]
                if fq_score > candidate_map[mid]["score_q2q"]:
                    candidate_map[mid]["score_q2q"] = fq_score

                fq_text = hit["text"]
                already = any(t == fq_text for t, _, _ in candidate_map[mid]["matched_fqs"])
                if not already:
                    candidate_map[mid]["matched_fqs"].append((fq_text, sq.text, fq_score))
                    matched_fq_details[mid].append(hit)

        if not candidate_map:
            logger.info("[ChromaDB] Q2Q found no candidates")
            return []

        # --- Phase 1.5: DAG Expansion ---
        dag_contexts: dict[str, list[list[VersionChainNode]]] = {}

        for mid, details in matched_fq_details.items():
            hit_qids = [h["query_id"] for h in details if h.get("parent_ids")]
            if hit_qids:
                dag_contexts[mid] = self._expand_dag_context(hit_qids)
            else:
                dag_contexts[mid] = []

        # --- Phase 2: Sort candidates by Q2Q score ---
        sorted_candidates = sorted(
            candidate_map.items(),
            key=lambda x: x[1]["score_q2q"],
            reverse=True,
        )

        logger.info(f"[ChromaDB] Q2Q phase: {len(sorted_candidates)} candidate sessions")

        # --- Phase 3: Q2C rerank (only if candidates > top_n) ---
        if len(sorted_candidates) <= self.top_n:
            results = self._build_results(store, sorted_candidates, dag_contexts)
            logger.info(
                f"[ChromaDB] Candidates ({len(sorted_candidates)}) <= top_n ({self.top_n}), "
                f"skipping Q2C, returning all"
            )
            return results
        else:
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

            for mid, info in sorted_candidates:
                info["final_score"] = (
                    self.alpha * info["score_q2q"]
                    + (1 - self.alpha) * info.get("score_q2c", 0.0)
                )

            sorted_candidates.sort(key=lambda x: x[1]["final_score"], reverse=True)
            top_candidates = sorted_candidates[:self.top_n]

            results = self._build_results(store, top_candidates, dag_contexts)
            logger.info(
                f"[ChromaDB] Q2C rerank: {len(sorted_candidates)} candidates -> top {self.top_n}"
            )
            return results

    # ========== Brute-force path ==========

    def _retrieve_bruteforce(self, sub_queries: list[SubQuery]) -> list[RetrievalResult]:
        memories = self.memory_store.load_all()
        if not memories:
            logger.warning("Memory store is empty, no results to retrieve")
            return []

        # --- Phase 1: Q2Q scoring ---
        candidate_map: dict[str, dict] = {}
        matched_fq_details: dict[str, list[dict]] = {}

        for memory in memories:
            mid = memory.memory_id
            best_q2q = 0.0
            matched_fqs: list[tuple[str, str, float]] = []
            fq_details: list[dict] = []

            for sq in sub_queries:
                if sq.embedding is None:
                    continue
                score, fq_text, fq_query_id, fq_parent_ids, fq_depth = self._compute_q2q_score(sq, memory)
                if score > best_q2q:
                    best_q2q = score
                if score > 0.0 and fq_text:
                    already = any(t == fq_text for t, _, _ in matched_fqs)
                    if not already:
                        matched_fqs.append((fq_text, sq.text, score))
                        fq_details.append({
                            "query_id": fq_query_id,
                            "parent_ids": fq_parent_ids,
                            "depth": fq_depth,
                        })

            if best_q2q > 0.0:
                candidate_map[mid] = {
                    "score_q2q": best_q2q,
                    "matched_fqs": matched_fqs,
                    "memory": memory,
                }
                matched_fq_details[mid] = fq_details

        if not candidate_map:
            logger.warning("Q2Q found no candidates")
            return []

        # --- Phase 1.5: DAG Expansion ---
        dag_contexts: dict[str, list[list[VersionChainNode]]] = {}
        for mid, details in matched_fq_details.items():
            hit_qids = [h["query_id"] for h in details if h.get("parent_ids")]
            if hit_qids:
                dag_contexts[mid] = self._expand_dag_context(hit_qids)
            else:
                dag_contexts[mid] = []

        # --- Phase 2: Sort by Q2Q score ---
        sorted_candidates = sorted(
            candidate_map.items(),
            key=lambda x: x[1]["score_q2q"],
            reverse=True,
        )

        logger.info(f"[BruteForce] Q2Q phase: {len(sorted_candidates)} candidate sessions")

        # --- Phase 3: Q2C rerank (only if candidates > top_n) ---
        if len(sorted_candidates) <= self.top_n:
            results = self._build_results(None, sorted_candidates, dag_contexts)
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

            results = self._build_results(None, top_candidates, dag_contexts)
            logger.info(
                f"[BruteForce] Q2C rerank: {len(sorted_candidates)} candidates -> top {self.top_n}"
            )
            return results

    # ========== Shared result builder ==========

    def _build_results(
        self,
        store: ChromaDBStore | None,
        candidates: list[tuple[str, dict]],
        dag_contexts: dict[str, list[list[VersionChainNode]]] | None = None,
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for mid, info in candidates:
            if "memory" in info:
                memory = info["memory"]
            elif store:
                memory = store.get_by_id(mid)
            else:
                continue
            if memory is None:
                continue

            fq_list = sorted(info["matched_fqs"], key=lambda x: x[2], reverse=True)
            matched_fqs = [t for t, _, _ in fq_list]
            matched_sqs = list(dict.fromkeys(s for _, s, _ in fq_list))

            final_score = info.get("final_score", info["score_q2q"])

            version_chain_context = dag_contexts.get(mid, []) if dag_contexts else []

            results.append(RetrievalResult(
                memory=memory,
                score_q2q=info["score_q2q"],
                score_q2c=info.get("score_q2c", 0.0),
                final_score=final_score,
                matched_fake_queries=matched_fqs,
                matched_sub_queries=matched_sqs,
                version_chain_context=version_chain_context,
            ))
        return results

    # ========== Score computation ==========

    def _compute_q2q_score(
        self, sq: SubQuery, memory: MemoryEntry
    ) -> tuple[float, str, str, list[str], int]:
        if not memory.fake_queries or sq.embedding is None:
            return 0.0, "", "", [], 0
        best_score = -1.0
        best_text = ""
        best_query_id = ""
        best_parent_ids: list[str] = []
        best_depth = 0
        for fq in memory.fake_queries:
            if fq.embedding is None:
                continue
            sim = BaseEmbeddingProvider.cosine_similarity(sq.embedding, fq.embedding)
            if sim > best_score:
                best_score = sim
                best_text = fq.text
                best_query_id = fq.query_id
                best_parent_ids = fq.parent_ids
                best_depth = fq.depth
        return max(best_score, 0.0), best_text, best_query_id, best_parent_ids, best_depth

    def _compute_q2c_score(self, sq: SubQuery, memory: MemoryEntry) -> float:
        if not memory.content_embeddings or sq.embedding is None:
            return 0.0
        scores = []
        for chunk_emb in memory.content_embeddings:
            sim = BaseEmbeddingProvider.cosine_similarity(sq.embedding, chunk_emb)
            scores.append(sim)
        scores.sort(reverse=True)
        top_scores = scores[:self.top_k_q2c]
        return max(sum(top_scores) / len(top_scores), 0.0) if top_scores else 0.0

    def _enrich_low_confidence_results(
        self, results: list[RetrievalResult], sub_queries: list[SubQuery]
    ) -> None:
        for r in results:
            if r.score_q2q >= self.fq_confidence_threshold:
                continue

            best_paragraphs: dict[str, float] = {}
            for sq in sub_queries:
                if sq.embedding is None:
                    continue
                hits = self.memory_store.search_paragraphs_for_memory(
                    sq.embedding, r.memory.memory_id, top_k=self.paragraph_top_k
                )
                for hit in hits:
                    text = hit["text"]
                    score = hit["score"]
                    if text not in best_paragraphs or score > best_paragraphs[text]:
                        best_paragraphs[text] = score

            sorted_paras = sorted(best_paragraphs.items(), key=lambda x: x[1], reverse=True)
            r.matched_paragraphs = [text for text, _ in sorted_paras[:self.paragraph_top_k]]

            if r.matched_paragraphs:
                logger.debug(
                    f"  Enriched memory {r.memory.memory_id[:12]} with "
                    f"{len(r.matched_paragraphs)} paragraphs (low confidence path)"
                )
