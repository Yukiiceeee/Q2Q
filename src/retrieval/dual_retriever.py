"""Dual-path retriever: Q2Q + Q2C with bidirectional DAG expansion and proposition reranking."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

import numpy as np

from src.schemas.models import SubQuery, FakeQuery, Proposition, RetrievalResult
from src.embedding.base import BaseEmbeddingProvider
from src.storage.base import BaseMemoryStore
from src.memory.graph_manager import MemoryGraphManager

logger = logging.getLogger(__name__)


class DualRetriever:

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        memory_store: BaseMemoryStore,
        graph_manager: MemoryGraphManager,
        top_k_per_sub: int = 20,
        top_n: int = 5,
        top_k_q2c: int = 3,
        chain_depth: int = 3,
        prop_top_k_per_fq: int = 3,
    ):
        self.embedding_provider = embedding_provider
        self.memory_store = memory_store
        self.graph_manager = graph_manager
        self.top_k_per_sub = top_k_per_sub
        self.top_n = top_n
        self.top_k_q2c = top_k_q2c
        self.chain_depth = chain_depth
        self.prop_top_k_per_fq = prop_top_k_per_fq

    async def retrieve(self, sub_queries: list[SubQuery]) -> RetrievalResult:
        valid_sqs = [sq for sq in sub_queries if sq.embedding is not None]
        if not valid_sqs:
            return RetrievalResult()

        # === Q2Q Path ===
        q2q_result = await self._q2q_retrieve(valid_sqs)

        # === Q2C Path ===
        q2c_result = await self._q2c_retrieve(valid_sqs)

        # === Q2Q Chain Propositions (from chain FQs) ===
        chain_props = await self._rerank_propositions(
            q2q_result["chain_fqs"], valid_sqs
        )

        # === Direct FQ → session compressed texts ===
        direct_session_ids: set[str] = set()
        direct_session_texts: list[str] = []
        for fq in q2q_result["direct_fqs"]:
            if fq.session_id and fq.session_id not in direct_session_ids:
                direct_session_ids.add(fq.session_id)
                entry = await self.memory_store.get_by_id(fq.session_id)
                if entry and entry.pre_session_text:
                    direct_session_texts.append(entry.pre_session_text)
                elif entry and entry.session_text:
                    direct_session_texts.append(entry.session_text[:8000])

        # === Q2C → propositions from matched sessions (excluding direct hits) ===
        q2c_props = await self._fetch_q2c_propositions(q2c_result, direct_session_ids, valid_sqs)

        # === Merge chain_props + q2c_props for backward compat (ranked_propositions) ===
        merged_props = self._merge_propositions(chain_props, q2c_props)

        # Legacy session_texts for backward compat
        all_session_texts = direct_session_texts

        result = RetrievalResult(
            matched_fqs=q2q_result["direct_fqs"],
            ranked_propositions=merged_props,
            session_texts=all_session_texts[:self.top_n],
            q2q_scores=q2q_result["scores"],
            q2c_scores={h["session_id"]: h["score"] for h in q2c_result},
            chain_fqs=q2q_result["chain_fqs"],
            direct_session_texts=direct_session_texts,
            chain_propositions=chain_props,
            q2c_propositions=q2c_props,
        )

        logger.info(
            f"Retrieval done: {len(result.matched_fqs)} direct FQs → "
            f"{len(direct_session_texts)} session texts, "
            f"{len(result.chain_fqs)} chain FQs → "
            f"{len(chain_props)} chain props, "
            f"{len(q2c_props)} Q2C props, "
            f"{len(merged_props)} merged (ranked) props total"
        )
        return result

    async def _q2q_retrieve(self, sub_queries: list[SubQuery]) -> dict:
        all_embeddings = [sq.embedding for sq in sub_queries]
        batch_results = await self.memory_store.batch_search_fake_queries(
            all_embeddings, top_k=self.top_k_per_sub
        )

        fq_score_map: dict[str, float] = {}
        fq_data_map: dict[str, dict] = {}

        for sq, fq_hits in zip(sub_queries, batch_results):
            for hit in fq_hits:
                qid = hit["query_id"]
                score = hit["score"]
                if qid not in fq_score_map or score > fq_score_map[qid]:
                    fq_score_map[qid] = score
                    fq_data_map[qid] = hit

        sorted_fqs = sorted(fq_score_map.items(), key=lambda x: x[1], reverse=True)
        top_fq_ids = [qid for qid, _ in sorted_fqs[:self.top_n]]
        top_fq_data = [fq_data_map[qid] for qid in top_fq_ids]

        # Bidirectional chain expansion
        chain_nodes = await self.graph_manager.get_chain_context(
            top_fq_ids, depth=self.chain_depth
        )

        direct_fq_ids = set(top_fq_ids)
        chain_fq_data = [n for n in chain_nodes if n["query_id"] not in direct_fq_ids]

        # Deduplicate
        seen = set(top_fq_ids)
        deduped_chain = []
        for node in chain_fq_data:
            if node["query_id"] not in seen:
                seen.add(node["query_id"])
                deduped_chain.append(node)

        # Convert to FakeQuery objects
        direct_fqs = [self._dict_to_fakequery(d) for d in top_fq_data]
        chain_fqs = [self._dict_to_fakequery(d) for d in deduped_chain]

        return {
            "direct_fqs": direct_fqs,
            "chain_fqs": chain_fqs,
            "all_fqs": direct_fqs + chain_fqs,
            "scores": dict(sorted_fqs[:self.top_n]),
        }

    async def _q2c_retrieve(self, sub_queries: list[SubQuery]) -> list[dict]:
        if self.top_k_q2c <= 0:
            return []
        all_embeddings = [sq.embedding for sq in sub_queries]
        hits = await self.memory_store.batch_search_content(
            all_embeddings, top_k=self.top_k_q2c
        )
        return hits[:self.top_n] if hits else []

    async def _rerank_propositions(
        self, all_fqs: list[FakeQuery], sub_queries: list[SubQuery]
    ) -> list[Proposition]:
        if not all_fqs:
            return []

        # Group by session_id, collect all prop_ids
        session_prop_ids: dict[str, set[str]] = defaultdict(set)
        for fq in all_fqs:
            for pid in fq.answer_source:
                session_prop_ids[fq.session_id].add(pid)

        # Fetch all propositions
        all_props: list[Proposition] = []
        fetch_tasks = []
        for session_id, prop_ids in session_prop_ids.items():
            fetch_tasks.append(
                self.memory_store.get_propositions_by_ids(list(prop_ids), session_id)
            )

        fetched_results = await asyncio.gather(*fetch_tasks)
        for prop_dicts in fetched_results:
            for pd in prop_dicts:
                all_props.append(Proposition.from_dict(pd))

        if not all_props:
            return []

        # Embed propositions that don't have embeddings
        props_to_embed = [p for p in all_props if p.embedding is None]
        if props_to_embed:
            texts = [p.text for p in props_to_embed]
            embeddings = await self.embedding_provider.embed_batch(texts)
            for prop, emb in zip(props_to_embed, embeddings):
                prop.embedding = emb

        # Score each proposition against sub_queries
        scored: list[tuple[float, Proposition]] = []
        for prop in all_props:
            if prop.embedding is None:
                continue
            best_sim = max(
                BaseEmbeddingProvider.cosine_similarity(sq.embedding, prop.embedding)
                for sq in sub_queries
                if sq.embedding is not None
            )
            scored.append((best_sim, prop))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by text
        seen_texts: set[str] = set()
        ranked: list[Proposition] = []
        for _, prop in scored:
            if prop.text not in seen_texts:
                seen_texts.add(prop.text)
                ranked.append(prop)

        max_props = self.prop_top_k_per_fq * len(all_fqs)
        return ranked[:max_props]

    async def _fetch_q2c_propositions(
        self, q2c_result: list[dict], exclude_session_ids: set[str], sub_queries: list[SubQuery]
    ) -> list[Proposition]:
        if not q2c_result:
            return []

        target_sids = [
            hit["session_id"] for hit in q2c_result
            if hit["session_id"] not in exclude_session_ids
        ][:self.top_k_q2c]

        if not target_sids:
            return []

        all_props: list[Proposition] = []
        for sid in target_sids:
            entry = await self.memory_store.get_by_id(sid)
            if entry and entry.propositions:
                all_props.extend(entry.propositions)

        if not all_props:
            return []

        # Embed propositions that don't have embeddings
        props_to_embed = [p for p in all_props if p.embedding is None]
        if props_to_embed:
            texts = [p.text for p in props_to_embed]
            embeddings = await self.embedding_provider.embed_batch(texts)
            for prop, emb in zip(props_to_embed, embeddings):
                prop.embedding = emb

        # Score against sub_queries and sort
        scored: list[tuple[float, Proposition]] = []
        for prop in all_props:
            if prop.embedding is None:
                continue
            best_sim = max(
                BaseEmbeddingProvider.cosine_similarity(sq.embedding, prop.embedding)
                for sq in sub_queries
                if sq.embedding is not None
            )
            scored.append((best_sim, prop))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [prop for _, prop in scored]

    def _merge_propositions(
        self, chain_props: list[Proposition], q2c_props: list[Proposition]
    ) -> list[Proposition]:
        seen_texts: set[str] = set()
        merged: list[Proposition] = []

        for prop in chain_props:
            key = prop.text.strip()
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(prop)

        for prop in q2c_props:
            key = prop.text.strip()
            if key not in seen_texts:
                seen_texts.add(key)
                merged.append(prop)

        return merged

    def _dict_to_fakequery(self, data: dict) -> FakeQuery:
        return FakeQuery(
            query_id=data.get("query_id", ""),
            text=data.get("text", ""),
            answer_source=data.get("answer_source", []),
            session_id=data.get("session_id", ""),
            time=data.get("time", ""),
            parent_ids=data.get("parent_ids", []),
            child_ids=data.get("child_ids", []),
            depth=data.get("depth", 0),
            created_at=data.get("created_at", ""),
        )
