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
from src.retrieval.focus import FocusWindow

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
        max_fqs_per_session: int = 3,
        session_text_budget_tokens: int = 0,
        focus_window: FocusWindow | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.memory_store = memory_store
        self.graph_manager = graph_manager
        self.top_k_per_sub = top_k_per_sub
        self.top_n = top_n
        self.top_k_q2c = top_k_q2c
        self.chain_depth = chain_depth
        self.prop_top_k_per_fq = prop_top_k_per_fq
        self.max_fqs_per_session = max_fqs_per_session
        # 0 keeps the retriever's historical behaviour: session texts leave
        # untrimmed and whoever consumes them decides what fits.
        self.session_text_budget_tokens = session_text_budget_tokens
        self.focus_window = focus_window or FocusWindow(
            embedding_provider=embedding_provider
        )

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
                text = await self._get_session_text(fq.session_id)
                if text:
                    direct_session_texts.append(text)

        # === DAG chain → session texts not already covered by direct hits ===
        chain_session_texts: list[str] = []
        chain_session_ids: set[str] = set()
        for fq in q2q_result["chain_fqs"]:
            sid = fq.session_id
            if not sid or sid in direct_session_ids or sid in chain_session_ids:
                continue
            chain_session_ids.add(sid)
            text = await self._get_session_text(sid)
            if text:
                chain_session_texts.append(text)

        # === Q2C → propositions from matched sessions ===
        q2c_props = await self._fetch_q2c_propositions(q2c_result, valid_sqs)

        # === Q2C → session texts for chunks neither Q2Q nor the chain reached ===
        covered = direct_session_ids | chain_session_ids
        q2c_session_texts: list[str] = []
        seen_q2c: set[str] = set()
        for hit in q2c_result:
            sid = hit.get("session_id", "")
            if not sid or sid in covered or sid in seen_q2c:
                continue
            seen_q2c.add(sid)
            text = await self._get_session_text(sid)
            if text:
                q2c_session_texts.append(text)

        # === Trim session texts to a query-relevant window before they leave ===
        if self.session_text_budget_tokens > 0:
            remaining = self.session_text_budget_tokens
            direct_session_texts, remaining = await self._window_texts(
                direct_session_texts, remaining, valid_sqs
            )
            chain_session_texts, remaining = await self._window_texts(
                chain_session_texts, remaining, valid_sqs
            )
            q2c_session_texts, remaining = await self._window_texts(
                q2c_session_texts, remaining, valid_sqs
            )

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
            q2c_session_texts=q2c_session_texts,
            chain_session_texts=chain_session_texts,
        )

        logger.info(
            f"Retrieval done: {len(result.matched_fqs)} direct FQs → "
            f"{len(direct_session_texts)} session texts, "
            f"{len(result.chain_fqs)} chain FQs → "
            f"{len(chain_session_texts)} chain texts / {len(chain_props)} chain props, "
            f"{len(q2c_session_texts)} Q2C texts / {len(q2c_props)} Q2C props, "
            f"{len(merged_props)} merged (ranked) props total"
        )
        return result

    async def _window_texts(
        self,
        texts: list[str],
        budget: int,
        sub_queries: list[SubQuery],
    ) -> tuple[list[str], int]:
        """Fit `texts` into `budget` tokens, keeping query-relevant parts.

        Tiers are windowed in priority order (direct → chain → Q2C) and each
        tier is handed the budget its predecessors left over, so the passage the
        Q2Q path matched directly is never squeezed to make room for a
        content-similarity hit further down. Within a tier every block of every
        passage competes for the tier's whole budget, which is what keeps the
        two dataset shapes from needing different settings: a lone oversized
        trajectory simply gets all of it, while five dialogue chunks let the
        budget settle on whichever of them the question is about.

        Trimming here rather than at the consumer is what actually saves money.
        The consumer sees whole passages and can only take or drop them, so a
        19k-token trace is either paid for in full or lost entirely; the
        retriever can hand over the 8k tokens of it the sub-queries point at.
        """
        if not texts:
            return texts, budget
        if budget <= 0:
            return [], 0

        before = sum(self.focus_window.count_tokens(t) for t in texts)
        kept = await self.focus_window.focus_many(texts, budget, sub_queries)
        used = sum(self.focus_window.count_tokens(t) for t in kept)

        if used < before:
            logger.info(
                f"Windowed session texts: {len(texts)} passages -> {len(kept)}, "
                f"{before} -> {used} tokens (budget {budget})"
            )
        return kept, max(budget - used, 0)

    async def _get_session_text(self, session_id: str) -> str:
        entry = await self.memory_store.get_by_id(session_id)
        if not entry:
            return ""
        text = entry.pre_session_text or entry.session_text or ""
        if not text:
            return ""
        # Compression protects date-bearing sentences, but the session header is
        # not guaranteed to survive sentence selection. Re-stating it costs a
        # handful of tokens and keeps "when was this recorded" answerable.
        date = entry.metadata.get("session_date", "")
        if date and date not in text[:200]:
            text = f"[Recorded: {date}]\n{text}"
        return text

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

        # Session-level aggregation: top_n bounds the number of distinct SESSIONS,
        # not raw FQ count. Several near-duplicate FQs from one session would
        # otherwise consume the whole budget and leave other sessions unread.
        #
        # That per-session cap only means something while there are other sessions
        # to protect. When the candidate pool spans fewer distinct sessions than
        # top_n -- which happens whenever a whole document is stored as one
        # session, as full-trajectory injection does -- the cap has nothing to
        # protect and degrades into plain truncation, discarding candidates no
        # other session could have supplied. A pool with the usual many sessions
        # is unaffected, since spreading the budget over min(pool, top_n)
        # recipients returns the configured cap exactly.
        pool_sessions = {fq_data_map[qid].get("session_id", "") for qid, _ in sorted_fqs}
        recipients = max(min(len(pool_sessions), self.top_n), 1)
        budget = self.top_n * self.max_fqs_per_session
        per_session_cap = max(self.max_fqs_per_session, -(-budget // recipients))

        direct_fq_ids: list[str] = []
        per_session_count: dict[str, int] = {}
        for qid, _score in sorted_fqs:
            sid = fq_data_map[qid].get("session_id", "")
            if sid not in per_session_count:
                if len(per_session_count) >= self.top_n:
                    continue
                per_session_count[sid] = 0
            if per_session_count[sid] >= per_session_cap:
                continue
            per_session_count[sid] += 1
            direct_fq_ids.append(qid)

        top_fq_data = [fq_data_map[qid] for qid in direct_fq_ids]

        # Chain seeds stay capped at top_n highest-scoring FQs so that widening
        # the direct set does not blow up DAG expansion breadth.
        chain_seed_ids = direct_fq_ids[:self.top_n]
        chain_nodes = await self.graph_manager.get_chain_context(
            chain_seed_ids, depth=self.chain_depth
        )

        direct_fq_id_set = set(direct_fq_ids)
        chain_fq_data = [n for n in chain_nodes if n["query_id"] not in direct_fq_id_set]

        # Deduplicate
        seen = set(direct_fq_ids)
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
            "scores": {qid: fq_score_map[qid] for qid in direct_fq_ids},
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
        self, q2c_result: list[dict], sub_queries: list[SubQuery]
    ) -> list[Proposition]:
        if not q2c_result:
            return []

        # Sessions already covered by the direct path are skipped in the Q2C
        # *text* tier, so the same raw passage is never duplicated. Their
        # propositions are a different granularity — distilled anchors the raw
        # text does not surface on its own — so they must not be skipped here.
        # Excluding them left the proposition tier empty precisely when Q2Q and
        # Q2C agreed on the session, i.e. when retrieval was working best.
        target_sids = [hit["session_id"] for hit in q2c_result][:self.top_k_q2c]

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
            created_at=data.get("created_at", ""),
        )
