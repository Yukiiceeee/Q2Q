"""Q2Q Agent - top-level orchestrator for memorize/query operations."""
from __future__ import annotations

import asyncio
import logging

from src.config import Q2QConfig
from src.schemas.models import MemoryEntry
from src.utils.llm_client import create_llm_client, BaseLLMClient, get_usage_tracker
from src.embedding.factory import create_embedding_provider
from src.embedding.base import BaseEmbeddingProvider
from src.storage.base import BaseMemoryStore
from src.storage.chromadb_store import ChromaDBStore
from src.preprocessing.compressor import PerplexityCompressor
from src.preprocessing.proposition_extractor import PropositionExtractor
from src.memory.query_generator import FakeQueryGenerator
from src.memory.constructor import MemoryConstructor
from src.memory.graph_manager import MemoryGraphManager
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.dual_retriever import DualRetriever
from src.retrieval.answerer import Answerer

logger = logging.getLogger(__name__)


class Q2QAgent:

    def __init__(self, config: Q2QConfig):
        self.config = config

        # --- Infrastructure ---
        self.llm_client: BaseLLMClient = create_llm_client(
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            api_key=config.llm.api_key or None,
            base_url=config.llm.base_url or None,
        )

        self.embedding_provider: BaseEmbeddingProvider = create_embedding_provider(
            provider=config.embedding.provider,
            model_name=config.embedding.model_name,
            device=config.embedding.device,
        )

        self.memory_store: BaseMemoryStore = self._create_store(config)

        # --- Phase 1: Preprocessing ---
        self.compressor = PerplexityCompressor(
            model_path=config.preprocessing.compressor_model,
            device=config.preprocessing.compressor_device,
            threshold=config.preprocessing.perplexity_threshold,
        )

        self.prop_extractor = PropositionExtractor(
            mode=config.preprocessing.proposition_mode,
            llm_client=self.llm_client,
            nlp_inference_model_path=config.preprocessing.nlp_inference_model,
            language=config.language,
            device=config.preprocessing.compressor_device,
        )

        # --- Phase 2: Memory Construction ---
        self.fq_generator = FakeQueryGenerator(
            llm_client=self.llm_client,
            embedding_provider=self.embedding_provider,
            num_queries=config.retrieval.num_fake_queries,
            language=config.language,
            binding_verify_threshold=config.retrieval.binding_verify_threshold,
            max_extra_bindings=config.retrieval.max_extra_bindings,
            style=config.retrieval.fq_style,
            role=config.retrieval.fq_role,
        )

        self.memory_constructor = MemoryConstructor(
            compressor=self.compressor,
            prop_extractor=self.prop_extractor,
            query_generator=self.fq_generator,
            embedding_provider=self.embedding_provider,
            memory_store=self.memory_store,
        )

        # --- Phase 3: Graph Management ---
        self.graph_manager = MemoryGraphManager(
            store=self.memory_store,
            embedding_provider=self.embedding_provider,
            evolve_threshold=config.graph.evolve_threshold,
            diverge_low=config.graph.diverge_low,
            diverge_high=config.graph.diverge_high,
            link_similarity=config.graph.link_similarity,
            max_parents=config.graph.max_parents,
            similarity_top_k=config.graph.similarity_top_k,
            maintenance_interval=config.graph.maintenance_interval,
            merge_overlap_threshold=config.graph.merge_overlap_threshold,
            max_chain_length=config.graph.max_chain_length,
            chain_sample_interval=config.graph.chain_sample_interval,
            max_chain_nodes=config.graph.max_chain_nodes,
        )

        # --- Phase 4: Retrieval & Answer ---
        self.query_decomposer = QueryDecomposer(
            llm_client=self.llm_client,
            embedding_provider=self.embedding_provider,
            language=config.language,
        )

        self.dual_retriever = DualRetriever(
            embedding_provider=self.embedding_provider,
            memory_store=self.memory_store,
            graph_manager=self.graph_manager,
            top_k_per_sub=config.retrieval.top_k_per_sub,
            top_n=config.retrieval.top_n,
            top_k_q2c=config.retrieval.top_k_q2c,
            chain_depth=config.retrieval.chain_depth,
            prop_top_k_per_fq=config.retrieval.prop_top_k_per_fq,
            max_fqs_per_session=config.retrieval.max_fqs_per_session,
        )

        self.answerer = Answerer(
            llm_client=self.llm_client,
            language=config.language,
            max_context_tokens=config.retrieval.max_context_tokens,
            embedding_provider=self.embedding_provider,
        )

        self.usage_tracker = get_usage_tracker()

        logger.info(
            f"Q2QAgent initialized | model={config.llm.model} | "
            f"embedding={config.embedding.model_name} | "
            f"storage={config.storage.backend} | "
            f"prop_mode={config.preprocessing.proposition_mode} | "
            f"lang={config.language}"
        )

    def _create_store(self, config: Q2QConfig) -> BaseMemoryStore:
        if config.storage.backend == "chromadb":
            return ChromaDBStore(
                persist_directory=config.storage.chromadb_path,
                collection_name=config.storage.chromadb_collection,
            )
        raise ValueError(f"Unsupported storage backend: {config.storage.backend}")

    async def memorize(self, session_text: str, metadata: dict | None = None) -> MemoryEntry:
        self.usage_tracker.set_phase("memorize")
        logger.info(f"--- Memorize Start ({len(session_text)} chars) ---")

        # Phase 1-2: Build memory (preprocessing + FQ generation + embedding)
        entry = await self.memory_constructor.build_memory(session_text, metadata)

        # Phase 3: Integrate into graph
        if entry.fake_queries:
            edges = await self.graph_manager.integrate_new_fqs(entry.fake_queries)
            logger.info(f"  Graph integration: {len(edges)} new edges")

        logger.info(
            f"  Memory {entry.session_id} complete: "
            f"{len(entry.fake_queries)} FQs, "
            f"{len(entry.propositions)} props"
        )
        logger.info(f"--- Memorize Complete ---")
        return entry

    async def query(
        self,
        raw_query: str,
        history: str = "",
        return_answer: bool = True,
    ) -> dict:
        self.usage_tracker.set_phase("query")
        logger.info(f"--- Query Start: {raw_query} ---")

        # Decompose query
        sub_queries = await self.query_decomposer.decompose(raw_query, history)
        logger.info(f"  Sub-queries ({len(sub_queries)}):")
        for sq in sub_queries:
            logger.info(f"    [{sq.index}] {sq.text}")

        # Dual retrieval
        retrieval_result = await self.dual_retriever.retrieve(sub_queries)
        logger.info(
            f"  Retrieval: {len(retrieval_result.matched_fqs)} direct FQs, "
            f"{len(retrieval_result.ranked_propositions)} props, "
            f"{len(retrieval_result.session_texts)} sessions"
        )

        # Answer generation
        answer = ""
        if return_answer and (retrieval_result.ranked_propositions or retrieval_result.session_texts):
            answer = await self.answerer.answer(
                raw_query, retrieval_result, history, sub_queries
            )
            logger.info(f"  Answer generated ({len(answer)} chars)")

        response = {
            "raw_query": raw_query,
            "sub_queries": [sq.text for sq in sub_queries],
            "num_fqs": len(retrieval_result.matched_fqs),
            "num_propositions": len(retrieval_result.ranked_propositions),
            "num_sessions": len(retrieval_result.session_texts),
            "matched_fqs": [
                {"query_id": fq.query_id, "text": fq.text, "session_id": fq.session_id}
                for fq in retrieval_result.matched_fqs[:5]
            ],
            "top_propositions": [
                {"prop_id": p.prop_id, "text": p.text, "session_id": p.session_id}
                for p in retrieval_result.ranked_propositions[:10]
            ],
            "answer": answer,
        }

        logger.info(f"--- Query Complete ---")
        return response

    async def batch_memorize(
        self,
        sessions: list[str],
        max_concurrent: int = 3,
    ) -> list[MemoryEntry]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _memorize_one(text: str) -> MemoryEntry:
            async with semaphore:
                return await self.memorize(text)

        return list(await asyncio.gather(*[_memorize_one(s) for s in sessions]))

    def get_stats(self) -> dict:
        return {
            "llm_usage": self.usage_tracker.get_stats(),
        }
