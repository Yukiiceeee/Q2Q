from __future__ import annotations
import logging
import json
from pathlib import Path
from datetime import datetime

from src.config import Q2QConfig
from src.schemas.models import MemoryEntry, RetrievalResult
from src.utils.llm_client import create_llm_client, BaseLLMClient, get_usage_tracker
from src.embedding.factory import create_embedding_provider
from src.embedding.base import BaseEmbeddingProvider
from src.storage.base import BaseMemoryStore
from src.storage.json_store import JsonMemoryStore
from src.storage.chromadb_store import ChromaDBStore
from src.memory.query_generator import FakeQueryGenerator
from src.memory.constructor import MemoryConstructor
from src.memory.version_detector import VersionDetector
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.dual_retriever import DualRetriever
from src.retrieval.answerer import Answerer

logger = logging.getLogger(__name__)


class Q2QAgent:

    def __init__(self, config: Q2QConfig):
        self.config = config

        # LLM client
        self.llm_client: BaseLLMClient = create_llm_client(
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            api_key=config.llm.api_key or None,
            base_url=config.llm.base_url or None,
        )

        # Embedding provider
        self.embedding_provider: BaseEmbeddingProvider = create_embedding_provider(
            provider=config.embedding.provider,
            model_name=config.embedding.model_name,
            device=config.embedding.device,
        )

        # Storage
        self.memory_store: BaseMemoryStore = self._create_store(config)

        # Version detector
        self.version_detector = VersionDetector(
            memory_store=self.memory_store,
            embedding_provider=self.embedding_provider,
            threshold=config.retrieval.version_threshold,
        )

        # Memory construction
        self.query_generator = FakeQueryGenerator(
            llm_client=self.llm_client,
            num_queries=config.retrieval.num_fake_queries,
            language=config.language,
        )
        self.memory_constructor = MemoryConstructor(
            query_generator=self.query_generator,
            embedding_provider=self.embedding_provider,
            memory_store=self.memory_store,
            version_detector=self.version_detector,
        )

        # Retrieval
        self.query_decomposer = QueryDecomposer(
            llm_client=self.llm_client,
            embedding_provider=self.embedding_provider,
            language=config.language,
        )
        self.dual_retriever = DualRetriever(
            embedding_provider=self.embedding_provider,
            memory_store=self.memory_store,
            alpha=config.retrieval.alpha,
            top_k_per_sub=config.retrieval.top_k_per_sub,
            top_n=config.retrieval.top_n,
            top_k_q2c=config.retrieval.top_k_q2c,
            version_chain_depth=config.retrieval.version_chain_depth,
            fq_confidence_threshold=config.retrieval.fq_confidence_threshold,
            paragraph_top_k=config.retrieval.paragraph_top_k,
        )
        self.answerer = Answerer(
            llm_client=self.llm_client,
            language=config.language,
            max_context_tokens=config.retrieval.max_context_tokens,
            fq_confidence_threshold=config.retrieval.fq_confidence_threshold,
        )

        # Usage tracker
        self.usage_tracker = get_usage_tracker()

        logger.info(
            f"Q2QAgent initialized | model={config.llm.model} | "
            f"embedding={config.embedding.model_name} | "
            f"storage={config.storage.backend} | lang={config.language} | "
            f"version_threshold={config.retrieval.version_threshold}"
        )

    def _create_store(self, config: Q2QConfig) -> BaseMemoryStore:
        if config.storage.backend == "chromadb":
            return ChromaDBStore(
                persist_directory=config.storage.chromadb_path,
                collection_name=config.storage.chromadb_collection,
            )
        elif config.storage.backend == "json":
            return JsonMemoryStore(storage_path=config.storage.json_path)
        raise ValueError(f"Unsupported storage backend: {config.storage.backend}")

    def memorize(self, session_text: str, metadata: dict | None = None) -> MemoryEntry:
        """Build and store a memory entry from a session text."""
        self.usage_tracker.set_phase("memorize")
        logger.info(f"--- Memorize Start ({len(session_text)} chars) ---")

        entry = self.memory_constructor.build_memory(session_text, metadata)

        logger.info(f"  Memory ID: {entry.memory_id}")
        logger.info(f"  Fake Queries ({len(entry.fake_queries)}):")
        for i, fq in enumerate(entry.fake_queries):
            dag_info = f" [parents={len(fq.parent_ids)}, d{fq.depth}]" if fq.parent_ids else ""
            logger.info(f"    [{i}] {fq.text}{dag_info}")
        logger.info(f"  Knowledge Points: {len(entry.knowledge_points)}")
        logger.info(f"  Paragraphs: {len(entry.paragraphs)}")
        logger.info(f"  Content Chunks: {len(entry.content_embeddings)}")
        logger.info(f"--- Memorize Complete ---")

        return entry

    def query(
        self,
        raw_query: str,
        history: str = "",
        return_answer: bool = True,
    ) -> dict:
        """Retrieve relevant memories and optionally generate an answer."""
        self.usage_tracker.set_phase("query")
        logger.info(f"--- Query Start ---")
        logger.info(f"  Raw Query: {raw_query}")

        # Step 1: decompose
        sub_queries = self.query_decomposer.decompose(raw_query, history)
        logger.info(f"  Sub-queries ({len(sub_queries)}):")
        for sq in sub_queries:
            logger.info(f"    [{sq.index}] {sq.text}")

        # Step 2: dual-path retrieval
        results = self.dual_retriever.retrieve(sub_queries)
        logger.info(f"  Retrieval Results: {len(results)}")
        for i, r in enumerate(results):
            matched_fqs_preview = r.matched_fake_queries[0][:60] if r.matched_fake_queries else ""
            chain_count = len(r.version_chain_context)
            logger.info(
                f"    [{i}] score={r.final_score:.4f} "
                f"(q2q={r.score_q2q:.4f}, q2c={r.score_q2c:.4f}) "
                f"memory={r.memory.memory_id[:12]}... "
                f"chains={chain_count} "
                f"matched_fq=\"{matched_fqs_preview}\""
            )

        # Step 3: generate answer (optional)
        answer = ""
        if return_answer and results:
            answer = self.answerer.answer(raw_query, results, history)
            logger.info(f"  Answer generated ({len(answer)} chars)")

        response = {
            "raw_query": raw_query,
            "sub_queries": [sq.text for sq in sub_queries],
            "num_results": len(results),
            "results": [
                {
                    "memory_id": r.memory.memory_id,
                    "score_q2q": round(r.score_q2q, 4),
                    "score_q2c": round(r.score_q2c, 4),
                    "final_score": round(r.final_score, 4),
                    "matched_fake_queries": r.matched_fake_queries,
                    "matched_sub_queries": r.matched_sub_queries,
                    "content_preview": r.memory.session_text[:200],
                    "version_chains": len(r.version_chain_context),
                }
                for r in results
            ],
            "answer": answer,
        }

        logger.info(f"--- Query Complete ---")
        return response

    def get_stats(self) -> dict:
        return {
            "memory_count": self.memory_store.count(),
            "llm_usage": self.usage_tracker.get_stats(),
        }
