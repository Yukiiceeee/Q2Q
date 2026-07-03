from __future__ import annotations
import logging

from src.schemas.models import MemoryEntry, FakeQuery
from src.embedding.base import BaseEmbeddingProvider
from src.memory.query_generator import FakeQueryGenerator
from src.storage.base import BaseMemoryStore

logger = logging.getLogger(__name__)


class MemoryConstructor:

    def __init__(
        self,
        query_generator: FakeQueryGenerator,
        embedding_provider: BaseEmbeddingProvider,
        memory_store: BaseMemoryStore,
    ):
        self.query_generator = query_generator
        self.embedding_provider = embedding_provider
        self.memory_store = memory_store

    def build_memory(self, session_text: str, metadata: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(session_text=session_text, metadata=metadata or {})

        # Step 1: generate fake queries with answers (two-stage LLM pipeline)
        query_answer_pairs = self.query_generator.generate(session_text)
        logger.info(f"Generated {len(query_answer_pairs)} fake query-answer pairs for memory {entry.memory_id}")

        # Step 2: compute embeddings for fake queries (query + answer concatenated for richer vectors)
        if query_answer_pairs:
            embed_texts = []
            for pair in query_answer_pairs:
                combined = pair["query"]
                if pair.get("answer"):
                    combined += " " + pair["answer"]
                embed_texts.append(combined)

            query_embeddings = self.embedding_provider.embed_batch(embed_texts)
            for pair, emb in zip(query_answer_pairs, query_embeddings):
                fq = FakeQuery(
                    text=pair["query"],
                    answer=pair.get("answer", ""),
                    embedding=emb,
                    memory_id=entry.memory_id,
                )
                entry.fake_queries.append(fq)

        # Step 3: compute content embeddings (sliding window for long text)
        content_embeddings = self.embedding_provider.embed_long_text(session_text)
        entry.content_embeddings = content_embeddings

        # Step 4: persist
        self.memory_store.save(entry)
        logger.info(
            f"Memory {entry.memory_id} built: "
            f"{len(entry.fake_queries)} queries, "
            f"{len(entry.content_embeddings)} content chunks"
        )
        return entry
