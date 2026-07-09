from __future__ import annotations
import uuid
import logging

from src.schemas.models import MemoryEntry, FakeQuery
from src.embedding.base import BaseEmbeddingProvider
from src.memory.query_generator import FakeQueryGenerator
from src.memory.version_detector import VersionDetector
from src.storage.base import BaseMemoryStore

logger = logging.getLogger(__name__)


class MemoryConstructor:

    def __init__(
        self,
        query_generator: FakeQueryGenerator,
        embedding_provider: BaseEmbeddingProvider,
        memory_store: BaseMemoryStore,
        version_detector: VersionDetector | None = None,
    ):
        self.query_generator = query_generator
        self.embedding_provider = embedding_provider
        self.memory_store = memory_store
        self.version_detector = version_detector

    def build_memory(self, session_text: str, metadata: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(session_text=session_text, metadata=metadata or {})

        # Step 1: LLM Call 1 - generate fake query texts
        query_texts = self.query_generator._generate_queries(session_text)
        logger.info(f"Step 1: Generated {len(query_texts)} fake query texts for memory {entry.memory_id}")

        if not query_texts:
            content_embeddings = self.embedding_provider.embed_long_text(session_text)
            entry.content_embeddings = content_embeddings
            self.memory_store.save(entry)
            return entry

        # Step 2: Embed query texts (for version detection)
        query_only_embeddings = self.embedding_provider.embed_batch(query_texts)
        logger.info(f"Step 2: Computed {len(query_only_embeddings)} query embeddings")

        # Step 3: Version detection
        version_histories = None
        if self.version_detector:
            version_histories = self.version_detector.detect_versions(
                query_texts, query_only_embeddings
            )
            logger.info(f"Step 3: Version detection completed")

        # Step 4: LLM Call 2 - generate answers (version-aware if applicable)
        if version_histories:
            answers = self.query_generator._generate_answers_with_history(
                session_text, query_texts, version_histories
            )
        else:
            answers = self.query_generator._generate_answers(session_text, query_texts)
        logger.info(f"Step 4: Generated {len(answers)} answers")

        # Step 5: Compute final embeddings (query + answer concatenated)
        embed_texts = []
        for query, answer in zip(query_texts, answers):
            combined = query
            if answer:
                combined += " " + answer
            embed_texts.append(combined)

        final_embeddings = self.embedding_provider.embed_batch(embed_texts)

        # Step 6: Build FakeQuery objects with version chain metadata
        for i, (query, answer, emb) in enumerate(zip(query_texts, answers, final_embeddings)):
            fq = FakeQuery(
                text=query,
                answer=answer,
                embedding=emb,
                memory_id=entry.memory_id,
            )
            if version_histories and i < len(version_histories):
                vh = version_histories[i]
                fq.chain_id = vh["chain_id"]
                fq.version_seq = vh["version_seq"]
                fq.supersedes = vh["supersedes"]
            else:
                fq.chain_id = uuid.uuid4().hex[:12]
                fq.version_seq = 0
                fq.supersedes = ""

            entry.fake_queries.append(fq)

        # Step 7: Compute content embeddings (unchanged)
        content_embeddings = self.embedding_provider.embed_long_text(session_text)
        entry.content_embeddings = content_embeddings

        # Step 8: Persist
        self.memory_store.save(entry)
        logger.info(
            f"Memory {entry.memory_id} built: "
            f"{len(entry.fake_queries)} queries, "
            f"{len(entry.content_embeddings)} content chunks"
        )
        return entry
