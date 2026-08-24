"""Memory Constructor - orchestrates preprocessing, FQ generation, embedding, and storage."""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from src.schemas.models import MemoryEntry, FakeQuery, Proposition
from src.embedding.base import BaseEmbeddingProvider
from src.preprocessing.compressor import PerplexityCompressor
from src.preprocessing.proposition_extractor import PropositionExtractor
from src.memory.query_generator import FakeQueryGenerator
from src.storage.base import BaseMemoryStore

logger = logging.getLogger(__name__)


class MemoryConstructor:

    def __init__(
        self,
        compressor: PerplexityCompressor,
        prop_extractor: PropositionExtractor,
        query_generator: FakeQueryGenerator,
        embedding_provider: BaseEmbeddingProvider,
        memory_store: BaseMemoryStore,
    ):
        self.compressor = compressor
        self.prop_extractor = prop_extractor
        self.query_generator = query_generator
        self.embedding_provider = embedding_provider
        self.memory_store = memory_store

    async def build_memory(self, session_text: str, metadata: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(session_text=session_text, metadata=metadata or {})
        session_id = entry.session_id

        logger.info(f"--- MemoryConstructor.build_memory ({len(session_text)} chars) ---")

        # Phase 1: Parallel - compression + proposition extraction
        compress_task = asyncio.create_task(
            asyncio.to_thread(self.compressor.compress, session_text)
        )
        props_task = asyncio.create_task(
            self.prop_extractor.extract(session_text, session_id)
        )

        pre_session_text, propositions = await asyncio.gather(compress_task, props_task)
        entry.pre_session_text = pre_session_text
        entry.propositions = propositions

        logger.info(
            f"  Phase 1 done: compressed={len(pre_session_text)} chars, "
            f"propositions={len(propositions)}"
        )

        if not propositions:
            logger.warning("  No propositions extracted, storing minimal entry")
            entry.content_embeddings = await self.embedding_provider.embed_long_text(
                pre_session_text or session_text
            )
            await self.memory_store.save(entry)
            return entry

        # Phase 2: FQ generation (uses propositions)
        fake_queries = await self.query_generator.generate(
            propositions, session_text, session_id
        )
        entry.fake_queries = fake_queries
        logger.info(f"  Phase 2 done: {len(fake_queries)} fake queries generated")

        # Phase 2.5: Embed propositions + content chunks (parallel)
        prop_embed_task = asyncio.create_task(
            self.embedding_provider.embed_batch([p.text for p in propositions])
        )
        content_embed_task = asyncio.create_task(
            self.embedding_provider.embed_long_text(pre_session_text or session_text)
        )

        prop_embeddings, content_embeddings = await asyncio.gather(
            prop_embed_task, content_embed_task
        )

        for i, prop in enumerate(entry.propositions):
            prop.embedding = prop_embeddings[i]

        entry.content_embeddings = content_embeddings
        logger.info(
            f"  Phase 2.5 done: {len(prop_embeddings)} prop embeddings, "
            f"{len(content_embeddings)} content chunks"
        )

        # Phase 3: Persist
        await self.memory_store.save(entry)
        logger.info(
            f"  Memory {session_id} built: "
            f"{len(entry.fake_queries)} FQs, "
            f"{len(entry.propositions)} props, "
            f"{len(entry.content_embeddings)} chunks"
        )
        return entry
