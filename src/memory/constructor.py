"""Memory Constructor - orchestrates preprocessing, FQ generation, embedding, and storage."""
from __future__ import annotations

import asyncio
import logging
import re

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

    def _extract_session_date(self, session_text: str, metadata: dict) -> str:
        if metadata.get("session_date"):
            return metadata["session_date"]
        if metadata.get("date"):
            return metadata["date"]
        if metadata.get("timestamp"):
            return metadata["timestamp"][:10]

        date_patterns = [
            re.compile(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'),
            re.compile(
                r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4})\b',
                re.IGNORECASE,
            ),
            re.compile(r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})\b', re.IGNORECASE),
        ]
        for pattern in date_patterns:
            match = pattern.search(session_text[:500])
            if match:
                return match.group(1)
        return ""

    async def build_memory(self, session_text: str, metadata: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(session_text=session_text, metadata=metadata or {})
        session_id = entry.session_id
        session_date = self._extract_session_date(session_text, metadata or {})

        logger.info(f"--- MemoryConstructor.build_memory ({len(session_text)} chars, date={session_date}) ---")

        # Phase 1: Parallel - compression + proposition extraction
        compress_task = asyncio.create_task(
            asyncio.to_thread(self.compressor.compress, session_text)
        )
        props_task = asyncio.create_task(
            self.prop_extractor.extract(session_text, session_id, session_date=session_date)
        )

        pre_session_text, propositions = await asyncio.gather(compress_task, props_task)
        entry.pre_session_text = pre_session_text
        entry.propositions = propositions

        # Debug: compression stats
        compress_ratio = len(pre_session_text) / max(len(session_text), 1)
        logger.info(
            f"  [DEBUG] Compression: {len(session_text)} → {len(pre_session_text)} chars "
            f"(ratio={compress_ratio:.2f})"
        )

        # Debug: proposition stats
        time_filled = sum(1 for p in propositions if p.time)
        entity_filled = sum(1 for p in propositions if p.entities)
        logger.info(
            f"  [DEBUG] Propositions: {len(propositions)} extracted, "
            f"time_filled={time_filled}/{len(propositions)}, "
            f"entity_filled={entity_filled}/{len(propositions)}"
        )
        for p in propositions[:5]:
            logger.info(f"    [PROP] {p.prop_id}: {p.text[:120]}")

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

        # Debug: FQ stats
        logger.info(
            f"  [DEBUG] FQ generation: {len(fake_queries)} FQs from {len(propositions)} props "
            f"(target={max(self.query_generator.num_queries, len(propositions))})"
        )
        if not fake_queries:
            logger.error(
                f"  [CRITICAL] Zero FQs generated for session {session_id} "
                f"with {len(propositions)} propositions!"
            )
        for fq in fake_queries[:5]:
            logger.info(f"    [FQ] {fq.query_id}: {fq.text[:120]} (sources={fq.answer_source})")

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
            f"  [DEBUG] Embeddings: {len(prop_embeddings)} prop embeddings (dim={len(prop_embeddings[0]) if prop_embeddings else 0}), "
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
