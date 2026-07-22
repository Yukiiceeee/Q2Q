from __future__ import annotations
import asyncio
import re
import logging

from src.schemas.models import MemoryEntry, FakeQuery, KnowledgePoint
from src.embedding.base import BaseEmbeddingProvider
from src.memory.query_generator import FakeQueryGenerator
from src.memory.version_detector import VersionDetector
from src.storage.base import BaseMemoryStore

import numpy as np

logger = logging.getLogger(__name__)

PARAGRAPH_MIN_CHARS = 50
PARAGRAPH_MAX_CHARS = 500
LINKED_KP_TOP_K = 10


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

    async def build_memory(self, session_text: str, metadata: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(session_text=session_text, metadata=metadata or {})

        # Fire-and-forget: content + paragraph embeddings (only depend on session_text)
        content_task = asyncio.create_task(
            self.embedding_provider.embed_long_text(session_text)
        )
        paragraphs = self._split_into_paragraphs(session_text)
        para_task = (
            asyncio.create_task(self.embedding_provider.embed_batch(paragraphs))
            if paragraphs else None
        )

        # Step 1: LLM Call 1 - generate fake query texts
        query_texts = await self.query_generator._generate_queries(session_text)
        logger.info(f"Step 1: Generated {len(query_texts)} fake query texts for memory {entry.memory_id}")

        if not query_texts:
            entry.content_embeddings = await content_task
            if para_task:
                entry.paragraph_embeddings = await para_task
                entry.paragraphs = paragraphs
            await self.memory_store.save(entry)
            return entry

        # Step 2: Embed query texts
        query_only_embeddings = await self.embedding_provider.embed_batch(query_texts)
        logger.info(f"Step 2: Computed {len(query_only_embeddings)} query embeddings")

        # Step 3: Version detection
        version_histories = None
        if self.version_detector:
            version_histories = await self.version_detector.detect_versions(
                query_texts, query_only_embeddings
            )
            logger.info(f"Step 3: Version detection completed")

        # Step 4: LLM Call 2 - extract knowledge points
        if version_histories:
            linked_kps = await self._gather_linked_knowledge_points(
                version_histories, query_only_embeddings
            )
            if linked_kps:
                kp_dicts = await self.query_generator.extract_knowledge_points_with_context(
                    session_text, linked_kps
                )
            else:
                kp_dicts = await self.query_generator.extract_knowledge_points(session_text)
        else:
            kp_dicts = await self.query_generator.extract_knowledge_points(session_text)

        entry.knowledge_points = [KnowledgePoint.from_dict(kp) for kp in kp_dicts]
        logger.info(f"Step 4: Extracted {len(entry.knowledge_points)} knowledge points")

        # Step 4.5: Embed knowledge points
        if entry.knowledge_points:
            kp_texts = [
                f"[{kp.time}] {kp.subject}: {kp.fact} ({kp.entities_or_values})"
                for kp in entry.knowledge_points
            ]
            entry.kp_embeddings = await self.embedding_provider.embed_batch(kp_texts)
            logger.info(f"Step 4.5: Computed {len(entry.kp_embeddings)} KP embeddings")

        # Step 5: Final embeddings
        final_embeddings = query_only_embeddings
        logger.info(f"Step 5: Using query-only embeddings ({len(final_embeddings)} vectors)")

        # Step 6: Build FakeQuery objects
        for i, (query, emb) in enumerate(zip(query_texts, final_embeddings)):
            fq = FakeQuery(text=query, answer="", embedding=emb, memory_id=entry.memory_id)
            if version_histories and i < len(version_histories):
                vh = version_histories[i]
                fq.parent_ids = vh["parent_ids"]
                fq.depth = vh["depth"]
            else:
                fq.parent_ids = []
                fq.depth = 0
            entry.fake_queries.append(fq)

        # Step 7: Await content embeddings (already running in parallel)
        entry.content_embeddings = await content_task

        # Step 7.5: Await paragraph embeddings
        if para_task:
            entry.paragraph_embeddings = await para_task
            entry.paragraphs = paragraphs
            logger.info(f"Step 7.5: Split into {len(paragraphs)} paragraphs")

        # Step 8: Persist
        await self.memory_store.save(entry)
        logger.info(
            f"Memory {entry.memory_id} built: "
            f"{len(entry.fake_queries)} queries, "
            f"{len(entry.knowledge_points)} knowledge_points, "
            f"{len(entry.paragraphs)} paragraphs, "
            f"{len(entry.content_embeddings)} content chunks"
        )
        return entry

    async def _gather_linked_knowledge_points(
        self, version_histories: list[dict], query_embeddings: list[np.ndarray]
    ) -> list[dict]:
        seen_memory_ids = set()
        memory_ids_to_fetch = []

        for vh in version_histories:
            if not vh.get("related_history"):
                continue
            for node in vh["related_history"]:
                mid = node.get("memory_id", "")
                if mid and mid not in seen_memory_ids:
                    seen_memory_ids.add(mid)
                    memory_ids_to_fetch.append(mid)

        if not memory_ids_to_fetch:
            return []

        # Concurrent fetch of all related memories
        memories = await asyncio.gather(*[
            self.memory_store.get_by_id(mid) for mid in memory_ids_to_fetch
        ])

        all_kp_candidates: list[tuple[dict, np.ndarray]] = []
        for mem in memories:
            if mem and mem.knowledge_points and mem.kp_embeddings:
                for kp, emb in zip(mem.knowledge_points, mem.kp_embeddings):
                    all_kp_candidates.append((kp.to_dict(), emb))
            elif mem and mem.knowledge_points:
                for kp in mem.knowledge_points:
                    all_kp_candidates.append((kp.to_dict(), None))

        if not all_kp_candidates:
            return []

        scored_kps: list[tuple[float, dict]] = []
        for kp_dict, kp_emb in all_kp_candidates:
            if kp_emb is None or not query_embeddings:
                scored_kps.append((0.0, kp_dict))
                continue
            best_sim = max(
                float(BaseEmbeddingProvider.cosine_similarity(q_emb, kp_emb))
                for q_emb in query_embeddings
            )
            scored_kps.append((best_sim, kp_dict))

        scored_kps.sort(key=lambda x: x[0], reverse=True)
        top_kps = [kp_dict for _, kp_dict in scored_kps[:LINKED_KP_TOP_K]]

        logger.info(
            f"Linked KPs: {len(all_kp_candidates)} candidates -> top {len(top_kps)} selected"
        )
        return top_kps

    def _split_into_paragraphs(self, session_text: str) -> list[str]:
        segments = re.split(r'\n{2,}', session_text)
        paragraphs = []
        buffer = ""
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if len(buffer) + len(seg) + 1 < PARAGRAPH_MIN_CHARS:
                buffer = (buffer + "\n" + seg).strip() if buffer else seg
                continue
            if buffer:
                if len(buffer) >= PARAGRAPH_MIN_CHARS:
                    paragraphs.append(buffer)
                else:
                    buffer = (buffer + "\n" + seg).strip()
                    if len(buffer) >= PARAGRAPH_MIN_CHARS:
                        paragraphs.append(buffer)
                        buffer = ""
                    continue
                buffer = ""
            if len(seg) <= PARAGRAPH_MAX_CHARS:
                paragraphs.append(seg)
            else:
                chunks = self._split_long_segment(seg)
                paragraphs.extend(chunks)
        if buffer and len(buffer) >= PARAGRAPH_MIN_CHARS:
            paragraphs.append(buffer)
        elif buffer and paragraphs:
            paragraphs[-1] = paragraphs[-1] + "\n" + buffer
        return paragraphs

    def _split_long_segment(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[。！？.!?\n])', text)
        chunks = []
        current = ""
        for sent in sentences:
            if not sent.strip():
                continue
            if len(current) + len(sent) > PARAGRAPH_MAX_CHARS and current:
                chunks.append(current.strip())
                current = sent
            else:
                current += sent
        if current.strip():
            chunks.append(current.strip())
        return [c for c in chunks if len(c) >= PARAGRAPH_MIN_CHARS]
