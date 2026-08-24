"""Fake Query Generator - proposition-based with binding verification."""
from __future__ import annotations

import json
import logging
import re

import numpy as np

from src.schemas.models import FakeQuery, Proposition
from src.embedding.base import BaseEmbeddingProvider
from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.fake_query_gen import build_fake_query_gen_prompt

logger = logging.getLogger(__name__)


class FakeQueryGenerator:

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_provider: BaseEmbeddingProvider,
        num_queries: int = 10,
        language: str = "zh",
        binding_verify_threshold: float = 0.8,
    ):
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.num_queries = num_queries
        self.language = language
        self.binding_verify_threshold = binding_verify_threshold

    async def generate(
        self,
        propositions: list[Proposition],
        session_text: str,
        session_id: str = "",
    ) -> list[FakeQuery]:
        if not propositions:
            logger.warning("No propositions provided, skipping FQ generation")
            return []

        props_text = self._format_propositions(propositions)
        prompt = build_fake_query_gen_prompt(props_text, self.num_queries, self.language)
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.7)

        raw_fqs = self._parse_response(response.content)
        if not raw_fqs:
            logger.warning("FQ generation returned empty, retrying with fallback")
            return []

        prop_id_set = {p.prop_id for p in propositions}
        fake_queries = []
        for item in raw_fqs:
            query_text = item.get("query", "")
            answer_source = item.get("answer_source", [])
            answer_source = [pid for pid in answer_source if pid in prop_id_set]

            if not query_text.strip():
                continue

            fq = FakeQuery(
                text=query_text,
                answer_source=answer_source,
                session_id=session_id,
            )
            fake_queries.append(fq)

        logger.info(f"Generated {len(fake_queries)} fake queries from {len(propositions)} propositions")

        fake_queries = await self._verify_bindings(fake_queries, propositions)
        return fake_queries

    async def _verify_bindings(
        self,
        fake_queries: list[FakeQuery],
        propositions: list[Proposition],
    ) -> list[FakeQuery]:
        if not fake_queries or not propositions:
            return fake_queries

        fq_texts = [fq.text for fq in fake_queries]
        prop_texts = [p.text for p in propositions]

        fq_embeddings = await self.embedding_provider.embed_batch(fq_texts)
        prop_embeddings = await self.embedding_provider.embed_batch(prop_texts)

        for i, fq in enumerate(fake_queries):
            fq.embedding = fq_embeddings[i]

        for i, prop in enumerate(propositions):
            prop.embedding = prop_embeddings[i]

        added_count = 0
        for i, fq in enumerate(fake_queries):
            existing_sources = set(fq.answer_source)
            for j, prop in enumerate(propositions):
                if prop.prop_id in existing_sources:
                    continue
                sim = BaseEmbeddingProvider.cosine_similarity(
                    fq_embeddings[i], prop_embeddings[j]
                )
                if sim >= self.binding_verify_threshold:
                    fq.answer_source.append(prop.prop_id)
                    added_count += 1

        if added_count > 0:
            logger.info(f"Binding verification added {added_count} new links")

        return fake_queries

    def _format_propositions(self, propositions: list[Proposition]) -> str:
        lines = []
        for p in propositions:
            line = f"[{p.prop_id}] {p.text}"
            if p.time:
                line += f" (time: {p.time})"
            if p.entities:
                line += f" (entities: {', '.join(p.entities)})"
            lines.append(line)
        return "\n".join(lines)

    def _parse_response(self, content: str) -> list[dict]:
        content = content.strip()

        parsed = self._try_json(content)
        if parsed is not None:
            return parsed

        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if match:
            parsed = self._try_json(match.group(1))
            if parsed is not None:
                return parsed

        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            parsed = self._try_json(match.group(0))
            if parsed is not None:
                return parsed

        logger.warning("Failed to parse FQ generation response")
        return []

    def _try_json(self, text: str) -> list[dict] | None:
        try:
            result = json.loads(text)
            if isinstance(result, list):
                valid = []
                for item in result:
                    if isinstance(item, dict) and "query" in item:
                        if "answer_source" not in item:
                            item["answer_source"] = []
                        elif isinstance(item["answer_source"], str):
                            item["answer_source"] = [item["answer_source"]]
                        valid.append(item)
                return valid if valid else None
        except json.JSONDecodeError:
            pass
        return None
