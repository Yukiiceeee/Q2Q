"""Fake Query Generator - CoT multi-dimension routing with style/role control."""
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


FALLBACK_PROMPT_EN = """Given these propositions extracted from a conversation, generate {num_queries} short questions that a user might ask to retrieve this information.

Propositions:
{propositions_text}

Output a JSON array of objects with "query" and "answer_source" fields. "answer_source" is a list of proposition IDs (e.g. ["P1","P3"]).
Output ONLY the JSON array, no other text.

Example:
[
  {{"query": "When did Emma visit the museum?", "answer_source": ["P1"]}},
  {{"query": "What book did Sarah recommend?", "answer_source": ["P2", "P3"]}}
]"""

FALLBACK_PROMPT_ZH = """给定以下从对话中提取的命题，生成 {num_queries} 个用户可能会提出的简短问题。

命题列表：
{propositions_text}

输出 JSON 数组，每个元素包含 "query" 和 "answer_source" 字段。"answer_source" 是命题 ID 列表。
只输出 JSON 数组，不要其他文本。

示例：
[
  {{"query": "Emma 什么时候去了博物馆？", "answer_source": ["P1"]}},
  {{"query": "Sarah 推荐了什么书？", "answer_source": ["P2", "P3"]}}
]"""


class FakeQueryGenerator:

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_provider: BaseEmbeddingProvider,
        num_queries: int = 15,
        language: str = "zh",
        binding_verify_threshold: float = 0.8,
        style: str = "formal",
        role: str = "user",
    ):
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.num_queries = num_queries
        self.language = language
        self.binding_verify_threshold = binding_verify_threshold
        self.style = style
        self.role = role

    async def generate(
        self,
        propositions: list[Proposition],
        session_text: str,
        session_id: str = "",
    ) -> list[FakeQuery]:
        if not propositions:
            logger.warning("No propositions provided, skipping FQ generation")
            return []

        dynamic_num = max(self.num_queries, min(len(propositions) * 2, 40))
        logger.info(
            f"FQ generation: {len(propositions)} props → target {dynamic_num} queries "
            f"(config={self.num_queries})"
        )

        props_text = self._format_propositions(propositions)

        # Attempt 1: full prompt
        prompt = build_fake_query_gen_prompt(
            props_text,
            dynamic_num,
            self.language,
            style=self.style,
            role=self.role,
        )
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.7)

        raw_fqs = self._parse_response(response.content)

        # Attempt 2: simplified fallback prompt if first attempt failed
        if not raw_fqs:
            logger.warning(
                f"[FQ] Primary generation failed (response length={len(response.content)}). "
                f"Response preview: {response.content[:300]}..."
            )
            logger.info("[FQ] Retrying with simplified fallback prompt...")
            fallback_prompt = self._build_fallback_prompt(props_text, dynamic_num)
            fallback_messages = format_messages(user_message=fallback_prompt)
            try:
                fallback_response = await self.llm_client.chat(fallback_messages, temperature=0.5)
                raw_fqs = self._parse_response(fallback_response.content)
                if raw_fqs:
                    logger.info(f"[FQ] Fallback prompt succeeded: {len(raw_fqs)} queries parsed")
                else:
                    logger.warning(
                        f"[FQ] Fallback also failed. Response preview: "
                        f"{fallback_response.content[:300]}..."
                    )
            except Exception as e:
                logger.warning(f"[FQ] Fallback LLM call failed: {e}")

        # Attempt 3: template-based generation as last resort
        if not raw_fqs:
            logger.warning("[FQ] All LLM attempts failed. Using template-based fallback.")
            raw_fqs = self._generate_template_fqs(propositions)

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

        logger.info(
            f"Generated {len(fake_queries)} fake queries from {len(propositions)} propositions "
            f"(style={self.style}, role={self.role})"
        )

        fake_queries = await self._verify_bindings(fake_queries, propositions)
        return fake_queries

    def _build_fallback_prompt(self, props_text: str, num_queries: int) -> str:
        template = FALLBACK_PROMPT_ZH if self.language == "zh" else FALLBACK_PROMPT_EN
        return template.format(propositions_text=props_text, num_queries=num_queries)

    def _generate_template_fqs(self, propositions: list[Proposition]) -> list[dict]:
        templates_en = [
            "What happened regarding {subject}?",
            "When did {subject} {action}?",
            "What did {subject} do?",
        ]
        templates_zh = [
            "关于{subject}发生了什么？",
            "{subject}什么时候{action}？",
            "{subject}做了什么？",
        ]
        templates = templates_zh if self.language == "zh" else templates_en

        result = []
        for prop in propositions:
            subject = prop.subject or (prop.entities[0] if prop.entities else "this person")
            fact_words = prop.object_value.split() if prop.object_value else prop.text.split()
            action = " ".join(fact_words[:5]) if fact_words else "that"

            for tmpl in templates[:2]:
                try:
                    q = tmpl.format(subject=subject, action=action)
                except (KeyError, IndexError):
                    q = tmpl.format(subject=subject, action="that")
                result.append({
                    "query": q,
                    "answer_source": [prop.prop_id],
                })

        logger.info(f"[FQ] Template fallback generated {len(result)} queries")
        return result

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

        # Try parsing as the CoT JSON format: {"_reasoning": ..., "queries": [...]}
        parsed = self._try_cot_json(content)
        if parsed is not None:
            return parsed

        # Fallback: try extracting JSON from code blocks
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            parsed = self._try_cot_json(match.group(1))
            if parsed is not None:
                return parsed

        # Fallback: try extracting bare JSON object with "queries" key
        match = re.search(r'\{[^{}]*"queries"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
        if match:
            parsed = self._try_cot_json(match.group(0))
            if parsed is not None:
                return parsed

        # Fallback: try extracting a JSON object greedily (handles nested braces)
        match = re.search(r'\{.*"queries"\s*:\s*\[.*\].*\}', content, re.DOTALL)
        if match:
            parsed = self._try_cot_json(match.group(0))
            if parsed is not None:
                return parsed

        # Fallback: try legacy array format
        parsed = self._try_legacy_array(content)
        if parsed is not None:
            return parsed

        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if match:
            parsed = self._try_legacy_array(match.group(1))
            if parsed is not None:
                return parsed

        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            parsed = self._try_legacy_array(match.group(0))
            if parsed is not None:
                return parsed

        # Fallback: try fixing truncated JSON (missing closing brackets)
        parsed = self._try_fix_truncated_json(content)
        if parsed is not None:
            return parsed

        logger.warning(f"Failed to parse FQ generation response (length={len(content)})")
        return []

    def _try_cot_json(self, text: str) -> list[dict] | None:
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "queries" in result:
                queries = result["queries"]
                if isinstance(queries, list):
                    reasoning = result.get("_reasoning", "")
                    if reasoning:
                        logger.info(f"FQ reasoning: {reasoning[:200]}")
                    valid = []
                    for item in queries:
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

    def _try_legacy_array(self, text: str) -> list[dict] | None:
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

    def _try_fix_truncated_json(self, content: str) -> list[dict] | None:
        # Try to find the start of a JSON structure and fix truncation
        for start_pattern in [r'"queries"\s*:\s*\[', r'^\s*\[']:
            match = re.search(start_pattern, content)
            if not match:
                continue

            # Extract from the array start
            if '"queries"' in match.group(0):
                arr_start = content.find('[', match.start())
            else:
                arr_start = match.start()

            if arr_start < 0:
                continue

            fragment = content[arr_start:]

            # Find the last complete object (ends with })
            last_brace = fragment.rfind('}')
            if last_brace < 0:
                continue

            # Take everything up to the last complete object, close the array
            candidate = fragment[:last_brace + 1] + ']'
            parsed = self._try_legacy_array(candidate)
            if parsed:
                logger.info(f"[FQ] Recovered {len(parsed)} queries from truncated JSON")
                return parsed

        return None
