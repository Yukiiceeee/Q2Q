from __future__ import annotations
import logging

import tiktoken

from src.schemas.models import RetrievalResult, KnowledgePoint, SubQuery
from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.answer_gen import build_answer_gen_prompt

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 10000
DEFAULT_FQ_CONFIDENCE_THRESHOLD = 0.80


class Answerer:

    def __init__(
        self,
        llm_client: BaseLLMClient,
        language: str = "zh",
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        fq_confidence_threshold: float = DEFAULT_FQ_CONFIDENCE_THRESHOLD,
    ):
        self.llm_client = llm_client
        self.language = language
        self.max_context_tokens = max_context_tokens
        self.fq_confidence_threshold = fq_confidence_threshold
        try:
            self._tokenizer = tiktoken.encoding_for_model(llm_client.model)
        except (KeyError, AttributeError):
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens]) + "..."

    async def answer(
        self,
        raw_query: str,
        results: list[RetrievalResult],
        history: str = "",
        sub_queries: list[SubQuery] | None = None,
    ) -> str:
        memories_context = self._build_memories_context(results)
        prompt = build_answer_gen_prompt(raw_query, memories_context, history, self.language)
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.7)
        return response.content

    def _build_memories_context(self, results: list[RetrievalResult]) -> str:
        if not results:
            return "(无相关记忆)" if self.language == "zh" else "(No relevant memories found)"

        all_below_threshold = all(
            r.final_score < self.fq_confidence_threshold for r in results
        )

        parts = []
        total_tokens = 0
        per_memory_token_limit = max(self.max_context_tokens // len(results), 1500)

        for i, r in enumerate(results, 1):
            memory_content_parts = []

            if r.memory.knowledge_points:
                kp_text = self._format_knowledge_points(
                    r.memory.knowledge_points,
                    per_memory_token_limit,
                )
                memory_content_parts.append(kp_text)

            if not all_below_threshold and r.version_chain_context:
                for chain_idx, chain in enumerate(r.version_chain_context):
                    if len(chain) > 1:
                        chain_header = (
                            f"\n[知识演变 (链 {chain_idx + 1})]:"
                            if self.language == "zh"
                            else f"\n[Version History (chain {chain_idx + 1})]:"
                        )
                        chain_lines = [chain_header]
                        for node in chain:
                            chain_lines.append(f"  [d{node.depth}] {node.text}")
                        memory_content_parts.append("\n".join(chain_lines))

            if all_below_threshold and r.matched_paragraphs:
                para_header = (
                    "\n[相关片段]:" if self.language == "zh" else "\n[Relevant Excerpts]:"
                )
                para_lines = [para_header]
                for j, para in enumerate(r.matched_paragraphs, 1):
                    para_lines.append(f"  --- [{j}] ---")
                    para_lines.append(f"  {para}")
                memory_content_parts.append("\n".join(para_lines))

            if not memory_content_parts:
                continue

            content = "\n".join(memory_content_parts)
            content_tokens = self._count_tokens(content)
            if content_tokens > per_memory_token_limit:
                content = self._truncate_to_tokens(content, per_memory_token_limit)

            confidence_label = "HIGH" if r.final_score >= self.fq_confidence_threshold else "LOW"
            entry_text = (
                f"### Memory {i} (relevance: {r.final_score:.3f}, confidence: {confidence_label})\n"
                f"{content}"
            )

            entry_tokens = self._count_tokens(entry_text)
            if total_tokens + entry_tokens > self.max_context_tokens:
                remaining = self.max_context_tokens - total_tokens
                if remaining > 200:
                    entry_text = self._truncate_to_tokens(entry_text, remaining)
                    parts.append(entry_text)
                break
            parts.append(entry_text)
            total_tokens += entry_tokens

        return "\n\n".join(parts)

    def _format_knowledge_points(
        self,
        kps: list[KnowledgePoint],
        token_limit: int,
    ) -> str:
        header = "Relevant Knowledge Points:" if self.language == "en" else "相关知识点:"
        lines = [header]
        total_tokens = self._count_tokens(header)

        for kp in kps:
            line = f"  - [{kp.time}] {kp.subject}: {kp.fact}"
            if kp.entities_or_values:
                line += f" ({kp.entities_or_values})"
            line_tokens = self._count_tokens(line)
            if total_tokens + line_tokens > token_limit:
                break
            lines.append(line)
            total_tokens += line_tokens

        return "\n".join(lines)
