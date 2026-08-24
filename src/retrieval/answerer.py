"""Two-stage answerer: propositions first, fallback to session text."""
from __future__ import annotations

import logging

import tiktoken

from src.schemas.models import RetrievalResult, Proposition, SubQuery
from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.answer_gen import (
    build_answer_gen_prompt_stage1,
    build_answer_gen_prompt_stage2,
)

logger = logging.getLogger(__name__)

INSUFFICIENT_MARKER = "[INSUFFICIENT]"


class Answerer:

    def __init__(
        self,
        llm_client: BaseLLMClient,
        language: str = "zh",
        max_context_tokens: int = 10000,
    ):
        self.llm_client = llm_client
        self.language = language
        self.max_context_tokens = max_context_tokens
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
        retrieval_result: RetrievalResult,
        history: str = "",
        sub_queries: list[SubQuery] | None = None,
    ) -> str:
        # === Stage 1: Answer from propositions ===
        props_context = self._format_propositions(retrieval_result.ranked_propositions)

        if not props_context.strip():
            if retrieval_result.session_texts:
                return await self._stage2_answer(raw_query, "", retrieval_result, history)
            return "未提及" if self.language == "zh" else "Not mentioned"

        prompt_s1 = build_answer_gen_prompt_stage1(
            raw_query, props_context, history, self.language
        )
        messages_s1 = format_messages(user_message=prompt_s1)
        response_s1 = await self.llm_client.chat(messages_s1, temperature=0.3)
        answer_s1 = response_s1.content.strip()

        logger.info(f"Stage 1 answer ({len(answer_s1)} chars)")

        # Check if sufficient
        if not self._is_insufficient(answer_s1):
            return answer_s1

        # === Stage 2: Supplement with session text ===
        logger.info("Stage 1 insufficient, entering Stage 2")
        clean_answer = answer_s1.replace(INSUFFICIENT_MARKER, "").strip()
        return await self._stage2_answer(raw_query, props_context, retrieval_result, history)

    async def _stage2_answer(
        self,
        raw_query: str,
        props_context: str,
        retrieval_result: RetrievalResult,
        history: str,
    ) -> str:
        session_context = self._format_session_texts(retrieval_result.session_texts)

        prompt_s2 = build_answer_gen_prompt_stage2(
            raw_query, props_context, session_context, history, self.language
        )
        messages_s2 = format_messages(user_message=prompt_s2)
        response_s2 = await self.llm_client.chat(messages_s2, temperature=0.3)
        answer_s2 = response_s2.content.strip()

        # Clean any remaining markers
        answer_s2 = answer_s2.replace(INSUFFICIENT_MARKER, "").strip()
        logger.info(f"Stage 2 answer ({len(answer_s2)} chars)")
        return answer_s2

    def _is_insufficient(self, response: str) -> bool:
        return INSUFFICIENT_MARKER in response

    def _format_propositions(self, propositions: list[Proposition]) -> str:
        if not propositions:
            return ""

        parts = []
        total_tokens = 0
        token_budget = self.max_context_tokens // 2

        for i, prop in enumerate(propositions, 1):
            line = f"[{i}] {prop.text}"
            if prop.time:
                line += f" (time: {prop.time})"
            if prop.subject:
                line += f" (subject: {prop.subject})"

            line_tokens = self._count_tokens(line)
            if total_tokens + line_tokens > token_budget:
                break
            parts.append(line)
            total_tokens += line_tokens

        return "\n".join(parts)

    def _format_session_texts(self, session_texts: list[str]) -> str:
        if not session_texts:
            return ""

        parts = []
        total_tokens = 0
        token_budget = self.max_context_tokens // 2

        for i, text in enumerate(session_texts, 1):
            header = f"--- Session {i} ---"
            content = self._truncate_to_tokens(text, token_budget // max(len(session_texts), 1))
            section = f"{header}\n{content}"

            section_tokens = self._count_tokens(section)
            if total_tokens + section_tokens > token_budget:
                break
            parts.append(section)
            total_tokens += section_tokens

        return "\n\n".join(parts)
