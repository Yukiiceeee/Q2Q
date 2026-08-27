"""Two-stage answerer: Stage 1 uses direct session texts + chain propositions; Stage 2 supplements with Q2C."""
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
        # === Stage 1: Direct FQ session texts + chain propositions ===
        stage1_context = self._build_stage1_context(retrieval_result)

        if not stage1_context.strip():
            # No Q2Q results at all, fall directly to Q2C
            if retrieval_result.q2c_session_texts:
                return await self._stage2_answer(raw_query, "", retrieval_result, history)
            return "未提及" if self.language == "zh" else "Not mentioned"

        prompt_s1 = build_answer_gen_prompt_stage1(
            raw_query, stage1_context, history, self.language
        )
        messages_s1 = format_messages(user_message=prompt_s1)
        response_s1 = await self.llm_client.chat(messages_s1, temperature=0.3)
        answer_s1 = response_s1.content.strip()

        logger.info(f"Stage 1 answer ({len(answer_s1)} chars)")

        if not self._is_insufficient(answer_s1):
            return answer_s1

        # === Stage 2: Supplement with Q2C session texts ===
        logger.info("Stage 1 insufficient, entering Stage 2 with Q2C context")
        return await self._stage2_answer(raw_query, stage1_context, retrieval_result, history)

    def _build_stage1_context(self, retrieval_result: RetrievalResult) -> str:
        parts = []
        total_tokens = 0
        token_budget = self.max_context_tokens

        # Part A: Direct FQ → session compressed texts (primary, high-information)
        direct_texts = retrieval_result.direct_session_texts
        if direct_texts:
            session_budget = int(token_budget * 0.7)
            per_session_budget = session_budget // max(len(direct_texts), 1)

            for i, text in enumerate(direct_texts, 1):
                header = f"--- Session {i} (Direct Match) ---"
                content = self._truncate_to_tokens(text, per_session_budget)
                section = f"{header}\n{content}"

                section_tokens = self._count_tokens(section)
                if total_tokens + section_tokens > token_budget:
                    break
                parts.append(section)
                total_tokens += section_tokens

        # Part B: Chain FQ → propositions (supplementary structured info)
        chain_props = retrieval_result.chain_propositions
        if chain_props:
            prop_budget = token_budget - total_tokens
            if prop_budget > 200:
                parts.append("\n--- Chain-linked Propositions ---")
                for j, prop in enumerate(chain_props, 1):
                    line = f"[C{j}] {prop.text}"
                    line_tokens = self._count_tokens(line)
                    if total_tokens + line_tokens > token_budget:
                        break
                    parts.append(line)
                    total_tokens += line_tokens

        return "\n".join(parts)

    async def _stage2_answer(
        self,
        raw_query: str,
        stage1_context: str,
        retrieval_result: RetrievalResult,
        history: str,
    ) -> str:
        # Use Q2C session texts as supplementary context
        q2c_context = self._format_q2c_sessions(retrieval_result.q2c_session_texts)

        if not q2c_context.strip() and not stage1_context.strip():
            return "未提及" if self.language == "zh" else "Not mentioned"

        prompt_s2 = build_answer_gen_prompt_stage2(
            raw_query, stage1_context, q2c_context, history, self.language
        )
        messages_s2 = format_messages(user_message=prompt_s2)
        response_s2 = await self.llm_client.chat(messages_s2, temperature=0.3)
        answer_s2 = response_s2.content.strip()

        answer_s2 = answer_s2.replace(INSUFFICIENT_MARKER, "").strip()
        logger.info(f"Stage 2 answer ({len(answer_s2)} chars)")
        return answer_s2

    def _is_insufficient(self, response: str) -> bool:
        return INSUFFICIENT_MARKER in response

    def _format_q2c_sessions(self, q2c_texts: list[str]) -> str:
        if not q2c_texts:
            return ""

        parts = []
        total_tokens = 0
        token_budget = self.max_context_tokens // 2

        for i, text in enumerate(q2c_texts, 1):
            header = f"--- Q2C Session {i} ---"
            per_budget = token_budget // max(len(q2c_texts), 1)
            content = self._truncate_to_tokens(text, per_budget)
            section = f"{header}\n{content}"

            section_tokens = self._count_tokens(section)
            if total_tokens + section_tokens > token_budget:
                break
            parts.append(section)
            total_tokens += section_tokens

        return "\n\n".join(parts)
