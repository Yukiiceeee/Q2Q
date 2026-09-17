"""Single-pass answerer: direct session texts + merged propositions → one LLM call."""
from __future__ import annotations

import logging

import tiktoken

from src.schemas.models import RetrievalResult, Proposition, SubQuery
from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.answer_gen import build_answer_gen_prompt

logger = logging.getLogger(__name__)


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
        context = self._build_context(retrieval_result)

        if not context.strip():
            return "信息不充分" if self.language == "zh" else "Insufficient information"

        prompt = build_answer_gen_prompt(
            raw_query, context, history, self.language
        )
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.3)
        answer = response.content.strip()

        logger.info(f"Answer generated ({len(answer)} chars)")
        return answer

    def _build_context(self, retrieval_result: RetrievalResult) -> str:
        parts = []
        total_tokens = 0
        token_budget = self.max_context_tokens

        # Part A (50%): Direct FQ → session compressed texts
        direct_texts = retrieval_result.direct_session_texts
        session_budget = int(token_budget * 0.50)
        if direct_texts:
            per_session_budget = session_budget // max(len(direct_texts), 1)

            for i, text in enumerate(direct_texts, 1):
                header = f"--- Session {i} (Direct Match) ---"
                content = self._truncate_to_tokens(text, per_session_budget)
                section = f"{header}\n{content}"

                section_tokens = self._count_tokens(section)
                if total_tokens + section_tokens > session_budget:
                    break
                parts.append(section)
                total_tokens += section_tokens

        # Part B (25%): Chain propositions from DAG expansion
        chain_props = retrieval_result.chain_propositions
        chain_budget = int(token_budget * 0.25)
        if chain_props:
            parts.append("\n--- Chain Propositions (DAG Expansion) ---")
            spent_chain = 0
            for j, prop in enumerate(chain_props, 1):
                line = f"[C{j}] {prop.text}"
                line_tokens = self._count_tokens(line)
                if spent_chain + line_tokens > chain_budget:
                    break
                parts.append(line)
                spent_chain += line_tokens
            total_tokens += spent_chain

        # Part C (25%): Q2C propositions from content-matched sessions
        q2c_props = retrieval_result.q2c_propositions
        q2c_budget = int(token_budget * 0.25)
        if q2c_props:
            parts.append("\n--- Cross-Session Propositions (Q2C) ---")
            spent_q2c = 0
            for k, prop in enumerate(q2c_props, 1):
                line = f"[X{k}] {prop.text}"
                line_tokens = self._count_tokens(line)
                if spent_q2c + line_tokens > q2c_budget:
                    break
                parts.append(line)
                spent_q2c += line_tokens
            total_tokens += spent_q2c

        return "\n".join(parts)
