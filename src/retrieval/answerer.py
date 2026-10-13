"""Single-pass answerer: direct session texts + merged propositions → one LLM call."""
from __future__ import annotations

import logging

from src.schemas.models import RetrievalResult, Proposition, SubQuery
from src.embedding.base import BaseEmbeddingProvider
from src.retrieval.focus import FocusWindow
from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.answer_gen import build_answer_gen_prompt

logger = logging.getLogger(__name__)


class Answerer:

    def __init__(
        self,
        llm_client: BaseLLMClient,
        language: str = "zh",
        max_context_tokens: int = 10000,
        embedding_provider: BaseEmbeddingProvider | None = None,
        focus_window: FocusWindow | None = None,
    ):
        self.llm_client = llm_client
        self.language = language
        self.max_context_tokens = max_context_tokens
        self.embedding_provider = embedding_provider
        # Shared with the retriever when one is handed in, so a session text
        # already windowed upstream is measured on the same tokenizer and hits
        # the same block-embedding cache instead of being re-embedded here.
        self.focus_window = focus_window or FocusWindow(
            embedding_provider=embedding_provider,
            tokenizer_model=getattr(llm_client, "model", "gpt-4o-mini"),
        )

    def _count_tokens(self, text: str) -> int:
        return self.focus_window.count_tokens(text)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return self.focus_window.truncate_to_tokens(text, max_tokens)

    async def _focus_to_tokens(
        self, text: str, max_tokens: int, sub_queries: list[SubQuery] | None
    ) -> str:
        return await self.focus_window.focus(text, max_tokens, sub_queries)

    async def answer(
        self,
        raw_query: str,
        retrieval_result: RetrievalResult,
        history: str = "",
        sub_queries: list[SubQuery] | None = None,
    ) -> str:
        context = await self._build_context(retrieval_result, sub_queries)

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

    def _format_proposition(self, tag: str, prop: Proposition) -> str:
        """Render one proposition line, prefixed with the date it was recorded.

        Without this the LLM sees bare statements and can only date them from
        whatever coarse phrasing the extractor put in the text ("about May
        2023"), so "when was this recorded" is unanswerable from this tier.
        """
        if prop.record_date:
            return f"[{tag}] ({prop.record_date}) {prop.text}"
        return f"[{tag}] {prop.text}"

    async def _build_context(
        self,
        retrieval_result: RetrievalResult,
        sub_queries: list[SubQuery] | None = None,
    ) -> str:
        """Priority-tiered dynamic filling: direct sessions → chain props → Q2C props.

        No fixed percentage allocation — each tier consumes what it needs and
        remaining budget flows down to the next tier.
        """
        parts = []
        remaining = self.max_context_tokens

        # Tier 1: Direct FQ → session compressed texts (full fidelity first)
        for i, text in enumerate(retrieval_result.direct_session_texts, 1):
            if remaining <= 0:
                break
            header = f"--- Session {i} (Direct Match) ---"
            header_tokens = self._count_tokens(header) + 1
            content = await self._focus_to_tokens(
                text, max(remaining - header_tokens, 0), sub_queries
            )
            if not content:
                break
            section = f"{header}\n{content}"
            parts.append(section)
            remaining -= self._count_tokens(section) + 1

        # Tier 2: Chain propositions from DAG expansion
        chain_lines = []
        for j, prop in enumerate(retrieval_result.chain_propositions, 1):
            line = self._format_proposition(f"C{j}", prop)
            line_tokens = self._count_tokens(line) + 1
            if remaining - line_tokens <= 0:
                break
            chain_lines.append(line)
            remaining -= line_tokens
        if chain_lines:
            parts.append("\n--- Chain Propositions (DAG Expansion) ---")
            parts.extend(chain_lines)

        # Tier 3: Q2C propositions from content-matched sessions
        q2c_lines = []
        for k, prop in enumerate(retrieval_result.q2c_propositions, 1):
            line = self._format_proposition(f"X{k}", prop)
            line_tokens = self._count_tokens(line) + 1
            if remaining - line_tokens <= 0:
                break
            q2c_lines.append(line)
            remaining -= line_tokens
        if q2c_lines:
            parts.append("\n--- Cross-Session Propositions (Q2C) ---")
            parts.extend(q2c_lines)

        return "\n".join(parts)
