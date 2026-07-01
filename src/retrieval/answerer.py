from __future__ import annotations
import logging

from src.schemas.models import RetrievalResult, RetrievalResponse, SubQuery
from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.answer_gen import build_answer_gen_prompt

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6000


class Answerer:

    def __init__(self, llm_client: BaseLLMClient, language: str = "zh"):
        self.llm_client = llm_client
        self.language = language

    def answer(
        self,
        raw_query: str,
        results: list[RetrievalResult],
        history: str = "",
    ) -> str:
        memories_context = self._build_memories_context(results)
        prompt = build_answer_gen_prompt(raw_query, memories_context, history, self.language)
        messages = format_messages(user_message=prompt)
        response = self.llm_client.chat(messages, temperature=0.7)
        return response.content

    def _build_memories_context(self, results: list[RetrievalResult]) -> str:
        if not results:
            return "(无相关记忆)" if self.language == "zh" else "(No relevant memories found)"

        parts = []
        total_chars = 0
        per_memory_limit = max(MAX_CONTEXT_CHARS // len(results), 500)

        for i, r in enumerate(results, 1):
            text = r.memory.session_text
            if len(text) > per_memory_limit:
                text = text[:per_memory_limit] + "..."

            matched_fqs = ", ".join(r.matched_fake_queries) if r.matched_fake_queries else ""
            entry_text = (
                f"### Memory {i} (score: {r.final_score:.3f}, "
                f"q2q: {r.score_q2q:.3f}, q2c: {r.score_q2c:.3f})\n"
                f"Matched queries: {matched_fqs}\n"
                f"Content:\n{text}"
            )
            if total_chars + len(entry_text) > MAX_CONTEXT_CHARS:
                break
            parts.append(entry_text)
            total_chars += len(entry_text)

        return "\n\n".join(parts)
