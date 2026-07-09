from __future__ import annotations
import logging

from src.schemas.models import RetrievalResult, VersionChainNode
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
        per_memory_limit = max(MAX_CONTEXT_CHARS // len(results), 800)

        for i, r in enumerate(results, 1):
            matched_fqs_preview = ", ".join(r.matched_fake_queries[:3]) if r.matched_fake_queries else ""

            memory_content_parts = []

            # Section A: matched fake query + answer pairs
            for fq in r.memory.fake_queries:
                if fq.text in r.matched_fake_queries:
                    memory_content_parts.append(f"Q: {fq.text}\nA: {fq.answer}")

            # Section B: version chain context
            if r.version_chain_context:
                for chain_idx, chain in enumerate(r.version_chain_context):
                    if len(chain) > 1:
                        chain_header = (
                            f"\n[知识演变 (链 {chain_idx + 1})]:"
                            if self.language == "zh"
                            else f"\n[Version History (chain {chain_idx + 1})]:"
                        )
                        chain_lines = [chain_header]
                        for node in chain:
                            chain_lines.append(f"  v{node.version_seq}: Q: {node.text}")
                            chain_lines.append(f"         A: {node.answer}")
                        memory_content_parts.append("\n".join(chain_lines))

            # Fallback: use session_text snippet if no Q&A pairs matched
            if not memory_content_parts:
                text = r.memory.session_text
                fallback_limit = per_memory_limit // 2
                if len(text) > fallback_limit:
                    text = text[:fallback_limit] + "..."
                memory_content_parts.append(f"Content:\n{text}")

            content = "\n".join(memory_content_parts)
            if len(content) > per_memory_limit:
                content = content[:per_memory_limit] + "..."

            entry_text = (
                f"### Memory {i} (score: {r.final_score:.3f}, "
                f"q2q: {r.score_q2q:.3f}, q2c: {r.score_q2c:.3f})\n"
                f"Matched queries: {matched_fqs_preview}\n"
                f"{content}"
            )

            if total_chars + len(entry_text) > MAX_CONTEXT_CHARS:
                break
            parts.append(entry_text)
            total_chars += len(entry_text)

        return "\n\n".join(parts)
