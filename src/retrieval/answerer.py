from __future__ import annotations
import logging

import numpy as np
import tiktoken

from src.schemas.models import RetrievalResult, VersionChainNode, KnowledgePoint, SubQuery
from src.embedding.base import BaseEmbeddingProvider
from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.answer_gen import build_answer_gen_prompt

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 10000
DEFAULT_FQ_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_KP_TOP_K = 10


class Answerer:

    def __init__(
        self,
        llm_client: BaseLLMClient,
        language: str = "zh",
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        fq_confidence_threshold: float = DEFAULT_FQ_CONFIDENCE_THRESHOLD,
        kp_top_k: int = DEFAULT_KP_TOP_K,
    ):
        self.llm_client = llm_client
        self.language = language
        self.max_context_tokens = max_context_tokens
        self.fq_confidence_threshold = fq_confidence_threshold
        self.kp_top_k = kp_top_k
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

    def answer(
        self,
        raw_query: str,
        results: list[RetrievalResult],
        history: str = "",
        sub_queries: list[SubQuery] | None = None,
    ) -> str:
        memories_context = self._build_memories_context(results, sub_queries)
        prompt = build_answer_gen_prompt(raw_query, memories_context, history, self.language)
        messages = format_messages(user_message=prompt)
        response = self.llm_client.chat(messages, temperature=0.7)
        return response.content

    def _build_memories_context(
        self, results: list[RetrievalResult], sub_queries: list[SubQuery] | None = None
    ) -> str:
        if not results:
            return "(无相关记忆)" if self.language == "zh" else "(No relevant memories found)"

        # Collect sub-query embeddings for KP reranking
        sq_embeddings = []
        if sub_queries:
            sq_embeddings = [sq.embedding for sq in sub_queries if sq.embedding is not None]

        parts = []
        total_tokens = 0
        per_memory_token_limit = max(self.max_context_tokens // len(results), 1500)

        for i, r in enumerate(results, 1):
            is_high_confidence = r.score_q2q >= self.fq_confidence_threshold

            memory_content_parts = []

            # Knowledge Points with query-aware reranking
            if r.memory.knowledge_points:
                kp_text = self._format_reranked_knowledge_points(
                    r.memory.knowledge_points,
                    r.memory.kp_embeddings,
                    sq_embeddings,
                    per_memory_token_limit,
                )
                memory_content_parts.append(kp_text)

            # Version chain context (high confidence only)
            if is_high_confidence and r.version_chain_context:
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

            # Low confidence path: add matched paragraphs
            if not is_high_confidence and r.matched_paragraphs:
                para_header = (
                    "\n[相关片段]:" if self.language == "zh" else "\n[Relevant Excerpts]:"
                )
                para_lines = [para_header]
                for j, para in enumerate(r.matched_paragraphs, 1):
                    para_lines.append(f"  --- [{j}] ---")
                    para_lines.append(f"  {para}")
                memory_content_parts.append("\n".join(para_lines))

            # Fallback: if no knowledge points and no paragraphs, use session_text
            if not memory_content_parts:
                text = r.memory.session_text
                text = self._truncate_to_tokens(text, per_memory_token_limit)
                memory_content_parts.append(f"Content:\n{text}")

            content = "\n".join(memory_content_parts)
            content_tokens = self._count_tokens(content)
            if content_tokens > per_memory_token_limit:
                content = self._truncate_to_tokens(content, per_memory_token_limit)

            confidence_label = "HIGH" if is_high_confidence else "LOW"
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

    def _format_reranked_knowledge_points(
        self,
        kps: list[KnowledgePoint],
        kp_embeddings: list[np.ndarray],
        sq_embeddings: list[np.ndarray],
        token_limit: int,
    ) -> str:
        """Rank KPs by relevance to sub-queries, then format top-K within token budget."""
        if sq_embeddings and kp_embeddings and len(kp_embeddings) == len(kps):
            scored_kps = []
            for idx, (kp, kp_emb) in enumerate(zip(kps, kp_embeddings)):
                best_sim = max(
                    float(BaseEmbeddingProvider.cosine_similarity(sq_emb, kp_emb))
                    for sq_emb in sq_embeddings
                )
                scored_kps.append((best_sim, idx, kp))

            scored_kps.sort(key=lambda x: x[0], reverse=True)
            ranked_kps = [(kp, score) for score, _, kp in scored_kps[:self.kp_top_k]]
        else:
            # Fallback: no embeddings available, use original order
            ranked_kps = [(kp, 0.0) for kp in kps[:self.kp_top_k]]

        header = "Relevant Knowledge Points:" if self.language == "en" else "相关知识点:"
        lines = [header]
        total_tokens = self._count_tokens(header)

        for kp, score in ranked_kps:
            line = f"  - [{kp.time}] {kp.subject}: {kp.fact}"
            if kp.entities_or_values:
                line += f" ({kp.entities_or_values})"
            line_tokens = self._count_tokens(line)
            if total_tokens + line_tokens > token_limit:
                break
            lines.append(line)
            total_tokens += line_tokens

        return "\n".join(lines)
