"""Single-pass answerer: direct session texts + merged propositions → one LLM call."""
from __future__ import annotations

import hashlib
import logging
import os
import re

import tiktoken

from src.schemas.models import RetrievalResult, Proposition, SubQuery
from src.embedding.base import BaseEmbeddingProvider
from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.answer_gen import build_answer_gen_prompt

logger = logging.getLogger(__name__)


class Answerer:

    _BLOCK_TOKENS = 250
    _BLOCK_CACHE_SIZE = 8
    _ELISION = "[... omitted ...]"
    # Literal anchors a query can name and a passage can be checked against:
    # ordinal record references, bracketed reference ids, quoted spans, and bare
    # multi-digit values. Dense similarity is weakest exactly where these matter
    # -- every node of an accessibility tree embeds alike, so "which id is the
    # 'Clear all' button at Step 19" has no semantic gradient to follow, while
    # the literal "Step 19" pins the passage immediately. Blending the two is
    # ordinary hybrid retrieval, and the decomposer already reproduces
    # identifiers verbatim, so the anchors survive into the sub-queries.
    _ANCHOR_PATTERNS = (
        re.compile(r'\b(?:step|turn|round|message|episode|iteration)\s*#?\d+\b', re.I),
        re.compile(r'\[\d+\]'),
        re.compile(r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]{2,40})['\"\u2018\u2019\u201c\u201d]"),
        re.compile(r'\b\d{2,}\b'),
    )
    _ANCHOR_WEIGHT = float(os.getenv("Q2Q_ANCHOR_WEIGHT", "0.5"))

    def __init__(
        self,
        llm_client: BaseLLMClient,
        language: str = "zh",
        max_context_tokens: int = 10000,
        embedding_provider: BaseEmbeddingProvider | None = None,
    ):
        self.llm_client = llm_client
        self.language = language
        self.max_context_tokens = max_context_tokens
        self.embedding_provider = embedding_provider
        self._block_cache: dict[str, list] = {}
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

    def _split_blocks(self, text: str) -> list[str]:
        blocks: list[str] = []
        cur: list[str] = []
        cur_tokens = 0
        for line in text.split("\n"):
            n = self._count_tokens(line) + 1
            if cur and cur_tokens + n > self._BLOCK_TOKENS:
                blocks.append("\n".join(cur))
                cur, cur_tokens = [], 0
            cur.append(line)
            cur_tokens += n
        if cur:
            blocks.append("\n".join(cur))
        return blocks

    @classmethod
    def _extract_anchors(cls, texts: list[str]) -> list[re.Pattern]:
        found: set[str] = set()
        for text in texts:
            for pattern in cls._ANCHOR_PATTERNS:
                for match in pattern.finditer(text):
                    token = (match.group(1) if match.groups() else match.group(0)).strip()
                    if len(token) >= 2:
                        found.add(token.lower())
        # "Step 19" already implies "19", and "[43647]" implies "43647"; counting
        # both would weight one anchor twice and let a single matched reference
        # outvote several distinct ones.
        anchors = [
            a for a in found
            if not any(a != other and a in other for other in found)
        ]
        compiled = []
        for anchor in sorted(anchors):
            # Bare values need word boundaries, or "19" matches "[2190]".
            if anchor.isalnum():
                compiled.append(re.compile(r'\b' + re.escape(anchor) + r'\b'))
            else:
                compiled.append(re.compile(re.escape(anchor)))
        return compiled

    async def _embed_blocks(self, blocks: list[str]) -> list | None:
        key = hashlib.sha1("\n\n".join(blocks).encode("utf-8")).hexdigest()
        cached = self._block_cache.get(key)
        if cached is not None:
            return cached
        try:
            embeddings = await self.embedding_provider.embed_batch(blocks)
        except Exception as exc:
            logger.warning(f"Block embedding failed, keeping prefix instead: {exc}")
            return None
        if len(self._block_cache) >= self._BLOCK_CACHE_SIZE:
            self._block_cache.pop(next(iter(self._block_cache)))
        self._block_cache[key] = embeddings
        return embeddings

    async def _focus_to_tokens(
        self, text: str, max_tokens: int, sub_queries: list[SubQuery] | None
    ) -> str:
        """Fit `text` into `max_tokens`, keeping the parts the query asks about.

        A session can exceed the budget on its own: one AMA-Bench web trajectory
        is a ~25k-token accessibility-tree trace stored as a single session, and
        several dialogue chunks together overrun the default budget too. Keeping
        a prefix then answers only questions about the start of the record. On
        AMA-Bench WEB at the 10k-token default, scoring line-blocks against the
        sub-queries and re-emitting the winners in reading order raised the share
        of gold-quoted element ids present in the context from 56% to 87% at the
        same budget -- 81% from dense similarity alone, the rest from the literal
        anchors below. Embeddings are local, so this costs no extra LLM calls,
        and blocks are cached because every query in an episode re-reads the same
        session.

        Note this is inert under the MedMemoryBench harness, which overrides
        max_context_tokens to 120000: no session text there comes close to the
        budget, so every call returns the text untouched. The loss measured on
        that path has a different cause -- ChromaDBStore caps the stored
        pre_session_text at 60000 chars -- and is not addressed here.
        """
        if max_tokens <= 0:
            return ""
        if self._count_tokens(text) <= max_tokens:
            return text

        query_embeddings = [
            sq.embedding for sq in (sub_queries or []) if sq.embedding is not None
        ]
        if self.embedding_provider is None or not query_embeddings:
            return self._truncate_to_tokens(text, max_tokens)

        blocks = self._split_blocks(text)
        if len(blocks) < 2:
            return self._truncate_to_tokens(text, max_tokens)
        block_embeddings = await self._embed_blocks(blocks)
        if block_embeddings is None:
            return self._truncate_to_tokens(text, max_tokens)

        anchors = self._extract_anchors([sq.text for sq in (sub_queries or [])])
        lowered = [b.lower() for b in blocks] if anchors else []

        def score(i: int) -> float:
            dense = max(
                BaseEmbeddingProvider.cosine_similarity(qe, block_embeddings[i])
                for qe in query_embeddings
            )
            if not anchors:
                return dense
            hits = sum(1 for a in anchors if a.search(lowered[i]))
            return dense + self._ANCHOR_WEIGHT * (hits / len(anchors))

        by_relevance = sorted(range(len(blocks)), key=score, reverse=True)
        # Block 0 is taken unconditionally: it holds the session header -- the
        # recorded date, and for traces the task statement -- that the rest of
        # the record is read relative to.
        order = [0] + [i for i in by_relevance if i != 0]

        chosen: set[int] = set()
        used = 0
        for i in order:
            cost = self._count_tokens(blocks[i]) + 1
            if used + cost > max_tokens:
                continue
            chosen.add(i)
            used += cost
        if not chosen:
            return self._truncate_to_tokens(text, max_tokens)

        parts: list[str] = []
        prev = -1
        for i in sorted(chosen):
            if prev >= 0 and i != prev + 1:
                parts.append(self._ELISION)
            parts.append(blocks[i])
            prev = i
        if prev < len(blocks) - 1:
            parts.append(self._ELISION)

        logger.info(
            f"Focused session text: {len(blocks)} blocks -> {len(chosen)} kept "
            f"({used}/{max_tokens} tokens)"
        )
        # Final guard so the tier budget stays exact despite the elision markers.
        return self._truncate_to_tokens("\n".join(parts), max_tokens)

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
