"""Relevance-guided windowing of long passages, shared by retrieval and answering.

Both ends of the pipeline need the same operation: fit a passage that overruns a
token budget into that budget while keeping the parts the query actually asks
about. The answerer needs it to respect its own context tier; the retriever
needs it so a session's raw text is already trimmed before it leaves the
retriever at all. Keeping one implementation means the scoring -- dense
similarity blended with literal anchors -- cannot drift between the two.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re

import tiktoken

from src.schemas.models import SubQuery
from src.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class FocusWindow:

    _BLOCK_TOKENS = 250
    _BLOCK_CACHE_SIZE = 8
    ELISION = "[... omitted ...]"
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
        re.compile(r"['\"‘’“”]([^'\"‘’“”]{2,40})['\"‘’“”]"),
        re.compile(r'\b\d{2,}\b'),
    )
    _ANCHOR_WEIGHT = float(os.getenv("Q2Q_ANCHOR_WEIGHT", "0.5"))

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider | None = None,
        tokenizer_model: str = "gpt-4o-mini",
    ):
        self.embedding_provider = embedding_provider
        self._block_cache: dict[str, list] = {}
        try:
            self._tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        except (KeyError, AttributeError):
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens]) + "..."

    def _split_blocks(self, text: str) -> list[str]:
        blocks: list[str] = []
        cur: list[str] = []
        cur_tokens = 0
        for line in text.split("\n"):
            n = self.count_tokens(line) + 1
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

    async def focus(
        self, text: str, max_tokens: int, sub_queries: list[SubQuery] | None
    ) -> str:
        """Fit `text` into `max_tokens`, keeping the parts the query asks about.

        Keeping a prefix instead answers only questions about the start of the
        record, which is the wrong half for most queries: an AMA-Bench web
        trajectory is a ~25k-token accessibility-tree trace, and the storage
        layer already clips it to a blind 60k-char prefix before retrieval sees
        it. On AMA-Bench WEB at a 10k budget, scoring line-blocks against the
        sub-queries and re-emitting the winners in reading order raised the
        share of gold-quoted element ids present in the context from 56% to 87%
        at the same budget -- 81% from dense similarity alone, the rest from the
        literal anchors. Embeddings are local, so this costs no extra LLM calls,
        and blocks are cached because every query in an episode re-reads the
        same session.
        """
        if max_tokens <= 0:
            return ""
        if self.count_tokens(text) <= max_tokens:
            return text

        query_embeddings = [
            sq.embedding for sq in (sub_queries or []) if sq.embedding is not None
        ]
        if self.embedding_provider is None or not query_embeddings:
            return self.truncate_to_tokens(text, max_tokens)

        blocks = self._split_blocks(text)
        if len(blocks) < 2:
            return self.truncate_to_tokens(text, max_tokens)
        block_embeddings = await self._embed_blocks(blocks)
        if block_embeddings is None:
            return self.truncate_to_tokens(text, max_tokens)

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
            cost = self.count_tokens(blocks[i]) + 1
            if used + cost > max_tokens:
                continue
            chosen.add(i)
            used += cost
        if not chosen:
            return self.truncate_to_tokens(text, max_tokens)

        logger.debug(
            f"Focused passage: {len(blocks)} blocks -> {len(chosen)} kept "
            f"({used}/{max_tokens} tokens)"
        )
        # Final guard so the budget stays exact despite the elision markers.
        return self.truncate_to_tokens(self._join_blocks(blocks, chosen), max_tokens)

    def _join_blocks(self, blocks: list[str], chosen: set[int]) -> str:
        """Re-emit the kept blocks in reading order, marking what was dropped.

        Reading order rather than relevance order because a record is written to
        be read forwards: an answer about what happened after some step needs the
        steps in sequence. The marker tells the model a gap is a gap, so it does
        not read two distant steps as consecutive.
        """
        parts: list[str] = []
        prev = -1
        for i in sorted(chosen):
            if prev >= 0 and i != prev + 1:
                parts.append(self.ELISION)
            parts.append(blocks[i])
            prev = i
        if prev < len(blocks) - 1:
            parts.append(self.ELISION)
        return "\n".join(parts)

    def _waterfill_truncate(
        self, texts: list[str], counts: list[int], budget: int
    ) -> list[str]:
        """Degraded path: size-based shares and a blind cut, no relevance signal."""
        allocation = self.allocate(counts, budget)
        out = []
        for text, count, share in zip(texts, counts, allocation):
            if share <= 0:
                continue
            out.append(text if count <= share else self.truncate_to_tokens(text, share))
        return out

    async def focus_many(
        self, texts: list[str], max_tokens: int, sub_queries: list[SubQuery] | None
    ) -> list[str]:
        """Fit several passages into one shared budget, by block relevance.

        `allocate` splits a budget by how large each passage is, which is the
        right call when nothing is known about them -- but once the blocks are
        scored anyway, passage size is the wrong currency. Five LoCoMo dialogue
        chunks get an equal fifth each under `allocate`, so the two chunks the
        question is actually about are cut as hard as the three that merely
        matched on topic. Letting every block in the tier compete for the whole
        budget instead spends it where the answer is, and a passage that wins no
        block drops out entirely rather than contributing a truncated head.

        A passage that wins any block also keeps its block 0, charged to the
        budget, so whatever survives still carries its own date and header.
        A single passage has nothing to arbitrate, so it is handed straight to
        `focus`.
        """
        if not texts or max_tokens <= 0:
            return []
        if len(texts) == 1:
            only = await self.focus(texts[0], max_tokens, sub_queries)
            return [only] if only else []

        counts = [self.count_tokens(t) for t in texts]
        if sum(counts) + len(texts) <= max_tokens:
            return list(texts)

        query_embeddings = [
            sq.embedding for sq in (sub_queries or []) if sq.embedding is not None
        ]
        if self.embedding_provider is None or not query_embeddings:
            return self._waterfill_truncate(texts, counts, max_tokens)

        per_passage = [self._split_blocks(t) for t in texts]
        flat = [
            (p, i) for p, blocks in enumerate(per_passage) for i in range(len(blocks))
        ]
        if len(flat) < 2:
            return self._waterfill_truncate(texts, counts, max_tokens)
        embeddings = await self._embed_blocks(
            [per_passage[p][i] for p, i in flat]
        )
        if embeddings is None:
            return self._waterfill_truncate(texts, counts, max_tokens)

        anchors = self._extract_anchors([sq.text for sq in (sub_queries or [])])
        lowered = (
            [per_passage[p][i].lower() for p, i in flat] if anchors else []
        )
        # Each kept block can introduce at most one preceding elision marker, and
        # each surviving passage at most one trailing marker. Charging that up
        # front is what keeps the assembled text inside the budget without a
        # second truncation pass that would cut a block mid-sentence.
        marker = self.count_tokens(self.ELISION) + 1
        header = {p: k for k, (p, i) in enumerate(flat) if i == 0}
        costs = [
            self.count_tokens(per_passage[p][i]) + 1 + marker * (2 if i == 0 else 1)
            for p, i in flat
        ]

        def score(k: int) -> float:
            dense = max(
                BaseEmbeddingProvider.cosine_similarity(qe, embeddings[k])
                for qe in query_embeddings
            )
            if not anchors:
                return dense
            hits = sum(1 for a in anchors if a.search(lowered[k]))
            return dense + self._ANCHOR_WEIGHT * (hits / len(anchors))

        chosen: set[int] = set()
        used = 0
        for k in sorted(range(len(flat)), key=score, reverse=True):
            need = {k, header[flat[k][0]]} - chosen
            if not need:
                continue
            cost = sum(costs[j] for j in need)
            if used + cost > max_tokens:
                continue
            chosen |= need
            used += cost
        if not chosen:
            return self._waterfill_truncate(texts, counts, max_tokens)

        kept_by_passage: dict[int, set[int]] = {}
        for k in chosen:
            p, i = flat[k]
            kept_by_passage.setdefault(p, set()).add(i)

        out = []
        for p, blocks in enumerate(per_passage):
            idx = kept_by_passage.get(p)
            if not idx:
                continue
            out.append(self._join_blocks(blocks, idx))
        logger.debug(
            f"Focused {len(texts)} passages: {len(flat)} blocks -> {len(chosen)} kept "
            f"({used}/{max_tokens} tokens), {len(out)} passages survive"
        )
        return out

    def allocate(self, token_counts: list[int], budget: int) -> list[int]:
        """Water-fill `budget` across passages, by their own token counts.

        Equal shares would waste budget on passages that are already smaller
        than their share while starving the one oversized passage next to them
        -- exactly the LoCoMo (five ~2.6k-token chunks) versus AMA (one
        ~19k-token trace) split. Passages under their share keep everything and
        hand the remainder back, so a lone long passage receives the whole
        budget and a set of uniform ones receives an even split.
        """
        n = len(token_counts)
        if n == 0:
            return []
        allocation = [0] * n
        pending = set(range(n))
        remaining = budget
        while pending:
            share = remaining // len(pending)
            if share <= 0:
                for i in pending:
                    allocation[i] = 0
                break
            small = [i for i in pending if token_counts[i] <= share]
            if not small:
                for i in pending:
                    allocation[i] = share
                break
            for i in small:
                allocation[i] = token_counts[i]
                remaining -= token_counts[i]
                pending.discard(i)
        return allocation
