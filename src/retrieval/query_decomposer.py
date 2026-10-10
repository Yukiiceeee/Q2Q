from __future__ import annotations
import json
import re
import logging

from src.schemas.models import SubQuery
from src.embedding.base import BaseEmbeddingProvider
from src.utils.llm_client import BaseLLMClient, format_messages
from src.utils.query_text import sanitize_query
from src.prompts.query_decompose import build_query_decompose_prompt

logger = logging.getLogger(__name__)


class QueryDecomposer:

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_provider: BaseEmbeddingProvider,
        language: str = "zh",
    ):
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.language = language

    async def decompose(self, raw_query: str, history: str = "") -> list[SubQuery]:
        # Callers may hand over the whole prompt they intend to send to the
        # answer model; only the question itself is a retrieval signal.
        # Sanitising here (rather than trusting the caller) keeps every
        # downstream embedding free of template scaffolding.
        raw_query = sanitize_query(raw_query)
        if not raw_query:
            return []

        prompt = build_query_decompose_prompt(raw_query, history, self.language)
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.3)
        parsed = self._parse_sub_queries(response.content)
        # The model can echo scaffolding back into a sub-query; the same
        # sanitiser applies to whatever it returns.
        parsed = [
            {**item, "query": sanitize_query(item["query"])}
            for item in parsed
            if item.get("query")
        ]
        parsed = [item for item in parsed if item["query"]]

        if not parsed:
            parsed = [{"query": raw_query, "keywords": ""}]

        # Always include the raw query itself as a safety-net sub-query:
        # guarantees retrieval quality even if decomposition distorts intent
        # (costs one extra local embedding, zero LLM calls).
        norm_raw = raw_query.strip().lower()
        if not any(item["query"].strip().lower() == norm_raw for item in parsed):
            parsed.append({"query": raw_query, "keywords": ""})

        embed_texts = []
        for item in parsed:
            combined = item["query"]
            if item.get("keywords"):
                combined += " " + item["keywords"]
            embed_texts.append(combined)

        embeddings = await self.embedding_provider.embed_batch(embed_texts)
        sub_queries = []
        for i, (item, emb) in enumerate(zip(parsed, embeddings)):
            sub_queries.append(SubQuery(
                text=item["query"],
                keywords=item.get("keywords", ""),
                embedding=emb,
                index=i,
            ))

        logger.info(f"QueryDecomposer: {len(sub_queries)} sub-queries")
        return sub_queries

    def _parse_sub_queries(self, content: str) -> list[dict]:
        content = content.strip()

        # Strip <think>...</think> reasoning block if present
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        parsed = self._try_json_parse(content)
        if parsed:
            return parsed

        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if match:
            parsed = self._try_json_parse(match.group(1))
            if parsed:
                return parsed

        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            parsed = self._try_json_parse(match.group(0))
            if parsed:
                return parsed

        lines = [line.strip().strip('-•*').strip() for line in content.split('\n')]
        return [{"query": line, "keywords": ""} for line in lines if line and len(line) > 10][:4]

    def _try_json_parse(self, text: str) -> list[dict] | None:
        try:
            result = json.loads(text)
            if isinstance(result, list) and result:
                output = []
                for item in result:
                    if isinstance(item, str) and item.strip():
                        output.append({"query": item.strip(), "keywords": ""})
                    elif isinstance(item, dict) and "query" in item:
                        output.append({
                            "query": str(item["query"]).strip(),
                            "keywords": str(item.get("keywords", "")).strip(),
                        })
                return output if output else None
        except (json.JSONDecodeError, TypeError):
            pass
        return None
