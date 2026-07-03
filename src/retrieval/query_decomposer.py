from __future__ import annotations
import json
import re
import logging

from src.schemas.models import SubQuery
from src.embedding.base import BaseEmbeddingProvider
from src.utils.llm_client import BaseLLMClient, format_messages
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

    def decompose(self, raw_query: str, history: str = "") -> list[SubQuery]:
        prompt = build_query_decompose_prompt(raw_query, history, self.language)
        messages = format_messages(user_message=prompt)
        response = self.llm_client.chat(messages, temperature=0.3)
        parsed = self._parse_sub_queries(response.content)

        if not parsed:
            parsed = [{"query": raw_query, "keywords": ""}]

        embed_texts = []
        for item in parsed:
            combined = item["query"]
            if item.get("keywords"):
                combined += " " + item["keywords"]
            embed_texts.append(combined)

        embeddings = self.embedding_provider.embed_batch(embed_texts)
        sub_queries = []
        for i, (item, emb) in enumerate(zip(parsed, embeddings)):
            sub_queries.append(SubQuery(
                text=item["query"],
                keywords=item.get("keywords", ""),
                embedding=emb,
                index=i,
            ))

        logger.info(f"QueryDecomposer: {len(sub_queries)} sub-queries")
        for sq in sub_queries:
            logger.debug(f"  SubQuery[{sq.index}]: {sq.text} | keywords: {sq.keywords}")
        return sub_queries

    def _parse_sub_queries(self, content: str) -> list[dict]:
        content = content.strip()

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
