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
        query_texts = self._parse_sub_queries(response.content)

        if not query_texts:
            query_texts = [raw_query]

        embeddings = self.embedding_provider.embed_batch(query_texts)
        sub_queries = []
        for i, (text, emb) in enumerate(zip(query_texts, embeddings)):
            sub_queries.append(SubQuery(text=text, embedding=emb, index=i))

        logger.info(f"QueryDecomposer: {len(sub_queries)} sub-queries")
        for sq in sub_queries:
            logger.debug(f"  SubQuery[{sq.index}]: {sq.text}")
        return sub_queries

    def _parse_sub_queries(self, content: str) -> list[str]:
        content = content.strip()
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return [str(q).strip() for q in result if str(q).strip()]
        except json.JSONDecodeError:
            pass

        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return [str(q).strip() for q in result if str(q).strip()]
            except json.JSONDecodeError:
                pass

        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return [str(q).strip() for q in result if str(q).strip()]
            except json.JSONDecodeError:
                pass

        lines = [line.strip().strip('-•*').strip() for line in content.split('\n')]
        return [line for line in lines if line and len(line) > 10][:4]
