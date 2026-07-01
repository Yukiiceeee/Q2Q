from __future__ import annotations
import json
import re
import logging

from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.fake_query_gen import build_fake_query_gen_prompt

logger = logging.getLogger(__name__)


class FakeQueryGenerator:

    def __init__(self, llm_client: BaseLLMClient, num_queries: int = 5, language: str = "zh"):
        self.llm_client = llm_client
        self.num_queries = num_queries
        self.language = language

    def generate(self, session_text: str) -> list[str]:
        prompt = build_fake_query_gen_prompt(session_text, self.num_queries, self.language)
        messages = format_messages(user_message=prompt)
        response = self.llm_client.chat(messages, temperature=0.7)
        queries = self._parse_queries(response.content)
        logger.info(f"FakeQueryGenerator produced {len(queries)} queries")
        for i, q in enumerate(queries):
            logger.debug(f"  FakeQuery[{i}]: {q}")
        return queries

    def _parse_queries(self, content: str) -> list[str]:
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
        queries = [line for line in lines if line and len(line) > 10]
        logger.warning(f"JSON parse failed, extracted {len(queries)} queries from plain text")
        return queries[:self.num_queries]
