from __future__ import annotations
import json
import re
import logging

from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.fake_query_gen import build_fake_query_gen_prompt
from src.prompts.knowledge_extraction import (
    build_knowledge_extraction_prompt,
    build_knowledge_extraction_with_context_prompt,
)

logger = logging.getLogger(__name__)


class FakeQueryGenerator:

    def __init__(self, llm_client: BaseLLMClient, num_queries: int = 10, language: str = "zh"):
        self.llm_client = llm_client
        self.num_queries = num_queries
        self.language = language

    async def _generate_queries(self, session_text: str) -> list[str]:
        prompt = build_fake_query_gen_prompt(session_text, self.num_queries, self.language)
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.7)
        return self._parse_string_list(response.content)

    async def extract_knowledge_points(self, session_text: str) -> list[dict]:
        prompt = build_knowledge_extraction_prompt(session_text, self.language)
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.3)
        return self._parse_knowledge_points(response.content)

    async def extract_knowledge_points_with_context(
        self, session_text: str, linked_kps: list[dict]
    ) -> list[dict]:
        prompt = build_knowledge_extraction_with_context_prompt(
            session_text, linked_kps, self.language
        )
        messages = format_messages(user_message=prompt)
        response = await self.llm_client.chat(messages, temperature=0.3)
        return self._parse_knowledge_points(response.content)

    def _parse_knowledge_points(self, content: str) -> list[dict]:
        content = content.strip()
        required_keys = {"time", "subject", "fact", "entities_or_values"}

        parsed = self._try_parse_json_array(content)
        if parsed is not None:
            return self._validate_kp_list(parsed, required_keys)

        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if match:
            parsed = self._try_parse_json_array(match.group(1))
            if parsed is not None:
                return self._validate_kp_list(parsed, required_keys)

        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            parsed = self._try_parse_json_array(match.group(0))
            if parsed is not None:
                return self._validate_kp_list(parsed, required_keys)

        logger.warning("Failed to parse knowledge points from LLM response")
        return []

    def _try_parse_json_array(self, text: str) -> list | None:
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        return None

    def _validate_kp_list(self, items: list, required_keys: set) -> list[dict]:
        valid = []
        for item in items:
            if isinstance(item, dict) and required_keys.issubset(item.keys()):
                valid.append({
                    "time": str(item.get("time", "")),
                    "subject": str(item.get("subject", "")),
                    "fact": str(item.get("fact", "")),
                    "entities_or_values": str(item.get("entities_or_values", "")),
                })
        if len(valid) < len(items):
            logger.warning(f"Knowledge points validation: {len(valid)}/{len(items)} valid")
        return valid

    def _parse_string_list(self, content: str) -> list[str]:
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
        items = [line for line in lines if line and len(line) > 10]
        if items:
            logger.warning(f"JSON parse failed, extracted {len(items)} items from plain text")
        return items
