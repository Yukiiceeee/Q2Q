from __future__ import annotations
import json
import re
import logging

from src.utils.llm_client import BaseLLMClient, format_messages
from src.prompts.fake_query_gen import build_fake_query_gen_prompt
from src.prompts.fake_answer_gen import (
    build_fake_answer_gen_prompt,
    build_fake_answer_gen_with_history_prompt,
)

logger = logging.getLogger(__name__)


class FakeQueryGenerator:

    def __init__(self, llm_client: BaseLLMClient, num_queries: int = 10, language: str = "zh"):
        self.llm_client = llm_client
        self.num_queries = num_queries
        self.language = language

    def generate(self, session_text: str) -> list[dict]:
        """Two-stage pipeline: generate fake queries, then generate answers.
        Returns list of {"query": str, "answer": str}.
        """
        # Stage 1: generate fake queries
        query_texts = self._generate_queries(session_text)
        logger.info(f"FakeQueryGenerator Stage 1: {len(query_texts)} queries generated")

        if not query_texts:
            return []

        # Stage 2: generate answer sequences
        answers = self._generate_answers(session_text, query_texts)
        logger.info(f"FakeQueryGenerator Stage 2: {len(answers)} answers generated")

        results = []
        for i, query in enumerate(query_texts):
            answer = answers[i] if i < len(answers) else ""
            results.append({"query": query, "answer": answer})
            logger.debug(f"  FakeQuery[{i}]: Q={query} | A={answer[:60]}")

        return results

    def _generate_queries(self, session_text: str) -> list[str]:
        prompt = build_fake_query_gen_prompt(session_text, self.num_queries, self.language)
        messages = format_messages(user_message=prompt)
        response = self.llm_client.chat(messages, temperature=0.7)
        return self._parse_string_list(response.content)

    def _generate_answers(self, session_text: str, queries: list[str]) -> list[str]:
        prompt = build_fake_answer_gen_prompt(session_text, queries, self.language)
        messages = format_messages(user_message=prompt)
        response = self.llm_client.chat(messages, temperature=0.3)
        answers = self._parse_string_list(response.content)

        while len(answers) < len(queries):
            answers.append("")
        return answers[:len(queries)]

    def _generate_answers_with_history(
        self,
        session_text: str,
        queries: list[str],
        version_histories: list[dict],
    ) -> list[str]:
        """Generate answers with version-aware context.

        Splits queries into those with history and those without,
        uses different prompts accordingly.
        """
        has_history_indices = []
        no_history_indices = []

        for i, vh in enumerate(version_histories):
            if vh.get("related_history"):
                has_history_indices.append(i)
            else:
                no_history_indices.append(i)

        answers = [""] * len(queries)

        # Queries without history: standard path
        if no_history_indices:
            no_hist_queries = [queries[i] for i in no_history_indices]
            no_hist_answers = self._generate_answers(session_text, no_hist_queries)
            for idx, ans in zip(no_history_indices, no_hist_answers):
                answers[idx] = ans

        # Queries with history: version-aware prompt
        if has_history_indices:
            hist_queries = [queries[i] for i in has_history_indices]
            hist_contexts = [version_histories[i] for i in has_history_indices]

            prompt = build_fake_answer_gen_with_history_prompt(
                session_text, hist_queries, hist_contexts, self.language
            )
            messages = format_messages(user_message=prompt)
            response = self.llm_client.chat(messages, temperature=0.3)
            hist_answers = self._parse_string_list(response.content)

            while len(hist_answers) < len(hist_queries):
                hist_answers.append("")

            for idx, ans in zip(has_history_indices, hist_answers[:len(has_history_indices)]):
                answers[idx] = ans

        logger.info(
            f"FakeQueryGenerator answers: "
            f"{len(no_history_indices)} standard, {len(has_history_indices)} with history"
        )
        return answers

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
