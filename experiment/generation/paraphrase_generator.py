"""Indirect query generator for experiment preprocessing.

Generates semantically indirect queries that probe the same memory
but use completely different surface-level wording and approach
the topic from oblique angles.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from experiment.generation.prompts.paraphrase_prompt import (
    build_indirect_query_prompt,
    INDIRECT_QUERY_STYLES,
)

logger = logging.getLogger(__name__)


class IndirectQueryGenerator:

    def __init__(
        self,
        llm_client,
        language: str = "en",
        output_dir: str = "experiment/generation/outputs",
        max_concurrent: int = 5,
        save_interval: int = 20,
    ):
        self.llm_client = llm_client
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.save_interval = save_interval

    async def generate_for_queries(
        self,
        queries: list[dict],
        dataset_name: str,
        resume: bool = True,
    ) -> dict[str, dict]:
        """Generate indirect queries for all true queries.

        Args:
            queries: List of dicts with 'query_id', 'text', 'answer', 'context'.
            dataset_name: Name for output file.
            resume: If True, skip queries already generated.

        Returns:
            Dict mapping query_id -> {
                "original": str,
                "answer": str,
                "indirect_queries": {"implication": str, "scenario": str, ...}
            }
        """
        output_path = self.output_dir / f"{dataset_name}_indirect_queries.json"
        results = {}
        if resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Resumed {len(results)} queries from {output_path}")

        pending = [q for q in queries if q["query_id"] not in results]
        if not pending:
            logger.info("All queries already have indirect queries, nothing to do.")
            return results

        logger.info(
            f"Generating indirect queries for {len(pending)} queries "
            f"(already done: {len(results)}, total: {len(queries)})"
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0

        async def _generate_one(query: dict) -> tuple[str, dict]:
            nonlocal completed
            async with semaphore:
                qid = query["query_id"]
                text = query["text"]
                answer = query.get("answer", "")
                context = query.get("context", "")
                indirect = await self._generate_indirect(text, answer, context)
                completed += 1
                if completed % self.save_interval == 0:
                    self._save_results(results, output_path)
                    logger.info(f"  Progress: {completed}/{len(pending)} (saved)")
                return qid, {
                    "original": text,
                    "answer": answer,
                    "indirect_queries": indirect,
                }

        tasks = [_generate_one(q) for q in pending]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Indirect query generation failed: {result}")
                continue
            qid, data = result
            results[qid] = data

        self._save_results(results, output_path)
        logger.info(f"Saved {len(results)} indirect queries to {output_path}")
        return results

    async def _generate_indirect(
        self, query_text: str, answer: str, context: str
    ) -> dict[str, str]:
        prompt = build_indirect_query_prompt(
            query_text, answer, context, self.language
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.chat(messages, temperature=0.8)
        return self._parse_indirect_queries(response.content)

    def _parse_indirect_queries(self, content: str) -> dict[str, str]:
        content = content.strip()
        items = self._try_json_array(content)
        if not items:
            match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
            if match:
                items = self._try_json_array(match.group(1))
        if not items:
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                items = self._try_json_array(match.group(0))

        if not items or len(items) < 5:
            lines = [l.strip().strip('-•*"').strip() for l in content.split('\n')]
            items = [l for l in lines if l and len(l) > 10][:5]

        result = {}
        for i, style in enumerate(INDIRECT_QUERY_STYLES):
            if i < len(items):
                result[style] = items[i]
            else:
                result[style] = ""
        return result

    def _try_json_array(self, text: str) -> list[str] | None:
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(item).strip() for item in result if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return None

    def _save_results(self, results: dict, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
