"""Anchor-query generator for experiment preprocessing.

Generates anchor-query candidates (``fake queries`` in the code) from session
text alone, using the experiment-local prompts in
``prompts/fake_query_prompt.py``.

This deliberately does not call the construction path in
``src.memory.query_generator``. That one conditions generation on extracted
propositions and verifies every candidate against an embedding provider, so it
returns ``FakeQuery`` objects carrying bindings. The motivation analysis needs
the opposite: generation blind to the true query, from raw session text, and a
plain list of query strings that ``experiment/run.py --step embed`` can hand
straight to ``embed_batch``. Mixing the two would leak proposition-level
supervision into a measurement that is supposed to be independent of it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from experiment.generation.prompts.fake_query_prompt import (
    build_experiment_fq_prompt,
)

logger = logging.getLogger(__name__)


class ExperimentFQGenerator:

    def __init__(
        self,
        llm_client,
        num_queries: int = 10,
        language: str = "en",
        output_dir: str = "experiment/generation/outputs",
        max_concurrent: int = 5,
        save_interval: int = 10,
    ):
        self.llm_client = llm_client
        self.num_queries = num_queries
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.save_interval = save_interval

    async def generate_for_dataset(
        self,
        sessions: list[dict],
        dataset_name: str,
        resume: bool = True,
    ) -> dict[str, list[str]]:
        """Generate fake queries for all sessions in a dataset.

        Args:
            sessions: List of dicts with keys 'session_id' and 'text'.
            dataset_name: Name for output file.
            resume: If True, skip sessions already generated.

        Returns:
            Dict mapping session_id -> list of fake query strings.
        """
        output_path = self.output_dir / f"{dataset_name}_fake_queries.json"
        results = {}
        if resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Resumed {len(results)} sessions from {output_path}")

        pending = [s for s in sessions if s["session_id"] not in results]
        if not pending:
            logger.info("All sessions already generated, nothing to do.")
            return results

        logger.info(
            f"Generating FQs for {len(pending)} sessions "
            f"(already done: {len(results)}, total: {len(sessions)})"
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0

        async def _generate_one(session: dict) -> tuple[str, list[str]]:
            nonlocal completed
            async with semaphore:
                sid = session["session_id"]
                text = session["text"]
                queries = await self._generate(text)
                completed += 1
                if completed % self.save_interval == 0:
                    self._save_results(results, output_path)
                    logger.info(f"  Progress: {completed}/{len(pending)} (saved)")
                return sid, queries

        tasks = [_generate_one(s) for s in pending]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Generation failed: {result}")
                continue
            sid, queries = result
            results[sid] = queries

        self._save_results(results, output_path)
        logger.info(f"Saved {len(results)} sessions to {output_path}")
        return results

    async def _generate(self, session_text: str) -> list[str]:
        prompt = build_experiment_fq_prompt(
            session_text, self.num_queries, self.language
        )
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.chat(messages, temperature=0.7)
        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> list[str]:
        content = content.strip()

        parsed = self._try_json_array(content)
        if parsed is None:
            match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
            if match:
                parsed = self._try_json_array(match.group(1))
        if parsed is None:
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                parsed = self._try_json_array(match.group(0))
        if parsed is None:
            logger.warning("Failed to parse FQ response, returning empty list")
            return []

        return self._validate_queries(parsed)

    def _validate_queries(self, items: list) -> list[str]:
        """Keep non-empty strings, deduplicate, and cap at ``num_queries``.

        The model is asked for exactly ``num_queries`` strings but is not bound
        to comply. Trimming here keeps the anchor budget identical across
        sessions, which experiment 4 relies on when it sweeps that budget.
        """
        queries: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(text)
        if len(queries) > self.num_queries:
            queries = queries[: self.num_queries]
        return queries

    def _try_json_array(self, text: str) -> list | None:
        try:
            result = json.loads(text)
            if isinstance(result, list) and len(result) > 0:
                return result
        except json.JSONDecodeError:
            pass
        return None

    def _save_results(self, results: dict, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
