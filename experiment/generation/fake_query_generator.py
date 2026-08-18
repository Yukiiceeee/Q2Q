"""Fake Query generator for experiment preprocessing.

Directly reuses the main Q2Q project's FakeQueryGenerator to ensure
consistency between experiment analysis and production behavior.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

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
        from src.memory.query_generator import FakeQueryGenerator

        self._generator = FakeQueryGenerator(
            llm_client=llm_client,
            num_queries=num_queries,
            language=language,
        )
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
                queries = await self._generator._generate_queries(text)
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

    def _save_results(self, results: dict, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
