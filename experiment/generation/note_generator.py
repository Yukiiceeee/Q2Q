"""Memory note variant generator for experiment preprocessing.

Generates three types of memory notes from session conversations:
- Proposition: Atomic fact extraction (Mem0 / HippoRAG style)
- Note: Structured Zettelkasten notes (A-Mem style)
- Reflection: High-level insights (Generative Agents style)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from experiment.generation.prompts.note_prompt import (
    build_variant_prompt,
    NOTE_VARIANT_STYLES,
)

logger = logging.getLogger(__name__)


class NoteGenerator:

    def __init__(
        self,
        llm_client,
        language: str = "en",
        output_dir: str = "experiment/generation/outputs",
        max_concurrent: int = 5,
        save_interval: int = 10,
    ):
        self.llm_client = llm_client
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.save_interval = save_interval

    async def generate_for_dataset(
        self,
        sessions: list[dict],
        dataset_name: str,
        style: str,
        resume: bool = True,
    ) -> dict[str, list]:
        """Generate memory notes for all sessions in a given style.

        Args:
            sessions: List of dicts with 'session_id' and 'text'.
            dataset_name: Name for output file.
            style: One of 'proposition', 'note', 'reflection'.
            resume: If True, skip sessions already generated.

        Returns:
            Dict mapping session_id -> list of generated items.
        """
        if style not in NOTE_VARIANT_STYLES:
            raise ValueError(
                f"Unknown style '{style}', must be one of {NOTE_VARIANT_STYLES}"
            )

        output_path = self.output_dir / f"{dataset_name}_{style}s.json"
        results: dict[str, list] = {}
        if resume and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Resumed {len(results)} sessions from {output_path}")

        pending = [s for s in sessions if s["session_id"] not in results]
        if not pending:
            logger.info(
                f"All sessions already have {style}s, nothing to do."
            )
            return results

        logger.info(
            f"Generating {style}s for {len(pending)} sessions "
            f"(already done: {len(results)}, total: {len(sessions)})"
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0

        async def _generate_one(session: dict) -> tuple[str, list]:
            nonlocal completed
            async with semaphore:
                sid = session["session_id"]
                text = session["text"]
                items = await self._generate(text, style)
                completed += 1
                if completed % self.save_interval == 0:
                    self._save_results(results, output_path)
                    logger.info(
                        f"  Progress: {completed}/{len(pending)} (saved)"
                    )
                return sid, items

        tasks = [_generate_one(s) for s in pending]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"{style} generation failed: {result}")
                continue
            sid, items = result
            results[sid] = items

        self._save_results(results, output_path)
        logger.info(
            f"Saved {len(results)} session {style}s to {output_path}"
        )
        return results

    async def _generate(self, session_text: str, style: str) -> list:
        prompt = build_variant_prompt(session_text, style, self.language)
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.chat(messages, temperature=0.7)
        return self._parse_response(response.content, style)

    def _parse_response(self, content: str, style: str) -> list:
        content = content.strip()

        parsed = self._try_json_array(content)
        if parsed is not None:
            return self._validate_items(parsed, style)

        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if match:
            parsed = self._try_json_array(match.group(1))
            if parsed is not None:
                return self._validate_items(parsed, style)

        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            parsed = self._try_json_array(match.group(0))
            if parsed is not None:
                return self._validate_items(parsed, style)

        logger.warning(f"Failed to parse {style} response, returning empty list")
        return []

    def _validate_items(self, items: list, style: str) -> list:
        if style == "note":
            validated = []
            for item in items:
                if isinstance(item, dict) and "title" in item and "content" in item:
                    tags = item.get("tags", [])
                    if not isinstance(tags, list):
                        tags = [str(tags)] if tags else []
                    validated.append({
                        "title": str(item.get("title") or ""),
                        "key_insight": str(item.get("key_insight") or ""),
                        "content": str(item.get("content") or ""),
                        "tags": tags,
                    })
            return validated
        return [str(item).strip() for item in items if str(item).strip()]

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
