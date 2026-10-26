"""LoCoMo loading and chunking, mirrored from the evaluation harness.

The harness (``MedMemoryBench/benchmarks/locomo/``) cannot be imported from
here: it ships its own ``src`` package, which collides with Q2Q's. So the two
pieces of its behaviour that decide what Q2Q actually sees -- the memory-text
rendering and the session-to-chunk packing -- are reproduced here verbatim.

Mirrored, not reimplemented. Any divergence makes probe measurements
unrepresentative of the real run, so the source of each rule is named:

* ``session_memory_text``  <- ``LoCoMoSession.to_memory_text``
* ``split_into_chunks``    <- ``LoCoMoEvaluator._split_sessions_into_chunks``
* ``parse_queries``        <- ``LoCoMoDataset._parse_queries``

Nothing here scores anything, and nothing here is on the evaluation path --
this module only reconstructs the *inputs* so retrieval can be measured
offline against a frozen store.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.paths import dataset_path

# Category id -> name, from LOCOMO_CATEGORY_MAPPING.
CATEGORY_NAMES: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

# Path relative to the dataset root (<project_root>/data, or DATA_DIR). LoCoMo
# is not redistributed with this repository; see README for how to fetch it.
DEFAULT_DATA_FILE = str(dataset_path("locomo/data/locomo10.json"))


@dataclass
class Turn:
    """One dialogue turn, keyed by the ``dia_id`` that gold evidence cites."""

    dia_id: str
    speaker: str
    text: str
    blip_caption: str = ""

    def render(self) -> str:
        """Render exactly as ``to_memory_text`` does for a single turn."""
        line = f'{self.speaker} said, "{self.text}"'
        if self.blip_caption:
            line += f" and shared {self.blip_caption}"
        return line


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: list[Turn] = field(default_factory=list)

    def memory_text(self) -> str:
        """Mirror of ``LoCoMoSession.to_memory_text``."""
        lines = [f"DATE: {self.date_time}", "CONVERSATION:"]
        lines.extend(turn.render() for turn in self.turns)
        return "\n".join(lines)


@dataclass
class Query:
    query_id: str
    question: str
    category: int
    answer: str
    evidence: list[str] = field(default_factory=list)

    @property
    def category_name(self) -> str:
        return CATEGORY_NAMES.get(self.category, "unknown")

    @property
    def is_adversarial(self) -> bool:
        return self.category == 5


@dataclass
class Chunk:
    """One memorize() call's worth of text: the harness's unit of injection."""

    chunk_index: int
    session_ids: list[int]
    text: str
    # Filled in after memorize(); the store's id for this chunk.
    q2q_session_id: str = ""


@dataclass
class Sample:
    sample_id: str
    sessions: list[Session]
    queries: list[Query]

    def turn_index(self) -> dict[str, Turn]:
        """dia_id -> Turn, across every session in the sample."""
        return {
            turn.dia_id: turn
            for session in self.sessions
            for turn in session.turns
            if turn.dia_id
        }

    def session_of_turn(self) -> dict[str, int]:
        """dia_id -> session_id, so evidence can be traced to a chunk."""
        return {
            turn.dia_id: session.session_id
            for session in self.sessions
            for turn in session.turns
            if turn.dia_id
        }


def _parse_sessions(conversation: dict[str, Any], include_images: bool) -> list[Session]:
    session_keys = sorted(
        (
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ),
        key=lambda key: int(key.split("_")[1]),
    )

    sessions: list[Session] = []
    for key in session_keys:
        number = int(key.split("_")[1])
        turns = [
            Turn(
                dia_id=raw.get("dia_id", ""),
                speaker=raw.get("speaker", "Unknown"),
                text=raw.get("text", ""),
                blip_caption=raw.get("blip_caption", "") if include_images else "",
            )
            for raw in conversation.get(key, [])
        ]
        sessions.append(
            Session(
                session_id=number,
                date_time=conversation.get(f"session_{number}_date_time", ""),
                turns=turns,
            )
        )
    return sessions


def _parse_queries(qa_list: list[dict[str, Any]], sample_id: str) -> list[Query]:
    """Mirror of ``LoCoMoDataset._parse_queries``.

    The index is the position in the *unfiltered* qa list, because that is
    what the harness uses to build ``query_id``; filtering happens later.
    """
    queries: list[Query] = []
    for index, qa in enumerate(qa_list):
        evidence = qa.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        queries.append(
            Query(
                query_id=f"{sample_id}_q{index}",
                question=qa.get("question", ""),
                category=qa.get("category", 1),
                answer=str(qa.get("answer", "")),
                evidence=list(evidence),
            )
        )
    return queries


def load_samples(
    data_file: str | Path = DEFAULT_DATA_FILE,
    max_samples: int | None = None,
    sample_ids: list[str] | None = None,
    include_images: bool = True,
) -> list[Sample]:
    """Load LoCoMo samples under the harness's selection rules.

    ``max_samples`` and ``sample_ids`` mirror the dataset config keys of the
    same name, so ``locomo_small.yaml`` (max_samples: 2) is reproducible.
    """
    with open(data_file, "r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    samples: list[Sample] = []
    for index, raw in enumerate(raw_data):
        sample_id = raw.get("sample_id", f"sample_{index}")
        if sample_ids and sample_id not in sample_ids:
            continue
        if max_samples and len(samples) >= max_samples:
            break
        samples.append(
            Sample(
                sample_id=sample_id,
                sessions=_parse_sessions(raw.get("conversation", {}), include_images),
                queries=_parse_queries(raw.get("qa", []), sample_id),
            )
        )
    return samples


def split_into_chunks(sessions: list[Session], memory_chunk_size: int) -> list[Chunk]:
    """Mirror of ``LoCoMoEvaluator._split_sessions_into_chunks``.

    A session is never split: it is packed whole, and a session that alone
    exceeds the limit still forms its own chunk. The limit counts characters
    of rendered memory text, excluding the ``"\\n\\n"`` joiners -- faithfully
    reproducing the harness's own off-by-a-joiner accounting.
    """
    chunks: list[Chunk] = []
    current_ids: list[int] = []
    current_texts: list[str] = []
    current_size = 0

    def flush() -> None:
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                session_ids=current_ids.copy(),
                text="\n\n".join(current_texts),
            )
        )

    for session in sessions:
        text = session.memory_text()
        if current_size + len(text) > memory_chunk_size and current_ids:
            flush()
            current_ids = []
            current_texts = []
            current_size = 0
        current_ids.append(session.session_id)
        current_texts.append(text)
        current_size += len(text)

    if current_ids:
        flush()
    return chunks


def iter_evaluated_queries(
    sample: Sample, category_filter: list[int] | None = None
) -> Iterator[Query]:
    """Queries the harness would score, in harness order."""
    for query in sample.queries:
        if category_filter and query.category not in category_filter:
            continue
        yield query
