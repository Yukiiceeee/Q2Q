"""Dataset loaders for LoCoMo and LongMemEval.

Unified interface providing:
- sessions: list of {session_id, text, date, metadata}
- queries: list of {query_id, text, answer, evidence_session_ids, category}
- get_answer_session(query_id) -> session_id mapping
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    session_id: str
    text: str
    date: str = ""
    turns: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryData:
    query_id: str
    text: str
    answer: str = ""
    evidence_session_ids: list[str] = field(default_factory=list)
    category: str = ""
    metadata: dict = field(default_factory=dict)


class BaseDataLoader(ABC):

    def __init__(self, config: dict):
        self.config = config
        self.sessions: list[SessionData] = []
        self.queries: list[QueryData] = []
        self._query_to_sessions: dict[str, list[str]] = {}

    @abstractmethod
    def load(self) -> None:
        ...

    def get_answer_sessions(self, query_id: str) -> list[str]:
        return self._query_to_sessions.get(query_id, [])

    def get_session_by_id(self, session_id: str) -> SessionData | None:
        for s in self.sessions:
            if s.session_id == session_id:
                return s
        return None

    def get_sessions_as_dicts(self) -> list[dict]:
        return [{"session_id": s.session_id, "text": s.text} for s in self.sessions]

    def get_queries_as_dicts(self) -> list[dict]:
        return [{"query_id": q.query_id, "text": q.text} for q in self.queries]

    def summary(self) -> str:
        return (
            f"Dataset: {self.config.get('name', 'unknown')}\n"
            f"  Sessions: {len(self.sessions)}\n"
            f"  Queries: {len(self.queries)}\n"
            f"  Query-Session mappings: {len(self._query_to_sessions)}"
        )


class LoCoMoLoader(BaseDataLoader):

    def load(self) -> None:
        path = Path(self.config["path"])
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        max_samples = self.config.get("max_samples", -1)
        categories = set(self.config.get("categories", []))
        min_turns = self.config.get("min_session_turns", 3)
        evidence_pattern = re.compile(
            self.config.get("evidence_pattern", r"D(\d+):(\d+)")
        )

        for sample_idx, sample in enumerate(raw_data):
            if 0 < max_samples <= sample_idx:
                break

            sample_id = sample.get("sample_id", f"sample_{sample_idx}")
            conv = sample.get("conversation", {})

            # Extract sessions
            session_map = {}
            session_idx = 1
            while True:
                key = f"session_{session_idx}"
                date_key = f"session_{session_idx}_date_time"
                if key not in conv:
                    break
                turns = conv[key]
                date = conv.get(date_key, "")
                if isinstance(turns, list) and len(turns) >= min_turns:
                    sid = f"{sample_id}_s{session_idx}"
                    text = self._turns_to_text(turns)
                    session_map[session_idx] = sid
                    self.sessions.append(SessionData(
                        session_id=sid,
                        text=text,
                        date=date,
                        turns=turns,
                        metadata={"sample_id": sample_id, "session_num": session_idx},
                    ))
                session_idx += 1

            # Extract QA pairs
            for qa_idx, qa in enumerate(sample.get("qa", [])):
                cat = qa.get("category", 0)
                if categories and cat not in categories:
                    continue

                qid = f"{sample_id}_q{qa_idx}"
                evidence = qa.get("evidence", [])
                evidence_sids = set()
                for ev in evidence:
                    m = evidence_pattern.match(str(ev))
                    if m:
                        sess_num = int(m.group(1))
                        if sess_num in session_map:
                            evidence_sids.add(session_map[sess_num])

                if not evidence_sids:
                    continue

                ev_list = sorted(evidence_sids)
                self.queries.append(QueryData(
                    query_id=qid,
                    text=qa["question"],
                    answer=str(qa.get("answer", "")),
                    evidence_session_ids=ev_list,
                    category=str(cat),
                    metadata={"sample_id": sample_id, "raw_evidence": evidence},
                ))
                self._query_to_sessions[qid] = ev_list

        logger.info(f"LoCoMo loaded: {self.summary()}")

    def _turns_to_text(self, turns: list[dict]) -> str:
        lines = []
        for turn in turns:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if text:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines)


class LongMemEvalLoader(BaseDataLoader):

    def load(self) -> None:
        path = Path(self.config["path"])
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        max_samples = self.config.get("max_samples", -1)
        question_types = set(self.config.get("question_types", []))
        min_turns = self.config.get("min_session_turns", 2)

        seen_sessions: dict[str, str] = {}

        for entry_idx, entry in enumerate(raw_data):
            if 0 < max_samples <= entry_idx:
                break

            qtype = entry.get("question_type", "")
            if question_types and qtype not in question_types:
                continue

            qid = entry.get("question_id", f"q_{entry_idx}")
            answer_session_ids = entry.get("answer_session_ids", [])
            haystack_ids = entry.get("haystack_session_ids", [])
            haystack_sessions = entry.get("haystack_sessions", [])
            haystack_dates = entry.get("haystack_dates", [])

            # Register sessions (deduplicate across entries)
            for i, (sid, sess_turns) in enumerate(zip(haystack_ids, haystack_sessions)):
                if sid in seen_sessions:
                    continue
                if not isinstance(sess_turns, list) or len(sess_turns) < min_turns:
                    continue
                text = self._session_to_text(sess_turns)
                date = haystack_dates[i] if i < len(haystack_dates) else ""
                seen_sessions[sid] = sid
                self.sessions.append(SessionData(
                    session_id=sid,
                    text=text,
                    date=date,
                    turns=sess_turns,
                    metadata={"question_type": qtype},
                ))

            # Map answer sessions
            evidence_sids = [
                sid for sid in answer_session_ids
                if sid in haystack_ids
            ]
            if not evidence_sids:
                continue

            self.queries.append(QueryData(
                query_id=qid,
                text=entry["question"],
                answer=entry.get("answer", ""),
                evidence_session_ids=evidence_sids,
                category=qtype,
                metadata={"question_date": entry.get("question_date", "")},
            ))
            self._query_to_sessions[qid] = evidence_sids

        logger.info(f"LongMemEval loaded: {self.summary()}")

    def _session_to_text(self, turns: list[dict]) -> str:
        lines = []
        for turn in turns:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)


def load_dataset(config: dict) -> BaseDataLoader:
    name = config.get("name", "").lower()
    if name == "locomo":
        loader = LoCoMoLoader(config)
    elif name == "longmemeval":
        loader = LongMemEvalLoader(config)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    loader.load()
    return loader
