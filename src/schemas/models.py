from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid
import numpy as np


@dataclass
class FakeQuery:
    text: str
    answer: str = ""
    embedding: Optional[np.ndarray] = None
    query_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    memory_id: str = ""
    supersedes: str = ""
    version_seq: int = 0
    chain_id: str = ""


@dataclass
class MemoryEntry:
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_text: str = ""
    content_embeddings: list[np.ndarray] = field(default_factory=list)
    fake_queries: list[FakeQuery] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "session_text": self.session_text,
            "content_embeddings": [e.tolist() for e in self.content_embeddings],
            "fake_queries": [
                {
                    "query_id": fq.query_id,
                    "text": fq.text,
                    "answer": fq.answer,
                    "memory_id": fq.memory_id,
                    "embedding": fq.embedding.tolist() if fq.embedding is not None else None,
                    "supersedes": fq.supersedes,
                    "version_seq": fq.version_seq,
                    "chain_id": fq.chain_id,
                }
                for fq in self.fake_queries
            ],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryEntry:
        entry = cls(
            memory_id=data["memory_id"],
            session_text=data["session_text"],
            content_embeddings=[np.array(e) for e in data.get("content_embeddings", [])],
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
        )
        for fq_data in data.get("fake_queries", []):
            emb = np.array(fq_data["embedding"]) if fq_data.get("embedding") is not None else None
            fq = FakeQuery(
                text=fq_data["text"],
                answer=fq_data.get("answer", ""),
                embedding=emb,
                query_id=fq_data.get("query_id", uuid.uuid4().hex[:12]),
                memory_id=fq_data.get("memory_id", entry.memory_id),
                supersedes=fq_data.get("supersedes", ""),
                version_seq=fq_data.get("version_seq", 0),
                chain_id=fq_data.get("chain_id", ""),
            )
            entry.fake_queries.append(fq)
        return entry


@dataclass
class VersionChainNode:
    query_id: str
    text: str
    answer: str
    memory_id: str
    version_seq: int
    created_at: str = ""


@dataclass
class SubQuery:
    text: str
    keywords: str = ""
    embedding: Optional[np.ndarray] = None
    index: int = 0


@dataclass
class RetrievalResult:
    memory: MemoryEntry
    score_q2q: float = 0.0
    score_q2c: float = 0.0
    final_score: float = 0.0
    matched_fake_queries: list[str] = field(default_factory=list)
    matched_sub_queries: list[str] = field(default_factory=list)
    version_chain_context: list[list[VersionChainNode]] = field(default_factory=list)


@dataclass
class RetrievalResponse:
    raw_query: str
    sub_queries: list[SubQuery] = field(default_factory=list)
    results: list[RetrievalResult] = field(default_factory=list)
