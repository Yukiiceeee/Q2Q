from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid
import numpy as np


class PropositionSource(Enum):
    RULE = "rule"
    MODEL = "model"
    LLM = "llm"


@dataclass
class Proposition:
    prop_id: str = field(default_factory=lambda: f"P_{uuid.uuid4().hex[:8]}")
    text: str = ""
    source: str = "llm"
    session_id: str = ""
    # structured fields (optional, populated by rule-based extraction)
    time: str = ""
    subject: str = ""
    predicate: str = ""
    object_value: str = ""
    entities: list[str] = field(default_factory=list)
    # embedding
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "prop_id": self.prop_id,
            "text": self.text,
            "source": self.source,
            "session_id": self.session_id,
            "time": self.time,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "entities": self.entities,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Proposition:
        emb = np.array(data["embedding"]) if data.get("embedding") is not None else None
        return cls(
            prop_id=data.get("prop_id", f"P_{uuid.uuid4().hex[:8]}"),
            text=data.get("text", ""),
            source=data.get("source", "llm"),
            session_id=data.get("session_id", ""),
            time=data.get("time", ""),
            subject=data.get("subject", ""),
            predicate=data.get("predicate", ""),
            object_value=data.get("object_value", ""),
            entities=data.get("entities", []),
            embedding=emb,
        )


@dataclass
class FakeQuery:
    query_id: str = field(default_factory=lambda: f"fq_{uuid.uuid4().hex[:10]}")
    text: str = ""
    answer_source: list[str] = field(default_factory=list)  # linked proposition_ids
    embedding: Optional[np.ndarray] = None
    session_id: str = ""
    time: str = ""
    # DAG node extended attributes
    last_accessed: str = ""
    access_count: int = 0
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    depth: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "answer_source": self.answer_source,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "session_id": self.session_id,
            "time": self.time,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "depth": self.depth,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FakeQuery:
        emb = np.array(data["embedding"]) if data.get("embedding") is not None else None
        return cls(
            query_id=data.get("query_id", f"fq_{uuid.uuid4().hex[:10]}"),
            text=data.get("text", ""),
            answer_source=data.get("answer_source", []),
            embedding=emb,
            session_id=data.get("session_id", ""),
            time=data.get("time", ""),
            last_accessed=data.get("last_accessed", ""),
            access_count=data.get("access_count", 0),
            parent_ids=data.get("parent_ids", []),
            child_ids=data.get("child_ids", []),
            depth=data.get("depth", 0),
            created_at=data.get("created_at", ""),
        )


class EdgeType(Enum):
    EVOLVES_TO = "EVOLVES_TO"
    DIVERGES_TO = "DIVERGES_TO"


@dataclass
class Edge:
    edge_id: str = field(default_factory=lambda: f"edge_{uuid.uuid4().hex[:10]}")
    src_id: str = ""
    dst_id: str = ""
    edge_type: str = "EVOLVES_TO"
    similarity: float = 0.0
    prop_overlap: float = 0.0

    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "edge_type": self.edge_type,
            "similarity": self.similarity,
            "prop_overlap": self.prop_overlap,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Edge:
        return cls(
            edge_id=data.get("edge_id", f"edge_{uuid.uuid4().hex[:10]}"),
            src_id=data.get("src_id", ""),
            dst_id=data.get("dst_id", ""),
            edge_type=data.get("edge_type", "EVOLVES_TO"),
            similarity=data.get("similarity", 0.0),
            prop_overlap=data.get("prop_overlap", 0.0),
        )


@dataclass
class MemoryEntry:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_text: str = ""
    pre_session_text: str = ""
    propositions: list[Proposition] = field(default_factory=list)
    fake_queries: list[FakeQuery] = field(default_factory=list)
    content_embeddings: list[np.ndarray] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_text": self.session_text,
            "pre_session_text": self.pre_session_text,
            "propositions": [p.to_dict() for p in self.propositions],
            "fake_queries": [fq.to_dict() for fq in self.fake_queries],
            "content_embeddings": [e.tolist() for e in self.content_embeddings],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryEntry:
        entry = cls(
            session_id=data.get("session_id", uuid.uuid4().hex[:16]),
            session_text=data.get("session_text", ""),
            pre_session_text=data.get("pre_session_text", ""),
            content_embeddings=[np.array(e) for e in data.get("content_embeddings", [])],
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
        )
        for p_data in data.get("propositions", []):
            entry.propositions.append(Proposition.from_dict(p_data))
        for fq_data in data.get("fake_queries", []):
            entry.fake_queries.append(FakeQuery.from_dict(fq_data))
        return entry


@dataclass
class SubQuery:
    text: str = ""
    keywords: str = ""
    embedding: Optional[np.ndarray] = None
    index: int = 0


@dataclass
class RetrievalResult:
    matched_fqs: list[FakeQuery] = field(default_factory=list)
    ranked_propositions: list[Proposition] = field(default_factory=list)
    session_texts: list[str] = field(default_factory=list)
    q2q_scores: dict = field(default_factory=dict)
    q2c_scores: dict = field(default_factory=dict)
    chain_fqs: list[FakeQuery] = field(default_factory=list)
    # Stage-aware context: direct FQ → session texts, chain FQ → propositions
    direct_session_texts: list[str] = field(default_factory=list)
    chain_propositions: list[Proposition] = field(default_factory=list)
    # Q2C fallback session texts (used only in Stage 2)
    q2c_session_texts: list[str] = field(default_factory=list)


@dataclass
class RetrievalResponse:
    raw_query: str = ""
    sub_queries: list[SubQuery] = field(default_factory=list)
    result: Optional[RetrievalResult] = None
    answer: str = ""
