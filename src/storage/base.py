from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.schemas.models import MemoryEntry, Edge


class BaseMemoryStore(ABC):

    @abstractmethod
    async def save(self, entry: MemoryEntry) -> None:
        ...

    @abstractmethod
    async def load_all(self) -> list[MemoryEntry]:
        ...

    @abstractmethod
    async def get_by_id(self, session_id: str) -> Optional[MemoryEntry]:
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...

    @abstractmethod
    async def clear(self) -> None:
        ...

    # --- Fake Query operations ---

    @abstractmethod
    async def search_fake_queries(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[dict]:
        ...

    @abstractmethod
    async def batch_search_fake_queries(
        self, query_embeddings: list[np.ndarray], top_k: int = 20
    ) -> list[list[dict]]:
        ...

    @abstractmethod
    async def get_fake_query_by_id(self, query_id: str) -> Optional[dict]:
        ...

    @abstractmethod
    async def get_fake_queries_by_ids(self, query_ids: list[str]) -> list[dict]:
        ...

    # --- Proposition operations ---

    @abstractmethod
    async def get_propositions_by_ids(self, prop_ids: list[str], session_id: str) -> list[dict]:
        ...

    async def get_proposition_embeddings(
        self, prop_ids: list[str], session_id: str
    ) -> dict[str, np.ndarray]:
        """Return {prop_id: embedding} for stored propositions. Optional capability."""
        return {}

    # --- Content (pre_session_text) operations ---

    @abstractmethod
    async def batch_search_content(
        self, query_embeddings: list[np.ndarray], top_k: int = 10
    ) -> list[dict]:
        ...

    # --- Edge (DAG) operations ---

    @abstractmethod
    async def save_edges(self, edges: list[Edge]) -> None:
        ...

    @abstractmethod
    async def get_all_edges(self) -> list[Edge]:
        ...

    @abstractmethod
    async def delete_edges_involving(self, node_id: str) -> None:
        ...

    @abstractmethod
    async def delete_node(self, query_id: str) -> None:
        ...

    @abstractmethod
    async def update_fq_metadata(self, query_id: str, updates: dict) -> None:
        ...
