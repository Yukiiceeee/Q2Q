from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from src.schemas.models import MemoryEntry


class BaseMemoryStore(ABC):

    @abstractmethod
    async def save(self, entry: MemoryEntry) -> None:
        ...

    @abstractmethod
    async def load_all(self) -> list[MemoryEntry]:
        ...

    @abstractmethod
    async def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...

    @abstractmethod
    async def clear(self) -> None:
        ...

    @abstractmethod
    async def search_all_fake_queries(
        self, query_embedding: np.ndarray, threshold: float = 0.80, max_results: int = 50
    ) -> list[dict]:
        ...

    @abstractmethod
    async def get_fake_query_by_id(self, query_id: str) -> dict | None:
        ...

    @abstractmethod
    async def get_fake_queries_by_ids(self, query_ids: list[str]) -> list[dict]:
        ...

    @abstractmethod
    async def search_paragraphs_for_memory(
        self, query_embedding: np.ndarray, memory_id: str, top_k: int = 3
    ) -> list[dict]:
        ...

    @abstractmethod
    async def search_kps_for_memory(
        self, query_embedding: np.ndarray, memory_id: str, top_k: int = 10
    ) -> list[dict]:
        ...
