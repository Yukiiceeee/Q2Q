from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from src.schemas.models import MemoryEntry


class BaseMemoryStore(ABC):

    @abstractmethod
    def save(self, entry: MemoryEntry) -> None:
        ...

    @abstractmethod
    def load_all(self) -> list[MemoryEntry]:
        ...

    @abstractmethod
    def get_by_id(self, memory_id: str) -> MemoryEntry | None:
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def search_all_fake_queries(
        self, query_embedding: np.ndarray, threshold: float = 0.80, max_results: int = 50
    ) -> list[dict]:
        ...

    @abstractmethod
    def get_fake_query_by_id(self, query_id: str) -> dict | None:
        ...

    @abstractmethod
    def get_fake_queries_by_ids(self, query_ids: list[str]) -> list[dict]:
        ...

    @abstractmethod
    def search_paragraphs_for_memory(
        self, query_embedding: np.ndarray, memory_id: str, top_k: int = 3
    ) -> list[dict]:
        ...
