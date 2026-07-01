from __future__ import annotations
from abc import ABC, abstractmethod

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
