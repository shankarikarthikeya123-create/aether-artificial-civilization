from datetime import datetime
from typing import List

from .models import MemoryRecord


class MemoryEngine:
    """
    AETHER's long-term memory system.

    Memory types:
    - episodic: experiences and events
    - semantic: learned facts and knowledge
    - procedural: learned ways of doing things
    """

    def __init__(self):
        self.memories: List[MemoryRecord] = []

    def remember(
        self,
        citizen_id: str,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ):
        record = MemoryRecord(
            id=f"memory_{len(self.memories) + 1:06d}",
            citizen_id=citizen_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            timestamp=datetime.now().isoformat(),
            tags=tags or [],
        )

        self.memories.append(record)
        return record

    def recall(
        self,
        citizen_id: str,
        query: str,
        limit: int = 5,
    ):
        citizen_memories = [
            memory
            for memory in self.memories
            if memory.citizen_id == citizen_id
        ]

        if not citizen_memories:
            return []

        words = set(query.lower().split())

        scored = []

        for memory in citizen_memories:
            content_words = set(
                memory.content.lower().split()
            )

            overlap = len(words & content_words)

            score = (
                overlap * 2
                + memory.importance
            )

            scored.append(
                (score, memory)
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            memory
            for _, memory in scored[:limit]
        ]

    def get_citizen_memories(
        self,
        citizen_id: str,
    ):
        return [
            memory
            for memory in self.memories
            if memory.citizen_id == citizen_id
        ]

    def get_all(self):
        return self.memories

    def count(self):
        return len(self.memories)