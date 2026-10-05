from pydantic import BaseModel, Field
from typing import List


class MemoryRecord(BaseModel):
    id: str
    citizen_id: str
    content: str
    memory_type: str = "episodic"
    importance: float = 0.5
    timestamp: str
    tags: List[str] = Field(default_factory=list)


class MemoryQuery(BaseModel):
    citizen_id: str
    query: str
    limit: int = 5