from pydantic import BaseModel, Field
from typing import List


class KnowledgeDocument(BaseModel):
    id: str
    title: str
    content: str
    source: str = "aether"
    knowledge_type: str = "general"
    tags: List[str] = Field(default_factory=list)
    importance: float = 0.5
    created_at: str


class KnowledgeQuery(BaseModel):
    query: str
    limit: int = 5