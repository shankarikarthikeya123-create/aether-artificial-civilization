from pydantic import BaseModel, Field
from typing import Dict, List


class CitizenNeeds(BaseModel):
    hunger: float = 20.0
    energy: float = 80.0
    social: float = 50.0
    money: float = 100.0
    safety: float = 90.0


class CitizenPersonality(BaseModel):
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    curiosity: float = 0.5


class CitizenMemory(BaseModel):
    memory_id: str
    content: str
    importance: float = 0.5
    timestamp: str


class Citizen(BaseModel):
    id: str
    name: str
    age: int

    occupation: str
    location_id: str

    personality: CitizenPersonality = Field(
        default_factory=CitizenPersonality
    )

    needs: CitizenNeeds = Field(
        default_factory=CitizenNeeds
    )

    skills: Dict[str, float] = Field(
        default_factory=dict
    )

    goals: List[str] = Field(
        default_factory=list
    )

    beliefs: List[str] = Field(
        default_factory=list
    )

    memories: List[CitizenMemory] = Field(
        default_factory=list
    )

    relationships: Dict[str, float] = Field(
        default_factory=dict
    )

    current_action: str = "idle"
    current_plan: List[str] = Field(
        default_factory=list
    )

    alive: bool = True