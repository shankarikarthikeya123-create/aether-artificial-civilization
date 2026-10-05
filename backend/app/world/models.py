from pydantic import BaseModel, Field
from typing import List


class Location(BaseModel):
    id: str
    name: str
    x: float
    y: float
    location_type: str
    population: int = 0


class Resource(BaseModel):
    name: str
    amount: float
    capacity: float


class WorldState(BaseModel):
    name: str

    # Real-world AETHER clock
    date: str = ""
    time: str = ""
    day_of_week: str = ""
    timezone: str = "Asia/Kolkata"

    # Kept for compatibility with the existing cognition system
    day: int = 1
    hour: int = 0
    minute: int = 0

    locations: List[Location] = Field(default_factory=list)
    resources: List[Resource] = Field(default_factory=list)
    population: int = 0
    events: List[str] = Field(default_factory=list)