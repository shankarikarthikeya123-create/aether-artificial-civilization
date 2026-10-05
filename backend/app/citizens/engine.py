import random
from datetime import datetime

from .models import (
    Citizen,
    CitizenMemory,
    CitizenNeeds,
    CitizenPersonality,
)


class CitizenEngine:

    FIRST_NAMES = [
        "Ari", "Mira", "Kael", "Nova", "Iris",
        "Leo", "Zara", "Orin", "Luna", "Evan",
        "Rhea", "Niko", "Sia", "Arin", "Kira",
        "Theo", "Maya", "Ravi", "Asha", "Noah",
    ]

    LAST_NAMES = [
        "Vale", "Stone", "Ray", "Morgan", "Ash",
        "Reed", "Carter", "Shaw", "Wells", "Rao",
    ]

    OCCUPATIONS = [
        "farmer",
        "engineer",
        "doctor",
        "teacher",
        "scientist",
        "builder",
        "merchant",
        "researcher",
        "artist",
        "mechanic",
        "software_developer",
        "miner",
        "pilot",
        "guard",
        "journalist",
    ]

    LOCATIONS = [
        "capital",
        "north",
        "industrial",
        "farmland",
        "harbor",
        "forest",
        "mountains",
    ]

    def __init__(self, population: int = 100):
        self.citizens = self._generate_population(population)

    def _generate_population(self, population: int):
        citizens = []

        for index in range(population):

            first_name = random.choice(self.FIRST_NAMES)
            last_name = random.choice(self.LAST_NAMES)

            occupation = random.choice(self.OCCUPATIONS)
            location = random.choice(self.LOCATIONS)

            personality = CitizenPersonality(
                openness=round(random.uniform(0.2, 1.0), 2),
                conscientiousness=round(random.uniform(0.2, 1.0), 2),
                extraversion=round(random.uniform(0.2, 1.0), 2),
                agreeableness=round(random.uniform(0.2, 1.0), 2),
                curiosity=round(random.uniform(0.2, 1.0), 2),
            )

            needs = CitizenNeeds(
                hunger=round(random.uniform(10, 40), 2),
                energy=round(random.uniform(50, 100), 2),
                social=round(random.uniform(30, 90), 2),
                money=round(random.uniform(50, 500), 2),
                safety=round(random.uniform(60, 100), 2),
            )

            skills = {
                occupation: round(random.uniform(0.4, 1.0), 2),
                "communication": round(random.uniform(0.3, 1.0), 2),
                "problem_solving": round(random.uniform(0.3, 1.0), 2),
                "adaptability": round(random.uniform(0.3, 1.0), 2),
            }

            goals = [
                f"Become better at {occupation}",
                "Maintain a stable life",
                "Build relationships",
            ]

            memory = CitizenMemory(
                memory_id=f"memory_{index + 1}_001",
                content="I was born into the civilization of AETHER.",
                importance=0.7,
                timestamp=datetime.now().isoformat(),
            )

            citizen = Citizen(
                id=f"citizen_{index + 1:03d}",
                name=f"{first_name} {last_name}",
                age=random.randint(18, 70),
                occupation=occupation,
                location_id=location,
                personality=personality,
                needs=needs,
                skills=skills,
                goals=goals,
                beliefs=[
                    "The civilization should continue to grow.",
                    "Knowledge is valuable.",
                ],
                memories=[memory],
                relationships={},
                current_action="idle",
                current_plan=[],
                alive=True,
            )

            citizens.append(citizen)

        return citizens

    def get_all(self):
        return self.citizens

    def get_citizen(self, citizen_id: str):
        for citizen in self.citizens:
            if citizen.id == citizen_id:
                return citizen

        return None

    def get_population(self):
        return len(self.citizens)