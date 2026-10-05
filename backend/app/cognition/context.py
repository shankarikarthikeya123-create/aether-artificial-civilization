from typing import Any


class ContextBuilder:

    def build_citizen_context(
        self,
        citizen: Any,
        world: Any,
        memories: list[Any] | None = None,
        knowledge_context: str = "",
    ) -> str:

        memories = memories or []

        personality = citizen.personality
        needs = citizen.needs

        memory_text = "\n".join(
            f"- {memory.content}"
            for memory in memories
        ) or "- No relevant memories available."

        goals = (
            ", ".join(citizen.goals)
            if citizen.goals
            else "None"
        )

        beliefs = (
            ", ".join(citizen.beliefs)
            if citizen.beliefs
            else "None"
        )

        knowledge_context = (
            knowledge_context.strip()
            if knowledge_context
            else "No relevant external knowledge retrieved."
        )

        return f"""
AETHER CITIZEN COGNITIVE CONTEXT

CITIZEN
Name: {citizen.name}
Age: {citizen.age}
Occupation: {citizen.occupation}
Location: {citizen.location_id}

CURRENT STATE
Current action: {citizen.current_action}
Goals: {goals}
Beliefs: {beliefs}

NEEDS
Hunger: {needs.hunger:.1f}
Energy: {needs.energy:.1f}
Social: {needs.social:.1f}
Money: {needs.money:.1f}
Safety: {needs.safety:.1f}

PERSONALITY
Openness: {personality.openness:.2f}
Conscientiousness: {personality.conscientiousness:.2f}
Agreeableness: {personality.agreeableness:.2f}
Extraversion: {personality.extraversion:.2f}
Curiosity: {personality.curiosity:.2f}

SKILLS
{citizen.skills}

RELEVANT LONG-TERM MEMORY
{memory_text}

RETRIEVED AETHER KNOWLEDGE
{knowledge_context}

WORLD
Civilization: {world.name}
Day: {world.day}
Time: {world.hour:02d}:{world.minute:02d}
Population: {world.population}

RESOURCES
{world.resources}

WORLD EVENTS
{world.events[-10:]}
""".strip()


context_builder = ContextBuilder()