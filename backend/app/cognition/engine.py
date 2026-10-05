from typing import Any

from .agent import CitizenAgent


class CognitionEngine:
    """
    Coordinates cognitive processing for AETHER citizens.

    The engine connects the civilization's:
    
    perception
        ↓
    citizen agent
        ↓
    reasoning
        ↓
    decision
        ↓
    memory
    """

    def __init__(
        self,
        citizen_engine: Any,
        world_engine: Any,
        memory_engine: Any,
    ):
        self.citizen_engine = citizen_engine
        self.world_engine = world_engine
        self.memory_engine = memory_engine

        self.agent = CitizenAgent(
            citizen_engine=citizen_engine,
            world_engine=world_engine,
            memory_engine=memory_engine,
        )

        self.last_decisions: list[dict] = []

    # ============================================================
    # THINK FOR ONE CITIZEN
    # ============================================================

    def think(
        self,
        citizen_id: str,
        situation: str,
    ):
        result = self.agent.think(
            citizen_id=citizen_id,
            situation=situation,
        )

        self.last_decisions.append(result)

        # Keep only recent decisions.
        self.last_decisions = self.last_decisions[-100:]

        return result

    # ============================================================
    # SYMBOLIC ANALYSIS
    # ============================================================

    def analyze(
        self,
        citizen_id: str,
    ):
        return self.agent.analyze_symbolically(
            citizen_id=citizen_id,
        )

    # ============================================================
    # PERCEPTION
    # ============================================================

    def perceive(
        self,
        citizen_id: str,
    ):
        return self.agent.perceive(
            citizen_id=citizen_id,
        )

    # ============================================================
    # RAG
    # ============================================================

    def retrieve_knowledge(
        self,
        situation: str,
        limit: int = 5,
    ):
        return self.agent.retrieve_knowledge(
            situation=situation,
            limit=limit,
        )

    # ============================================================
    # MEMORY
    # ============================================================

    def retrieve_memories(
        self,
        citizen_id: str,
        situation: str,
        limit: int = 5,
    ):
        return self.agent.retrieve_memories(
            citizen_id=citizen_id,
            situation=situation,
            limit=limit,
        )

    # ============================================================
    # RECENT DECISIONS
    # ============================================================

    def get_recent_decisions(
        self,
        limit: int = 20,
    ):
        return self.last_decisions[-limit:]

    # ============================================================
    # STATUS
    # ============================================================

    def status(self):
        return {
            "status": "online",
            "system": "AETHER COGNITION ENGINE",
            "components": {
                "perception": True,
                "memory": True,
                "rag": True,
                "symbolic_reasoning": True,
                "bayesian_reasoning": True,
                "planning": True,
                "astar_navigation": True,
                "local_llm": True,
                "learning": True,
            },
            "decisions_recorded": len(
                self.last_decisions
            ),
        }


cognition_engine = None