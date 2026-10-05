from typing import Any

from ..knowledge.engine import knowledge_engine
from ..memory.engine import MemoryEngine
from ..reasoning.bayesian import bayesian_reasoner
from ..reasoning.logic import rule_engine
from ..reasoning.planner import goal_planner
from ..reasoning.search import astar
from .context import context_builder
from ..llm import llm


class CitizenAgent:
    """
    AETHER cognitive agent.

    Cognitive pipeline:

    PERCEIVE
        ↓
    RETRIEVE MEMORY
        ↓
    RETRIEVE KNOWLEDGE / RAG
        ↓
    SYMBOLIC REASONING
        ↓
    BAYESIAN REASONING
        ↓
    PLANNING
        ↓
    PATH SEARCH
        ↓
    LLM DECISION
        ↓
    ACTION
        ↓
    MEMORY
    """

    def __init__(
        self,
        citizen_engine: Any,
        world_engine: Any,
        memory_engine: MemoryEngine,
    ):
        self.citizen_engine = citizen_engine
        self.world_engine = world_engine
        self.memory_engine = memory_engine

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def get_citizen(self, citizen_id: str):
        citizen = self.citizen_engine.get_citizen(citizen_id)

        if citizen is None:
            raise ValueError(f"Citizen '{citizen_id}' not found")

        return citizen

    def get_world(self):
        return self.world_engine.get_world()

    # ============================================================
    # PERCEPTION
    # ============================================================

    def perceive(self, citizen_id: str):
        """
        Converts the current world into information relevant
        to one citizen.
        """

        citizen = self.get_citizen(citizen_id)
        world = self.get_world()

        resources = {}

        for resource in world.resources:
            percentage = 0.0

            if resource.capacity > 0:
                percentage = (
                    resource.amount / resource.capacity
                ) * 100

            resources[resource.name] = {
                "amount": resource.amount,
                "capacity": resource.capacity,
                "percentage": round(percentage, 2),
            }

        return {
            "citizen": {
                "id": citizen.id,
                "name": citizen.name,
                "occupation": citizen.occupation,
                "location": citizen.location_id,
            },
            "needs": {
                "hunger": citizen.needs.hunger,
                "energy": citizen.needs.energy,
                "social": citizen.needs.social,
                "money": citizen.needs.money,
                "safety": citizen.needs.safety,
            },
            "world": {
                "day": world.day,
                "time": f"{world.hour:02d}:{world.minute:02d}",
                "population": world.population,
                "resources": resources,
                "recent_events": world.events[-10:],
            },
        }

    # ============================================================
    # FACT GENERATION
    # ============================================================

    def _build_facts(self, citizen, world):
        facts = set()

        # Population
        if world.population > 0:
            facts.add("population_active")

        # --------------------------------------------------------
        # Citizen needs
        # --------------------------------------------------------

        if citizen.needs.hunger >= 70:
            facts.add("hunger_high")

        if citizen.needs.energy <= 30:
            facts.add("energy_low_personal")

        if citizen.needs.social <= 30:
            facts.add("social_low")

        if citizen.needs.money <= 20:
            facts.add("money_low")

        if citizen.needs.safety <= 30:
            facts.add("safety_low")

        # --------------------------------------------------------
        # Civilization resources
        # --------------------------------------------------------

        resource_map = {
            resource.name: resource
            for resource in world.resources
        }

        food = resource_map.get("food")
        water = resource_map.get("water")
        energy = resource_map.get("energy")

        if food and food.capacity > 0:
            if food.amount / food.capacity <= 0.25:
                facts.add("food_low")

        if water and water.capacity > 0:
            if water.amount / water.capacity <= 0.25:
                facts.add("water_low")

        if energy and energy.capacity > 0:
            if energy.amount / energy.capacity <= 0.25:
                facts.add("energy_low")

        return facts

    # ============================================================
    # SYMBOLIC REASONING
    # ============================================================

    def _symbolic_reasoning(self, citizen, world):
        facts = self._build_facts(citizen, world)

        inference = rule_engine.infer(
            facts=facts,
            max_iterations=20,
        )

        all_facts = set(inference["facts"])

        # --------------------------------------------------------
        # Determine the citizen's immediate goal
        # --------------------------------------------------------

        goal = None

        if "food_required" in all_facts:
            goal = "hunger_satisfied"

        elif "rest_required" in all_facts:
            goal = "energy_restored"

        elif "social_interaction_required" in all_facts:
            goal = "social_need_satisfied"

        elif "food_crisis" in all_facts:
            goal = "food_crisis"

        elif "water_crisis" in all_facts:
            goal = "water_crisis"

        elif "energy_crisis" in all_facts:
            goal = "energy_crisis"

        else:
            goal = "maintain_wellbeing"

        # --------------------------------------------------------
        # Planning
        # --------------------------------------------------------

        plan = goal_planner.plan(
            current_state=all_facts,
            goal=goal,
            max_steps=10,
        )

        # --------------------------------------------------------
        # Bayesian reasoning
        # --------------------------------------------------------

        resource_likelihoods = {
            "resource_shortage": 0.2,
            "normal_consumption": 0.8,
        }

        if (
            "food_crisis" in all_facts
            or "water_crisis" in all_facts
            or "energy_crisis" in all_facts
        ):
            resource_likelihoods = {
                "resource_shortage": 0.9,
                "normal_consumption": 0.1,
            }

        posterior = bayesian_reasoner.update(
            resource_likelihoods
        )

        most_likely = (
            max(
                posterior,
                key=posterior.get,
            )
            if posterior
            else None
        )

        # --------------------------------------------------------
        # A* route planning
        # --------------------------------------------------------

        locations = {
            location.id: location
            for location in world.locations
        }

        route = {
            "found": False,
            "path": [],
            "cost": None,
        }

        if citizen.location_id in locations and "capital" in locations:

            def neighbors(location_id):
                current = locations[location_id]

                result = []

                for other_id, other in locations.items():

                    if other_id == location_id:
                        continue

                    distance = (
                        (current.x - other.x) ** 2
                        + (current.y - other.y) ** 2
                    ) ** 0.5

                    result.append(
                        (
                            other_id,
                            distance,
                        )
                    )

                return result

            def heuristic(location_id, target_id):
                current = locations[location_id]
                target = locations[target_id]

                return (
                    (current.x - target.x) ** 2
                    + (current.y - target.y) ** 2
                ) ** 0.5

            route = astar.search(
                start=citizen.location_id,
                goal="capital",
                neighbors=neighbors,
                heuristic=heuristic,
            )

        return {
            "facts": sorted(all_facts),
            "new_facts": inference["new_facts"],
            "goal": goal,
            "plan": plan,
            "bayesian": {
                "posterior": posterior,
                "most_likely": most_likely,
            },
            "navigation": route,
        }

    # ============================================================
    # RAG
    # ============================================================

    def retrieve_knowledge(
        self,
        situation: str,
        limit: int = 4,
    ):
        return knowledge_engine.query_rag(
            query=situation,
            limit=limit,
        )

    # ============================================================
    # MEMORY RETRIEVAL
    # ============================================================

    def retrieve_memories(
        self,
        citizen_id: str,
        situation: str,
        limit: int = 5,
    ):
        return self.memory_engine.recall(
            citizen_id=citizen_id,
            query=situation,
            limit=limit,
        )

    # ============================================================
    # COMPLETE THINKING PIPELINE
    # ============================================================

    def think(
        self,
        citizen_id: str,
        situation: str,
    ):
        citizen = self.get_citizen(citizen_id)
        world = self.get_world()

        # --------------------------------------------------------
        # 1. PERCEPTION
        # --------------------------------------------------------

        perception = self.perceive(citizen_id)

        # --------------------------------------------------------
        # 2. MEMORY RETRIEVAL
        # --------------------------------------------------------

        memories = self.retrieve_memories(
            citizen_id=citizen_id,
            situation=situation,
            limit=5,
        )

        # --------------------------------------------------------
        # 3. RAG
        # --------------------------------------------------------

        rag_result = self.retrieve_knowledge(
            situation=situation,
            limit=4,
        )

        knowledge_context = rag_result.get(
            "context",
            "",
        )

        # --------------------------------------------------------
        # 4. SYMBOLIC AI
        # --------------------------------------------------------

        symbolic = self._symbolic_reasoning(
            citizen=citizen,
            world=world,
        )

        # --------------------------------------------------------
        # 5. BUILD CONTEXT
        # --------------------------------------------------------

        context = context_builder.build_citizen_context(
            citizen=citizen,
            world=world,
            memories=memories,
            knowledge_context=knowledge_context,
        )

        symbolic_context = f"""
SYMBOLIC REASONING

Facts:
{symbolic["facts"]}

Newly inferred facts:
{symbolic["new_facts"]}

Primary goal:
{symbolic["goal"]}

Plan:
{symbolic["plan"]}

Bayesian assessment:
{symbolic["bayesian"]}

Navigation:
{symbolic["navigation"]}
""".strip()

        # --------------------------------------------------------
        # 6. LOCAL LLM
        # --------------------------------------------------------

        prompt = f"""
You are the cognitive decision system of an autonomous citizen
inside the artificial civilization AETHER.

You are NOT controlling the whole civilization.

You are deciding what ONE citizen should do next.

Follow this reasoning architecture:

PERCEIVE
→ REMEMBER
→ RETRIEVE KNOWLEDGE
→ REASON
→ PLAN
→ DECIDE
→ ACT
→ LEARN

CURRENT SITUATION
{situation}

{context}

{symbolic_context}

RETRIEVED KNOWLEDGE
{knowledge_context or "No external knowledge retrieved."}

Your task:

1. Understand the citizen's current state.
2. Consider relevant memories.
3. Use retrieved AETHER knowledge.
4. Respect the symbolic reasoning results.
5. Choose one practical next action.
6. Keep the action realistic for this citizen.
7. Do not invent impossible resources.
8. Do not control other citizens.
9. Return concise structured reasoning.

Return EXACTLY in this format:

ACTION: <one action>

REASON: <one or two sentences>

PLAN: <short plan>

LEARNING: <what the citizen should remember from this situation>
""".strip()

        decision = llm.generate(prompt).strip()

        # --------------------------------------------------------
        # 7. PARSE DECISION
        # --------------------------------------------------------

        action = self._extract_field(
            decision,
            "ACTION",
        )

        reason = self._extract_field(
            decision,
            "REASON",
        )

        plan_text = self._extract_field(
            decision,
            "PLAN",
        )

        learning = self._extract_field(
            decision,
            "LEARNING",
        )

        # --------------------------------------------------------
        # 8. FALLBACK
        # --------------------------------------------------------

        if not action:
            action = self._fallback_action(
                citizen=citizen,
                symbolic=symbolic,
            )

        if not reason:
            reason = (
                "The citizen selected this action based on "
                "current needs and symbolic reasoning."
            )

        if not plan_text:
            plan_text = self._plan_to_text(
                symbolic.get("plan")
            )

        if not learning:
            learning = (
                f"Situation considered: {situation}"
            )

        # --------------------------------------------------------
        # 9. UPDATE CITIZEN STATE
        # --------------------------------------------------------

        citizen.current_action = action

        if symbolic.get("plan", {}).get("steps"):
            citizen.current_plan = [
                step["action"]
                for step in symbolic["plan"]["steps"]
            ]
        else:
            citizen.current_plan = [action]

        # --------------------------------------------------------
        # 10. STORE MEMORY
        # --------------------------------------------------------

        memory_content = (
            f"Situation: {situation}. "
            f"Decision: {action}. "
            f"Reason: {reason}. "
            f"Learning: {learning}"
        )

        new_memory = self.memory_engine.remember(
            citizen_id=citizen_id,
            content=memory_content,
            memory_type="episodic",
            importance=0.7,
            tags=[
                "decision",
                "cognition",
                "learning",
            ],
        )

        return {
            "citizen_id": citizen_id,
            "citizen_name": citizen.name,
            "situation": situation,

            "perception": perception,

            "memory": {
                "retrieved": memories,
                "new_memory": new_memory,
            },

            "rag": {
                "query": rag_result.get("query"),
                "results": rag_result.get("results", []),
            },

            "symbolic_reasoning": symbolic,

            "decision": {
                "raw": decision,
                "action": action,
                "reason": reason,
                "plan": plan_text,
                "learning": learning,
            },

            "architecture": [
                "perception",
                "memory_retrieval",
                "rag",
                "symbolic_reasoning",
                "bayesian_reasoning",
                "planning",
                "astar_navigation",
                "local_llm_decision",
                "action",
                "learning",
                "memory_storage",
            ],
        }

    # ============================================================
    # SYMBOLIC-ONLY ANALYSIS
    # ============================================================

    def analyze_symbolically(
        self,
        citizen_id: str,
    ):
        citizen = self.get_citizen(citizen_id)
        world = self.get_world()

        return self._symbolic_reasoning(
            citizen=citizen,
            world=world,
        )

    # ============================================================
    # FIELD PARSER
    # ============================================================

    @staticmethod
    def _extract_field(
        text: str,
        field: str,
    ) -> str:

        prefix = f"{field}:"

        for line in text.splitlines():

            cleaned = line.strip()

            if cleaned.upper().startswith(
                prefix
            ):
                return cleaned[
                    len(prefix):
                ].strip()

        return ""

    # ============================================================
    # FALLBACK ACTION
    # ============================================================

    @staticmethod
    def _fallback_action(
        citizen,
        symbolic,
    ):

        goal = symbolic.get(
            "goal",
            "",
        )

        if goal == "hunger_satisfied":
            return "find_food"

        if goal == "energy_restored":
            return "rest"

        if goal == "social_need_satisfied":
            return "socialize"

        if goal == "food_crisis":
            return "support_food_response"

        if goal == "water_crisis":
            return "support_water_response"

        if goal == "energy_crisis":
            return "support_energy_response"

        return "continue_daily_activity"

    # ============================================================
    # PLAN FORMATTER
    # ============================================================

    @staticmethod
    def _plan_to_text(
        plan,
    ):

        if not plan:
            return "No explicit plan."

        steps = plan.get(
            "steps",
            [],
        )

        if not steps:
            return "No explicit plan."

        return " → ".join(
            step.get(
                "action",
                "unknown",
            )
            for step in steps
        )


citizen_agent = None