


from typing import Any
import re

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
    # SITUATION / LIVE-LIFE CONTEXT
    # ============================================================

    @staticmethod
    def _build_effective_situation(citizen, world, situation):
        explicit = (situation or "").strip()
        if explicit:
            return explicit

        recent_events = world.events[-3:] if getattr(world, "events", None) else []
        event_text = "; ".join(str(event) for event in recent_events) or "no major recent event recorded"
        return (
            f"Autonomous current-life check for {citizen.name}. "
            f"{citizen.name} is a {citizen.occupation} in {citizen.location_id}. "
            f"Personal state: hunger {citizen.needs.hunger:.0f}, "
            f"energy {citizen.needs.energy:.0f}, social {citizen.needs.social:.0f}, "
            f"safety {citizen.needs.safety:.0f}, money {citizen.needs.money:.0f}. "
            f"AETHER is on simulation day {world.day} at {world.hour:02d}:{world.minute:02d}. "
            f"Recent civilization events: {event_text}. "
            "Determine the most relevant thing happening in this citizen's life now "
            "from the live state, memories, knowledge, needs, occupation and world conditions."
        )

    # ============================================================
    # SYMBOLIC REASONING
    # ============================================================

    def _symbolic_reasoning(self, citizen, world, situation=""):
        facts = self._build_facts(citizen, world)

        situation_text = (situation or "").lower()
        situation_rules = []
        if any(word in situation_text for word in ["fire", "burning", "flame", "smoke", "explosion", "attack", "danger", "emergency", "threat", "accident", "earthquake", "flood", "storm", "collapse", "injured", "hurt"]):
            facts.add("external_danger_detected")
            situation_rules.append("The user-provided situation contains an immediate external danger or emergency.")
        elif any(word in situation_text for word in ["water", "thirst", "river", "lake", "leak", "drought", "flooding"]):
            facts.add("external_water_context")
            situation_rules.append("The user-provided situation concerns water, water access, or a water-related event.")
        elif any(word in situation_text for word in ["food", "meal", "restaurant", "hungry", "harvest", "crop", "farm"]):
            facts.add("external_food_context")
            situation_rules.append("The user-provided situation concerns food, farming, or food availability.")
        elif any(word in situation_text for word in ["relationship", "friend", "friendship", "family", "love", "argument", "conflict", "lonely", "social", "meet someone"]):
            facts.add("external_relationship_context")
            situation_rules.append("The user-provided situation concerns relationships or social wellbeing.")
        elif any(word in situation_text for word in ["job", "work", "occupation", "career", "project", "task", "build", "research", "teach", "trade"]):
            facts.add("external_work_context")
            situation_rules.append("The user-provided situation concerns work, occupation, or a task.")
        elif any(word in situation_text for word in ["health", "sick", "ill", "disease", "fever", "pain", "doctor", "medical", "injury"]):
            facts.add("external_health_context")
            situation_rules.append("The user-provided situation concerns health or medical wellbeing.")
        elif any(word in situation_text for word in ["discover", "discovery", "unknown", "mystery", "new place", "artifact", "signal", "experiment"]):
            facts.add("external_discovery_context")
            situation_rules.append("The user-provided situation concerns discovery, investigation, or something unknown.")
        elif any(word in situation_text for word in ["money", "market", "price", "trade", "economy", "economic", "wealth", "payment"]):
            facts.add("external_economic_context")
            situation_rules.append("The user-provided situation concerns money, trade, markets, or the economy.")
        elif any(word in situation_text for word in ["water crisis", "food crisis", "energy crisis", "resource", "shortage", "scarcity", "blackout"]):
            facts.add("external_resource_context")
            situation_rules.append("The user-provided situation concerns civilization resources or scarcity.")
        elif any(word in situation_text for word in ["lost", "stranded", "cannot find", "where am i", "navigate", "navigation"]):
            facts.add("external_navigation_problem")
            situation_rules.append("The user-provided situation concerns navigation or being lost.")

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

        if "external_danger_detected" in facts:
            goal = "citizen_safe"
            plan = {"steps": [{"action": "assess_danger", "reason": "Identify the immediate threat."}, {"action": "seek_safety", "reason": "Move away from the danger."}, {"action": "remain_safe", "reason": "Avoid returning until conditions improve."}]}
        elif "external_water_context" in facts:
            goal = "respond_to_water_situation"
            plan = {"steps": [{"action": "assess_water", "reason": "Determine whether the situation is safe and whether water is available."}, {"action": "respond_to_water", "reason": "Take the most appropriate water-related action."}, {"action": "reassess", "reason": "Check the citizen and civilization state again."}]}
        elif "external_relationship_context" in facts:
            goal = "maintain_relationship"
            plan = {"steps": [{"action": "understand_relationship", "reason": "Assess the social situation."}, {"action": "communicate", "reason": "Respond appropriately to the people involved."}, {"action": "reassess_relationship", "reason": "Evaluate the outcome."}]}
        elif "external_work_context" in facts:
            goal = "complete_work"
            plan = {"steps": [{"action": "assess_task", "reason": "Understand the work requirement."}, {"action": "work", "reason": "Use the citizen's occupation and skills to act."}, {"action": "reassess", "reason": "Check progress and remaining needs."}]}
        elif "external_health_context" in facts:
            goal = "protect_health"
            plan = {"steps": [{"action": "assess_health", "reason": "Understand the health condition."}, {"action": "seek_help", "reason": "Use an appropriate health response."}, {"action": "recover", "reason": "Monitor the citizen until stable."}]}
        elif "external_discovery_context" in facts:
            goal = "investigate_discovery"
            plan = {"steps": [{"action": "investigate", "reason": "Gather evidence about the unknown situation."}, {"action": "research", "reason": "Use knowledge and memory to understand it."}, {"action": "learn", "reason": "Store the result as knowledge."}]}
        elif "external_economic_context" in facts:
            goal = "respond_to_economy"
            plan = {"steps": [{"action": "assess_market", "reason": "Understand the economic condition."}, {"action": "work_or_trade", "reason": "Choose an economically appropriate response."}, {"action": "reassess", "reason": "Monitor the result."}]}
        elif "external_food_context" in facts:
            goal = "respond_to_food_situation"
            plan = {"steps": [{"action": "assess_food", "reason": "Determine food availability and risk."}, {"action": "obtain_or_protect_food", "reason": "Respond to the food situation."}, {"action": "reassess", "reason": "Check needs and resources again."}]}

        return {
            "facts": sorted(all_facts),
            "situation_rules": situation_rules,
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
        # 2. INTERPRET THE INPUT
        # --------------------------------------------------------
        # An empty situation means: let the citizen reason about their
        # actual current life. An explicit situation becomes high-priority
        # external context and must influence the cognition pipeline.
        effective_situation = self._build_effective_situation(
            citizen=citizen,
            world=world,
            situation=situation,
        )

        # --------------------------------------------------------
        # 3. MEMORY RETRIEVAL
        # --------------------------------------------------------

        memories = self.retrieve_memories(
            citizen_id=citizen_id,
            situation=effective_situation,
            limit=5,
        )

        # --------------------------------------------------------
        # 3. RAG
        # --------------------------------------------------------

        rag_result = self.retrieve_knowledge(
            situation=effective_situation,
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
            situation=effective_situation,
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

CURRENT SITUATION (HIGH-PRIORITY EXTERNAL CONTEXT)
{situation}

The situation above is an explicit scenario supplied by the user. It MUST influence the decision. If it describes an immediate danger, emergency, threat, accident, fire, attack, flood, earthquake, or other external hazard, prioritize responding to that situation over routine needs such as hunger, money, or social activity. Do not ignore the situation merely because the citizen has another internal need.

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

        situation_action = self._situation_action(situation)
        if situation_action:
            action = situation_action
            reason = self._situation_reason(situation, citizen, situation_action)
            plan_text = "Prioritize the external situation, respond to the immediate condition, and then reassess personal needs."

        # If the user described something genuinely novel that does not match
        # a known category, do not discard it and fall back to "observe".
        # Investigate it instead; the LLM/RAG context still explains it.
        if (situation or "").strip() and not situation_action and action in {"", "observe"}:
            action = "explore"
            reason = f"{citizen.name} is treating the unfamiliar situation as something to investigate before choosing a more specific response."

        if not action:
            action = self._fallback_action(
                citizen=citizen,
                symbolic=symbolic,
            )

        # Validate LLM choices against the citizen's actual needs.
        # Need values use one consistent 0-100 scale: higher hunger/social/safety/money means more satisfied; energy is higher-is-better.
        if not situation_action:
            needs = citizen.needs
            if action in {"find_food", "eat"} and needs.hunger < 60:
                action = self._fallback_action(citizen=citizen, symbolic={})
                reason = f"{citizen.name} is not sufficiently hungry, so food is not the priority."
            elif action == "rest" and needs.energy > 45:
                action = self._fallback_action(citizen=citizen, symbolic={})
                reason = f"{citizen.name} has adequate energy, so rest is not the priority."
            elif action == "socialize" and needs.social > 40:
                action = self._fallback_action(citizen=citizen, symbolic={})
                reason = f"{citizen.name} has adequate social wellbeing, so socializing is not the priority."
            elif action == "seek_safety" and needs.safety > 40:
                action = self._fallback_action(citizen=citizen, symbolic={})
                reason = f"{citizen.name} is currently safe, so seeking safety is not the priority."

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
    # EXTERNAL SITUATION PRIORITY
    # ============================================================

    @staticmethod
    def _situation_action(situation: str):
        text = (situation or "").lower()

        if not text.strip():
            return ""

        if any(word in text for word in [
            "fire", "burning", "flame", "smoke", "explosion", "attack",
            "danger", "emergency", "threat", "accident", "earthquake",
            "flood", "storm", "collapse", "injured", "hurt"
        ]):
            return "seek_safety"

        if any(word in text for word in [
            "lost", "stranded", "cannot find", "where am i", "navigate", "navigation"
        ]):
            return "explore"

        if any(word in text for word in [
            "water", "thirst", "river", "lake", "leak", "drought", "water crisis"
        ]):
            return "respond_to_water"

        if any(word in text for word in [
            "food", "meal", "restaurant", "hungry", "harvest", "crop", "farm"
        ]):
            return "find_food"

        if any(word in text for word in [
            "relationship", "friend", "friendship", "family", "love",
            "argument", "conflict", "lonely", "social", "meet someone"
        ]):
            return "socialize"

        if any(word in text for word in [
            "job", "work", "occupation", "career", "project", "task"
        ]):
            return "work"

        if any(word in text for word in [
            "research", "discover", "discovery", "unknown", "mystery",
            "artifact", "signal", "experiment"
        ]):
            return "research"

        if any(word in text for word in [
            "health", "sick", "ill", "disease", "fever", "pain",
            "doctor", "medical", "injury"
        ]):
            return "seek_health_help"

        if any(word in text for word in [
            "money", "market", "price", "trade", "economy", "economic",
            "wealth", "payment"
        ]):
            return "work_or_trade"

        if any(word in text for word in [
            "resource", "shortage", "scarcity", "blackout", "energy crisis",
            "food crisis"
        ]):
            return "support_resources"

        return ""

    @staticmethod
    def _situation_reason(situation: str, citizen, action: str):
        reasons = {
            "seek_safety": f"{citizen.name} prioritizes the immediate hazard because safety overrides routine needs.",
            "explore": f"{citizen.name} investigates the environment because the situation indicates a navigation or location problem.",
            "respond_to_water": f"{citizen.name} treats the water-related situation as the immediate context and responds before unrelated routine activity.",
            "find_food": f"{citizen.name} responds to the food-related situation using the citizen's current needs and available civilization resources.",
            "socialize": f"{citizen.name} prioritizes the relationship or social situation and considers communication and wellbeing.",
            "work": f"{citizen.name} responds to the work situation using their occupation, skills, needs, and current world conditions.",
            "research": f"{citizen.name} investigates the new or unknown situation using memory, knowledge, and reasoning.",
            "seek_health_help": f"{citizen.name} prioritizes the health situation and seeks an appropriate response before routine activity.",
            "work_or_trade": f"{citizen.name} responds to the economic situation using their occupation, money state, and the live market context.",
            "support_resources": f"{citizen.name} responds to the civilization resource situation and considers its effect on personal and world wellbeing.",
        }
        return reasons.get(
            action,
            f"{citizen.name} is responding to the explicit situation provided by the user.",
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

        # A healthy citizen should not collapse into the same generic
        # "observe" action. Choose a meaningful routine action from the
        # citizen's actual state and occupation.
        needs = getattr(citizen, "needs", None)
        if needs:
            if needs.hunger >= 60:
                return "find_food"
            if needs.energy <= 45:
                return "rest"
            if needs.social <= 40:
                return "socialize"
            if needs.safety <= 40:
                return "seek_safety"
            if needs.money <= 40:
                return "work"

        occupation = (getattr(citizen, "occupation", "") or "").lower()
        occupation_actions = {
            "software_developer": "develop",
            "developer": "develop",
            "researcher": "research",
            "teacher": "teach",
            "builder": "build",
            "merchant": "trade",
            "farmer": "farm",
            "doctor": "treat_patients",
            "engineer": "engineer",
            "artist": "create",
            "guard": "patrol",
            "scientist": "research",
            "cook": "prepare_food",
            "mechanic": "repair",
        }

        return occupation_actions.get(
            occupation,
            "work",
        )

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
