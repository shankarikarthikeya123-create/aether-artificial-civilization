import random
import re
from datetime import datetime
from typing import Any

from .world.engine import CivilizationEngine
from .citizens.engine import CitizenEngine
from .memory.engine import MemoryEngine
from .automation.engine import AutomationEngine
from .cognition.agent import CitizenAgent
from .knowledge.engine import knowledge_engine
from .learning_store import learning_store
from .reasoning.fai_engine import (
    uniform_cost_search,
    a_star_search,
    best_first_search,
    forward_chain,
    state_space_plan,
    bayes_update,
    coordinate_agents,
    QLearner,
    decision_tree_predict,
)


class SimulationEngine:

    def __init__(
        self,
        world_engine: CivilizationEngine | None = None,
        citizen_engine: CitizenEngine | None = None,
        memory_engine: MemoryEngine | None = None,
    ):

        self.world_engine = world_engine or CivilizationEngine()

        self.citizen_engine = citizen_engine or CitizenEngine(
            population=self.world_engine.get_world().population
        )

        self.memory_engine = memory_engine or MemoryEngine()

        self.citizen_agent = CitizenAgent(
            citizen_engine=self.citizen_engine,
            world_engine=self.world_engine,
            memory_engine=self.memory_engine,
        )

        # Real automation engine connected to the civilization.
        self.automation_engine = AutomationEngine(
            world_engine=self.world_engine,
            citizen_engine=self.citizen_engine,
            memory_engine=self.memory_engine,
        )

        self.tick_count = 0

        self.events: list[dict[str, Any]] = [
            {
                "type": "system",
                "message": "AETHER civilization initialized.",
                "tick": 0,
                "timestamp": datetime.now().isoformat(),
            }
        ]

        self.decision_history: list[dict[str, Any]] = []

        self.total_decisions = 0

        self.total_memories_created = 0

        self.total_relationships = 0

        self.total_discoveries = 0

        self.total_crises = 0

        self.total_career_changes = 0

        # Persistent learning model shared by the civilization.
        self.q_learner = QLearner(alpha=0.25, gamma=0.9)
        self.learning_updates = 0
        self.learning_history: list[dict[str, Any]] = []
        self.learning_dataset: list[dict[str, Any]] = learning_store.recent(500)
        self.learning_history = self.learning_dataset[-200:]
        self.last_council: dict[str, Any] = {}
        self.learning_updates = learning_store.count()
        self.q_learner.q.update(learning_store.load_q())

        self.started_at = datetime.now().isoformat()

        self._initialize_memories()

        # Seed the knowledge layer with the initial live civilization state.
        knowledge_engine.sync_live_knowledge(
            self.world_engine.get_world(),
            self.citizen_engine.get_all(),
            self.events,
            self.decision_history,
        )

    def _initialize_memories(self):

        for citizen in self.citizen_engine.get_all():

            self.memory_engine.remember(
                citizen_id=citizen.id,
                content=(
                    f"I am {citizen.name}, a {citizen.occupation}. "
                    f"I live in {citizen.location_id}."
                ),
                memory_type="semantic",
                importance=0.8,
                tags=["identity", "occupation", "origin"],
            )

            self.memory_engine.remember(
                citizen_id=citizen.id,
                content=(
                    "AETHER is a young artificial civilization "
                    "built around knowledge, cooperation and discovery."
                ),
                memory_type="semantic",
                importance=0.9,
                tags=["civilization", "knowledge"],
            )

    def tick(self, minutes: int = 10):

        self.tick_count += 1

        all_citizens = [
            citizen
            for citizen in self.citizen_engine.get_all()
            if citizen.alive
        ]
        # Process a rotating cohort so a 100+ citizen civilization stays
        # responsive while every citizen gets regular turns.
        cohort_size = min(5, len(all_citizens))
        if all_citizens and cohort_size:
            start_index = ((self.tick_count - 1) * cohort_size) % len(all_citizens)
            citizens = [
                all_citizens[(start_index + offset) % len(all_citizens)]
                for offset in range(cohort_size)
            ]
        else:
            citizens = []

        decisions_this_tick = []

        # =====================================================
        # CITIZEN COGNITIVE LOOP
        # =====================================================

        # The live civilization tick is deterministic and fast by default.
        # The explicit Cognition screen invokes the local LLM on demand.
        # Set AETHER_TICK_LLM=1 if a slower LLM-controlled citizen should
        # participate in every tick.
        import os
        llm_citizen = None
        if os.getenv("AETHER_TICK_LLM", "0") == "1" and citizens:
            llm_citizen = citizens[(self.tick_count - 1) % len(citizens)]

        for citizen in citizens:

            perception = self._perceive(citizen)

            memories = self._retrieve_memories(
                citizen,
                perception,
            )

            # Real lexical RAG retrieval is part of the live cognitive trace.
            rag_query = (
                f"{perception['dominant_need']} "
                f"{citizen.occupation} "
                f"{citizen.location_id} "
                f"food water energy civilization"
            )
            rag_result = knowledge_engine.query_rag(rag_query, limit=3)

            # Run the FAI layer for every citizen so the observable trace
            # contains the same symbolic/search/probabilistic evidence even
            # when an optional local LLM controls the final action.
            fai = self._fai_reasoning(citizen, perception)

            llm_controlled = (
                llm_citizen is not None
                and citizen.id == llm_citizen.id
            )

            llm_action = None

            if llm_controlled:

                try:

                    reasoning = self.citizen_agent.think(
                        citizen.id,
                        (
                            f"Current situation: dominant need is "
                            f"{perception['dominant_need']}. "
                            f"Resources: {perception['resources']}. "
                            "Choose the most appropriate first action. "
                            "The DECISION must be exactly one of: "
                            "eat, rest, socialize, work, research, explore."
                        ),
                    )

                    # CitizenAgent.think() returns a dictionary.
                    # _extract_llm_decision() safely extracts the
                    # actual LLM decision from that dictionary.
                    llm_action = self._extract_llm_decision(
                        reasoning
                    )

                except Exception as exc:

                    reasoning = (
                        f"Local LLM unavailable for this tick: {exc}"
                    )

            if llm_action:

                action = llm_action

                plan = self._plan_for_action(
                    action
                )

            else:

                reasoning = (
                    self._reason(
                        citizen,
                        perception,
                        memories,
                    )
                    if not llm_controlled
                    else reasoning
                )

                # Every symbolic citizen now passes through the same
                # FAI reasoning layer used by the reasoning laboratory.
                reasoning = f"{reasoning} FAI pipeline: {fai['summary']}"

                plan = self._create_plan(
                    citizen,
                    reasoning,
                )

                action = self._decide(
                    citizen,
                    reasoning,
                    plan,
                    perception,
                )

                # FAI planning can override a weak symbolic choice when
                # the civilization has a concrete resource crisis.
                if fai["recommended_action"]:
                    action = fai["recommended_action"]
                    plan = fai["plan"]

            citizen.current_plan = plan

            citizen.current_action = action

            learning_before = {
                "hunger": citizen.needs.hunger,
                "energy": citizen.needs.energy,
                "social": citizen.needs.social,
                "money": citizen.needs.money,
            }

            result = self._execute_action(
                citizen,
                action,
            )

            self._learn_from_action(
                citizen,
                action,
                result,
                learning_before,
            )

            self._create_memory(
                citizen,
                action,
                result,
                reasoning,
            )

            cognitive_trace = {
                "perception": {
                    "dominant_need": perception["dominant_need"],
                    "need_values": perception["need_values"],
                    "location": perception["location"],
                    "occupation": perception["occupation"],
                    "resources": perception["resources"],
                    "world": {
                        "population": perception["population"],
                        "day": perception["day"],
                        "hour": perception["hour"],
                    },
                },
                "memory": {
                    "query": f"{perception['dominant_need']} {citizen.occupation} {citizen.location_id}",
                    "count": len(memories),
                    "items": [
                        {
                            "content": getattr(memory, "content", str(memory)),
                            "type": getattr(memory, "memory_type", "unknown"),
                            "importance": getattr(memory, "importance", None),
                        }
                        for memory in memories[:5]
                    ],
                },
                "rag": {
                    "query": rag_result.get("query", rag_query),
                    "results": rag_result.get("results", []),
                },
                "reasoning": {
                    "facts": sorted(fai.get("facts", [])),
                    "route": fai.get("route", {}),
                    "a_star": fai.get("a_star", {}),
                    "best_first": fai.get("best_first", {}),
                    "posterior": fai.get("posterior", {}),
                    "summary": fai.get("summary", ""),
                    "text": reasoning,
                },
                "planning": {
                    "goal": sorted(list(fai.get("plan", []) and [])),
                    "plan": plan,
                    "recommended_action": fai.get("recommended_action"),
                },
                "decision": {
                    "action": action,
                    "llm_controlled": bool(llm_action),
                    "reason": self._decision_reason(citizen, perception, action),
                    "scores": self._decision_scores(citizen, perception),
                },
                "action": {
                    "result": result,
                },
                "learning": {
                    "state_before": learning_before,
                    "persistent_experiences": learning_store.count(),
                    "q_states": len(self.q_learner.q),
                    "updates": self.learning_updates,
                },
            }

            # Preserve the actual FAI goal separately when available.
            cognitive_trace["planning"]["goal"] = sorted(
                list(fai.get("goal", []))
            )

            decision = {
                "citizen_id": citizen.id,
                "citizen": citizen.name,
                "occupation": citizen.occupation,
                "dominant_need": perception["dominant_need"],
                "action": action,
                "result": result,
                "memories_retrieved": len(memories),
                "reasoning": reasoning,
                "plan": plan,
                "llm_controlled": bool(llm_action),
                "cognitive_trace": cognitive_trace,
                "tick": self.tick_count,
                "timestamp": datetime.now().isoformat(),
            }

            decisions_this_tick.append(
                decision
            )

            self.decision_history.append(
                decision
            )

            self.total_decisions += 1

        # =====================================================
        # WORLD DYNAMICS
        # =====================================================

        self._apply_world_dynamics()

        # =====================================================
        # EMERGENT CIVILIZATION BEHAVIOUR
        # =====================================================

        self._emergent_relationships()

        self._emergent_career_changes()

        self._emergent_discoveries()

        self._detect_crises()

        self._generate_world_events()

        # =====================================================
        # AUTOMATION ENGINE
        # =====================================================

        automation_results = (
            self.automation_engine.run_cycle()
        )

        # Convert automation events into normal simulation events
        # so the frontend can display them.
        for workflow in automation_results:

            if workflow.result:

                self.events.append(
                    {
                        "type": "automation",
                        "message": workflow.result,
                        "workflow_id": workflow.id,
                        "tick": self.tick_count,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # The specialist council evaluates the whole civilization after
        # the local citizen decisions and before time advances.
        try:
            from .integrations.multi_agent import run_council
            self.last_council = run_council(self)
        except Exception:
            self.last_council = {}

        # =====================================================
        # ADVANCE WORLD TIME
        # =====================================================

        self.world_engine.advance_time(
            minutes
        )

        # =====================================================
        # SYNCHRONIZE LOCATIONS AND POPULATION
        # =====================================================

        self._sync_locations()

        self.decision_history = (
            self.decision_history[-500:]
        )

        self.events = (
            self.events[-300:]
        )

        world = self.world_engine.get_world()

        self.events.append(
            {
                "type": "simulation",
                "message": (
                    f"Simulation tick {self.tick_count} completed. "
                    f"{len(decisions_this_tick)} citizens made decisions."
                ),
                "tick": self.tick_count,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Every completed tick becomes current civilization knowledge.
        knowledge_engine.sync_live_knowledge(
            world,
            self.citizen_engine.get_all(),
            self.events,
            self.decision_history,
        )

        return {
            "status": "success",
            "tick": self.tick_count,
            "day": world.day,
            "hour": world.hour,
            "minute": world.minute,
            "population": world.population,
            "decisions": decisions_this_tick,
            "automation": [
                workflow.__dict__
                for workflow in automation_results
            ],
            "recent_events": self.events[-20:],
            "council": self.last_council,
        }

    def run(self, minutes: int = 10):
        """Compatibility wrapper used by the API simulation endpoint."""
        return self.tick(minutes)

    def run_tick(self, minutes: int = 10):
        """API compatibility alias for running one simulation tick."""
        return self.tick(minutes)

    # =========================================================
    # PERCEPTION
    # =========================================================

    def _perceive(self, citizen):

        needs = citizen.needs

        need_values = {
            "hunger": needs.hunger,
            "energy": 100 - needs.energy,
            "social": 100 - needs.social,
            "money": 100 - min(
                needs.money / 5,
                100,
            ),
            "safety": 100 - needs.safety,
        }

        dominant_need = max(
            need_values,
            key=need_values.get,
        )

        world = self.world_engine.get_world()

        resources = {
            resource.name: resource.amount
            for resource in world.resources
        }

        return {
            "dominant_need": dominant_need,
            "need_values": need_values,
            "location": citizen.location_id,
            "occupation": citizen.occupation,
            "resources": resources,
            "population": world.population,
            "day": world.day,
            "hour": world.hour,
        }

    # =========================================================
    # MEMORY RETRIEVAL
    # =========================================================

    def _retrieve_memories(
        self,
        citizen,
        perception,
    ):

        query = (
            f"{perception['dominant_need']} "
            f"{citizen.occupation} "
            f"{citizen.location_id}"
        )

        return self.memory_engine.recall(
            citizen_id=citizen.id,
            query=query,
            limit=5,
        )

    # =========================================================
    # UNIFIED FAI REASONING
    # =========================================================

    def _fai_reasoning(self, citizen, perception):
        """Apply core FAI algorithms to each symbolic citizen decision."""
        resources = perception["resources"]
        facts = set()
        if citizen.needs.hunger >= 65: facts.add("citizen_hungry")
        if 100 - citizen.needs.energy >= 65: facts.add("citizen_tired")
        if 100 - citizen.needs.social >= 65: facts.add("citizen_social_need")
        if resources.get("food", 0) < 200: facts.add("food_shortage")
        if resources.get("water", 0) < 400: facts.add("water_shortage")
        if resources.get("energy", 0) < 300: facts.add("energy_shortage")

        rules = [
            ({"food_shortage"}, "food_crisis"),
            ({"water_shortage"}, "water_crisis"),
            ({"energy_shortage"}, "energy_crisis"),
            ({"citizen_hungry"}, "food_required"),
            ({"citizen_tired"}, "rest_required"),
            ({"citizen_social_need"}, "social_interaction_required"),
        ]
        inference = forward_chain(facts, rules)

        graph = {
            "citizen": [("food", 2), ("water", 3), ("energy", 4), ("community", 2)],
            "food": [("hub", 2)], "water": [("hub", 2)],
            "energy": [("hub", 2)], "community": [("hub", 1)], "hub": [],
        }
        heuristic = {"citizen": 3, "food": 2, "water": 2, "energy": 2, "community": 1, "hub": 0}
        route = uniform_cost_search(graph, "citizen", "hub")
        astar_route = a_star_search(graph, "citizen", "hub", heuristic)
        best_first_route = best_first_search(
            {key: [node for node, _ in value] for key, value in graph.items()},
            "citizen",
            "hub",
            heuristic,
        )

        posterior = bayes_update(
            {"shortage": 0.4, "normal": 0.6},
            {"shortage": 0.9 if facts else 0.1, "normal": 0.2 if facts else 0.9},
        )

        action = None
        goal = set()
        actions = []
        if "food_required" in inference["facts"] or "food_crisis" in inference["facts"]:
            action, goal, actions = "eat", {"food_acquired"}, [
                {"name": "find_food", "preconditions": ["food_required"], "effects": ["food_found"]},
                {"name": "acquire_food", "preconditions": ["food_found"], "effects": ["food_acquired"]},
            ]
        elif "rest_required" in inference["facts"] or "energy_crisis" in inference["facts"]:
            action, goal, actions = "rest", {"energy_restored"}, [
                {"name": "find_rest", "preconditions": ["citizen_tired"], "effects": ["rest_area_found"]},
                {"name": "rest", "preconditions": ["rest_area_found"], "effects": ["energy_restored"]},
            ]
        elif "social_interaction_required" in inference["facts"]:
            action, goal, actions = "socialize", {"social_need_satisfied"}, [
                {"name": "find_group", "preconditions": ["citizen_social_need"], "effects": ["social_group_found"]},
                {"name": "socialize", "preconditions": ["social_group_found"], "effects": ["social_need_satisfied"]},
            ]

        planned = state_space_plan(set(inference["facts"]), goal, actions, 4) if goal else {"plan": []}
        plan = planned.get("plan") or (self._plan_for_action(action) if action else [])
        summary = f"forward-chain={len(inference['fired'])}; UCS={route['cost']}; A*={astar_route['cost']}; BestFirst={'ok' if best_first_route['found'] else 'fail'}; Bayesian={max(posterior, key=posterior.get)}"
        return {
            "summary": summary,
            "facts": inference["facts"],
            "route": route,
            "a_star": astar_route,
            "best_first": best_first_route,
            "posterior": posterior,
            "plan": plan,
            "goal": goal,
            "recommended_action": action,
        }

    # =========================================================
    # REASONING
    # =========================================================

    def _reason(
        self,
        citizen,
        perception,
        memories,
    ):

        dominant_need = perception["dominant_need"]

        memory_text = [
            memory.content
            for memory in memories
        ]

        if dominant_need == "hunger":

            reasoning = (
                f"{citizen.name} detects increasing hunger. "
                f"As a {citizen.occupation}, they evaluate "
                f"food-related actions before making a decision."
            )

        elif dominant_need == "energy":

            reasoning = (
                f"{citizen.name} is becoming tired and evaluates "
                f"whether rest or reduced activity is appropriate."
            )

        elif dominant_need == "social":

            reasoning = (
                f"{citizen.name} detects a social need and evaluates "
                f"interaction with nearby citizens."
            )

        elif dominant_need == "money":

            reasoning = (
                f"{citizen.name} detects financial pressure and "
                f"evaluates productive or economic activities."
            )

        elif dominant_need == "safety":

            reasoning = (
                f"{citizen.name} evaluates environmental safety "
                f"and considers protective actions."
            )

        else:

            reasoning = (
                f"{citizen.name} evaluates the current state "
                f"of the civilization."
            )

        if memory_text:

            reasoning += (
                f" Relevant memories retrieved: "
                f"{len(memory_text)}."
            )

        try:
            from .integrations.orchestration import build_cognitive_context
            context = build_cognitive_context(
                situation=reasoning,
                world=perception.get("resources", {}),
                memories=memories,
                knowledge="Live AETHER knowledge and retrieved civilization state.",
                rules=["Protect immediate citizen welfare.", "Respect civilization resources.", "Learn from outcomes."],
            )
            reasoning += f" Context orchestrated through {context['framework']}."
        except Exception:
            reasoning += " Context orchestration fallback active."

        return reasoning

    # =========================================================
    # PLANNING
    # =========================================================

    def _create_plan(
        self,
        citizen,
        reasoning,
    ):

        need_scores = {
            "hunger": citizen.needs.hunger,
            "energy": 100 - citizen.needs.energy,
            "social": 100 - citizen.needs.social,
            "money": 100 - min(
                citizen.needs.money / 5,
                100,
            ),
            "safety": 100 - citizen.needs.safety,
        }

        dominant_need = max(
            need_scores,
            key=need_scores.get,
        )

        if dominant_need == "hunger":

            return [
                "Locate available food",
                "Acquire food",
                "Consume food",
                "Update hunger state",
            ]

        if dominant_need == "energy":

            return [
                "Reduce activity",
                "Find a safe place",
                "Rest",
                "Restore energy",
            ]

        if dominant_need == "social":

            return [
                "Identify nearby citizens",
                "Initiate interaction",
                "Build relationship",
                "Update social state",
            ]

        if dominant_need == "money":

            return [
                "Evaluate economic opportunities",
                "Perform productive work",
                "Receive economic reward",
                "Update financial state",
            ]

        if dominant_need == "safety":

            return [
                "Inspect local conditions",
                "Evaluate threats",
                "Move toward safer conditions",
                "Update safety state",
            ]

        return [
            "Observe environment",
            "Evaluate available actions",
            "Select useful action",
        ]

    # =========================================================
    # LLM DECISION PARSING
    # =========================================================

    def _extract_llm_decision(
        self,
        reasoning,
    ):
        """
        Extract one safe action from the local LLM response.

        CitizenAgent.think() returns a dictionary:

            {
                "decision": {
                    "raw": "...",
                    "action": "...",
                    "reason": "...",
                    ...
                }
            }

        Older versions of the simulation expected a string.
        This method now supports both formats.
        """

        if not reasoning:
            return None

        valid_actions = (
            "eat",
            "rest",
            "socialize",
            "work",
            "research",
            "explore",
        )

        # -----------------------------------------------------
        # NEW FORMAT: CitizenAgent returns a dictionary
        # -----------------------------------------------------

        if isinstance(reasoning, dict):

            decision_data = reasoning.get(
                "decision",
                {},
            )

            if isinstance(
                decision_data,
                dict,
            ):

                # The agent already parsed the action.
                action = decision_data.get(
                    "action"
                )

                if isinstance(
                    action,
                    str,
                ):

                    action = (
                        action
                        .strip()
                        .lower()
                    )

                    if action in valid_actions:
                        return action

                # If structured action wasn't available,
                # inspect the raw LLM output.
                raw = decision_data.get(
                    "raw",
                    "",
                )

            else:

                raw = ""

        # -----------------------------------------------------
        # OLD FORMAT: direct string
        # -----------------------------------------------------

        elif isinstance(
            reasoning,
            str,
        ):

            raw = reasoning

        else:

            return None

        if not isinstance(
            raw,
            str,
        ):
            return None

        text = raw.lower()

        # -----------------------------------------------------
        # Preferred format:
        #
        # ACTION: research
        # -----------------------------------------------------

        match = re.search(
            r"action\s*[:\-]?\s*[`*_ ]*([a-z_]+)",
            text,
            flags=re.IGNORECASE,
        )

        if match:

            candidate = (
                match.group(1)
                .strip()
                .lower()
            )

            if candidate in valid_actions:
                return candidate

        # -----------------------------------------------------
        # Backward-compatible format:
        #
        # DECISION: research
        # -----------------------------------------------------

        match = re.search(
            r"decision\s*[:\-]?\s*[`*_ ]*([a-z_]+)",
            text,
            flags=re.IGNORECASE,
        )

        if match:

            candidate = (
                match.group(1)
                .strip()
                .lower()
            )

            if candidate in valid_actions:
                return candidate

        # -----------------------------------------------------
        # Last fallback: search near ACTION / DECISION
        # -----------------------------------------------------

        for action in valid_actions:

            if re.search(
                rf"(?:action|decision)[^\n]{{0,80}}\b"
                rf"{re.escape(action)}\b",
                text,
                flags=re.IGNORECASE,
            ):
                return action

        return None

    def _plan_for_action(
        self,
        action: str,
    ):

        plans = {

            "eat": [
                "Locate available food",
                "Acquire food",
                "Consume food",
                "Update hunger state",
            ],

            "rest": [
                "Reduce activity",
                "Find a safe place",
                "Rest",
                "Restore energy",
            ],

            "socialize": [
                "Identify nearby citizens",
                "Initiate interaction",
                "Build relationship",
                "Update social state",
            ],

            "work": [
                "Evaluate economic opportunities",
                "Perform productive work",
                "Receive economic reward",
                "Update financial state",
            ],

            "research": [
                "Investigate a useful question",
                "Perform research",
                "Increase knowledge",
                "Record the result",
            ],

            "explore": [
                "Observe the local environment",
                "Explore the region",
                "Evaluate discoveries",
                "Record the result",
            ],
        }

        return plans.get(
            action,
            [
                "Observe environment",
                "Evaluate available actions",
                "Select useful action",
            ],
        )

    # =========================================================
    # DECISION MAKING
    # =========================================================

    def _decision_scores(self, citizen, perception):
        """Score actions using the same snapshot shown in the cognitive trace.

        Personality values are stored on a 0-100 scale, so they must be
        normalized before they influence the decision. Immediate needs get
        priority over curiosity/exploration preferences.
        """
        needs = citizen.needs
        pressure = perception.get("need_values", {})
        scores = {
            "work": max(0.0, pressure.get("money", 0.0)) * 1.5,
            "rest": max(0.0, pressure.get("energy", 0.0)) * 1.2,
            "eat": max(0.0, pressure.get("hunger", 0.0)) * 1.2,
            "socialize": max(0.0, pressure.get("social", 0.0)) * 1.0,
            "research": 0.0,
            "explore": 0.0,
        }
        curiosity = float(getattr(citizen.personality, "curiosity", 0)) / 100.0
        openness = float(getattr(citizen.personality, "openness", 0)) / 100.0
        extraversion = float(getattr(citizen.personality, "extraversion", 0)) / 100.0
        conscientiousness = float(getattr(citizen.personality, "conscientiousness", 0)) / 100.0
        scores["research"] += curiosity * 20.0
        scores["explore"] += openness * 12.0
        scores["socialize"] += extraversion * 8.0
        scores["work"] += conscientiousness * 8.0

        # Avoid random noise overwhelming an urgent physiological/economic need.
        dominant = perception.get("dominant_need")
        if dominant in {"money", "hunger", "energy", "social", "safety"}:
            preferred = {
                "money": "work",
                "hunger": "eat",
                "energy": "rest",
                "social": "socialize",
                "safety": "explore",
            }.get(dominant)
            if preferred:
                scores[preferred] += 30.0
        return scores

    def _decision_reason(self, citizen, perception, action):
        dominant = perception.get("dominant_need", "wellbeing")
        explanations = {
            "work": f"{citizen.name} chose work because money pressure is the strongest current need, and work can directly increase personal money.",
            "eat": f"{citizen.name} chose eating because hunger is the strongest current need.",
            "rest": f"{citizen.name} chose rest because low energy is the strongest current need.",
            "socialize": f"{citizen.name} chose socializing because social pressure is the strongest current need.",
            "research": f"{citizen.name} chose research because curiosity currently outweighs immediate needs.",
            "explore": f"{citizen.name} chose exploration because it is the best available response to the current safety/environment state.",
        }
        return explanations.get(action, f"{citizen.name} chose {action} while responding to {dominant} pressure.")

    def _decide(
        self,
        citizen,
        reasoning,
        plan,
        perception=None,
    ):
        perception = perception or self._perceive(citizen)
        scores = self._decision_scores(citizen, perception)
        return max(scores, key=scores.get)

    # =========================================================
    # ACTION EXECUTION
    # =========================================================

    def _execute_action(
        self,
        citizen,
        action,
    ):

        needs = citizen.needs

        if action == "eat":

            needs.hunger = max(
                0,
                needs.hunger - random.uniform(
                    15,
                    30,
                ),
            )

            result = (
                f"{citizen.name} consumed food "
                f"and reduced hunger."
            )

        elif action == "rest":

            needs.energy = min(
                100,
                needs.energy + random.uniform(
                    10,
                    25,
                ),
            )

            result = (
                f"{citizen.name} rested "
                f"and recovered energy."
            )

        elif action == "socialize":

            needs.social = min(
                100,
                needs.social + random.uniform(
                    8,
                    20,
                ),
            )

            result = (
                f"{citizen.name} interacted "
                f"with another citizen."
            )

        elif action == "work":

            needs.money += random.uniform(
                5,
                20,
            )

            needs.energy = max(
                0,
                needs.energy - random.uniform(
                    3,
                    8,
                ),
            )

            result = (
                f"{citizen.name} worked as a "
                f"{citizen.occupation} "
                f"and earned resources."
            )

        elif action == "research":

            needs.energy = max(
                0,
                needs.energy - random.uniform(
                    2,
                    6,
                ),
            )

            result = (
                f"{citizen.name} conducted research "
                f"and increased civilization knowledge."
            )

            self._increase_resource(
                "knowledge",
                random.uniform(
                    1,
                    8,
                ),
            )

        elif action == "explore":

            needs.energy = max(
                0,
                needs.energy - random.uniform(
                    4,
                    10,
                ),
            )

            result = (
                f"{citizen.name} explored the "
                f"{citizen.location_id} region."
            )

        else:

            result = (
                f"{citizen.name} remained idle."
            )

        citizen.needs = needs

        return result

    # =========================================================
    # LEARNING FROM ACTION
    # =========================================================

    def _learn_from_action(
        self,
        citizen,
        action,
        result,
    ):

        current_skill = citizen.skills.get(
            citizen.occupation,
            0.5,
        )

        improvement = random.uniform(
            0.001,
            0.01,
        )

        citizen.skills[citizen.occupation] = min(
            1.0,
            current_skill + improvement,
        )

        current_problem_solving = citizen.skills.get(
            "problem_solving",
            0.5,
        )

        citizen.skills["problem_solving"] = min(
            1.0,
            current_problem_solving
            + random.uniform(
                0.0005,
                0.005,
            ),
        )

    # =========================================================
    # MEMORY CREATION
    # =========================================================

    def _create_memory(
        self,
        citizen,
        action,
        result,
        reasoning,
    ):

        importance = 0.4

        if action in [
            "research",
            "explore",
        ]:

            importance = 0.6

        if citizen.needs.hunger > 80:

            importance = 0.8

        if citizen.needs.safety < 30:

            importance = 0.9

        self.memory_engine.remember(
            citizen_id=citizen.id,
            content=(
                f"I decided to {action}. "
                f"{result}"
            ),
            memory_type="episodic",
            importance=importance,
            tags=[
                "decision",
                action,
            ],
        )

        self.total_memories_created += 1

    # =========================================================
    # RELATIONSHIPS
    # =========================================================

    def _create_relationship(
        self,
        citizen,
        other,
    ):

        existing = citizen.relationships.get(
            other.id,
            0.0,
        )

        strength = min(
            1.0,
            existing + random.uniform(
                0.03,
                0.12,
            ),
        )

        citizen.relationships[
            other.id
        ] = strength

        other.relationships[
            citizen.id
        ] = strength

        self.total_relationships += 1

        self.memory_engine.remember(
            citizen_id=citizen.id,
            content=(
                f"I interacted with {other.name} "
                f"and strengthened our relationship."
            ),
            memory_type="episodic",
            importance=0.6,
            tags=[
                "relationship",
                "social",
            ],
        )

    def _emergent_relationships(self):

        citizens = [
            citizen
            for citizen in self.citizen_engine.get_all()
            if citizen.alive
        ]

        if len(citizens) < 2:
            return

        interaction_count = min(
            5,
            max(
                1,
                len(citizens) // 20,
            ),
        )

        for _ in range(
            interaction_count
        ):

            citizen = random.choice(
                citizens
            )

            candidates = [
                other
                for other in citizens
                if (
                    other.id != citizen.id
                    and other.location_id
                    == citizen.location_id
                )
            ]

            if not candidates:
                continue

            other = random.choice(
                candidates
            )

            relationship = citizen.relationships.get(
                other.id,
                0.0,
            )

            if relationship > 0:

                change = random.uniform(
                    0.01,
                    0.05,
                )

                new_strength = min(
                    1.0,
                    relationship + change,
                )

                citizen.relationships[
                    other.id
                ] = new_strength

                other.relationships[
                    citizen.id
                ] = new_strength

            else:

                if random.random() < 0.35:

                    strength = random.uniform(
                        0.05,
                        0.15,
                    )

                    citizen.relationships[
                        other.id
                    ] = strength

                    other.relationships[
                        citizen.id
                    ] = strength

                    self.total_relationships += 1

                    self.memory_engine.remember(
                        citizen_id=citizen.id,
                        content=(
                            f"I met {other.name} "
                            f"and formed a new "
                            f"social connection."
                        ),
                        memory_type="episodic",
                        importance=0.6,
                        tags=[
                            "relationship",
                            "social",
                        ],
                    )

                    self.memory_engine.remember(
                        citizen_id=other.id,
                        content=(
                            f"I met {citizen.name} "
                            f"and formed a new "
                            f"social connection."
                        ),
                        memory_type="episodic",
                        importance=0.6,
                        tags=[
                            "relationship",
                            "social",
                        ],
                    )

                    self.events.append(
                        {
                            "type": "relationship",
                            "message": (
                                f"{citizen.name} met "
                                f"{other.name}. "
                                f"A new relationship formed."
                            ),
                            "tick": self.tick_count,
                            "timestamp": (
                                datetime.now().isoformat()
                            ),
                        }
                    )

    # =========================================================
    # CAREER CHANGES
    # =========================================================

    def _emergent_career_changes(self):

        citizens = self.citizen_engine.get_all()

        for citizen in citizens:

            if not citizen.alive:
                continue

            if random.random() > 0.015:
                continue

            old_occupation = citizen.occupation

            candidate_occupations = [
                occupation
                for occupation in CitizenEngine.OCCUPATIONS
                if occupation != old_occupation
            ]

            if not candidate_occupations:
                continue

            new_occupation = random.choice(
                candidate_occupations
            )

            citizen.occupation = new_occupation

            citizen.skills.setdefault(
                new_occupation,
                random.uniform(
                    0.35,
                    0.65,
                ),
            )

            citizen.goals.insert(
                0,
                f"Become better at {new_occupation}",
            )

            citizen.goals = citizen.goals[:10]

            self.total_career_changes += 1

            self.memory_engine.remember(
                citizen_id=citizen.id,
                content=(
                    f"I changed my occupation from "
                    f"{old_occupation} to "
                    f"{new_occupation}."
                ),
                memory_type="episodic",
                importance=0.9,
                tags=[
                    "career",
                    "change",
                ],
            )

            self.events.append(
                {
                    "type": "career",
                    "message": (
                        f"{citizen.name} changed career "
                        f"from {old_occupation} "
                        f"to {new_occupation}."
                    ),
                    "tick": self.tick_count,
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                }
            )

    # =========================================================
    # DISCOVERIES
    # =========================================================

    def _emergent_discoveries(self):

        citizens = self.citizen_engine.get_all()

        researchers = [
            citizen
            for citizen in citizens
            if citizen.alive
            and citizen.occupation in [
                "scientist",
                "researcher",
                "software_developer",
                "engineer",
            ]
        ]

        if not researchers:
            return

        if random.random() > 0.08:
            return

        citizen = random.choice(
            researchers
        )

        discovery_types = [
            "new agricultural technique",
            "energy optimization",
            "medical insight",
            "software algorithm",
            "engineering method",
            "environmental observation",
            "economic model",
            "scientific hypothesis",
        ]

        discovery = random.choice(
            discovery_types
        )

        knowledge_gain = random.uniform(
            5,
            25,
        )

        self._increase_resource(
            "knowledge",
            knowledge_gain,
        )

        self.total_discoveries += 1

        memory = (
            f"I discovered a {discovery}. "
            f"This increased the civilization's knowledge."
        )

        self.memory_engine.remember(
            citizen_id=citizen.id,
            content=memory,
            memory_type="semantic",
            importance=0.95,
            tags=[
                "discovery",
                "research",
                "knowledge",
            ],
        )

        self.events.append(
            {
                "type": "discovery",
                "message": (
                    f"{citizen.name} discovered "
                    f"a {discovery}."
                ),
                "citizen_id": citizen.id,
                "tick": self.tick_count,
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }
        )

    # =========================================================
    # WORLD RESOURCES
    # =========================================================

    def _apply_world_dynamics(self):

        world = self.world_engine.get_world()

        population = max(
            world.population,
            1,
        )

        food_consumption = (
            population
            * random.uniform(
                0.3,
                0.8,
            )
        )

        self._change_resource(
            "food",
            -food_consumption,
        )

        water_consumption = (
            population
            * random.uniform(
                0.5,
                1.2,
            )
        )

        self._change_resource(
            "water",
            -water_consumption,
        )

        energy_consumption = (
            population
            * random.uniform(
                0.3,
                0.7,
            )
        )

        self._change_resource(
            "energy",
            -energy_consumption,
        )

        farmers = sum(
            1
            for citizen in self.citizen_engine.get_all()
            if (
                citizen.alive
                and citizen.occupation == "farmer"
            )
        )

        if farmers:

            self._increase_resource(
                "food",
                farmers * random.uniform(
                    0.4,
                    1.2,
                ),
            )

        engineers = sum(
            1
            for citizen in self.citizen_engine.get_all()
            if (
                citizen.alive
                and citizen.occupation == "engineer"
            )
        )

        if engineers:

            self._increase_resource(
                "energy",
                engineers * random.uniform(
                    0.1,
                    0.5,
                ),
            )

        merchants = sum(
            1
            for citizen in self.citizen_engine.get_all()
            if (
                citizen.alive
                and citizen.occupation == "merchant"
            )
        )

        if merchants:

            self._increase_resource(
                "money",
                merchants * random.uniform(
                    10,
                    40,
                ),
            )

        researchers = sum(
            1
            for citizen in self.citizen_engine.get_all()
            if (
                citizen.alive
                and citizen.occupation
                in [
                    "researcher",
                    "scientist",
                ]
            )
        )

        if researchers:

            self._increase_resource(
                "knowledge",
                researchers * random.uniform(
                    0.2,
                    0.8,
                ),
            )

        self._increase_resource(
            "water",
            population * 0.05,
        )

    # =========================================================
    # RESOURCE HELPERS
    # =========================================================

    def _change_resource(
        self,
        resource_name: str,
        amount: float,
    ):

        world = self.world_engine.get_world()

        for resource in world.resources:

            if resource.name == resource_name:

                resource.amount = max(
                    0,
                    min(
                        resource.capacity,
                        resource.amount + amount,
                    ),
                )

                return

    def _increase_resource(
        self,
        resource_name: str,
        amount: float,
    ):

        self._change_resource(
            resource_name,
            abs(amount),
        )

    # =========================================================
    # CRISIS DETECTION
    # =========================================================

    def _detect_crises(self):

        world = self.world_engine.get_world()

        resources = {
            resource.name: resource.amount
            for resource in world.resources
        }

        crisis_messages = []

        food = resources.get(
            "food",
            0,
        )

        water = resources.get(
            "water",
            0,
        )

        energy = resources.get(
            "energy",
            0,
        )

        if food < 200:

            crisis_messages.append(
                "FOOD CRISIS: Food reserves are critically low."
            )

        if water < 400:

            crisis_messages.append(
                "WATER CRISIS: Water reserves are critically low."
            )

        if energy < 300:

            crisis_messages.append(
                "ENERGY CRISIS: Energy reserves are critically low."
            )

        population = max(
            world.population,
            1,
        )

        unemployed = sum(
            1
            for citizen in self.citizen_engine.get_all()
            if (
                citizen.alive
                and citizen.occupation
                in [
                    "artist",
                    "journalist",
                ]
            )
        )

        unemployment_ratio = (
            unemployed / population
        )

        if unemployment_ratio > 0.15:

            crisis_messages.append(
                "EMPLOYMENT ALERT: "
                "Employment imbalance detected."
            )

        for message in crisis_messages:

            self.total_crises += 1

            self.events.append(
                {
                    "type": "crisis",
                    "message": message,
                    "tick": self.tick_count,
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                }
            )

    # =========================================================
    # WORLD EVENTS
    # =========================================================

    def _generate_world_events(self):

        if random.random() > 0.10:
            return

        events = [
            (
                "festival",
                "AETHER citizens organized a spontaneous festival.",
            ),
            (
                "storm",
                "A powerful storm passed through the region.",
            ),
            (
                "market",
                "A major market exchange occurred.",
            ),
            (
                "migration",
                "Several citizens moved toward a new district.",
            ),
            (
                "research",
                "Researchers opened a new collaborative project.",
            ),
            (
                "construction",
                "A new construction project was initiated.",
            ),
            (
                "education",
                "A new educational initiative spread through AETHER.",
            ),
        ]

        event_type, message = random.choice(
            events
        )

        self.events.append(
            {
                "type": event_type,
                "message": message,
                "tick": self.tick_count,
                "timestamp": (
                    datetime.now().isoformat()
                ),
            }
        )

        if event_type == "storm":

            self._change_resource(
                "energy",
                -random.uniform(
                    10,
                    40,
                ),
            )

            self._change_resource(
                "food",
                -random.uniform(
                    5,
                    20,
                ),
            )

        elif event_type == "market":

            self._increase_resource(
                "money",
                random.uniform(
                    100,
                    1000,
                ),
            )

        elif event_type == "research":

            self._increase_resource(
                "knowledge",
                random.uniform(
                    10,
                    50,
                ),
            )

        elif event_type == "education":

            for citizen in self.citizen_engine.get_all():

                citizen.skills[
                    "problem_solving"
                ] = min(
                    1.0,
                    citizen.skills.get(
                        "problem_solving",
                        0.5,
                    ) + 0.01,
                )

    # =========================================================
    # LOCATION SYNCHRONIZATION
    # =========================================================

    def _sync_locations(self):

        world = self.world_engine.get_world()

        for location in world.locations:

            location.population = sum(
                1
                for citizen in self.citizen_engine.get_all()
                if (
                    citizen.alive
                    and citizen.location_id
                    == location.id
                )
            )

        world.population = sum(
            1
            for citizen in self.citizen_engine.get_all()
            if citizen.alive
        )

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self):

        world = self.world_engine.get_world()

        resources = {
            resource.name: {
                "amount": round(
                    resource.amount,
                    2,
                ),
                "capacity": resource.capacity,
            }
            for resource in world.resources
        }

        automation_status = (
            self.automation_engine.get_status()
        )

        return {
            "civilization": world.name,
            "status": "running",
            "tick": self.tick_count,
            "day": world.day,
            "hour": world.hour,
            "minute": world.minute,
            "population": world.population,
            "resources": resources,
            "total_decisions": self.total_decisions,
            "total_memories": self.memory_engine.count(),
            "total_relationships": self.total_relationships,
            "total_discoveries": self.total_discoveries,
            "total_crises": self.total_crises,
            "total_career_changes": self.total_career_changes,
            "event_count": len(self.events),
            "started_at": self.started_at,
            "automation": automation_status,
        }

    # =========================================================
    # HISTORY
    # =========================================================

    def get_history(self):

        return {
            "events": self.events[-100:],
            "decisions": self.decision_history[-100:],
            "tick": self.tick_count,
            "automation": self.automation_engine.get_events(),
        }

    # =========================================================
    # CITIZEN MEMORIES
    # =========================================================

    def get_citizen_memories(
        self,
        citizen_id: str,
    ):

        return self.memory_engine.get_citizen_memories(
            citizen_id
        )


    # =========================================================
    # CIVILIZATION OBSERVATORY / LIVE SCENARIOS
    # =========================================================

    def cognitive_stream(self) -> dict[str, Any]:
        """Return the latest *actual* cognitive trace, not a UI-only summary."""
        world = self.world_engine.get_world()
        latest = self.decision_history[-1] if self.decision_history else {}
        trace = latest.get("cognitive_trace", {})
        citizen_id = latest.get("citizen_id")
        citizen = self.citizen_engine.get_citizen(citizen_id) if citizen_id else None
        action = latest.get("action", "observe")

        perception = trace.get("perception", {})
        memory = trace.get("memory", {})
        rag = trace.get("rag", {})
        reasoning = trace.get("reasoning", {})
        planning = trace.get("planning", {})
        decision = trace.get("decision", {})
        action_trace = trace.get("action", {})
        learning = trace.get("learning", {})

        stages = [
            {
                "stage": "PERCEPTION",
                "state": f"Dominant need: {perception.get('dominant_need', latest.get('dominant_need', 'unknown'))}",
                "value": perception,
            },
            {
                "stage": "MEMORY",
                "state": f"{memory.get('count', latest.get('memories_retrieved', 0))} relevant experiences retrieved",
                "value": memory,
            },
            {
                "stage": "RAG",
                "state": f"{len(rag.get('results', []))} knowledge documents retrieved",
                "value": rag,
            },
            {
                "stage": "REASONING",
                "state": reasoning.get("summary", "Symbolic + probabilistic analysis"),
                "value": reasoning,
            },
            {
                "stage": "PLANNING",
                "state": f"Goal: {', '.join(planning.get('goal', [])) or 'none'}",
                "value": planning,
            },
            {
                "stage": "DECISION",
                "state": "Local LLM selected action" if decision.get("llm_controlled") else "FAI + citizen decision layer selected action",
                "value": decision.get("action", action),
            },
            {
                "stage": "ACTION",
                "state": "World-changing action executed",
                "value": action_trace.get("result", latest.get("result", "Awaiting next action")),
            },
            {
                "stage": "LEARNING",
                "state": f"{learning.get('persistent_experiences', self.learning_updates)} persistent experiences",
                "value": learning,
            },
        ]

        return {
            "tick": self.tick_count,
            "citizen": citizen.name if citizen else "AETHER",
            "citizen_id": citizen.id if citizen else None,
            "stages": stages,
            "trace": trace,
            "world": {
                "population": world.population,
                "day": world.day,
                "time": world.time,
                "resources": {r.name: r.amount for r in world.resources},
            },
            "council": self.last_council,
        }

    def run_civilization_scenario(self, days: int = 30) -> dict[str, Any]:
        days = max(1, min(days, 30))
        world = self.world_engine.get_world()
        snapshots = []

        # Create a visible crisis so the demo starts with a genuine decision.
        food = next((r for r in world.resources if r.name == "food"), None)
        if food:
            # Every trial starts from a controlled baseline so the experiment
            # is repeatable even after previous trials.
            food.amount = food.capacity * 0.14
        self.events.append({
            "type": "scenario",
            "message": "30-DAY CIVILIZATION TRIAL: food reserves deliberately stressed.",
            "tick": self.tick_count,
            "timestamp": datetime.now().isoformat(),
        })

        for day_index in range(1, days + 1):
            # A simulated day is much faster than wall-clock time, so reset
            # real-time workflow cooldowns between experiment days.
            self.automation_engine.cooldowns.clear()
            self.tick(minutes=1440)
            current = self.world_engine.get_world()
            snapshots.append({
                "day": day_index,
                "population": current.population,
                "food": next((r.amount for r in current.resources if r.name == "food"), 0),
                "water": next((r.amount for r in current.resources if r.name == "water"), 0),
                "energy": next((r.amount for r in current.resources if r.name == "energy"), 0),
                "knowledge": next((r.amount for r in current.resources if r.name == "knowledge"), 0),
                "decisions": self.total_decisions,
                "memories": self.total_memories_created,
                "learning": self.learning_updates,
            })

        current = self.world_engine.get_world()
        return {
            "status": "success",
            "scenario": "AETHER 30-DAY AUTONOMOUS CIVILIZATION TRIAL",
            "days": days,
            "start": snapshots[0] if snapshots else {},
            "end": snapshots[-1] if snapshots else {},
            "timeline": snapshots,
            "events": self.events[-12:],
            "learning": {
                "updates": self.learning_updates,
                "persistent_experiences": learning_store.count(),
                "q_states": len(self.q_learner.q),
            },
            "cognitive_stream": self.cognitive_stream(),
        }

    # =========================================================
    # LEARNING AGENT
    # =========================================================

    def _learning_state(self, citizen) -> str:
        hunger = "high" if citizen.needs.hunger >= 60 else "low"
        energy = "low" if citizen.needs.energy <= 40 else "high"
        social = "low" if citizen.needs.social <= 40 else "high"
        return f"hunger_{hunger}|energy_{energy}|social_{social}"

    def _learning_reward(self, citizen, action: str, before: dict[str, float]) -> float:
        after = {
            "hunger": citizen.needs.hunger,
            "energy": citizen.needs.energy,
            "social": citizen.needs.social,
            "money": citizen.needs.money,
        }
        improvement = 0.0
        if action == "eat": improvement = before["hunger"] - after["hunger"]
        elif action == "rest": improvement = after["energy"] - before["energy"]
        elif action == "socialize": improvement = after["social"] - before["social"]
        elif action == "work": improvement = after["money"] - before["money"]
        elif action == "research": improvement = 2.0
        elif action == "explore": improvement = 1.5
        return round(max(-10.0, min(10.0, improvement / 10.0)), 3)

    def _learn_from_action(self, citizen, action: str, result: str, before: dict[str, float] | None = None):
        before = before or {
            "hunger": citizen.needs.hunger,
            "energy": citizen.needs.energy,
            "social": citizen.needs.social,
            "money": citizen.needs.money,
        }
        state = self._learning_state(citizen)
        reward = self._learning_reward(citizen, action, before)
        next_state = self._learning_state(citizen)
        actions = ["eat", "rest", "socialize", "work", "research", "explore"]
        q_value = self.q_learner.update(state, action, reward, next_state, actions)
        self.learning_updates += 1
        self.learning_history.append({
            "citizen_id": citizen.id,
            "citizen": citizen.name,
            "state": state,
            "action": action,
            "reward": reward,
            "q_value": round(q_value, 4),
            "learned_best_action": self.q_learner.best_action(state, actions),
            "result": result,
            "tick": self.tick_count,
            "timestamp": datetime.now().isoformat(),
        })
        self.learning_history = self.learning_history[-200:]

        # Store every learning experience as a training example.
        self.learning_dataset.append({
            "citizen_id": citizen.id,
            "citizen": citizen.name,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "result": result,
            "tick": self.tick_count,
            "timestamp": datetime.now().isoformat(),
        })
        self.learning_dataset = self.learning_dataset[-500:]
        learning_store.add_experience(self.learning_dataset[-1])
        learning_store.save_q(self.q_learner.q)


# =============================================================
# SHARED LAZY SIMULATION INSTANCE
# =============================================================

simulation = None


def get_simulation():

    global simulation

    if simulation is None:
        simulation = SimulationEngine()

    return simulation