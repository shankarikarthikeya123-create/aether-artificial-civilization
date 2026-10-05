from math import sqrt

from fastapi import APIRouter, HTTPException, Query

from ..knowledge.engine import knowledge_engine


router = APIRouter()

# ============================================================
# FULL FAI REASONING LABORATORY
# ============================================================

from ..reasoning.fai_engine import (
    run_demo,
    uniform_cost_search,
    depth_limited_search,
    iterative_deepening_search,
    hill_climbing,
    beam_search,
    forward_chain,
    backward_chain,
    propositional_resolution,
    state_space_plan,
    bayes_update,
    coordinate_agents,
    strategic_crisis_decision,
)

@router.get("/reasoning/lab")
def reasoning_lab():
    """Run a deterministic AETHER demonstration across core FAI topics."""
    return {"status": "success", "civilization": "AETHER", "laboratory": run_demo()}

@router.post("/reasoning/search")
def reasoning_search(payload: dict):
    algorithm = str(payload.get("algorithm", "ucs")).lower()
    graph = payload.get("graph", {})
    start = str(payload.get("start", "capital"))
    goal = str(payload.get("goal", "hub"))
    if algorithm == "ucs":
        weighted = {k: [(str(n), float(c)) for n, c in v] for k, v in graph.items()}
        return uniform_cost_search(weighted, start, goal)
    unweighted = {k: [str(n) for n in v] for k, v in graph.items()}
    if algorithm == "dls":
        return depth_limited_search(unweighted, start, goal, int(payload.get("limit", 5)))
    if algorithm == "iddfs":
        return iterative_deepening_search(unweighted, start, goal, int(payload.get("max_depth", 20)))
    if algorithm == "hill":
        h = payload.get("heuristic", {})
        return hill_climbing(start, goal, lambda n: unweighted.get(n, []), lambda n: float(h.get(n, 99)))
    if algorithm == "beam":
        h = payload.get("heuristic", {})
        return beam_search(start, goal, lambda n: unweighted.get(n, []), lambda n: float(h.get(n, 99)), int(payload.get("width", 2)))
    raise HTTPException(status_code=400, detail="Unknown search algorithm")

@router.post("/reasoning/logic")
def reasoning_logic(payload: dict):
    facts = set(map(str, payload.get("facts", [])))
    rules = [(set(map(str, r.get("conditions", []))), str(r.get("conclusion"))) for r in payload.get("rules", [])]
    mode = str(payload.get("mode", "forward")).lower()
    if mode == "forward":
        return forward_chain(facts, rules)
    if mode == "backward":
        return backward_chain(str(payload.get("goal")), facts, rules)
    if mode == "resolution":
        return propositional_resolution([set(map(str, c)) for c in payload.get("clauses", [])], str(payload.get("query")))
    raise HTTPException(status_code=400, detail="Unknown logic mode")

@router.post("/reasoning/plan")
def reasoning_plan(payload: dict):
    initial = set(map(str, payload.get("initial", [])))
    goal = set(map(str, payload.get("goal", [])))
    return state_space_plan(initial, goal, payload.get("actions", []), int(payload.get("max_depth", 10)))

@router.post("/reasoning/bayesian")
def reasoning_bayesian(payload: dict):
    return {"posterior": bayes_update(payload.get("priors", {}), payload.get("likelihoods", {}))}

@router.post("/reasoning/agents")
def reasoning_agents(payload: dict):
    return coordinate_agents(payload.get("agents", []), list(map(str, payload.get("tasks", []))))


@router.get("/learning")
def learning_status(limit: int = Query(default=30, ge=1, le=200)):
    simulation = get_simulation()
    return {
        "status": "success",
        "algorithm": "Q-Learning",
        "updates": simulation.learning_updates,
        "recent_learning": simulation.learning_history[-limit:],
        "dataset_size": len(simulation.learning_dataset),
        "q_table_size": len(simulation.q_learner.q),
    }


@router.get("/reasoning/strategy")
def strategic_reasoning():
    """Use Minimax, Alpha-Beta, MCTS and CSP on the current civilization state."""
    simulation = get_simulation()
    world = simulation.world_engine.get_world()
    resources = {r.name.lower(): float(r.amount) for r in world.resources}
    result = strategic_crisis_decision(resources, world.population)
    return {
        "status": "success",
        "civilization": "AETHER",
        "resources": resources,
        "population": world.population,
        "strategy": result,
    }


@router.get("/civilization/intelligence")
def civilization_intelligence():
    """Single live snapshot showing how world, events, knowledge, decisions and learning connect."""
    simulation = get_simulation()
    world = simulation.world_engine.get_world()
    return {
        "status": "success",
        "world": world,
        "events": simulation.events[-10:],
        "decisions": simulation.decision_history[-10:],
        "knowledge": knowledge_engine.statistics(),
        "learning": {
            "updates": simulation.learning_updates,
            "dataset_size": len(simulation.learning_dataset),
            "q_table_size": len(simulation.q_learner.q),
        },
        "automation": simulation.automation_engine.get_cooldown_status(),
    }


@router.get("/civilization/cognitive-stream")
def civilization_cognitive_stream():
    simulation = get_simulation()
    return {"status": "success", **simulation.cognitive_stream()}


@router.post("/civilization/run-trial")
def civilization_run_trial(payload: dict | None = None):
    simulation = get_simulation()
    days = int((payload or {}).get("days", 30))
    return simulation.run_civilization_scenario(days)


@router.get("/civilization/timeline")
def civilization_timeline(limit: int = Query(default=30, ge=1, le=100)):
    simulation = get_simulation()
    return {
        "status": "success",
        "tick": simulation.tick_count,
        "timeline": simulation.events[-limit:],
        "decisions": simulation.decision_history[-limit:],
        "learning": simulation.learning_history[-limit:],
    }


@router.get("/integrations/council")
def integration_council():
    from ..integrations.multi_agent import run_council
    return {"status": "success", **run_council(get_simulation())}


@router.get("/integrations/langchain/context")
def langchain_context():
    from ..integrations.orchestration import build_cognitive_context
    simulation = get_simulation()
    world = simulation.world_engine.get_world()
    citizen = simulation.citizen_engine.get_all()[0]
    memories = simulation.memory_engine.recall(citizen.id, f"{citizen.occupation} {citizen.location_id}", limit=5)
    rag = knowledge_engine.query_rag(f"{citizen.occupation} {citizen.location_id}", limit=3)
    context = build_cognitive_context(
        situation=f"{citizen.name} is in {citizen.location_id} and has hunger {citizen.needs.hunger:.1f}.",
        world=world.model_dump(),
        memories=memories,
        knowledge=rag.get("context", ""),
        rules=["Prioritize immediate needs.", "Protect civilization resources.", "Learn from outcomes."],
    )
    return {"status": "success", **context}


@router.get("/integrations/status")
def integration_status():
    from ..integrations.orchestration import LANGCHAIN_AVAILABLE
    try:
        import mcp  # noqa: F401
        mcp_available = True
    except ImportError:
        mcp_available = False
    try:
        import crewai  # noqa: F401
        crew_available = True
    except ImportError:
        crew_available = False
    return {
        "status": "success",
        "layers": {
            "langchain": {"enabled": LANGCHAIN_AVAILABLE, "purpose": "context orchestration + grounded prompt construction"},
            "mcp": {"enabled": mcp_available, "purpose": "standardized world, knowledge and action tools"},
            "multi_agent_council": {"enabled": True, "purpose": "resource, citizen, planning, governance and learning agents"},
            "crew_ai": {"enabled": crew_available, "purpose": "CrewAI-compatible specialized agent orchestration"},
        },
    }


# ============================================================
# KNOWLEDGE REPRESENTATION / FOL / WUMPUS
# ============================================================

@router.post("/reasoning/fol")
def fol_reasoning(payload: dict):
    from ..reasoning.fai_engine import fol_inference, unify, knowledge_based_agent
    facts = [
        (str(item.get("predicate")), tuple(item.get("args", [])))
        for item in payload.get("facts", [])
    ]
    rules = payload.get("rules", [])
    query_item = payload.get("query", {})
    query = (str(query_item.get("predicate")), tuple(query_item.get("args", [])))
    result = fol_inference(facts, rules, query)
    return {"status": "success", "algorithm": "first_order_logic", "result": result}


@router.post("/reasoning/unify")
def unification(payload: dict):
    from ..reasoning.fai_engine import unify
    return {
        "status": "success",
        "algorithm": "unification",
        "bindings": unify(payload.get("pattern"), payload.get("value")),
    }


@router.get("/reasoning/knowledge-agent")
def knowledge_agent_demo():
    from ..reasoning.fai_engine import knowledge_based_agent
    facts = {"population_active", "food_low"}
    rules = [
        ({"food_low", "population_active"}, "food_crisis"),
        ({"food_crisis"}, "activate_emergency_protocol"),
    ]
    return {
        "status": "success",
        "algorithm": "knowledge_based_agent",
        "result": knowledge_based_agent(facts, rules, "activate_emergency_protocol"),
    }


@router.get("/reasoning/representation")
def knowledge_representation():
    from ..reasoning.fai_engine import build_aether_ontology, represent_frame
    return {
        "status": "success",
        "ontology": build_aether_ontology(),
        "frames": {
            "Citizen": represent_frame("Citizen", {
                "needs": ["hunger", "energy", "social", "money", "safety"],
                "mind": ["beliefs", "memories", "goals", "personality"],
                "behavior": ["current_action", "current_plan"],
            }),
            "Resource": represent_frame("Resource", {
                "state": ["amount", "capacity"],
            }),
        },
    }


@router.get("/reasoning/wumpus")
def wumpus_world():
    # Small deterministic Wumpus World used as the syllabus environment demo.
    grid = {
        "A1": {"safe": True},
        "B1": {"safe": True, "breeze": True},
        "C1": {"safe": True, "pit": True},
        "A2": {"safe": True, "stench": True},
        "B2": {"safe": True, "wumpus": True},
        "C2": {"safe": True, "pit": True},
        "A3": {"safe": True, "gold": True},
        "B3": {"safe": True},
        "C3": {"safe": True},
    }
    rules = [
        {"observation": "breeze", "inference": "adjacent_cell_may_contain_pit"},
        {"observation": "stench", "inference": "adjacent_cell_may_contain_wumpus"},
        {"observation": "no_breeze", "inference": "adjacent_cells_are_not_pits"},
        {"observation": "no_stench", "inference": "adjacent_cells_are_not_wumpus"},
    ]
    safe_route = ["A1", "A2", "A3"]
    return {
        "status": "success",
        "environment": "Wumpus World",
        "grid": grid,
        "percepts": rules,
        "agent_goal": "find_gold_and_return_safely",
        "safe_route": safe_route,
        "inference": "A2 has stench, so the agent treats neighboring B2 as a Wumpus candidate; A3 contains the goal gold.",
    }


# ============================================================
# HELPER
# ============================================================

def get_simulation():
    from ..simulation import get_simulation

    return get_simulation()


# ============================================================
# WORLD
# ============================================================

@router.get("/world")
def get_world():
    simulation = get_simulation()

    return simulation.world_engine.get_world()


# ============================================================
# CITIZENS
# ============================================================

@router.get("/citizens")
def get_citizens():
    simulation = get_simulation()

    citizens = simulation.citizen_engine.get_all()

    return {
        "status": "success",
        "count": len(citizens),
        "citizens": citizens,
    }


@router.get("/citizens/{citizen_id}")
def get_citizen(citizen_id: str):
    simulation = get_simulation()

    citizen = simulation.citizen_engine.get_citizen(citizen_id)

    if citizen is None:
        raise HTTPException(
            status_code=404,
            detail="Citizen not found",
        )

    return {
        "status": "success",
        "citizen": citizen,
    }


# ============================================================
# SIMULATION
# ============================================================

@router.post("/simulation/tick")
def simulation_tick(
    minutes: int = Query(
        default=10,
        ge=1,
        le=1440,
    )
):
    simulation = get_simulation()

    return simulation.run(minutes)


@router.post("/simulate")
def simulate(
    minutes: int = Query(
        default=10,
        ge=1,
        le=1440,
    )
):
    simulation = get_simulation()

    return simulation.run(minutes)


# ============================================================
# SIMULATION STATUS
# ============================================================

@router.get("/simulation/status")
def simulation_status():
    simulation = get_simulation()

    return simulation.get_status()


@router.get("/simulation/events")
def simulation_events(limit: int = Query(default=100, ge=1, le=500)):
    simulation = get_simulation()
    # Events belong to the living simulation, not the static world model.
    events = simulation.events[-limit:]
    return {
        "status": "success",
        "count": len(events),
        "events": events,
    }


@router.get("/status")
def get_status():
    simulation = get_simulation()

    return simulation.get_status()


# ============================================================
# SIMULATION HISTORY
# ============================================================

@router.get("/simulation/history")
def simulation_history():
    simulation = get_simulation()

    return simulation.get_history()


@router.get("/history")
def history():
    simulation = get_simulation()

    return simulation.get_history()


# ============================================================
# DECISIONS
# ============================================================

@router.get("/decisions")
def get_decisions(
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    )
):
    simulation = get_simulation()

    decisions = simulation.decision_history[-limit:]

    return {
        "status": "success",
        "count": len(decisions),
        "decisions": decisions,
    }


# ============================================================
# CITIZEN MEMORIES
# ============================================================

@router.get("/citizens/{citizen_id}/memories")
def citizen_memories(citizen_id: str):
    simulation = get_simulation()

    citizen = simulation.citizen_engine.get_citizen(
        citizen_id
    )

    if citizen is None:
        raise HTTPException(
            status_code=404,
            detail="Citizen not found",
        )

    memories = simulation.get_citizen_memories(
        citizen_id
    )

    return {
        "status": "success",
        "citizen_id": citizen_id,
        "count": len(memories),
        "memories": memories,
    }


# ============================================================
# COGNITION
# ============================================================

@router.post("/citizens/{citizen_id}/think")
def citizen_think(
    citizen_id: str,
    situation: str,
):
    simulation = get_simulation()

    try:
        result = simulation.citizen_agent.think(
            citizen_id=citizen_id,
            situation=situation,
        )

        return {
            "status": "success",
            "citizen_id": citizen_id,
            "situation": situation,
            "reasoning": result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


# ============================================================
# KNOWLEDGE / RAG
# ============================================================

@router.get("/knowledge")
def get_knowledge():
    return {
        "status": "success",
        "count": knowledge_engine.count(),
        "documents": knowledge_engine.get_all(),
    }


@router.get("/knowledge/stats")
def knowledge_statistics():
    return {
        "status": "success",
        **knowledge_engine.statistics(),
    }


@router.get("/knowledge/search")
def search_knowledge(
    q: str = Query(
        ...,
        min_length=1,
    ),
    limit: int = Query(
        default=5,
        ge=1,
        le=20,
    ),
):
    result = knowledge_engine.query_rag(
        query=q,
        limit=limit,
    )

    return {
        "status": "success",
        **result,
    }


@router.post("/knowledge/ask")
def ask_knowledge(query: str = Query(..., min_length=1)):
    """Ask AETHER about its live civilization using RAG + local LLM."""
    simulation = get_simulation()
    return {
        "status": "success",
        **knowledge_engine.answer_question(
            query=query,
            world=simulation.world_engine.get_world(),
            citizens=simulation.citizen_engine.get_all(),
            events=simulation.events,
            decisions=simulation.decision_history,
        ),
    }


@router.post("/knowledge")
def add_knowledge(
    title: str,
    content: str,
    source: str = "aether",
    knowledge_type: str = "general",
    importance: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
    ),
):
    document = knowledge_engine.add_document(
        title=title,
        content=content,
        source=source,
        knowledge_type=knowledge_type,
        importance=importance,
    )

    return {
        "status": "success",
        "document": document,
    }


@router.delete("/knowledge/{document_id}")
def delete_knowledge(document_id: str):
    deleted = knowledge_engine.delete_document(
        document_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Knowledge document not found",
        )

    return {
        "status": "success",
        "deleted": document_id,
    }


# ============================================================
# CITIZEN KNOWLEDGE
# ============================================================

@router.get("/citizens/{citizen_id}/knowledge")
def citizen_knowledge(
    citizen_id: str,
    q: str = Query(
        ...,
        min_length=1,
    ),
    limit: int = Query(
        default=5,
        ge=1,
        le=10,
    ),
):
    simulation = get_simulation()

    citizen = simulation.citizen_engine.get_citizen(
        citizen_id
    )

    if citizen is None:
        raise HTTPException(
            status_code=404,
            detail="Citizen not found",
        )

    result = knowledge_engine.query_rag(
        query=q,
        limit=limit,
    )

    return {
        "status": "success",
        "citizen_id": citizen_id,
        **result,
    }


# ============================================================
# SYMBOLIC REASONING
# ============================================================

@router.get("/reasoning/citizen/{citizen_id}")
def citizen_symbolic_reasoning(
    citizen_id: str,
):
    simulation = get_simulation()

    try:
        result = simulation.citizen_agent.analyze_symbolically(
            citizen_id
        )

        return {
            "status": "success",
            "citizen_id": citizen_id,
            "reasoning": result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


# ============================================================
# A* SEARCH
# ============================================================

@router.get("/reasoning/search")
def reasoning_search(
    start: str,
    goal: str,
):
    simulation = get_simulation()

    from ..reasoning.search import astar

    world = simulation.world_engine.get_world()

    locations = {
        location.id: location
        for location in world.locations
    }

    if start not in locations:
        raise HTTPException(
            status_code=404,
            detail=f"Start location '{start}' not found",
        )

    if goal not in locations:
        raise HTTPException(
            status_code=404,
            detail=f"Goal location '{goal}' not found",
        )

    def neighbors(location_id):
        current = locations[location_id]

        result = []

        for other_id, other in locations.items():

            if other_id == location_id:
                continue

            distance = sqrt(
                (current.x - other.x) ** 2
                + (current.y - other.y) ** 2
            )

            result.append(
                (other_id, distance)
            )

        return result

    def heuristic(location_id, target_id):
        current = locations[location_id]
        target = locations[target_id]

        return sqrt(
            (current.x - target.x) ** 2
            + (current.y - target.y) ** 2
        )

    result = astar.search(
        start=start,
        goal=goal,
        neighbors=neighbors,
        heuristic=heuristic,
    )

    return {
        "status": "success",
        "algorithm": "A*",
        "start": start,
        "goal": goal,
        "result": result,
    }


# ============================================================
# FORWARD CHAINING
# ============================================================

@router.post("/reasoning/infer")
def symbolic_inference(
    facts: list[str],
):
    from ..reasoning.logic import rule_engine

    result = rule_engine.infer(
        facts=set(facts)
    )

    return {
        "status": "success",
        "algorithm": "forward_chaining",
        "input_facts": facts,
        "result": result,
    }


# ============================================================
# GOAL PLANNING
# ============================================================

@router.post("/reasoning/plan")
def symbolic_plan(
    current_state: list[str],
    goal: str,
):
    from ..reasoning.planner import goal_planner

    result = goal_planner.plan(
        current_state=set(current_state),
        goal=goal,
    )

    return {
        "status": "success",
        "algorithm": "goal_based_planning",
        "goal": goal,
        "result": result,
    }


# ============================================================
# BAYESIAN REASONING
# ============================================================

@router.post("/reasoning/bayesian")
def bayesian_inference(
    likelihoods: dict[str, float],
):
    from ..reasoning.bayesian import bayesian_reasoner

    posterior = bayesian_reasoner.update(
        likelihoods
    )

    most_likely = (
        max(
            posterior,
            key=posterior.get,
        )
        if posterior
        else None
    )

    return {
        "status": "success",
        "algorithm": "bayesian_inference",
        "likelihoods": likelihoods,
        "posterior": posterior,
        "most_likely": most_likely,
    }


# ============================================================
# AUTOMATION
# ============================================================

@router.get("/automation/events")
def automation_events():
    simulation = get_simulation()

    return {
        "status": "success",
        "events": simulation.automation_engine.events,
    }


@router.get("/automation/cooldowns")
def automation_cooldowns():
    simulation = get_simulation()

    return {
        "status": "success",
        "cooldowns": simulation.automation_engine.get_cooldown_status(),
    }