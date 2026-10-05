"""Safe read-oriented MCP tools for AETHER."""
from typing import Any

def world_state(simulation) -> dict[str, Any]:
    return simulation.world_engine.get_world().model_dump()

def citizen_snapshot(simulation, citizen_id: str) -> dict[str, Any]:
    citizen = simulation.citizen_engine.get_citizen(citizen_id)
    return {"error": "Citizen not found"} if citizen is None else citizen.model_dump()

def knowledge_search(query: str, limit: int = 5) -> dict[str, Any]:
    from ..knowledge.engine import knowledge_engine
    return knowledge_engine.query_rag(query=query, limit=limit)

def resource_snapshot(simulation) -> dict[str, float]:
    world = simulation.world_engine.get_world()
    return {r.name: r.amount for r in world.resources}

def inspect_location(simulation, location_id: str) -> dict[str, Any]:
    world = simulation.world_engine.get_world()
    location = next((x for x in world.locations if x.id == location_id), None)
    if location is None:
        return {"error": "Location not found"}
    citizens = [c.model_dump() for c in simulation.citizen_engine.get_all() if c.location_id == location_id and c.alive]
    return {"location": location.model_dump(), "citizens": citizens}

def execute_citizen_action(simulation, citizen_id: str, action: str) -> dict[str, Any]:
    allowed = {"eat", "rest", "socialize", "work", "research", "explore"}
    if action not in allowed:
        return {"error": f"Action must be one of {sorted(allowed)}"}
    citizen = simulation.citizen_engine.get_citizen(citizen_id)
    if citizen is None:
        return {"error": "Citizen not found"}
    before = {"hunger": citizen.needs.hunger, "energy": citizen.needs.energy, "social": citizen.needs.social, "money": citizen.needs.money}
    result = simulation._execute_action(citizen, action)
    simulation._learn_from_action(citizen, action, result, before)
    simulation._create_memory(citizen, action, result, "MCP tool action")
    simulation.events.append({"type": "tool_action", "message": f"MCP executed {action} for {citizen.name}.", "tick": simulation.tick_count})
    return {"citizen": citizen.name, "action": action, "result": result, "needs": citizen.needs.model_dump()}

def civilization_summary(simulation) -> dict[str, Any]:
    world = simulation.world_engine.get_world()
    return {
        "civilization": world.name, "population": world.population,
        "day": world.day, "time": f"{world.hour:02d}:{world.minute:02d}",
        "resources": {r.name: r.amount for r in world.resources},
        "recent_events": simulation.events[-5:],
        "recent_decisions": simulation.decision_history[-5:],
    }
