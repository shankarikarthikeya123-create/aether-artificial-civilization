"""Local multi-agent civilization council, CrewAI-ready and provider-independent."""
ROLES = {
    "resource_agent": "Monitor food, water and energy and flag shortages.",
    "citizen_agent": "Evaluate citizen needs, memories and welfare.",
    "planning_agent": "Convert facts into a feasible action plan.",
    "governance_agent": "Check population-wide consequences and priorities.",
    "learning_agent": "Review outcomes and learning signals.",
}

def run_council(simulation):
    world = simulation.world_engine.get_world()
    resources = {r.name: r.amount for r in world.resources}
    thresholds = {"food": 200, "water": 400, "energy": 300}
    alerts = [f"{n} low" for n, a in resources.items() if n in thresholds and a < thresholds[n]]
    citizens = simulation.citizen_engine.get_all()
    hungry = sum(1 for c in citizens if c.needs.hunger >= 65)
    tired = sum(1 for c in citizens if c.needs.energy <= 35)
    social = sum(1 for c in citizens if c.needs.social <= 35)
    if alerts: priority = "stabilize_resources"
    elif hungry > max(5, world.population * .2): priority = "food_welfare"
    elif tired > max(5, world.population * .2): priority = "rest_and_energy"
    elif social > max(5, world.population * .2): priority = "social_welfare"
    else: priority = "continue_development"
    return {
        "framework": "AETHER Multi-Agent Council",
        "roles": ROLES,
        "agents": [
            {"role": "resource_agent", "finding": alerts or ["resources stable"]},
            {"role": "citizen_agent", "finding": {"hungry": hungry, "tired": tired, "social_need": social}},
            {"role": "governance_agent", "finding": {"population": world.population, "priority": priority}},
            {"role": "planning_agent", "finding": f"Recommend plan for {priority}"},
            {"role": "learning_agent", "finding": f"{len(simulation.learning_history)} learning records available"},
        ],
        "consensus": {"priority": priority, "confidence": .9 if alerts else .75},
    }
