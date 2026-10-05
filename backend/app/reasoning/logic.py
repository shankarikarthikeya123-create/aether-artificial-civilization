from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Rule:
    name: str
    conditions: list[str]
    conclusion: str
    priority: float = 0.5


class RuleEngine:
    """
    Simple symbolic inference engine.

    Supports forward chaining:

    facts + rules -> new facts
    """

    def __init__(self):
        self.rules: list[Rule] = []

    def add_rule(
        self,
        name: str,
        conditions: list[str],
        conclusion: str,
        priority: float = 0.5,
    ):

        self.rules.append(
            Rule(
                name=name,
                conditions=conditions,
                conclusion=conclusion,
                priority=priority,
            )
        )

    def infer(
        self,
        facts: set[str] | list[str],
        max_iterations: int = 20,
    ):

        known = set(facts)
        fired_rules = []

        for _ in range(max_iterations):

            changed = False

            applicable = sorted(
                self.rules,
                key=lambda rule: rule.priority,
                reverse=True,
            )

            for rule in applicable:

                if rule.conclusion in known:
                    continue

                if all(
                    condition in known
                    for condition in rule.conditions
                ):

                    known.add(rule.conclusion)

                    fired_rules.append(
                        {
                            "rule": rule.name,
                            "conditions": rule.conditions,
                            "conclusion": rule.conclusion,
                        }
                    )

                    changed = True

            if not changed:
                break

        return {
            "facts": sorted(known),
            "new_facts": fired_rules,
        }

    def clear(self):
        self.rules.clear()


rule_engine = RuleEngine()

# ------------------------------------------------------------
# AETHER CIVILIZATION RULES
# ------------------------------------------------------------

rule_engine.add_rule(
    name="food_shortage",
    conditions=[
        "food_low",
        "population_active",
    ],
    conclusion="food_crisis",
    priority=1.0,
)

rule_engine.add_rule(
    name="water_shortage",
    conditions=[
        "water_low",
        "population_active",
    ],
    conclusion="water_crisis",
    priority=1.0,
)

rule_engine.add_rule(
    name="energy_shortage",
    conditions=[
        "energy_low",
        "population_active",
    ],
    conclusion="energy_crisis",
    priority=1.0,
)

rule_engine.add_rule(
    name="citizen_hungry",
    conditions=[
        "hunger_high",
    ],
    conclusion="food_required",
    priority=0.9,
)

rule_engine.add_rule(
    name="citizen_tired",
    conditions=[
        "energy_low_personal",
    ],
    conclusion="rest_required",
    priority=0.9,
)

rule_engine.add_rule(
    name="citizen_social_need",
    conditions=[
        "social_low",
    ],
    conclusion="social_interaction_required",
    priority=0.7,
)