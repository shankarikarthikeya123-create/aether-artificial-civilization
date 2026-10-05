from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class PlanStep:
    action: str
    description: str
    cost: float = 0.0


class GoalPlanner:
    """
    Goal-oriented state-space planner.

    Converts:

        current state + goal

    into:

        ordered sequence of actions
    """

    def __init__(self):
        self.actions: dict[str, dict[str, Any]] = {}

    def add_action(
        self,
        name: str,
        preconditions: list[str],
        effects: list[str],
        cost: float = 1.0,
    ):

        self.actions[name] = {
            "preconditions": set(preconditions),
            "effects": set(effects),
            "cost": cost,
        }

    def plan(
        self,
        current_state: set[str],
        goal: str,
        max_steps: int = 10,
    ):

        state = set(current_state)
        steps = []

        if goal in state:
            return {
                "success": True,
                "goal": goal,
                "steps": [],
            }

        for _ in range(max_steps):

            candidates = []

            for name, action in self.actions.items():

                preconditions = action[
                    "preconditions"
                ]

                if not preconditions.issubset(state):
                    continue

                effects = action["effects"]

                useful_effects = effects - state

                if not useful_effects:
                    continue

                candidates.append(
                    (
                        len(useful_effects),
                        -action["cost"],
                        name,
                    )
                )

            if not candidates:
                break

            candidates.sort(
                reverse=True
            )

            _, _, selected_name = candidates[0]

            selected = self.actions[
                selected_name
            ]

            steps.append(
                PlanStep(
                    action=selected_name,
                    description=(
                        f"Execute {selected_name}"
                    ),
                    cost=selected["cost"],
                )
            )

            state.update(
                selected["effects"]
            )

            if goal in state:

                return {
                    "success": True,
                    "goal": goal,
                    "steps": [
                        {
                            "action": step.action,
                            "description": step.description,
                            "cost": step.cost,
                        }
                        for step in steps
                    ],
                }

        return {
            "success": False,
            "goal": goal,
            "steps": [
                {
                    "action": step.action,
                    "description": step.description,
                    "cost": step.cost,
                }
                for step in steps
            ],
        }


goal_planner = GoalPlanner()

# ------------------------------------------------------------
# AETHER ACTIONS
# ------------------------------------------------------------

goal_planner.add_action(
    name="find_food",
    preconditions=[
        "food_required",
    ],
    effects=[
        "food_found",
    ],
    cost=1,
)

goal_planner.add_action(
    name="acquire_food",
    preconditions=[
        "food_found",
    ],
    effects=[
        "food_acquired",
    ],
    cost=1,
)

goal_planner.add_action(
    name="consume_food",
    preconditions=[
        "food_acquired",
    ],
    effects=[
        "hunger_satisfied",
    ],
    cost=1,
)

goal_planner.add_action(
    name="find_rest_area",
    preconditions=[
        "rest_required",
    ],
    effects=[
        "rest_area_found",
    ],
    cost=1,
)

goal_planner.add_action(
    name="rest",
    preconditions=[
        "rest_area_found",
    ],
    effects=[
        "energy_restored",
    ],
    cost=1,
)

goal_planner.add_action(
    name="find_social_group",
    preconditions=[
        "social_interaction_required",
    ],
    effects=[
        "social_group_found",
    ],
    cost=1,
)

goal_planner.add_action(
    name="socialize",
    preconditions=[
        "social_group_found",
    ],
    effects=[
        "social_need_satisfied",
    ],
    cost=1,
)