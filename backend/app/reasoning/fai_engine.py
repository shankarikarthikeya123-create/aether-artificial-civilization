"""Unified FAI reasoning laboratory used by AETHER.

This module keeps the syllabus algorithms executable and composable.
It is deliberately dependency-free so the civilization can run locally.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Any, Callable


# ------------------------- Search -------------------------

def a_star_search(graph: dict[str, list[tuple[str, float]]], start: str, goal: str, heuristic: dict[str, float]) -> dict[str, Any]:
    frontier: list[tuple[float, float, str, list[str]]] = [(float(heuristic.get(start, 0)), 0.0, start, [start])]
    best: dict[str, float] = {start: 0.0}
    while frontier:
        frontier.sort(key=lambda x: x[0])
        _, cost, node, path = frontier.pop(0)
        if node == goal:
            return {"found": True, "path": path, "cost": cost, "expanded": len(best)}
        for nxt, edge in graph.get(node, []):
            new_cost = cost + float(edge)
            if new_cost < best.get(nxt, math.inf):
                best[nxt] = new_cost
                frontier.append((new_cost + float(heuristic.get(nxt, 0)), new_cost, nxt, path + [nxt]))
    return {"found": False, "path": [], "cost": None, "expanded": len(best)}


def best_first_search(graph: dict[str, list[str]], start: str, goal: str, heuristic: dict[str, float]) -> dict[str, Any]:
    frontier: list[tuple[float, str, list[str]]] = [(float(heuristic.get(start, 0)), start, [start])]
    visited: set[str] = set()
    while frontier:
        frontier.sort(key=lambda x: x[0])
        _, node, path = frontier.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return {"found": True, "path": path, "expanded": len(visited)}
        for nxt in graph.get(node, []):
            if nxt not in visited:
                frontier.append((float(heuristic.get(nxt, 0)), nxt, path + [nxt]))
    return {"found": False, "path": [], "expanded": len(visited)}


# ------------------------- Search -------------------------

def uniform_cost_search(graph: dict[str, list[tuple[str, float]]], start: str, goal: str) -> dict[str, Any]:
    frontier: list[tuple[float, str, list[str]]] = [(0.0, start, [start])]
    best: dict[str, float] = {start: 0.0}
    while frontier:
        frontier.sort(key=lambda x: x[0])
        cost, node, path = frontier.pop(0)
        if node == goal:
            return {"found": True, "path": path, "cost": cost, "expanded": len(best)}
        for nxt, edge in graph.get(node, []):
            new_cost = cost + float(edge)
            if new_cost < best.get(nxt, math.inf):
                best[nxt] = new_cost
                frontier.append((new_cost, nxt, path + [nxt]))
    return {"found": False, "path": [], "cost": None, "expanded": len(best)}


def depth_limited_search(graph: dict[str, list[str]], start: str, goal: str, limit: int) -> dict[str, Any]:
    expanded = 0
    def dfs(node: str, path: list[str], depth: int):
        nonlocal expanded
        expanded += 1
        if node == goal:
            return path
        if depth == 0:
            return None
        for nxt in graph.get(node, []):
            if nxt not in path:
                result = dfs(nxt, path + [nxt], depth - 1)
                if result:
                    return result
        return None
    path = dfs(start, [start], limit)
    return {"found": bool(path), "path": path or [], "expanded": expanded, "limit": limit}


def iterative_deepening_search(graph: dict[str, list[str]], start: str, goal: str, max_depth: int = 20) -> dict[str, Any]:
    total = 0
    for depth in range(max_depth + 1):
        result = depth_limited_search(graph, start, goal, depth)
        total += result["expanded"]
        if result["found"]:
            result["algorithm"] = "IDDFS"
            result["total_expanded"] = total
            return result
    return {"found": False, "path": [], "total_expanded": total}


def hill_climbing(start: str, goal: str, neighbors: Callable[[str], list[str]], heuristic: Callable[[str], float], max_steps: int = 50) -> dict[str, Any]:
    current = start
    path = [current]
    for _ in range(max_steps):
        if current == goal:
            return {"found": True, "path": path, "steps": len(path) - 1}
        options = [n for n in neighbors(current) if n not in path]
        if not options:
            break
        best = min(options, key=heuristic)
        if heuristic(best) >= heuristic(current):
            break
        current = best
        path.append(current)
    return {"found": current == goal, "path": path, "steps": len(path) - 1, "local_optimum": current != goal}


def beam_search(start: str, goal: str, neighbors: Callable[[str], list[str]], heuristic: Callable[[str], float], width: int = 2, max_depth: int = 20) -> dict[str, Any]:
    beam = [(start, [start])]
    expanded = 0
    for _ in range(max_depth + 1):
        candidates = []
        for node, path in beam:
            expanded += 1
            if node == goal:
                return {"found": True, "path": path, "expanded": expanded}
            for nxt in neighbors(node):
                if nxt not in path:
                    candidates.append((nxt, path + [nxt]))
        candidates.sort(key=lambda item: heuristic(item[0]))
        beam = candidates[:max(1, width)]
        if not beam:
            break
    return {"found": False, "path": [], "expanded": expanded}


# ------------------------- Games -------------------------

def minimax(tree: dict[str, Any], node: str, maximizing: bool = True) -> tuple[float, str | None]:
    data = tree[node]
    children = data.get("children", [])
    if not children:
        return float(data.get("value", 0)), node
    values = [(minimax(tree, child, not maximizing)[0], child) for child in children]
    return (max(values, key=lambda x: x[0]) if maximizing else min(values, key=lambda x: x[0]))


def alpha_beta(tree: dict[str, Any], node: str, depth: int, alpha: float = -math.inf, beta: float = math.inf, maximizing: bool = True) -> tuple[float, str | None, int]:
    data = tree[node]
    children = data.get("children", [])
    if depth == 0 or not children:
        return float(data.get("value", 0)), node, 1
    expanded = 1
    if maximizing:
        value, best = -math.inf, None
        for child in children:
            score, _, count = alpha_beta(tree, child, depth - 1, alpha, beta, False)
            expanded += count
            if score > value:
                value, best = score, child
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best, expanded
    value, best = math.inf, None
    for child in children:
        score, _, count = alpha_beta(tree, child, depth - 1, alpha, beta, True)
        expanded += count
        if score < value:
            value, best = score, child
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value, best, expanded


# ------------------------- CSP -------------------------

@dataclass
class CSPResult:
    assignment: dict[str, Any]
    backtracks: int


def solve_csp(variables: list[str], domains: dict[str, list[Any]], constraints: list[Callable[[dict[str, Any]], bool]]) -> CSPResult | None:
    assignment: dict[str, Any] = {}
    backtracks = 0

    def backtrack() -> bool:
        nonlocal backtracks
        if len(assignment) == len(variables):
            return True
        unassigned = [v for v in variables if v not in assignment]
        var = min(unassigned, key=lambda v: len([x for x in domains[v] if x not in assignment.values()]))
        for value in domains[var]:
            if value in assignment.values():
                continue
            assignment[var] = value
            if all(c(assignment) for c in constraints) and backtrack():
                return True
            assignment.pop(var)
            backtracks += 1
        return False

    if backtrack():
        return CSPResult(dict(assignment), backtracks)
    return None


# ------------------------- Logic -------------------------

def forward_chain(facts: set[str], rules: list[tuple[set[str], str]]) -> dict[str, Any]:
    known = set(facts)
    fired = []
    changed = True
    while changed:
        changed = False
        for conditions, conclusion in rules:
            if conditions.issubset(known) and conclusion not in known:
                known.add(conclusion)
                fired.append({"conditions": sorted(conditions), "conclusion": conclusion})
                changed = True
    return {"facts": sorted(known), "fired": fired}


def backward_chain(goal: str, facts: set[str], rules: list[tuple[set[str], str]]) -> dict[str, Any]:
    visited: set[str] = set()
    proof: list[str] = []
    def prove(target: str) -> bool:
        if target in facts:
            proof.append(target)
            return True
        if target in visited:
            return False
        visited.add(target)
        for conditions, conclusion in rules:
            if conclusion == target and all(prove(c) for c in conditions):
                proof.append(target)
                return True
        return False
    success = prove(goal)
    return {"proved": success, "goal": goal, "proof": list(reversed(proof))}


def propositional_resolution(clauses: list[set[str]], query: str) -> dict[str, Any]:
    # Refutation: add negated query and derive the empty clause.
    work = [set(c) for c in clauses] + [{f"not:{query}"}]
    seen = {frozenset(c) for c in work}
    while True:
        new: set[frozenset[str]] = set()
        current = list(seen)
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                a, b = set(current[i]), set(current[j])
                for lit in list(a):
                    opposite = lit[4:] if lit.startswith("not:") else f"not:{lit}"
                    if opposite in b:
                        resolvent = (a - {lit}) | (b - {opposite})
                        if not resolvent:
                            return {"entailed": True, "query": query}
                        new.add(frozenset(resolvent))
        if new.issubset(seen):
            return {"entailed": False, "query": query}
        seen |= new


# ------------------------- First-order logic / unification -------------------------

def unify(pattern: Any, value: Any, bindings: dict[str, Any] | None = None) -> dict[str, Any] | None:
    bindings = dict(bindings or {})
    if isinstance(pattern, str) and pattern.startswith("?"):
        bound = bindings.get(pattern)
        if bound is None:
            bindings[pattern] = value
            return bindings
        return bindings if bound == value else None
    if isinstance(pattern, (list, tuple)) and isinstance(value, (list, tuple)) and len(pattern) == len(value):
        for a, b in zip(pattern, value):
            bindings = unify(a, b, bindings)
            if bindings is None:
                return None
        return bindings
    return bindings if pattern == value else None


def fol_inference(facts: list[tuple[str, tuple[Any, ...]]], rules: list[dict[str, Any]], query: tuple[str, tuple[Any, ...]]) -> dict[str, Any]:
    known = set(facts)
    proofs = []
    changed = True
    while changed:
        changed = False
        for rule in rules:
            premise = rule["premise"]
            conclusion = rule["conclusion"]
            for fact in list(known):
                binding = unify(premise[1], fact[1]) if premise[0] == fact[0] else None
                if binding is not None:
                    args = tuple(binding.get(x, x) for x in conclusion[1])
                    derived = (conclusion[0], args)
                    if derived not in known:
                        known.add(derived)
                        proofs.append({"rule": rule.get("name", "rule"), "derived": derived})
                        changed = True
    return {"entailed": query in known, "query": query, "derived": proofs, "facts": sorted(known, key=str)}


def knowledge_based_agent(facts: set[str], rules: list[tuple[set[str], str]], goal: str) -> dict[str, Any]:
    inference = forward_chain(facts, rules)
    if goal in inference["facts"]:
        action = "execute_goal"
    elif "food_crisis" in inference["facts"]:
        action = "activate_food_protocol"
    elif "water_crisis" in inference["facts"]:
        action = "activate_water_protocol"
    elif "energy_crisis" in inference["facts"]:
        action = "activate_energy_protocol"
    else:
        action = "observe_and_learn"
    return {"action": action, "goal": goal, "inference": inference}


# ------------------------- Advanced planning -------------------------

def partial_order_plan(initial: set[str], goal: set[str], actions: list[dict[str, Any]]) -> dict[str, Any]:
    remaining = set(goal) - set(initial)
    selected = []
    available = list(actions)
    while remaining:
        candidates = [a for a in available if set(a.get("effects", [])) & remaining and set(a.get("preconditions", [])).issubset(initial | {e for s in selected for e in s.get("effects", [])})]
        if not candidates:
            return {"success": False, "plan": [a["name"] for a in selected], "unordered_goals": sorted(remaining)}
        action = max(candidates, key=lambda a: len(set(a.get("effects", [])) & remaining))
        selected.append(action)
        remaining -= set(action.get("effects", []))
    return {"success": True, "plan": [a["name"] for a in selected], "ordering_constraints": [[selected[i]["name"], selected[i+1]["name"]] for i in range(len(selected)-1)]}


def hierarchical_plan(goal: str, task_library: dict[str, list[str]]) -> dict[str, Any]:
    if goal not in task_library:
        return {"success": False, "goal": goal, "steps": []}
    return {"success": True, "goal": goal, "steps": task_library[goal], "method": "hierarchical decomposition"}


# ------------------------- Planning -------------------------

def state_space_plan(initial: set[str], goal: set[str], actions: list[dict[str, Any]], max_depth: int = 10) -> dict[str, Any]:
    queue = deque([(set(initial), [])])
    seen = {frozenset(initial)}
    while queue:
        state, plan = queue.popleft()
        if goal.issubset(state):
            return {"success": True, "plan": plan, "depth": len(plan)}
        if len(plan) >= max_depth:
            continue
        for action in actions:
            pre = set(action.get("preconditions", []))
            if pre.issubset(state):
                nxt = (state - set(action.get("delete", []))) | set(action.get("effects", []))
                key = frozenset(nxt)
                if key not in seen:
                    seen.add(key)
                    queue.append((nxt, plan + [action["name"]]))
    return {"success": False, "plan": [], "depth": None}


# ------------------------- Learning -------------------------

class QLearner:
    def __init__(self, alpha: float = 0.2, gamma: float = 0.9):
        self.alpha, self.gamma = alpha, gamma
        self.q: dict[tuple[str, str], float] = {}

    def update(self, state: str, action: str, reward: float, next_state: str, next_actions: list[str]) -> float:
        old = self.q.get((state, action), 0.0)
        future = max((self.q.get((next_state, a), 0.0) for a in next_actions), default=0.0)
        new = old + self.alpha * (reward + self.gamma * future - old)
        self.q[(state, action)] = new
        return new

    def best_action(self, state: str, actions: list[str]) -> str | None:
        if not actions:
            return None
        return max(actions, key=lambda a: self.q.get((state, a), 0.0))


def decision_tree_predict(rows: list[dict[str, Any]], features: list[str], target: str, sample: dict[str, Any]) -> Any:
    if not rows:
        return None
    # Compact ID3-style prediction: choose the feature with highest information gain,
    # then classify by the matching branch. This keeps the learning demonstration local.
    def entropy(items):
        counts = {}
        for row in items:
            counts[row[target]] = counts.get(row[target], 0) + 1
        total = len(items)
        return -sum((n/total) * math.log2(n/total) for n in counts.values() if n)
    base = entropy(rows)
    gains = {}
    for feature in features:
        groups = {}
        for row in rows:
            groups.setdefault(row[feature], []).append(row)
        remainder = sum(len(g)/len(rows) * entropy(g) for g in groups.values())
        gains[feature] = base - remainder
    best = max(gains, key=gains.get)
    matching = [r for r in rows if r[best] == sample.get(best)]
    if not matching:
        matching = rows
    counts = {}
    for row in matching:
        counts[row[target]] = counts.get(row[target], 0) + 1
    return max(counts, key=counts.get)


# ------------------------- Strategic decision-making -------------------------

def monte_carlo_tree_search(tree: dict[str, Any], root: str, simulations: int = 80) -> dict[str, Any]:
    children = tree.get(root, {}).get("children", [])
    if not children:
        return {"action": root, "value": float(tree.get(root, {}).get("value", 0)), "simulations": 0}

    def rollout(node: str) -> float:
        current = node
        for _ in range(20):
            options = tree.get(current, {}).get("children", [])
            if not options:
                return float(tree.get(current, {}).get("value", 0))
            current = random.choice(options)
        return float(tree.get(current, {}).get("value", 0))

    scores = {child: [] for child in children}
    for _ in range(max(1, simulations)):
        for child in children:
            scores[child].append(rollout(child))
    averages = {k: sum(v) / len(v) for k, v in scores.items()}
    best = max(averages, key=averages.get)
    return {"action": best, "value": averages[best], "scores": averages, "simulations": simulations}


def strategic_crisis_decision(resources: dict[str, float], population: int) -> dict[str, Any]:
    food = resources.get("food", 0.0)
    water = resources.get("water", 0.0)
    energy = resources.get("energy", 0.0)
    tree = {
        "root": {"children": ["food_response", "water_response", "energy_response"]},
        "food_response": {"children": ["food_a", "food_b"]},
        "water_response": {"children": ["water_a", "water_b"]},
        "energy_response": {"children": ["energy_a", "energy_b"]},
        "food_a": {"value": food * 1.0 + population * 0.05},
        "food_b": {"value": food * 0.8 + population * 0.02},
        "water_a": {"value": water * 1.0 + population * 0.04},
        "water_b": {"value": water * 0.8 + population * 0.03},
        "energy_a": {"value": energy * 1.0 + population * 0.03},
        "energy_b": {"value": energy * 0.8 + population * 0.02},
    }
    minimax_value, minimax_action = minimax(tree, "root")
    alpha_value, alpha_action, alpha_expanded = alpha_beta(tree, "root", depth=2)
    mcts = monte_carlo_tree_search(tree, "root", simulations=40)
    csp = solve_csp(
        ["food_team", "water_team", "energy_team"],
        {"food_team": ["A", "B", "C"], "water_team": ["A", "B", "C"], "energy_team": ["A", "B", "C"]},
        [lambda a: len(set(a.values())) == len(a)],
    )
    action = alpha_action or mcts["action"] or minimax_action
    return {
        "recommended_action": action,
        "minimax": {"value": minimax_value, "action": minimax_action},
        "alpha_beta": {"value": alpha_value, "action": alpha_action, "expanded": alpha_expanded},
        "mcts": mcts,
        "csp": {"assignment": csp.assignment, "backtracks": csp.backtracks} if csp else None,
    }


# ------------------------- Knowledge representation -------------------------

def build_aether_ontology() -> dict[str, Any]:
    return {
        "Citizen": ["needs", "goals", "beliefs", "skills", "memories"],
        "Location": ["resources", "population", "infrastructure"],
        "Resource": ["amount", "capacity", "consumption"],
        "Crisis": ["severity", "resource", "affected_population", "response"],
        "relationships": {"Citizen-livesIn-Location": True, "Location-has-Resource": True, "Crisis-affects-Location": True},
    }


def represent_frame(entity: str, slots: dict[str, Any]) -> dict[str, Any]:
    return {"frame": entity, "slots": slots}


# ------------------------- Bayesian + multi-agent -------------------------

def bayes_update(priors: dict[str, float], likelihoods: dict[str, float]) -> dict[str, float]:
    weighted = {k: max(0.0, priors.get(k, 0.0)) * max(0.0, likelihoods.get(k, 0.0)) for k in priors}
    total = sum(weighted.values())
    return {k: (v / total if total else 0.0) for k, v in weighted.items()}


def coordinate_agents(agents: list[dict[str, Any]], tasks: list[str]) -> dict[str, Any]:
    assignments = {}
    remaining = list(tasks)
    ranked = sorted(agents, key=lambda a: float(a.get("capability", 0.5)), reverse=True)
    for agent in ranked:
        if not remaining:
            break
        task = remaining.pop(0)
        assignments[agent["id"]] = task
    return {"assignments": assignments, "unassigned": remaining}


def run_demo() -> dict[str, Any]:
    graph = {
        "capital": [("food", 2), ("water", 3), ("energy", 4)],
        "food": [("farm", 2)],
        "water": [("reservoir", 1)],
        "energy": [("grid", 2)],
        "farm": [("hub", 2)],
        "reservoir": [("hub", 2)],
        "grid": [("hub", 1)],
        "hub": [],
    }
    unweighted = {k: [n for n, _ in v] for k, v in graph.items()}
    heuristic = {"capital": 4, "food": 2, "water": 2, "energy": 2, "farm": 1, "reservoir": 1, "grid": 1, "hub": 0}
    return {
        "search": {
            "ucs": uniform_cost_search(graph, "capital", "hub"),
            "a_star": a_star_search(graph, "capital", "hub", heuristic),
            "best_first": best_first_search(unweighted, "capital", "hub", heuristic),
            "iddfs": iterative_deepening_search(unweighted, "capital", "hub"),
            "hill_climbing": hill_climbing("capital", "hub", lambda n: unweighted.get(n, []), lambda n: heuristic.get(n, 99)),
            "beam_search": beam_search("capital", "hub", lambda n: unweighted.get(n, []), lambda n: heuristic.get(n, 99)),
        },
        "logic": forward_chain({"food_low", "population_active"}, [({"food_low", "population_active"}, "food_crisis"), ({"food_crisis"}, "activate_emergency_protocol")]),
        "planning": state_space_plan({"food_low"}, {"food_acquired"}, [
            {"name": "find_food", "preconditions": ["food_low"], "effects": ["food_found"]},
            {"name": "acquire_food", "preconditions": ["food_found"], "effects": ["food_acquired"]},
        ]),
        "bayesian": bayes_update({"shortage": 0.4, "normal": 0.6}, {"shortage": 0.9, "normal": 0.2}),
        "multi_agent": coordinate_agents([
            {"id": "city_ai", "capability": 0.95},
            {"id": "resource_ai", "capability": 0.85},
            {"id": "citizen_ai", "capability": 0.75},
        ], ["diagnose_crisis", "allocate_resources", "coordinate_citizens"]),
    }
