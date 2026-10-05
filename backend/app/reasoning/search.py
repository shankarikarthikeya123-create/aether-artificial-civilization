from heapq import heappop, heappush
from typing import Callable, Any


class AStarSearch:
    """
    A* graph search.

    f(n) = g(n) + h(n)

    g(n) = cost already spent
    h(n) = estimated cost to goal
    """

    def search(
        self,
        start: Any,
        goal: Any,
        neighbors: Callable[[Any], list[tuple[Any, float]]],
        heuristic: Callable[[Any, Any], float],
    ):

        open_set = []

        heappush(
            open_set,
            (heuristic(start, goal), 0, start),
        )

        came_from = {}
        cost_so_far = {
            start: 0
        }

        counter = 0

        while open_set:

            _, _, current = heappop(open_set)

            if current == goal:
                return self._reconstruct_path(
                    came_from,
                    current,
                    cost_so_far[current],
                )

            for next_node, movement_cost in neighbors(current):

                new_cost = (
                    cost_so_far[current]
                    + movement_cost
                )

                if (
                    next_node not in cost_so_far
                    or new_cost < cost_so_far[next_node]
                ):

                    cost_so_far[next_node] = new_cost
                    came_from[next_node] = current

                    priority = (
                        new_cost
                        + heuristic(next_node, goal)
                    )

                    counter += 1

                    heappush(
                        open_set,
                        (
                            priority,
                            counter,
                            next_node,
                        ),
                    )

        return {
            "found": False,
            "path": [],
            "cost": None,
        }

    @staticmethod
    def _reconstruct_path(
        came_from,
        current,
        cost,
    ):

        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        return {
            "found": True,
            "path": path,
            "cost": cost,
        }


class BestFirstSearch:

    def search(
        self,
        start: Any,
        goal: Any,
        neighbors: Callable[[Any], list[Any]],
        heuristic: Callable[[Any, Any], float],
    ):

        open_set = []

        heappush(
            open_set,
            (
                heuristic(start, goal),
                start,
            ),
        )

        came_from = {}
        visited = set()

        while open_set:

            _, current = heappop(open_set)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:

                path = [current]

                while current in came_from:
                    current = came_from[current]
                    path.append(current)

                path.reverse()

                return {
                    "found": True,
                    "path": path,
                }

            for next_node in neighbors(current):

                if next_node in visited:
                    continue

                if next_node not in came_from:
                    came_from[next_node] = current

                heappush(
                    open_set,
                    (
                        heuristic(next_node, goal),
                        next_node,
                    ),
                )

        return {
            "found": False,
            "path": [],
        }


astar = AStarSearch()
best_first = BestFirstSearch()