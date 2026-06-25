"""A* shortest paths; respects impassable edges via get_neighbors; h = Manhattan × min edge weight."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import core.constants as constants
from core.city_graph import CityGraph
from core.edge import Coord, Edge

_MIN_EDGE_WEIGHT: float = constants.RESIDENTIAL_WEIGHT  # scales h so it stays ≤ true cost per hop


class PathNotFoundError(Exception):
    """No traversable path between src and dst."""


class AStarSolver:
    """A* on the shared graph; stateless between find_path calls."""

    def __init__(self, graph: CityGraph) -> None:
        self.graph = graph

    def find_path(self, src: Coord, dst: Coord) -> list[Edge]:
        """Lowest-cost path as edges; cost = edge.weight × risk_multiplier at neighbor."""
        if not self.graph.is_valid_coord(*src):
            raise ValueError(f"Source {src!r} is outside grid bounds.")
        if not self.graph.is_valid_coord(*dst):
            raise ValueError(f"Destination {dst!r} is outside grid bounds.")

        if src == dst:
            return []

        g_score: dict[Coord, float] = {src: 0.0}
        came_from: dict[Coord, tuple[Coord, Edge]] = {}
        counter = 0
        h = self.graph.manhattan(src, dst) * _MIN_EDGE_WEIGHT
        open_heap: list[tuple[float, int, Coord]] = [(h, counter, src)]

        closed: set[Coord] = set()

        while open_heap:
            f, _, current = heapq.heappop(open_heap)

            if current == dst:
                return self._reconstruct(came_from, dst)

            if current in closed:
                continue
            closed.add(current)

            for neighbor in self.graph.get_neighbors(current):
                if neighbor in closed:
                    continue

                edge = self.graph.get_edge(current, neighbor)

                risk_mult = (
                    self.graph.get_node_attr(neighbor, "risk_multiplier") or 1.0
                )
                tentative_g = g_score[current] + edge.weight * risk_mult

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = (current, edge)
                    h_n = self.graph.manhattan(neighbor, dst) * _MIN_EDGE_WEIGHT
                    f_score = tentative_g + h_n
                    counter += 1
                    heapq.heappush(open_heap, (f_score, counter, neighbor))

        raise PathNotFoundError(
            f"No traversable path from {src!r} to {dst!r}. "
            "Some road segments may be blocked (is_impassable=True)."
        )

    def _reconstruct(
        self,
        came_from: dict[Coord, tuple[Coord, Edge]],
        dst: Coord,
    ) -> list[Edge]:
        """Walk came_from from dst back to src."""
        path: list[Edge] = []
        node = dst
        while node in came_from:
            parent, edge = came_from[node]
            path.append(edge)
            node = parent
        path.reverse()
        return path
