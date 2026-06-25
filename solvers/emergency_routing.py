"""Depot–hospital: Dijkstra primary, then Dijkstra again with primary edges locked (disjoint backup)."""

from __future__ import annotations

import heapq

import core.constants as constants
from core.city_graph import CityGraph
from core.edge import Coord, Edge, EdgeContext
from models.zone import ZoneType


class EmergencyRouter:
    """Two shortest paths; second avoids edges from the first."""

    def __init__(self, graph: CityGraph) -> None:
        self.graph = graph

    def _calculate_weight(self, node: Coord) -> float:
        """Step cost into node (residential cheaper)."""
        if self.graph.get_zone(node) is ZoneType.RESIDENTIAL:
            return constants.RESIDENTIAL_WEIGHT
        return constants.DEFAULT_WEIGHT

    def _dijkstra(self, start: Coord, target: Coord) -> list[Edge] | None:
        """Shortest path as edges, or None."""
        distances: dict[Coord, float] = {start: 0.0}
        previous: dict[Coord, tuple[Coord, Edge]] = {}
        heap: list[tuple[float, Coord]] = [(0.0, start)]
        visited: set[Coord] = set()

        while heap:
            current_distance, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current == target:
                path: list[Edge] = []
                cursor = target
                while cursor != start:
                    parent, edge = previous[cursor]
                    path.append(edge)
                    cursor = parent
                path.reverse()
                return path

            for neighbor in self.graph.get_neighbors(current):
                if neighbor in visited:
                    continue

                edge = self.graph.get_edge(current, neighbor)
                new_distance = current_distance + self._calculate_weight(neighbor)

                if new_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_distance
                    previous[neighbor] = (current, edge)
                    heapq.heappush(heap, (new_distance, neighbor))

        return None

    def find_routes(self) -> tuple[list[Edge], list[Edge]]:
        """(primary_edges, backup_edges)."""
        start = constants.DEPOT_COORD
        target = constants.HOSPITAL_COORD

        if start is None:
            raise ValueError("constants.DEPOT_COORD is not set.")
        if target is None:
            raise ValueError("constants.HOSPITAL_COORD is not set.")

        primary_route = self._dijkstra(start, target)
        if primary_route is None:
            raise ValueError("No primary route exists between depot and hospital.")

        with EdgeContext(primary_route):
            backup_route = self._dijkstra(start, target)

        if backup_route is None:
            raise ValueError("No edge-disjoint backup route exists.")

        return primary_route, backup_route
