"""Random initial zones + min-conflicts local search until constraints clear or cap hit."""

from __future__ import annotations

import random
from collections import deque

import core.constants as constants
from core.constants import MAX_ITERATIONS_CSP
from core.city_graph import CityGraph
from core.edge import Coord
from models.zone import ZoneType


class MinConflictsSolver:
    """Zoning CSP by iterative conflict reduction."""

    def __init__(self, graph: CityGraph) -> None:
        self.graph = graph
        self._all_nodes = graph.all_nodes()
        self._last_depot_count = 0

    def _initialize_random(self) -> set[Coord]:
        """Random zones; returns cells that start in conflict."""
        active_conflicts: set[Coord] = set()
        zones = list(ZoneType)

        for node in self._all_nodes:
            self.graph.set_zone(node, random.choice(zones))

        self._last_depot_count = sum(
            1
            for node in self._all_nodes
            if self.graph.get_zone(node) is ZoneType.DEPOT
        )

        for node in self._all_nodes:
            conflict_count = self._count_conflicts(node)
            self.graph.set_conflict_count(node, conflict_count)
            if conflict_count > 0:
                active_conflicts.add(node)

        return active_conflicts

    def _hop_distance_check(
        self,
        start_node: Coord,
        target_zones: set[ZoneType],
        max_hops: int,
    ) -> bool:
        """True if any target_zones cell within max_hops BFS."""
        queue: deque[tuple[Coord, int]] = deque([(start_node, 0)])
        visited: set[Coord] = {start_node}

        while queue:
            current, distance = queue.popleft()
            if self.graph.get_zone(current) in target_zones:
                return True
            if distance == max_hops:
                continue

            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return False

    def _count_conflicts(self, node: Coord) -> int:
        """Violation count for this cell’s zone vs neighbours and global depot rule."""
        conflicts = 0
        zone = self.graph.get_zone(node)

        for neighbor in self.graph.get_neighbors(node):
            neighbor_zone = self.graph.get_zone(neighbor)
            if (
                zone is ZoneType.INDUSTRIAL
                and neighbor_zone in {ZoneType.SCHOOL, ZoneType.HOSPITAL}
            ):
                conflicts += 1
            if (
                zone in {ZoneType.SCHOOL, ZoneType.HOSPITAL}
                and neighbor_zone is ZoneType.INDUSTRIAL
            ):
                conflicts += 1

        if zone is ZoneType.RESIDENTIAL and not self._hop_distance_check(
            node,
            {ZoneType.HOSPITAL},
            3,
        ):
            conflicts += 1

        if zone is ZoneType.POWER_PLANT and not self._hop_distance_check(
            node,
            {ZoneType.INDUSTRIAL},
            2,
        ):
            conflicts += 1

        depot_count = self._last_depot_count
        if depot_count == 0:
            conflicts += 1
        elif depot_count > 1 and zone is ZoneType.DEPOT:
            conflicts += depot_count - 1

        return conflicts

    def _update_local_conflicts(
        self,
        node: Coord,
        active_conflicts_set: set[Coord],
    ) -> None:
        """Recompute conflict counts for local patch (or whole grid if depot count changed)."""
        depot_count = sum(
            1
            for candidate in self._all_nodes
            if self.graph.get_zone(candidate) is ZoneType.DEPOT
        )

        if depot_count != self._last_depot_count:
            nodes_to_update = set(self._all_nodes)
            self._last_depot_count = depot_count
        else:
            nodes_to_update: set[Coord] = {node}
            queue: deque[tuple[Coord, int]] = deque([(node, 0)])
            visited: set[Coord] = {node}

            while queue:
                current, distance = queue.popleft()
                nodes_to_update.add(current)
                if distance == 3:
                    continue

                for neighbor in self.graph.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

        for candidate in nodes_to_update:
            conflict_count = self._count_conflicts(candidate)
            self.graph.set_conflict_count(candidate, conflict_count)
            if conflict_count > 0:
                active_conflicts_set.add(candidate)
            else:
                active_conflicts_set.discard(candidate)

    def solve(self) -> bool:
        """True if a full valid assignment was found."""
        active_conflicts = self._initialize_random()
        zones = list(ZoneType)
        iterations = 0

        while active_conflicts and iterations < MAX_ITERATIONS_CSP:
            node = random.choice(tuple(active_conflicts))
            active_conflicts.discard(node)

            current_zone = self.graph.get_zone(node)
            best_conflict_count: int | None = None
            best_zones: list[ZoneType] = []

            for zone in zones:
                self.graph.set_zone(node, zone)
                saved_depot_count = self._last_depot_count
                candidate_depot_count = saved_depot_count
                if current_zone is ZoneType.DEPOT:
                    candidate_depot_count -= 1
                if zone is ZoneType.DEPOT:
                    candidate_depot_count += 1
                self._last_depot_count = candidate_depot_count
                conflict_count = self._count_conflicts(node)
                self._last_depot_count = saved_depot_count
                if best_conflict_count is None or conflict_count < best_conflict_count:
                    best_conflict_count = conflict_count
                    best_zones = [zone]
                elif conflict_count == best_conflict_count:
                    best_zones.append(zone)

            self.graph.set_zone(node, current_zone)
            self.graph.set_zone(node, random.choice(best_zones))
            self._update_local_conflicts(node, active_conflicts)
            iterations += 1

        if active_conflicts:
            return False

        constants.DEPOT_COORD = next(
            node
            for node in self._all_nodes
            if self.graph.get_zone(node) is ZoneType.DEPOT
        )

        constants.HOSPITAL_COORD = None
        queue: deque[Coord] = deque([constants.DEPOT_COORD])
        visited: set[Coord] = {constants.DEPOT_COORD}

        while queue:
            current = queue.popleft()
            if self.graph.get_zone(current) is ZoneType.HOSPITAL:
                constants.HOSPITAL_COORD = current
                break

            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return constants.HOSPITAL_COORD is not None
