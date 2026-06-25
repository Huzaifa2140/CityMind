"""Kruskal MST: force primary+backup edges in first, then cheapest remaining edges."""

from __future__ import annotations

import core.constants as constants
from core.city_graph import CityGraph
from core.edge import Coord, Edge
from models.zone import ZoneType


class DisjointSet:
    """Union-find with path compression + rank."""

    def __init__(self, nodes: list[Coord]) -> None:
        self.parent: dict[Coord, Coord] = {node: node for node in nodes}
        self.rank: dict[Coord, int] = {node: 0 for node in nodes}

    def find(self, node: Coord) -> Coord:
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1: Coord, node2: Coord) -> bool:
        """True if two components merged."""
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1 == root2:
            return False

        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1

        return True


class RoadNetworkPlanner:
    """MST over the grid with emergency corridors pinned first."""

    def __init__(self, graph: CityGraph) -> None:
        self.graph = graph

    def _edge_weight(self, edge: Edge) -> float:
        """Mean endpoint weight (residential = cheaper to pave)."""
        node_a_weight = (
            constants.RESIDENTIAL_WEIGHT
            if self.graph.get_zone(edge.node_a) is ZoneType.RESIDENTIAL
            else constants.DEFAULT_WEIGHT
        )
        node_b_weight = (
            constants.RESIDENTIAL_WEIGHT
            if self.graph.get_zone(edge.node_b) is ZoneType.RESIDENTIAL
            else constants.DEFAULT_WEIGHT
        )
        return (node_a_weight + node_b_weight) / 2.0

    def build_mst(
        self,
        primary_route: list[Edge],
        backup_route: list[Edge],
    ) -> list[Edge]:
        disjoint_set = DisjointSet(self.graph.all_nodes())
        final_mst: list[Edge] = []

        for edge in primary_route:
            final_mst.append(edge)
            disjoint_set.union(edge.node_a, edge.node_b)

        for edge in backup_route:
            final_mst.append(edge)
            disjoint_set.union(edge.node_a, edge.node_b)

        sorted_edges = sorted(self.graph.get_all_edges(), key=self._edge_weight)

        for edge in sorted_edges:
            if disjoint_set.union(edge.node_a, edge.node_b):
                final_mst.append(edge)

        return final_mst
