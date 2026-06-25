"""Grid graph: shared Edge objects, zones, per-node attrs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.edge import Coord, Edge

if TYPE_CHECKING:
    from models.zone import ZoneType


class CityGraph:
    """4-neighbour grid; (x,y) nodes."""

    width: int
    height: int
    _adjacency: dict[int, list[Edge]]
    _zone_map: dict[Coord, ZoneType]
    _conflict_counts: dict[Coord, int]
    _node_attrs: dict[Coord, dict[str, Any]]

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._adjacency = {}
        self._zone_map = {}
        self._conflict_counts = {}
        self._node_attrs = {}

    def add_edge(self, node_a: Coord, node_b: Coord, weight: float) -> Edge:
        """One undirected edge stored on both adjacency lists."""
        edge = Edge(node_a, node_b, weight)
        self._adjacency.setdefault(self._perfect_hash(*node_a), []).append(edge)
        self._adjacency.setdefault(self._perfect_hash(*node_b), []).append(edge)
        return edge

    def get_edges(self, node: Coord) -> list[Edge]:
        """All incident edges (includes blocked)."""
        return self._adjacency.get(self._perfect_hash(*node), [])

    def get_edge(self, node_a: Coord, node_b: Coord) -> Edge:
        for edge in self.get_edges(node_a):
            if edge.other(node_a) == node_b:
                return edge
        raise ValueError(f"No edge exists between {node_a!r} and {node_b!r}.")

    def get_all_edges(self) -> set[Edge]:
        """Unique edges (by id)."""
        seen: set[int] = set()
        edges: set[Edge] = set()

        for bucket in self._adjacency.values():
            for edge in bucket:
                edge_id = id(edge)
                if edge_id not in seen:
                    seen.add(edge_id)
                    edges.add(edge)

        return edges

    def get_neighbors(self, node: Coord) -> list[Coord]:
        """Neighbours reachable through traversable edges only."""
        return [
            edge.other(node)
            for edge in self.get_edges(node)
            if edge.is_traversable
        ]

    def get_zone(self, node: Coord) -> ZoneType | None:
        return self._zone_map.get(node)

    def set_zone(self, node: Coord, zone: ZoneType) -> None:
        self._zone_map[node] = zone

    def all_nodes(self) -> list[Coord]:
        """All cells row-major."""
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
        ]

    def get_node_attr(self, node: Coord, key: str) -> Any:
        return self._node_attrs.get(node, {}).get(key)

    def set_node_attr(self, node: Coord, key: str, value: Any) -> None:
        self._node_attrs.setdefault(node, {})[key] = value

    def get_conflict_count(self, node: Coord) -> int:
        return self._conflict_counts.get(node, 0)

    def set_conflict_count(self, node: Coord, count: int) -> None:
        self._conflict_counts[node] = count

    def increment_conflict(self, node: Coord, delta: int = 1) -> None:
        self._conflict_counts[node] = self.get_conflict_count(node) + delta

    def is_valid_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _perfect_hash(self, x: int, y: int) -> int:
        return x * self.width + y

    def manhattan(self, a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
