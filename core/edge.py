"""Grid edge: endpoints, weight, lock/flood flags."""

from __future__ import annotations

from core.constants import DEFAULT_WEIGHT, GRID_WIDTH

Coord = tuple[int, int]


class Edge:
    """Undirected edge; same object lives in both nodes’ edge lists."""

    node_a: Coord
    node_b: Coord
    weight: float
    is_locked: bool
    is_impassable: bool
    blocked_direction: str | None

    def __init__(
        self,
        node_a: Coord,
        node_b: Coord,
        weight: float = DEFAULT_WEIGHT,
    ) -> None:
        """Endpoints ordered by stable hash tie-break."""
        hash_a = node_a[0] * GRID_WIDTH + node_a[1]
        hash_b = node_b[0] * GRID_WIDTH + node_b[1]

        if hash_a <= hash_b:
            self.node_a = node_a
            self.node_b = node_b
        else:
            self.node_a = node_b
            self.node_b = node_a

        self.weight = weight
        self.is_locked = False
        self.is_impassable = False
        self.blocked_direction = None

    @property
    def is_traversable(self) -> bool:
        """False if locked (e.g. reserved for disjoint path) or impassable."""
        return not self.is_locked and not self.is_impassable

    def other(self, node: Coord) -> Coord:
        """The other endpoint; ValueError if node isn’t on this edge."""
        if node == self.node_a:
            return self.node_b
        if node == self.node_b:
            return self.node_a
        raise ValueError(f"Node {node!r} is not an endpoint of {self!r}.")

    def __repr__(self) -> str:
        return (
            "Edge("
            f"node_a={self.node_a!r}, "
            f"node_b={self.node_b!r}, "
            f"weight={self.weight!r}, "
            f"is_locked={self.is_locked!r}, "
            f"is_impassable={self.is_impassable!r}, "
            f"blocked_direction={self.blocked_direction!r}"
            ")"
        )


class EdgeContext:
    """Context manager: sets is_locked on given edges, clears on exit."""

    def __init__(self, edges_to_lock: list[Edge]) -> None:
        self.edges_to_lock = edges_to_lock

    def __enter__(self) -> None:
        for edge in self.edges_to_lock:
            edge.is_locked = True

    def __exit__(self, *args: object) -> None:
        """Always unlock."""
        try:
            return None
        finally:
            for edge in self.edges_to_lock:
                edge.is_locked = False
