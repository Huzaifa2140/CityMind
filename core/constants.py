"""Grid size, traversal weights, CSP cap, depot/hospital coords (set after layout)."""

from typing import Optional

GRID_WIDTH: int = 40
GRID_HEIGHT: int = 40

DEFAULT_WEIGHT: float = 1.0
RESIDENTIAL_WEIGHT: float = 0.8

MAX_ITERATIONS_CSP: int = 10_000

HOSPITAL_COORD: Optional[tuple[int, int]] = None
DEPOT_COORD: Optional[tuple[int, int]] = None
