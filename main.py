"""CityMind sim: layout → roads → GA ambulances → risk ML → 20 steps (A*, floods); one shared CityGraph."""

from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

# UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import core.constants as constants
from core.city_graph import CityGraph
from core.edge import Coord, Edge
from models.zone import ZoneType
from solvers.ambulance_placement import AmbulancePlacementGA
from solvers.astar import AStarSolver, PathNotFoundError
from solvers.emergency_routing import EmergencyRouter
from solvers.min_conflicts import MinConflictsSolver
from solvers.risk_predictor import RISK_MULTIPLIERS, RiskPredictor
from solvers.road_network import RoadNetworkPlanner

ZONE_SYMBOLS: dict[ZoneType, str] = {
    ZoneType.RESIDENTIAL: "R",
    ZoneType.INDUSTRIAL:  "I",
    ZoneType.SCHOOL:      "S",
    ZoneType.POWER_PLANT: "W",
    ZoneType.HOSPITAL:    "H",
    ZoneType.DEPOT:       "D",
}

SIM_STEPS:        int = 20
NUM_AMBULANCES:   int = 3
FLOOD_PROB:       float = 0.30
RISK_UPDATE_STEP: int = 10
SEED:             int = 42


@dataclass
class LogEntry:
    step:    int
    event:   str
    detail:  str
    data:    dict = field(default_factory=dict)


@dataclass
class SimState:
    """Per-tick state for the UI."""
    step:              int   = 0
    ambulance_positions: list[Coord] = field(default_factory=list)
    active_floods:     list[tuple[Coord, Coord]] = field(default_factory=list)
    last_route:        list[Edge]  = field(default_factory=list)
    last_route_src:    Coord | None = None
    last_route_dst:    Coord | None = None
    risk_counts:       dict[str, int] = field(default_factory=dict)
    event_log:         list[LogEntry] = field(default_factory=list)


class CitySimulation:
    """Wires CSP, roads, GA, risk, and sim steps together."""

    def __init__(self, seed: int = SEED) -> None:
        self._rng = random.Random(seed)
        self.graph: CityGraph = CityGraph(
            constants.GRID_WIDTH, constants.GRID_HEIGHT
        )
        self._build_grid_edges()

        self.ga:        AmbulancePlacementGA | None = None
        self.predictor: RiskPredictor        | None = None
        self.astar:     AStarSolver          | None = None

        self._flooded_edges: list[Edge] = []
        self._step: int = 0
        self.state: SimState = SimState()
        self.event_log: list[LogEntry] = []

        self.mst_edges:     list[Edge] = []
        self.primary_route: list[Edge] = []
        self.backup_route:  list[Edge] = []
        self.active_routes: list[dict] = []

    def _build_grid_edges(self) -> None:
        """Full grid of unit-weight orthogonal edges."""
        for x in range(self.graph.width):
            for y in range(self.graph.height):
                if x + 1 < self.graph.width:
                    self.graph.add_edge((x, y), (x + 1, y), constants.DEFAULT_WEIGHT)
                if y + 1 < self.graph.height:
                    self.graph.add_edge((x, y), (x, y + 1), constants.DEFAULT_WEIGHT)

    def _log(self, event: str, detail: str, **data: Any) -> LogEntry:
        entry = LogEntry(step=self._step, event=event, detail=detail, data=data)
        self.event_log.append(entry)
        print(f"  [Step {self._step:02d}] [{event}] {detail}")
        return entry

    def _run_layout(self) -> None:
        print("\n[Phase 0] Challenge 1 — City Layout (Min-Conflicts CSP)...")
        solver = MinConflictsSolver(self.graph)
        t = perf_counter()
        ok = solver.solve()
        elapsed = perf_counter() - t
        assert ok, "Layout solver failed — cannot proceed."
        print(f"  Layout solved in {elapsed:.2f}s")
        print(f"  Depot    @ {constants.DEPOT_COORD}")
        print(f"  Hospital @ {constants.HOSPITAL_COORD}")

    def _run_road_network(self) -> None:
        print("\n[Phase 1] Challenge 2 — Road Network (Kruskal + Double-Dijkstra)...")
        router = EmergencyRouter(self.graph)
        primary, backup = router.find_routes()
        planner = RoadNetworkPlanner(self.graph)
        road_network = planner.build_mst(primary, backup)
        self.mst_edges     = road_network
        self.primary_route = primary
        self.backup_route  = backup
        print(f"  Road network built: {len(road_network)} edges in MST.")
        print(f"  Primary route      : {len(primary)} hops")
        print(f"  Backup route       : {len(backup)} hops")

    def _run_placement(self, label: str = "initial") -> list[Coord]:
        tag = "PLACEMENT" if label == "initial" else "REPLANNING"
        print(f"\n[Phase 2] Challenge 3 — Ambulance Placement (GA Minimax) [{label}]...")
        if self.ga is None:
            self.ga = AmbulancePlacementGA(
                self.graph,
                num_ambulances=NUM_AMBULANCES,
                max_generations=500,
            )
        else:
            self.ga.refresh()

        positions, score, history = self.ga.solve()
        improvement = history[0] - history[-1]
        print(f"  Positions : {positions}")
        print(f"  Minimax   : {score} hops  (improved {improvement} hops over {len(history)} generations)")
        self._log(tag, f"Ambulances placed at {positions}. Minimax={score} hops.",
                  positions=positions, score=score, history=history)
        self.state.ambulance_positions = list(positions)
        return positions

    def _run_risk_prediction(self, label: str = "initial") -> dict:
        tag = "RISK_UPDATE" if label != "initial" else "RISK_INIT"
        print(f"\n[Phase 3] Challenge 5 — Crime Risk Prediction [{label}]...")
        if self.predictor is None:
            self.predictor = RiskPredictor(self.graph, seed=SEED)
            stats = self.predictor.solve()
        else:
            stats = self.predictor.recalculate()

        acc = stats["train_accuracy"]
        rc  = stats["risk_counts"]
        print(f"  DT accuracy    : {acc*100:.1f}%")
        print(f"  Risk distribution: Low={rc.get('Low',0)}  "
              f"Medium={rc.get('Medium',0)}  High={rc.get('High',0)}")
        self._log(tag,
                  f"Risk updated. DT acc={acc*100:.1f}%. "
                  f"High={rc.get('High',0)} Medium={rc.get('Medium',0)} Low={rc.get('Low',0)}",
                  risk_counts=rc, accuracy=acc)
        self.state.risk_counts = dict(rc)
        return stats

    def _trigger_flood(self) -> list[Edge]:
        """Randomly blocks 1–3 edges."""
        all_edges = list(self.graph.get_all_edges())
        passable  = [e for e in all_edges if not e.is_impassable]
        if not passable:
            return []
        count   = self._rng.randint(1, min(3, len(passable)))
        flooded = self._rng.sample(passable, count)
        for edge in flooded:
            edge.is_impassable = True
        self._flooded_edges.extend(flooded)
        pairs = [(e.node_a, e.node_b) for e in flooded]
        self.state.active_floods = [(e.node_a, e.node_b) for e in self._flooded_edges]
        self._log("FLOOD",
                  f"{count} road(s) flooded: {pairs}",
                  flooded_edges=pairs)
        return flooded

    def _clear_floods(self) -> None:
        """Clears flood flags from all flooded edges."""
        for edge in self._flooded_edges:
            edge.is_impassable = False
        count = len(self._flooded_edges)
        self._flooded_edges.clear()
        self.state.active_floods = []
        if count:
            self._log("FLOOD_CLEAR", f"{count} flooded road(s) restored.")

    def _nearest_ambulance(self, dst: Coord) -> tuple[Coord, int]:
        """Closest ambulance to dst by Manhattan distance."""
        best_pos, best_idx, best_dist = None, 0, float("inf")
        for idx, pos in enumerate(self.state.ambulance_positions):
            d = abs(pos[0] - dst[0]) + abs(pos[1] - dst[1])
            if d < best_dist:
                best_dist, best_pos, best_idx = d, pos, idx
        return (best_pos or constants.DEPOT_COORD, best_idx)

    def _pick_spread_target(
        self, zone_type: ZoneType, already_picked: list[Coord]
    ) -> Coord | None:
        """Picks a zone cell spread out from previous picks (first: median-ish by x+y)."""
        candidates = [
            n for n in self.graph.all_nodes()
            if self.graph.get_zone(n) is zone_type
        ]
        if not candidates:
            return None
        if not already_picked:
            return sorted(candidates, key=lambda n: (n[0] + n[1]))[len(candidates) // 2]
        def min_dist(cand: Coord) -> int:
            return min(abs(cand[0] - p[0]) + abs(cand[1] - p[1]) for p in already_picked)
        return max(candidates, key=min_dist)

    def _route_to_civilian(self, src: Coord, dst: Coord, zone_name: str,
                           amb_idx: int = -1) -> list[Edge]:
        """A* from src to dst; logs route or NO_ROUTE."""
        if self.astar is None:
            self.astar = AStarSolver(self.graph)
        label = f"Amb{amb_idx+1}" if amb_idx >= 0 else "Depot"
        try:
            path = self.astar.find_path(src, dst)
            cost = sum(
                e.weight * (self.graph.get_node_attr(e.node_b, "risk_multiplier") or 1.0)
                for e in path
            )
            self._log("ROUTE",
                      f"A* routed {label}->{zone_name} @ {dst}: "
                      f"{len(path)} hops, cost={cost:.2f}",
                      src=src, dst=dst, hops=len(path), cost=cost)
            self.state.last_route     = path
            self.state.last_route_src = src
            self.state.last_route_dst = dst
            return path
        except PathNotFoundError:
            self._log("NO_ROUTE",
                      f"{label}->{zone_name} @ {dst} BLOCKED.",
                      src=src, dst=dst)
            self.state.last_route     = []
            self.state.last_route_src = src
            self.state.last_route_dst = dst
            return []

    def setup(self) -> SimState:
        """Layout, roads, ambulances, risk — call once before step()."""
        print("=" * 60)
        print("CityMind — System Initialisation")
        print("=" * 60)

        self._run_layout()
        self._run_road_network()
        self._run_placement("initial")
        self._run_risk_prediction("initial")

        self.astar = AStarSolver(self.graph)
        self._step = 0
        self.state.step = 0

        print("\n" + "=" * 60)
        print("Initialisation complete. Starting 20-step simulation.")
        print("=" * 60)
        return self.state

    def step(self) -> SimState:
        """One tick: optional risk+GA at step 10, random flood, sequential route advance, reroute if flooded."""
        self._step += 1
        self.state.step = self._step

        print(f"\n{'─'*50}")
        print(f"  SIMULATION STEP {self._step:02d} / {SIM_STEPS}")
        print(f"{'─'*50}")

        if self._step == RISK_UPDATE_STEP:
            self._run_risk_prediction("recalculate")
            self._run_placement("risk-adjusted")

        flood_this_step = self._rng.random() < FLOOD_PROB
        flooded = []
        if flood_this_step:
            flooded = self._trigger_flood()

        if self._step == 1:
            self._build_sequential_route()
        else:
            steps_per_leg = max(1, SIM_STEPS // 5)
            if self._step % steps_per_leg == 0:
                self._advance_sequential_leg()

        if flood_this_step:
            self._recalculate_current_leg()
        elif self._step != 1 and self._step % max(1, SIM_STEPS // 5) != 0:
            self._log("NO_EVENT", "No flood event this step.")

        self.state.event_log = list(self.event_log)
        return self.state

    def _path_to_nodes(self, src: Coord, path: list) -> list[list[int]]:
        """Edge list → [[x,y], ...] polyline."""
        if not path:
            return []
        nodes = [[src[0], src[1]]]
        cur = src
        for edge in path:
            nxt = edge.node_b if edge.node_a == cur else edge.node_a
            nodes.append([nxt[0], nxt[1]])
            cur = nxt
        return nodes

    def _build_sequential_route(self) -> None:
        """Chains H→S→R→I→W with spread targets; one team, legs stored in sequential_route."""
        civilian_zones = [
            ZoneType.HOSPITAL,
            ZoneType.SCHOOL,
            ZoneType.RESIDENTIAL,
            ZoneType.INDUSTRIAL,
            ZoneType.POWER_PLANT,
        ]
        already_picked: list[Coord] = []
        targets: list[tuple[Coord, ZoneType]] = []
        for zone_type in civilian_zones:
            target = self._pick_spread_target(zone_type, already_picked)
            if target is not None:
                already_picked.append(target)
                targets.append((target, zone_type))

        if not targets:
            return

        first_target = targets[0][0]
        team_pos, team_idx = self._nearest_ambulance(first_target)
        self.seq_team_amb_idx: int = team_idx
        self.seq_current_leg: int = 0
        self.sequential_route: list[dict] = []

        current_pos = team_pos
        for i, (target, zone_type) in enumerate(targets):
            path = self._route_to_civilian(current_pos, target, zone_type.name, team_idx)
            self.sequential_route.append({
                "zone":    zone_type.name,
                "src":     [current_pos[0], current_pos[1]],
                "dst":     [target[0], target[1]],
                "amb_idx": team_idx,
                "nodes":   self._path_to_nodes(current_pos, path),
                "blocked": len(path) == 0,
                "status":  "active" if i == 0 else "pending",
            })
            current_pos = target

        n = len(self.sequential_route)
        self._log("ROUTE",
                  f"Seq route built: {n} legs, team=Amb{team_idx+1} @ {team_pos}")
        self._sync_active_routes()

    def _advance_sequential_leg(self) -> None:
        """Completes current leg, starts next, or rebuilds the chain."""
        if not getattr(self, 'sequential_route', None):
            return
        if self.seq_current_leg < len(self.sequential_route):
            self.sequential_route[self.seq_current_leg]["status"] = "completed"
        self.seq_current_leg += 1
        if self.seq_current_leg < len(self.sequential_route):
            self.sequential_route[self.seq_current_leg]["status"] = "active"
            zone = self.sequential_route[self.seq_current_leg]["zone"]
            leg  = self.seq_current_leg + 1
            total = len(self.sequential_route)
            self._log("ROUTE",
                      f"Seq leg {leg}/{total}: heading to {zone}")
        else:
            self._build_sequential_route()
        self._sync_active_routes()

    def _recalculate_current_leg(self) -> None:
        """A* again for the active leg only."""
        if not getattr(self, 'sequential_route', None):
            return
        if self.seq_current_leg >= len(self.sequential_route):
            return
        leg = self.sequential_route[self.seq_current_leg]
        src = tuple(leg["src"])
        dst = tuple(leg["dst"])
        path = self._route_to_civilian(src, dst, leg["zone"],
                                       getattr(self, 'seq_team_amb_idx', -1))
        leg["nodes"]   = self._path_to_nodes(src, path)
        leg["blocked"] = len(path) == 0
        self._sync_active_routes()

    def _sync_active_routes(self) -> None:
        """Sets active_routes to the current leg (API/UI)."""
        if not getattr(self, 'sequential_route', None):
            self.active_routes = []
            return
        if self.seq_current_leg < len(self.sequential_route):
            self.active_routes = [dict(self.sequential_route[self.seq_current_leg])]
        else:
            self.active_routes = []

    def recalculate_active_routes(self) -> None:
        """API entry: reruns A* on the active leg."""
        self._recalculate_current_leg()

    def run(self) -> list[SimState]:
        """setup() then SIM_STEPS step() calls; clears floods and prints summary."""
        self.setup()
        states: list[SimState] = []
        for _ in range(SIM_STEPS):
            states.append(self.step())
        self._clear_floods()
        self._print_summary()
        return states

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("CityMind — Simulation Complete")
        print("=" * 60)
        counts: dict[str, int] = {}
        for e in self.event_log:
            counts[e.event] = counts.get(e.event, 0) + 1
        for event, n in sorted(counts.items()):
            print(f"  {event:<15}: {n} occurrence(s)")
        print(f"  Final ambulance positions : {self.state.ambulance_positions}")
        print(f"  Active floods at end      : {len(self.state.active_floods)}")
        print(f"  Total log entries         : {len(self.event_log)}")
        print("=" * 60)
        print("\nFull Event Log:")
        for e in self.event_log:
            print(f"  Step {e.step:02d}  [{e.event:<14}] {e.detail}")

    def render_grid(self) -> list[list[str]]:
        """ASCII grid: zone letters, 1–3 ambulances, X on flooded endpoints."""
        w, h = self.graph.width, self.graph.height
        grid = [["?" for _ in range(w)] for _ in range(h)]

        for y in range(h):
            for x in range(w):
                node = (x, y)
                zone = self.graph.get_zone(node)
                grid[y][x] = ZONE_SYMBOLS.get(zone, "?") if zone else "?"

        flooded_nodes: set[Coord] = set()
        for edge in self._flooded_edges:
            flooded_nodes.add(edge.node_a)
            flooded_nodes.add(edge.node_b)
        for node in flooded_nodes:
            x, y = node
            grid[y][x] = "X"

        for i, pos in enumerate(self.state.ambulance_positions):
            x, y = pos
            grid[y][x] = str(i + 1)

        return grid

    def print_grid(self) -> None:
        """Dumps render_grid() plus legend."""
        grid = self.render_grid()
        for row in grid:
            print(" ".join(row))
        print(f"\nLegend: R=Residential  I=Industrial  S=School")
        print(f"        W=PowerPlant  H=Hospital    D=Depot")
        print(f"        1/2/3=Ambulance   X=Flooded road endpoint")


def main() -> None:
    sim = CitySimulation(seed=SEED)
    sim.run()
    print("\n[Final Grid State]")
    sim.print_grid()


if __name__ == "__main__":
    main()
