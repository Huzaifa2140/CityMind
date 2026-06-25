# CityMind — AI-Driven City Emergency Simulation

A Python simulation that applies five classical AI algorithms to model a smart city's emergency response system on a configurable grid, paired with a real-time Flask + HTML/JS web dashboard.

---

## Table of Contents

- Overview
- Project Structure
- Algorithms in Depth
- How the Simulation Runs
- Getting Started
- Configuration
- Web API Reference

---

## Overview

CityMind builds a virtual city on a grid (up to 40×40 cells) and runs it through 20 time-steps. Each cell is assigned a zone type — residential, industrial, school, power plant, hospital, or depot. Roads connect every cell, ambulances are deployed strategically, and disaster events (floods) block roads at random, forcing the system to reroute in real time.

The project solves five interconnected challenges, each using a different AI technique:

| Challenge | Problem | Algorithm |
|-----------|---------|-----------|
| 1 | Assign valid zones to all grid cells | Min-Conflicts CSP |
| 2 | Connect the city and find emergency routes | Kruskal MST + Dijkstra |
| 3 | Place ambulances to minimise worst-case response time | Genetic Algorithm |
| 4 | Route ambulances around flooded roads | A* Search |
| 5 | Classify every cell by crime/risk level | K-Means + Decision Tree |

---

## Project Structure

```
citymind/
│
├── core/
│   ├── city_graph.py       # Graph data structure — nodes, edges, zone and attribute storage
│   ├── constants.py        # Global config: grid size, weights, CSP cap, depot/hospital coords
│   └── edge.py             # Edge and Coord type definitions
│
├── models/
│   └── zone.py             # ZoneType enum + population headcount per zone type
│
├── solvers/
│   ├── min_conflicts.py    # Challenge 1 — CSP solver for city layout
│   ├── road_network.py     # Challenge 2 — Kruskal MST builder
│   ├── emergency_routing.py# Challenge 2 — Dijkstra primary + backup route finder
│   ├── ambulance_placement.py  # Challenge 3 — Genetic Algorithm placement
│   ├── astar.py            # Challenge 4 — A* pathfinder with risk weighting
│   └── risk_predictor.py   # Challenge 5 — K-Means clustering + Decision Tree classifier
│
├── ui/
│   └── index.html          # Single-file web dashboard (HTML + JS + CSS, no build step)
│
├── app.py                  # Flask server and REST API
├── main.py                 # CitySimulation class + CLI entry point
├── verify_build.py         # Sanity-check: graph builds correctly
├── verify_spread.py        # Validates spread-target selection logic
└── README.md
```

---

## Algorithms in Depth

### Challenge 1 — City Layout via Min-Conflicts CSP

**File:** `solvers/min_conflicts.py`

The zone assignment problem is modelled as a Constraint Satisfaction Problem (CSP). Every grid cell is a variable; its domain is the set of six zone types. Constraints enforce realistic urban planning rules:

- Industrial cells may not be adjacent to Schools or Hospitals.
- Every Residential cell must have a Hospital within 3 hops (BFS).
- Every Power Plant must have an Industrial cell within 2 hops.
- Exactly one Depot must exist on the entire grid.

The solver uses **Min-Conflicts local search**: it initialises all cells randomly, identifies which cells violate constraints, then iteratively picks a conflicted cell and reassigns it to whichever zone type produces the fewest conflicts. This repeats until no conflicts remain or the iteration cap (`MAX_ITERATIONS_CSP = 10,000`) is hit.

After a valid layout is found, a BFS from the Depot locates the nearest Hospital — these two coordinates are stored in `constants.py` and used by every downstream solver.

---

### Challenge 2 — Road Network via Kruskal + Double Dijkstra

**Files:** `solvers/road_network.py`, `solvers/emergency_routing.py`

All grid cells are connected by a unit-weight orthogonal edge graph. The road network is then built in two steps:

**Minimum Spanning Tree (Kruskal's algorithm)**
All edges are sorted by weight and added greedily as long as they don't create a cycle (union-find). The result is the minimal set of roads that keeps every cell reachable — used by the UI to visualise the city's backbone.

**Primary + Backup Routes (Dijkstra × 2)**
Two independent shortest paths are computed from the Depot to the Hospital:
- The **primary route** is the globally shortest path.
- The **backup route** is computed after temporarily removing all edges used by the primary, guaranteeing a physically disjoint fallback if the main road is flooded.

---

### Challenge 3 — Ambulance Placement via Genetic Algorithm

**File:** `solvers/ambulance_placement.py`

The goal is to place 3 ambulances on the grid so that the **worst-case BFS distance** from any populated cell to its nearest ambulance is minimised (a minimax objective). This is NP-hard at scale, so it is solved with a Genetic Algorithm.

**Representation:** A chromosome is a list of 3 grid coordinates (one per ambulance).

**Fitness function:** Multi-source BFS from all 3 ambulance positions simultaneously. The fitness score is the maximum BFS distance reached before all populated nodes are covered. Lower is better.

**Evolutionary operators:**
- *Selection* — binary tournament (two random individuals, keep the better one).
- *Crossover* — per-slot uniform crossover from two parents; duplicate positions are resolved by drawing a random unused cell.
- *Mutation* — with 15% probability, one ambulance is relocated to a random unused cell.
- *Elitism* — the top 4 individuals carry over unchanged each generation.

The GA runs for up to 500 generations and stops early after 60 consecutive generations with no improvement. At simulation step 10, risk multipliers are updated and the GA is re-run on the current graph state.

---

### Challenge 4 — Emergency Routing via A*

**File:** `solvers/astar.py`

When an ambulance needs to reach a target cell, A* finds the optimal path. The heuristic is Manhattan distance. Edge traversal cost is the base edge weight multiplied by the destination cell's **risk multiplier** (set by the risk predictor), so A* naturally avoids high-risk zones when cheaper alternatives exist.

If a road is flooded (`edge.is_impassable = True`), A* treats it as a missing edge. This means every flood event automatically triggers a new A* call on the active route leg — if no path exists, a `PathNotFoundError` is raised and logged as `NO_ROUTE`.

The sequential routing pattern (H → S → R → I → W) picks spread-maximised targets so ambulances cover distinct parts of the city across legs.

---

### Challenge 5 — Risk Prediction via K-Means + Decision Tree

**File:** `solvers/risk_predictor.py`

Every cell is classified into one of three risk tiers — **Low**, **Medium**, or **High** — which then scales A* routing costs.

**Feature extraction (per cell):**
- Normalised population count (from zone type)
- Inverse BFS distance to the nearest Industrial cell (proximity to industry = higher risk)
- Zone base risk score (a fixed prior per zone type)

**Step 1 — K-Means clustering (k=3, stdlib only)**
Cells are clustered in 2D feature space (population + industrial proximity). Cluster centroids are ranked by a 50/50 weighted score to assign Low / Medium / High labels.

**Step 2 — Label noise injection**
~8% of labels are nudged ±1 tier to simulate real-world label uncertainty and prevent the decision tree from being trivially exact.

**Step 3 — Gini Decision Tree (max depth 6, stdlib only)**
A binary decision tree is trained on the noisy labels. At each split, the feature and threshold that minimise weighted Gini impurity are chosen. The trained tree predicts the final risk tier for every cell, and the result is written back to the graph as `risk_label` and `risk_multiplier`.

| Risk Label | A* Cost Multiplier |
|------------|-------------------|
| Low | 1.00× |
| Medium | 1.25× |
| High | 1.50× |

Both K-Means and the Decision Tree are implemented entirely from scratch — no NumPy, scikit-learn, or any external ML library is used.

---

## How the Simulation Runs

```
setup()
  ├── Min-Conflicts CSP      → valid zone layout
  ├── Kruskal + Dijkstra     → MST + primary/backup routes
  ├── Genetic Algorithm      → 3 ambulance positions (minimax)
  └── K-Means + Decision Tree → risk labels on every cell

step() × 20
  ├── Step 10 only: recalculate risk → rerun GA
  ├── ~30% chance: flood 1–3 random passable edges
  ├── Sequential route leg: one ambulance team visits H → S → R → I → W
  └── If flooded: A* reruns on the active leg only

After step 20: floods cleared, summary printed
```

**Zone types and populations:**

| Symbol | Zone | Headcount |
|--------|------|-----------|
| R | Residential | 1,000 |
| S | School | 800 |
| H | Hospital | 600 |
| I | Industrial | 300 |
| W | Power Plant | 75 |
| D | Depot | 0 |

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- pip

### Install

```bash
git clone https://github.com/your-username/citymind.git
cd citymind
pip install flask
```

The simulation core (`main.py` and everything in `core/`, `models/`, `solvers/`) has **no external dependencies** — pure Python stdlib throughout.

### Run in the terminal (CLI)

```bash
python -X utf8 main.py
```

Runs all 20 steps, printing phase logs, per-step events, and a final ASCII grid with a legend:

```
Legend: R=Residential  I=Industrial  S=School
        W=PowerPlant  H=Hospital    D=Depot
        1/2/3=Ambulance   X=Flooded road endpoint
```

### Run the web dashboard

```bash
python -X utf8 app.py
```

Open **http://localhost:5050** in your browser.

- Use the **grid size slider** to set city dimensions (10–40) before initialising.
- Click **Initialize** to run setup.
- Click **Next Step** to advance one tick, or **Auto-run** to play through all 20.
- Click any cell to **manually flood** its road.
- Use **Clear Floods** or **Reroute** to respond dynamically.

The event log panel on the right shows the last 30 timestamped events live.

---

## Configuration

**`core/constants.py`** — grid and solver settings:

| Constant | Default | Effect |
|----------|---------|--------|
| `GRID_WIDTH` / `GRID_HEIGHT` | 40 | Grid dimensions (UI slider overrides these) |
| `DEFAULT_WEIGHT` | 1.0 | Base road traversal cost |
| `RESIDENTIAL_WEIGHT` | 0.8 | Slightly cheaper roads through residential areas |
| `MAX_ITERATIONS_CSP` | 10,000 | Hard cap for Min-Conflicts iterations |

**Top of `main.py`** — simulation parameters:

| Constant | Default | Effect |
|----------|---------|--------|
| `SIM_STEPS` | 20 | Total simulation ticks |
| `NUM_AMBULANCES` | 3 | Ambulances placed by the GA |
| `FLOOD_PROB` | 0.30 | Per-step probability of a flood event |
| `RISK_UPDATE_STEP` | 10 | Step that triggers risk recalculation + GA rerun |
| `SEED` | 42 | RNG seed — change for different city layouts |

---

## Web API Reference

All endpoints are served by Flask at `http://localhost:5050`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the web dashboard |
| POST | `/api/setup` | Initialises a new simulation — body: `{"size": 40}` |
| GET | `/api/state` | Returns current simulation state; add `?grid=true` to include full grid data |
| POST | `/api/step` | Advances the simulation by one tick |
| POST | `/api/flood` | Manually floods a cell — body: `{"x": 5, "y": 10}` |
| POST | `/api/clear` | Clears all active floods and reroutes |
| POST | `/api/reroute` | Re-runs A* on the current active route leg |
