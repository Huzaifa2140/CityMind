# CityMind — AI-Driven City Emergency Simulation

A Python simulation that applies five classical AI algorithms to model a smart city's emergency response system on a configurable grid, paired with a real-time Flask + HTML/JS web dashboard.

---

## Table of Contents

- Overview
- Project Structure
- Algorithms
- How the Simulation Runs
- Getting Started
- Deployment
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
├── Dockerfile              # Container build instructions for Cloud Run
├── requirements.txt        # Python dependencies (flask, gunicorn)
└── README.md
```

---

## Algorithms

| Challenge | Algorithm | File |
|-----------|-----------|------|
| City Layout (CSP) | Min-Conflicts local search | `solvers/min_conflicts.py` |
| Road Network | Kruskal MST + Dijkstra (×2) | `solvers/road_network.py`, `solvers/emergency_routing.py` |
| Ambulance Placement | Genetic Algorithm (minimax BFS) | `solvers/ambulance_placement.py` |
| Emergency Routing | A* with risk-weighted edges | `solvers/astar.py` |
| Risk Prediction | K-Means clustering + Gini Decision Tree | `solvers/risk_predictor.py` |

**Key design decisions:**

- **CSP constraints** — Industrial zones cannot be adjacent to Schools or Hospitals; Residential zones must have a Hospital within 3 hops; exactly one Depot per grid.
- **Dual routing** — A disjoint backup route is computed by removing primary-route edges before running Dijkstra a second time, guaranteeing a flood-safe fallback.
- **GA fitness** — Multi-source BFS from all ambulance positions; score = maximum distance to the farthest populated cell (minimax). Stops after 60 stagnant generations.
- **A* heuristic** — Manhattan distance; edge cost is scaled by the cell's risk multiplier (Low 1.0×, Medium 1.25×, High 1.50×) so the router naturally avoids danger zones.
- **Risk pipeline** — K-Means labels cells by population density + industrial proximity; a Gini Decision Tree (depth 6) then classifies each cell. Both are built from scratch — no NumPy or scikit-learn.

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

## Deployment

The app is containerised and ready to deploy to **Google Cloud Run** for a permanent public URL.

### Deploy from local machine

Requires [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) and a GCP project with Cloud Run enabled.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
cd citymind
gcloud run deploy citymind --source . --region us-central1 --allow-unauthenticated --platform managed --clear-base-image
```

Cloud Run builds the `Dockerfile`, deploys the container, and returns a permanent URL like:
`https://citymind-xxxx-uc.a.run.app`

### Deploy from Cloud Shell (no local SDK needed)

```bash
git clone https://github.com/your-username/citymind.git
cd citymind
gcloud run deploy citymind --source . --region us-central1 --allow-unauthenticated --platform managed
```

### Run locally with Docker

```bash
docker build -t citymind .
docker run -p 5050:8080 citymind
```

Then open **http://localhost:5050**.

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
