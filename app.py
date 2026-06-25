"""Flask UI for CityMind: pip install flask && python -X utf8 app.py → http://localhost:5050"""

from __future__ import annotations

import json
import os
import sys
import threading
import core.constants as _core_constants

# UTF-8 console on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# pyrefly: ignore [missing-import]
from flask import Flask, jsonify, request, send_from_directory


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.edge import Edge
from main import SEED, SIM_STEPS, CitySimulation
from models.zone import ZoneType

app = Flask(__name__, static_folder="ui", static_url_path="/ui")

_lock: threading.Lock = threading.Lock()
_sim: CitySimulation | None = None
_setup_done: bool = False


def _node_to_list(n) -> list[int]:
    return [int(n[0]), int(n[1])]


def _edge_nodes(edge: Edge, cur) -> tuple:
    """Other endpoint of edge when walking from cur."""
    return edge.node_b if edge.node_a == cur else edge.node_a


def _serialize_grid(sim: CitySimulation) -> list[list[dict]]:
    g = sim.graph
    result = []
    for y in range(g.height):
        row = []
        for x in range(g.width):
            node = (x, y)
            zone = g.get_zone(node)
            row.append({
                "zone": zone.value if zone else None,
                "pop":  int(g.get_node_attr(node, "population") or 0),
                "risk": g.get_node_attr(node, "risk_label") or "Low",
                "mult": float(g.get_node_attr(node, "risk_multiplier") or 1.0),
            })
        result.append(row)
    return result


def _serialize_route(sim: CitySimulation) -> list[list[int]]:
    state = sim.state
    if not state.last_route or state.last_route_src is None:
        return []
    nodes = [_node_to_list(state.last_route_src)]
    cur = state.last_route_src
    for edge in state.last_route:
        nxt = _edge_nodes(edge, cur)
        nodes.append(_node_to_list(nxt))
        cur = nxt
    return nodes


def _serialize_edge_list(edges) -> list[dict]:
    """Edges → JSON dicts (dedupe by object id)."""
    seen: set[int] = set()
    out = []
    for edge in edges:
        if id(edge) not in seen:
            seen.add(id(edge))
            out.append({
                "a": _node_to_list(edge.node_a),
                "b": _node_to_list(edge.node_b),
                "w": float(edge.weight),
            })
    return out


def _serialize_route_edges(edges, start) -> list[list[int]]:
    """Ordered edges → polyline nodes for the map."""
    if not edges or start is None:
        return []
    nodes = [_node_to_list(start)]
    cur = start
    for edge in edges:
        nxt = _edge_nodes(edge, cur)
        nodes.append(_node_to_list(nxt))
        cur = nxt
    return nodes


def _build_state(sim: CitySimulation, include_grid: bool = True) -> dict:
    s = sim.state
    data: dict = {
        "ready":         True,
        "step":          s.step,
        "max_steps":     SIM_STEPS,
        "ambulances":    [_node_to_list(p) for p in s.ambulance_positions],
        "floods":        [[_node_to_list(a), _node_to_list(b)]
                          for a, b in s.active_floods],
        "route":         _serialize_route(sim),
        "route_src":     _node_to_list(s.last_route_src) if s.last_route_src else None,
        "route_dst":     _node_to_list(s.last_route_dst) if s.last_route_dst else None,
        "risk_counts":   s.risk_counts,
        "events": [
            {"step": e.step, "event": e.event, "detail": e.detail}
            for e in reversed(s.event_log[-30:])
        ],
        "grid_size":     _core_constants.GRID_WIDTH,
        "mst_edges":     _serialize_edge_list(sim.mst_edges),
        "primary_route": _serialize_route_edges(sim.primary_route, _core_constants.DEPOT_COORD),
        "backup_route":  _serialize_route_edges(sim.backup_route,  _core_constants.DEPOT_COORD),
        "sequential_route": getattr(sim, 'sequential_route', []),
        "seq_current_leg":  getattr(sim, 'seq_current_leg',  0),
        "seq_team_amb":     getattr(sim, 'seq_team_amb_idx', 0),
        "active_routes":    getattr(sim, 'active_routes',    []),
    }
    if include_grid:
        data["grid"] = _serialize_grid(sim)
    return data


@app.route("/")
def index():
    return send_from_directory("ui", "index.html")


@app.route("/prep")
def prep():
    """Serves demo-prep.html."""
    root = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(root, "demo-prep.html")


@app.route("/api/setup", methods=["POST"])
def api_setup():
    global _sim, _setup_done
    data = request.json or {}
    size = int(data.get("size", 40))
    size = max(10, min(40, size))   # clamp: 10–40
    with _lock:
        _core_constants.GRID_WIDTH    = size
        _core_constants.GRID_HEIGHT   = size
        _core_constants.DEPOT_COORD   = None   # reset so CSP can re-place
        _core_constants.HOSPITAL_COORD = None
        _sim = CitySimulation(seed=SEED)
        _sim.setup()
        _setup_done = True
        return jsonify(_build_state(_sim))


@app.route("/api/state", methods=["GET"])
def api_state():
    with _lock:
        if not _setup_done or _sim is None:
            return jsonify({"ready": False})
        include_grid = request.args.get("grid", "false").lower() == "true"
        return jsonify(_build_state(_sim, include_grid=include_grid))


@app.route("/api/step", methods=["POST"])
def api_step():
    with _lock:
        if not _setup_done or _sim is None:
            return jsonify({"error": "not ready"}), 400
        if _sim.state.step >= SIM_STEPS:
            return jsonify({"error": "simulation complete"}), 400
        _sim.step()
        return jsonify(_build_state(_sim, include_grid=False))


@app.route("/api/flood", methods=["POST"])
def api_flood():
    """Blocks first passable edge at the clicked cell."""
    data = request.json or {}
    x, y = int(data.get("x", 0)), int(data.get("y", 0))
    with _lock:
        if _sim is None:
            return jsonify({"error": "not ready"}), 400
        node = (x, y)
        edges = _sim.graph.get_edges(node)
        passable = [e for e in edges if not e.is_impassable]
        if passable:
            target = passable[0]
            target.is_impassable = True
            _sim._flooded_edges.append(target)
            _sim.state.active_floods = [
                (e.node_a, e.node_b) for e in _sim._flooded_edges
            ]
            _sim._log("FLOOD",
                      f"Manual flood: edge {target.node_a}-{target.node_b} blocked.",
                      flooded_edges=[[list(target.node_a), list(target.node_b)]])
            _sim.recalculate_active_routes()
    return jsonify(_build_state(_sim, include_grid=False))


@app.route("/api/clear", methods=["POST"])
def api_clear():
    with _lock:
        if _sim:
            _sim._clear_floods()
            _sim.recalculate_active_routes()
    return jsonify(_build_state(_sim, include_grid=False))



@app.route("/api/reroute", methods=["POST"])
def api_reroute():
    """Re-runs A* on the active sequential leg."""
    with _lock:
        if not _setup_done or _sim is None:
            return jsonify({"error": "not ready"}), 400
        try:
            _sim.recalculate_active_routes()
            _sim._log("REROUTE", "Manual reroute triggered. A* recalculated all active routes.")
            _sim.state.event_log = list(_sim.event_log)
            result = _build_state(_sim, include_grid=False)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
    return jsonify(result)


if __name__ == "__main__":
    os.makedirs("ui", exist_ok=True)
    port = int(os.environ.get("PORT", 5050))
    print(f"CityMind UI  →  http://localhost:{port}")
    print(f"Demo prep    →  http://localhost:{port}/prep")
    print("Run:  python -X utf8 app.py")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
