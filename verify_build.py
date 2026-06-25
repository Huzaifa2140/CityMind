from main import CitySimulation, SEED
from app import _build_state
sim = CitySimulation(seed=SEED)
sim.setup()
sim.step()
data = _build_state(sim, include_grid=False)
print("Keys:", data.keys())
print("sequential_route type:", type(data.get("sequential_route")))
print("sequential_route len:", len(data.get("sequential_route", [])))
print("seq_team_amb:", data.get("seq_team_amb"))
