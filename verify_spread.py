from main import CitySimulation, SEED
sim = CitySimulation(seed=SEED)
sim.setup()
sim.step()
seq = getattr(sim, 'sequential_route', None)
print(f"sequential_route type: {type(seq)}")
print(f"sequential_route len:  {len(seq) if seq else 'None'}")
if seq:
    for i, leg in enumerate(seq):
        print(f"  Leg {i+1}: zone={leg['zone']} status={leg['status']} "
              f"src={leg['src']} dst={leg['dst']} "
              f"nodes={len(leg['nodes'])} blocked={leg['blocked']}")
