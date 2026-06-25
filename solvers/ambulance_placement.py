"""GA for ambulance positions: minimise worst-case BFS hops from any populated cell to nearest ambulance."""

from __future__ import annotations

import random
from collections import deque

from core.city_graph import CityGraph
from core.edge import Coord
from models.zone import ZONE_POPULATION, ZoneType


POPULATED_ZONES: frozenset[ZoneType] = frozenset({
    ZoneType.RESIDENTIAL,
    ZoneType.SCHOOL,
    ZoneType.HOSPITAL,
    ZoneType.INDUSTRIAL,
    ZoneType.POWER_PLANT,
})

Chromosome = list[Coord]


class AmbulancePlacementGA:
    """Minimax distance to farthest populated node; multi-source BFS fitness."""

    def __init__(
        self,
        graph: CityGraph,
        num_ambulances: int = 3,
        population_size: int = 60,
        max_generations: int = 500,
        mutation_rate: float = 0.15,
        elite_count: int = 4,
        stagnation_limit: int = 60,
    ) -> None:
        self.graph = graph
        self._num_ambulances = num_ambulances
        self._population_size = population_size
        self._max_generations = max_generations
        self._mutation_rate = mutation_rate
        self._elite_count = elite_count
        self._stagnation_limit = stagnation_limit

        self._all_nodes: list[Coord] = graph.all_nodes()

        self._populated_nodes: list[Coord] = [
            node
            for node in self._all_nodes
            if graph.get_zone(node) in POPULATED_ZONES
        ]

        self._assign_population_density()

    def refresh(self) -> int:
        """Rebuilds node caches before a re-solve (e.g. after risk update)."""
        self._all_nodes = self.graph.all_nodes()
        self._populated_nodes = [
            node
            for node in self._all_nodes
            if self.graph.get_zone(node) in POPULATED_ZONES
        ]
        return len(self._populated_nodes)

    def _assign_population_density(self) -> None:
        """Writes ZONE_POPULATION onto each node as graph attrs."""
        for node in self._all_nodes:
            zone = self.graph.get_zone(node)
            pop = ZONE_POPULATION.get(zone, 0) if zone is not None else 0
            self.graph.set_node_attr(node, "population", pop)

    def _evaluate(self, chromosome: Chromosome) -> int:
        """Max BFS distance from any populated node to nearest ambulance (unreachable → 10_000)."""
        if not self._populated_nodes:
            return 0

        distances: dict[Coord, int] = {}
        queue: deque[tuple[Coord, int]] = deque()

        for pos in chromosome:
            if pos not in distances:
                distances[pos] = 0
                queue.append((pos, 0))

        while queue:
            current, dist = queue.popleft()
            for neighbor in self.graph.get_neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        max_dist = 0
        for node in self._populated_nodes:
            d = distances.get(node, 10_000)  # unreachable → very high penalty
            if d > max_dist:
                max_dist = d

        return max_dist

        # Optional tie-break (population-weighted avg) — not used; would need solve() return type change.
        #
        # total_pop = 0
        # weighted_sum = 0.0
        # for node in self._populated_nodes:
        #     d = distances.get(node, 10_000)
        #     pop = self.graph.get_node_attr(node, "population") or 0
        #     weighted_sum += d * pop
        #     total_pop += pop
        # weighted_avg = weighted_sum / total_pop if total_pop else 0.0
        # return (max_dist, weighted_avg)

    def _random_chromosome(self) -> Chromosome:
        """n distinct random grid cells."""
        return random.sample(self._all_nodes, self._num_ambulances)

    def _initialize_population(self) -> list[Chromosome]:
        return [self._random_chromosome() for _ in range(self._population_size)]

    def _tournament_select(
        self,
        population: list[Chromosome],
        scores: list[int],
    ) -> Chromosome:
        """Pick better of two random individuals."""
        i, j = random.sample(range(len(population)), 2)
        return population[i] if scores[i] <= scores[j] else population[j]

    def _crossover(self, parent_a: Chromosome, parent_b: Chromosome) -> Chromosome:
        """Per-ambulance uniform pick from parents; fixes duplicate slots with a random free cell."""
        child: Chromosome = []
        used: set[Coord] = set()

        for i in range(self._num_ambulances):
            primary = parent_a[i] if random.random() < 0.5 else parent_b[i]
            secondary = parent_b[i] if primary == parent_a[i] else parent_a[i]

            if primary not in used:
                candidate = primary
            elif secondary not in used:
                candidate = secondary
            else:
                available = [n for n in self._all_nodes if n not in used]
                candidate = random.choice(available) if available else primary

            child.append(candidate)
            used.add(candidate)

        return child

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """With mutation_rate, moves one ambulance to a random unused cell."""
        if random.random() >= self._mutation_rate:
            return chromosome

        used = set(chromosome)
        available = [n for n in self._all_nodes if n not in used]
        if not available:
            return chromosome

        mutated = chromosome[:]
        idx = random.randrange(self._num_ambulances)
        mutated[idx] = random.choice(available)
        return mutated

    def solve(self) -> tuple[Chromosome, int, list[int]]:
        """Returns (best chromosome, best score, per-gen best-so-far scores)."""
        population = self._initialize_population()
        scores = [self._evaluate(c) for c in population]

        best_idx = scores.index(min(scores))
        best_chromosome: Chromosome = population[best_idx][:]
        best_score: int = scores[best_idx]
        self.initial_best_score = best_score
        stagnation = 0

        generation_history: list[int] = [best_score]

        for _generation in range(self._max_generations):
            paired = sorted(zip(scores, population), key=lambda x: x[0])
            elites: list[Chromosome] = [chrom for _, chrom in paired[: self._elite_count]]

            new_population: list[Chromosome] = list(elites)
            while len(new_population) < self._population_size:
                parent_a = self._tournament_select(population, scores)
                parent_b = self._tournament_select(population, scores)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population
            scores = [self._evaluate(c) for c in population]

            gen_best_idx = scores.index(min(scores))
            gen_best_score = scores[gen_best_idx]

            generation_history.append(min(best_score, gen_best_score))

            if gen_best_score < best_score:
                best_score = gen_best_score
                best_chromosome = population[gen_best_idx][:]
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= self._stagnation_limit:
                break

        return best_chromosome, best_score, generation_history
