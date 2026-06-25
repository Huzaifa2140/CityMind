"""Risk tiers: K-means (pop + industrial proximity) → noisy labels → Gini decision tree; writes risk_* on nodes (stdlib only)."""

from __future__ import annotations

import math
import random
from collections import deque

from core.city_graph import CityGraph
from core.edge import Coord
from models.zone import ZONE_POPULATION, ZoneType

K_CLUSTERS: int = 3
IND_PROX_MAX_HOPS: int = 10
NOISE_SIGMA: float = 0.05
SEED: int = 42

RISK_MULTIPLIERS: dict[str, float] = {
    "Low":    1.00,
    "Medium": 1.25,
    "High":   1.50,
}

ZONE_BASE_RISK: dict[ZoneType, float] = {
    ZoneType.INDUSTRIAL:  0.65,
    ZoneType.RESIDENTIAL: 0.45,
    ZoneType.HOSPITAL:    0.40,
    ZoneType.SCHOOL:      0.25,
    ZoneType.POWER_PLANT: 0.15,
    ZoneType.DEPOT:       0.05,
}


class _KMeans:
    """K-means in feature space; refit() reruns from scratch."""

    def __init__(
        self,
        k: int = 3,
        max_iter: int = 200,
        tol: float = 1e-5,
        seed: int = SEED,
    ) -> None:
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self._rng = random.Random(seed)
        self.centroids: list[list[float]] = []
        self.labels_: list[int] = []

    @staticmethod
    def _dist(a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _assign(self, X: list[list[float]]) -> list[int]:
        labels = []
        for point in X:
            dists = [self._dist(point, c) for c in self.centroids]
            labels.append(dists.index(min(dists)))
        return labels

    def _update_centroids(
        self, X: list[list[float]], labels: list[int]
    ) -> list[list[float]]:
        n_feat = len(X[0])
        new_centroids: list[list[float]] = []
        for k in range(self.k):
            pts = [X[i] for i in range(len(X)) if labels[i] == k]
            if pts:
                new_centroids.append(
                    [sum(p[f] for p in pts) / len(pts) for f in range(n_feat)]
                )
            else:
                new_centroids.append(list(self.centroids[k]))
        return new_centroids

    def fit(self, X: list[list[float]]) -> "_KMeans":
        """Fit k-means to X."""
        self.centroids = [list(c) for c in self._rng.sample(X, self.k)]
        for _ in range(self.max_iter):
            labels = self._assign(X)
            new_centroids = self._update_centroids(X, labels)
            shift = max(
                self._dist(c, nc)
                for c, nc in zip(self.centroids, new_centroids)
            )
            self.centroids = new_centroids
            if shift < self.tol:
                break
        self.labels_ = labels
        return self

    def refit(self, X: list[list[float]]) -> "_KMeans":
        return self.fit(X)

    def predict(self, X: list[list[float]]) -> list[int]:
        return self._assign(X)


class _DTNode:
    """One DT node (split or leaf)."""

    __slots__ = (
        "feature_idx", "threshold", "left", "right", "label", "gini", "n_samples"
    )

    def __init__(self) -> None:
        self.feature_idx: int | None = None
        self.threshold: float | None = None
        self.left: "_DTNode | None" = None
        self.right: "_DTNode | None" = None
        self.label: str | None = None
        self.gini: float = 0.0
        self.n_samples: int = 0


class _DecisionTreeClassifier:
    """Binary decision tree, Gini splits."""

    def __init__(self, max_depth: int = 6, min_samples_split: int = 4) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: _DTNode | None = None
        self.feature_names: list[str] = []
        self.classes_: list[str] = []

    @staticmethod
    def _gini(labels: list[str]) -> float:
        n = len(labels)
        if n == 0:
            return 0.0
        counts: dict[str, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    @staticmethod
    def _majority(labels: list[str]) -> str:
        counts: dict[str, int] = {}
        for lbl in labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    def _best_split(
        self,
        X: list[list[float]],
        y: list[str],
        parent_gini: float,
    ) -> tuple[int | None, float | None, float]:
        best_gini = float("inf")
        best_feat: int | None = None
        best_thresh: float | None = None
        n = len(y)
        n_feat = len(X[0])

        for feat in range(n_feat):
            sorted_vals = sorted(set(row[feat] for row in X))
            thresholds = [
                (sorted_vals[i] + sorted_vals[i + 1]) / 2
                for i in range(len(sorted_vals) - 1)
            ]
            for thresh in thresholds:
                left_y  = [y[i] for i in range(n) if X[i][feat] <= thresh]
                right_y = [y[i] for i in range(n) if X[i][feat] >  thresh]
                if not left_y or not right_y:
                    continue
                w_gini = (
                    len(left_y)  / n * self._gini(left_y) +
                    len(right_y) / n * self._gini(right_y)
                )
                if w_gini < best_gini:
                    best_gini  = w_gini
                    best_feat  = feat
                    best_thresh = thresh

        return best_feat, best_thresh, best_gini

    def _build(
        self, X: list[list[float]], y: list[str], depth: int
    ) -> _DTNode:
        node = _DTNode()
        node.gini = self._gini(y)
        node.n_samples = len(y)

        pure      = len(set(y)) == 1
        too_small = len(y) < self.min_samples_split
        too_deep  = depth >= self.max_depth

        if pure or too_small or too_deep:
            node.label = self._majority(y)
            return node

        feat, thresh, split_gini = self._best_split(X, y, node.gini)

        if feat is None or split_gini >= node.gini:
            node.label = self._majority(y)
            return node

        node.feature_idx = feat
        node.threshold   = thresh

        left_idx  = [i for i in range(len(X)) if X[i][feat] <= thresh]
        right_idx = [i for i in range(len(X)) if X[i][feat] >  thresh]

        node.left  = self._build([X[i] for i in left_idx],  [y[i] for i in left_idx],  depth + 1)
        node.right = self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth + 1)
        return node

    def _predict_one(self, node: _DTNode, x: list[float]) -> str:
        if node.label is not None:
            return node.label
        assert node.feature_idx is not None and node.threshold is not None
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(node.left, x)   # type: ignore[arg-type]
        return self._predict_one(node.right, x)       # type: ignore[arg-type]

    def _count_leaves(self, node: _DTNode | None) -> int:
        if node is None or node.label is not None:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def fit(
        self,
        X: list[list[float]],
        y: list[str],
        feature_names: list[str] | None = None,
    ) -> "_DecisionTreeClassifier":
        self.classes_ = sorted(set(y))
        self.feature_names = feature_names or [f"f{i}" for i in range(len(X[0]))]
        self.root = self._build(X, y, 0)
        return self

    def predict(self, X: list[list[float]]) -> list[str]:
        assert self.root is not None, "Call fit() before predict()."
        return [self._predict_one(self.root, x) for x in X]

    @property
    def n_leaves(self) -> int:
        return self._count_leaves(self.root)


class RiskPredictor:
    """Fits k-means + tree, writes risk_label / risk_multiplier on the graph."""

    def __init__(self, graph: CityGraph, seed: int = SEED) -> None:
        self.graph = graph
        self._seed = seed
        self._rng = random.Random(seed)
        self._kmeans = _KMeans(k=K_CLUSTERS, seed=seed)
        self._tree = _DecisionTreeClassifier(max_depth=6, min_samples_split=4)
        self._nodes: list[Coord] = []
        self._X_norm: list[list[float]] = []
        self._labels: list[str] = []
        self._trained = False

    def _bfs_dist_to_industrial(self) -> dict[Coord, int]:
        """BFS hop count to nearest industrial cell."""
        sources = [
            n for n in self.graph.all_nodes()
            if self.graph.get_zone(n) is ZoneType.INDUSTRIAL
        ]
        dist: dict[Coord, int] = {}
        queue: deque[tuple[Coord, int]] = deque()
        for s in sources:
            if s not in dist:
                dist[s] = 0
                queue.append((s, 0))
        while queue:
            node, d = queue.popleft()
            if d >= IND_PROX_MAX_HOPS:
                continue
            for nb in self.graph.get_neighbors(node):
                if nb not in dist:
                    dist[nb] = d + 1
                    queue.append((nb, d + 1))
        return dist

    def _extract_features(self) -> tuple[
        list[Coord],
        list[list[float]],
        list[list[float]],
    ]:
        """Per node: population, capped inverse dist to industry, zone base risk — min-max normalised."""
        ind_dist = self._bfs_dist_to_industrial()
        nodes = self.graph.all_nodes()
        raw: list[list[float]] = []

        for node in nodes:
            zone     = self.graph.get_zone(node)
            pop      = float(self.graph.get_node_attr(node, "population") or
                            ZONE_POPULATION.get(zone, 0) if zone else 0)
            d_ind    = ind_dist.get(node, IND_PROX_MAX_HOPS)
            ind_prox = float(IND_PROX_MAX_HOPS - d_ind)
            base     = ZONE_BASE_RISK.get(zone, 0.0) if zone else 0.0
            raw.append([pop, ind_prox, base])

        n_feat = len(raw[0])
        mins   = [min(r[f] for r in raw) for f in range(n_feat)]
        maxs   = [max(r[f] for r in raw) for f in range(n_feat)]

        def norm(val: float, lo: float, hi: float) -> float:
            return (val - lo) / (hi - lo) if hi > lo else 0.0

        X_norm = [
            [norm(row[f], mins[f], maxs[f]) for f in range(n_feat)]
            for row in raw
        ]
        return nodes, raw, X_norm

    def _generate_labels(
        self,
        X_norm: list[list[float]],
        cluster_labels: list[int],
        centroids: list[list[float]],
    ) -> list[str]:
        """Cluster id → Low/Medium/High by ordering centroids on 0.5*pop + 0.5*ind_prox."""
        centroid_scores = [0.5 * c[0] + 0.5 * c[1] for c in centroids]
        ranked = sorted(range(K_CLUSTERS), key=lambda i: centroid_scores[i])
        cluster_to_tier = {ranked[0]: "Low", ranked[1]: "Medium", ranked[2]: "High"}
        return [cluster_to_tier[lbl] for lbl in cluster_labels]

    def _add_noise_to_labels(self, labels: list[str]) -> list[str]:
        """~8% labels nudged ±1 tier so the tree isn’t trivially exact."""
        tier_order = ["Low", "Medium", "High"]
        noisy: list[str] = []
        for lbl in labels:
            if self._rng.random() < 0.08:
                idx = tier_order.index(lbl)
                idx = max(0, min(2, idx + self._rng.choice([-1, 1])))
                noisy.append(tier_order[idx])
            else:
                noisy.append(lbl)
        return noisy

    def _write_to_graph(
        self, nodes: list[Coord], predicted: list[str]
    ) -> None:
        """Stores risk_label and risk_multiplier on each node."""
        for node, label in zip(nodes, predicted):
            self.graph.set_node_attr(node, "risk_label",      label)
            self.graph.set_node_attr(node, "risk_multiplier", RISK_MULTIPLIERS[label])

    @staticmethod
    def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true) if y_true else 0.0

    def solve(self) -> dict:
        """Full pipeline; returns n_nodes, cluster_counts, train_accuracy, n_leaves, risk_counts."""
        nodes, _raw, X_norm = self._extract_features()
        self._nodes  = nodes
        self._X_norm = X_norm

        self._kmeans.fit([[x[0], x[1]] for x in X_norm])
        cluster_labels = self._kmeans.labels_

        base_labels = self._generate_labels(
            X_norm, cluster_labels, self._kmeans.centroids
        )
        noisy_labels = self._add_noise_to_labels(base_labels)

        self._tree.fit(
            X_norm,
            noisy_labels,
            feature_names=["population_norm", "industrial_proximity_norm", "zone_base_risk"],
        )
        self._trained = True

        predicted = self._tree.predict(X_norm)
        self._labels = predicted

        train_acc = self._accuracy(noisy_labels, predicted)

        self._write_to_graph(nodes, predicted)

        cluster_counts = {}
        for c in cluster_labels:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

        risk_counts: dict[str, int] = {"Low": 0, "Medium": 0, "High": 0}
        for lbl in predicted:
            risk_counts[lbl] += 1

        return {
            "n_nodes":        len(nodes),
            "cluster_counts": cluster_counts,
            "train_accuracy": train_acc,
            "n_leaves":       self._tree.n_leaves,
            "risk_counts":    risk_counts,
        }

    def recalculate(self) -> dict:
        """Same as solve() but assumes _nodes may already exist; refits on current graph."""
        if not self._nodes:
            return self.solve()

        nodes, _raw, X_norm = self._extract_features()
        self._nodes  = nodes
        self._X_norm = X_norm

        self._kmeans.refit([[x[0], x[1]] for x in X_norm])
        cluster_labels = self._kmeans.labels_
        base_labels = self._generate_labels(
            X_norm, cluster_labels, self._kmeans.centroids
        )
        noisy_labels = self._add_noise_to_labels(base_labels)

        self._tree.fit(X_norm, noisy_labels)
        predicted = self._tree.predict(X_norm)
        self._labels = predicted
        self._write_to_graph(nodes, predicted)

        risk_counts: dict[str, int] = {"Low": 0, "Medium": 0, "High": 0}
        for lbl in predicted:
            risk_counts[lbl] += 1

        return {
            "n_nodes":        len(nodes),
            "cluster_counts": {c: cluster_labels.count(c) for c in range(K_CLUSTERS)},
            "train_accuracy": self._accuracy(noisy_labels, predicted),
            "n_leaves":       self._tree.n_leaves,
            "risk_counts":    risk_counts,
        }
