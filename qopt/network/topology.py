"""
Network topology generators for the Q-OPT digital twin.

Five topology families are provided, matching the roadmap:
    A: line (5 nodes)
    B: ring (6 nodes)
    C: mesh (10 nodes)
    D: random geometric graph (15 nodes)
    E: scale-free / Barabasi-Albert (20 nodes)

All generators return a ``networkx.Graph`` with per-edge physical
attributes (``length_km``) already attached, since link length feeds
directly into the QKD channel loss model in ``qopt.qkd.channel``.
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def _attach_lengths(G: nx.Graph, rng: np.random.Generator,
                     lo_km: float = 10.0, hi_km: float = 60.0) -> nx.Graph:
    for u, v in G.edges():
        G[u][v]["length_km"] = float(rng.uniform(lo_km, hi_km))
    return G


def topology_a_line(seed: int = 0) -> nx.Graph:
    """5-node line topology."""
    rng = np.random.default_rng(seed)
    G = nx.path_graph(5)
    return _attach_lengths(G, rng)


def topology_b_ring(seed: int = 0) -> nx.Graph:
    """6-node ring topology."""
    rng = np.random.default_rng(seed)
    G = nx.cycle_graph(6)
    return _attach_lengths(G, rng)


def topology_c_mesh(seed: int = 0) -> nx.Graph:
    """10-node partial mesh (~2.5x edges of a tree, still planar-ish sparse mesh)."""
    rng = np.random.default_rng(seed)
    G = nx.random_regular_graph(d=4, n=10, seed=seed)
    if not nx.is_connected(G):
        G = nx.connected_watts_strogatz_graph(10, 4, 0.3, seed=seed)
    return _attach_lengths(G, rng)


def topology_d_rgg(seed: int = 0, n: int = 15, radius: float = 0.35) -> nx.Graph:
    """15-node random geometric graph. Retries until connected."""
    for attempt in range(50):
        G = nx.random_geometric_graph(n, radius + 0.02 * attempt, seed=seed + attempt)
        if nx.is_connected(G):
            rng = np.random.default_rng(seed)
            return _attach_lengths(G, rng)
    raise RuntimeError("Could not generate a connected RGG topology D")


def topology_e_scale_free(seed: int = 0, n: int = 20, m: int = 2) -> nx.Graph:
    """20-node scale-free (Barabasi-Albert) topology."""
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return _attach_lengths(G, rng)


TOPOLOGIES = {
    "A": topology_a_line,
    "B": topology_b_ring,
    "C": topology_c_mesh,
    "D": topology_d_rgg,
    "E": topology_e_scale_free,
}


def get_topology(name: str, seed: int = 0) -> nx.Graph:
    """Fetch a named topology (A-E) with a directed-view attribute setup.

    Edge attributes ``length_km`` are always present. Nodes are relabeled
    to plain integers 0..N-1 for downstream indexing (QUBO variable maps).
    """
    if name not in TOPOLOGIES:
        raise ValueError(f"Unknown topology '{name}'. Choose from {list(TOPOLOGIES)}")
    G = TOPOLOGIES[name](seed=seed)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")
    G.graph["name"] = name
    G.graph["seed"] = seed
    return G


if __name__ == "__main__":
    for name in TOPOLOGIES:
        G = get_topology(name, seed=0)
        assert nx.is_connected(G), f"Topology {name} not connected!"
        print(f"Topology {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"connected={nx.is_connected(G)}")
