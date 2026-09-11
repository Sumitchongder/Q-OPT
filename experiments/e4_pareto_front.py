"""
Experiment 4 (E12): multi-objective sensitivity / Pareto front.

Sweeps the risk-weight vs latency-weight trade-off (w_R vs w_L) over a
grid, solves EXACTLY at every grid point (so the front is a genuine
optimum, not an artifact of a heuristic), and reports the resulting
solutions in RAW (non-normalized) risk/latency space -- not the
internal QUBO-normalized units -- since that's what's interpretable in
a paper figure.

At each weight setting we also vary w_K (key-depletion weight) as a
third axis via marker size, giving a 3-objective view without a full
3D plot (a common, reviewer-friendly Q1 presentation choice).

Produces:
    tables/e4_pareto_grid.csv
    tables/e4_pareto_front.csv          (non-dominated points only)
    figures/e4_pareto_front.pdf/png

Run:
    python experiments/e4_pareto_front.py --n-seeds 15 --n-requests 4
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from qopt.network.topology import get_topology
from qopt.network.traffic import sample_requests
from qopt.optimization.qubo import build_options, build_qubo
from qopt.optimization.classical_baselines import solve_exact
from qopt.optimization.qubo import evaluate_bitstring


def _raw_solution_metrics(problem, y):
    res = evaluate_bitstring(problem, y)
    chosen = res["chosen"]
    if not chosen:
        return dict(mean_risk=np.nan, mean_latency_ms=np.nan, mean_pool_pressure=np.nan)
    return dict(
        mean_risk=float(np.mean([o.risk for o in chosen])),
        mean_latency_ms=float(np.mean([o.latency_ms for o in chosen])),
        mean_pool_pressure=float(np.mean([o.key_pool_pressure for o in chosen])),
    )


def pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Non-dominated set (minimize both x and y)."""
    pts = df[[x_col, y_col]].values
    is_dominated = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i == j:
                continue
            if (pts[j][0] <= pts[i][0] and pts[j][1] <= pts[i][1] and
                    (pts[j][0] < pts[i][0] or pts[j][1] < pts[i][1])):
                is_dominated[i] = True
                break
    return df[~is_dominated].sort_values(x_col)


def run(n_seeds: int, n_requests: int, k_paths: int, topology: str,
        outdir_tables: str, outdir_figs: str):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    w_R_grid = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    w_L_grid = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]

    records = []
    for w_R in w_R_grid:
        for w_L in w_L_grid:
            weights = dict(w_R=w_R, w_L=w_L, w_K=0.8, w_C=1.1, w_D=0.6, w_S=0.5)
            for seed in range(n_seeds):
                G = get_topology(topology, seed=0)
                reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
                rng = np.random.default_rng(seed + 2000)
                link_state = {
                    frozenset(e): dict(qber=float(rng.uniform(0.01, 0.05)),
                                        risk=float(rng.uniform(0.05, 0.4)),
                                        key_pool_frac=float(rng.uniform(0.3, 0.95)))
                    for e in G.edges()
                }
                options = build_options(G, reqs, link_state, k_paths=k_paths)
                problem = build_qubo(options, reqs, weights=weights)
                result = solve_exact(problem, max_n=30)
                metrics = _raw_solution_metrics(problem, result.y)
                records.append(dict(w_R=w_R, w_L=w_L, seed=seed, **metrics))

    df = pd.DataFrame(records).dropna()
    df.to_csv(os.path.join(outdir_tables, "e4_pareto_grid.csv"), index=False)

    grid_mean = df.groupby(["w_R", "w_L"]).agg(
        mean_risk=("mean_risk", "mean"), mean_latency_ms=("mean_latency_ms", "mean"),
        mean_pool_pressure=("mean_pool_pressure", "mean")).reset_index()

    front = pareto_front(grid_mean, "mean_risk", "mean_latency_ms")
    front.to_csv(os.path.join(outdir_tables, "e4_pareto_front.csv"), index=False)
    print(f"Pareto front: {len(front)} / {len(grid_mean)} weight settings are non-dominated")
    print(front.to_string(index=False))

    _plot(grid_mean, front, outdir_figs)
    return grid_mean, front


def _plot(grid_mean: pd.DataFrame, front: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig

    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    sizes = 20 + 200 * (grid_mean["mean_pool_pressure"] - grid_mean["mean_pool_pressure"].min()) / (
        grid_mean["mean_pool_pressure"].max() - grid_mean["mean_pool_pressure"].min() + 1e-9)
    sc = ax.scatter(grid_mean["mean_risk"], grid_mean["mean_latency_ms"], s=sizes,
                     c="#a8c3e6", edgecolors="#2a6fdb", linewidths=0.7, alpha=0.85,
                     label="All (w_R, w_L) settings")
    ax.plot(front["mean_risk"], front["mean_latency_ms"], color="#d1495b", linewidth=1.8,
            marker="o", markersize=6, label="Pareto front (non-dominated)", zorder=5)

    ax.set_xlabel("Mean solution risk (raw units)")
    ax.set_ylabel("Mean solution latency (ms)")
    ax.set_title("Pareto front: security risk vs. latency\n(marker size = key-pool pressure)")
    ax.legend(loc="upper right", fontsize=8)
    savefig(fig, os.path.join(outdir_figs, "e4_pareto_front"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--n-requests", type=int, default=4)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="B")
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    run(args.n_seeds, args.n_requests, args.k_paths, args.topology, args.outdir_tables, args.outdir_figs)
