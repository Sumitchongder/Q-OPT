"""
Experiment 9 (E6): QAOA vs the Q-learning baseline, alongside the
existing classical baselines, on identical QUBO instances across seeds.

Produces:
    tables/e9_qaoa_vs_rl.csv
    figures/e9_qaoa_vs_rl.pdf/png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from qopt.network.topology import get_topology
from qopt.network.traffic import sample_requests
from qopt.optimization.qubo import build_options, build_qubo
from qopt.optimization.classical_baselines import solve_exact, solve_greedy, solve_simulated_annealing
from qopt.optimization.milp_baseline import solve_milp
from qopt.optimization.qaoa_qiskit import run_qaoa_aer
from qopt.optimization.rl_baseline import solve_q_learning


def run(n_seeds: int, n_requests: int, k_paths: int, topology: str, n_episodes: int,
        outdir_tables: str, outdir_figs: str):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    records = []
    for seed in range(n_seeds):
        G = get_topology(topology, seed=0)
        reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
        rng = np.random.default_rng(seed + 9000)
        link_state = {
            frozenset(e): dict(qber=float(rng.uniform(0.01, 0.05)),
                                risk=float(rng.uniform(0.05, 0.4)),
                                key_pool_frac=float(rng.uniform(0.3, 0.95)))
            for e in G.edges()
        }
        options = build_options(G, reqs, link_state, k_paths=k_paths)
        problem = build_qubo(options, reqs)

        exact = solve_exact(problem, max_n=26)
        opt_ref = exact.objective

        rl = solve_q_learning(problem, reqs, n_episodes=n_episodes, seed=seed)
        qaoa = run_qaoa_aer(problem, reps=2, shots=1024, maxiter=60, seed=seed, optimal_energy=opt_ref)

        for r in [exact, solve_greedy(problem), solve_milp(problem),
                  solve_simulated_annealing(problem, n_sweeps=1500, seed=seed), rl]:
            records.append(dict(seed=seed, solver=r.solver, objective=r.objective,
                                 runtime_s=r.runtime_s, violations=r.constraint_violations,
                                 approximation_ratio=r.objective / opt_ref if opt_ref > 0 else np.nan))
        records.append(dict(seed=seed, solver="QAOA(p=2)", objective=qaoa.objective,
                             runtime_s=qaoa.runtime_s, violations=qaoa.constraint_violations,
                             approximation_ratio=qaoa.approximation_ratio))
        print(f"seed {seed+1}/{n_seeds} done")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e9_qaoa_vs_rl.csv"), index=False)

    summary = df.groupby("solver").agg(
        mean_AR=("approximation_ratio", "mean"), std_AR=("approximation_ratio", "std"),
        mean_runtime_ms=("runtime_s", lambda x: 1000 * x.mean()),
        mean_violations=("violations", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))

    _plot(df, outdir_figs)
    return df, summary


def _plot(df: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig, PALETTE

    set_style()
    order = ["Greedy", "SimulatedAnnealing", "Q-Learning", "MILP", "Exact", "QAOA(p=2)"]
    order = [o for o in order if o in df["solver"].unique()]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    data = [df[df.solver == s]["approximation_ratio"].dropna().values for s in order]
    bp = ax.boxplot(data, tick_labels=order, showmeans=True, meanline=True, patch_artist=True)
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE.get(s, "#7a5195"))
        patch.set_alpha(0.55)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Approximation ratio")
    ax.set_title("QAOA vs. RL (Q-learning) vs. classical baselines")
    plt.xticks(rotation=20, ha="right")
    savefig(fig, os.path.join(outdir_figs, "e9_qaoa_vs_rl"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--n-requests", type=int, default=3)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="B")
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    run(args.n_seeds, args.n_requests, args.k_paths, args.topology, args.n_episodes,
        args.outdir_tables, args.outdir_figs)
