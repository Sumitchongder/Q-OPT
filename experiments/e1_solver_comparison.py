"""
Experiment 1 (maps to E1/E4/E5 in the design notes): compare Exact,
Greedy, MILP, Simulated Annealing and QAOA(p=1,2,3) on identical QUBO
instances across N random seeds, on a laptop-feasible qubit count.

Produces:
    tables/solver_comparison.csv
    figures/solver_comparison_objective.pdf/png
    figures/solver_comparison_runtime.pdf/png

Run:
    python experiments/e1_solver_comparison.py --n-seeds 30 --n-requests 2
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


def run(n_seeds: int, n_requests: int, k_paths: int, topology: str, outdir_tables: str, outdir_figs: str):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    records = []
    for seed in range(n_seeds):
        G = get_topology(topology, seed=0)  # fixed topology, vary traffic/link-state seed
        reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
        rng = np.random.default_rng(seed + 1000)
        link_state = {
            frozenset(e): dict(qber=float(rng.uniform(0.01, 0.05)),
                                risk=float(rng.uniform(0.05, 0.4)),
                                key_pool_frac=float(rng.uniform(0.3, 0.95)))
            for e in G.edges()
        }
        options = build_options(G, reqs, link_state, k_paths=k_paths)
        problem = build_qubo(options, reqs)
        n_qubits = len(options)

        exact = solve_exact(problem, max_n=24) if n_qubits <= 24 else None
        greedy = solve_greedy(problem)
        milp = solve_milp(problem)
        sa = solve_simulated_annealing(problem, n_sweeps=1500, seed=seed)

        opt_ref = exact.objective if exact is not None else milp.objective
        results = [greedy, milp, sa]
        if exact is not None:
            results.append(exact)

        for r in results:
            records.append(dict(seed=seed, n_qubits=n_qubits, solver=r.solver,
                                 objective=r.objective, runtime_s=r.runtime_s,
                                 violations=r.constraint_violations,
                                 approximation_ratio=r.objective / opt_ref if opt_ref > 0 else np.nan))

        for p in [1, 2, 3]:
            qr = run_qaoa_aer(problem, reps=p, shots=1024, maxiter=60, seed=seed,
                               optimal_energy=opt_ref)
            records.append(dict(seed=seed, n_qubits=n_qubits, solver=f"QAOA(p={p})",
                                 objective=qr.objective, runtime_s=qr.runtime_s,
                                 violations=qr.constraint_violations,
                                 approximation_ratio=qr.approximation_ratio))
        print(f"seed {seed+1}/{n_seeds} done (n_qubits={n_qubits})")

    df = pd.DataFrame(records)
    csv_path = os.path.join(outdir_tables, "solver_comparison.csv")
    df.to_csv(csv_path, index=False)

    summary = df.groupby("solver").agg(
        mean_AR=("approximation_ratio", "mean"), std_AR=("approximation_ratio", "std"),
        mean_runtime_ms=("runtime_s", lambda x: 1000 * x.mean()),
        std_runtime_ms=("runtime_s", lambda x: 1000 * x.std()),
        mean_violations=("violations", "mean"),
    ).reset_index()
    summary_path = os.path.join(outdir_tables, "solver_comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n=== Summary (Table 5/6 style) ===")
    print(summary.to_string(index=False))

    _make_figures(df, outdir_figs)
    return df, summary


def _make_figures(df: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig, PALETTE

    set_style()
    order = ["Greedy", "SimulatedAnnealing", "MILP", "Exact", "QAOA(p=1)", "QAOA(p=2)", "QAOA(p=3)"]
    order = [o for o in order if o in df["solver"].unique()]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    data = [df[df.solver == s]["approximation_ratio"].dropna().values for s in order]
    bp = ax.boxplot(data, tick_labels=order, showmeans=True, meanline=True, patch_artist=True)
    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE.get(s, "#999999"))
        patch.set_alpha(0.55)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Optimal (AR=1.0)")
    ax.set_ylabel("Approximation ratio (lower is better)")
    ax.set_title("Solver solution quality across random instances")
    plt.xticks(rotation=20, ha="right")
    ax.legend(loc="upper right")
    savefig(fig, os.path.join(outdir_figs, "solver_comparison_objective"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    means = df.groupby("solver")["runtime_s"].mean().reindex(order) * 1000
    stds = df.groupby("solver")["runtime_s"].std().reindex(order) * 1000
    ax.bar(order, means, yerr=stds, color=[PALETTE.get(s, "#999999") for s in order], alpha=0.85, capsize=3)
    ax.set_ylabel("Runtime (ms, mean ± std)")
    ax.set_yscale("log")
    ax.set_title("Solver runtime comparison")
    plt.xticks(rotation=20, ha="right")
    savefig(fig, os.path.join(outdir_figs, "solver_comparison_runtime"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--n-requests", type=int, default=2)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="A")
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    run(args.n_seeds, args.n_requests, args.k_paths, args.topology, args.outdir_tables, args.outdir_figs)
