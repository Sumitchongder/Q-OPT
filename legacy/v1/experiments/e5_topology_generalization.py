"""
Experiment 5 (E13): generalization to unseen/larger topologies.

Runs the full solver comparison (Exact/Greedy/MILP/SA/QAOA p=2) on all
five topologies (A: 5 nodes, B: 6, C: 10, D: 15 random-geometric, E: 20
scale-free), so you can report whether solver behavior (approximation
ratio, runtime scaling, qubit count) holds up as the network grows --
the "does this scale" question reviewers ask.

IMPORTANT (documented, not hidden): QAOA's classical outer-loop
(COBYLA) optimization is run fresh on EACH topology in this script --
this is architecture/pipeline generalization (same solver, same
weights, re-optimized per instance), not literal parameter transfer.
True angle-transfer across different qubit counts is not physically
meaningful (the parameter vector length depends on n_qubits), so this
is the honest way to frame "generalization" here; see
docs/experiment_specs.md E13 for the distinction.

Produces:
    tables/e5_topology_generalization.csv
    tables/e5_topology_generalization_summary.csv
    figures/e5_topology_generalization.pdf/png

Run:
    python experiments/e5_topology_generalization.py --n-seeds 10
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

TOPOLOGY_SIZES = {"A": 5, "B": 6, "C": 10, "D": 15, "E": 20}


def run(n_seeds: int, n_requests: int, k_paths: int, topologies: list,
        outdir_tables: str, outdir_figs: str, run_qaoa: bool = True,
        qaoa_max_qubits: int = 14):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    records = []
    for topo in topologies:
        n_qubits = None
        for seed in range(n_seeds):
            G = get_topology(topo, seed=0)
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            rng = np.random.default_rng(seed + 3000)
            link_state = {
                frozenset(e): dict(qber=float(rng.uniform(0.01, 0.05)),
                                    risk=float(rng.uniform(0.05, 0.4)),
                                    key_pool_frac=float(rng.uniform(0.3, 0.95)))
                for e in G.edges()
            }
            options = build_options(G, reqs, link_state, k_paths=k_paths)
            problem = build_qubo(options, reqs)
            n_qubits = len(options)

            exact = solve_exact(problem, max_n=26) if n_qubits <= 26 else None
            greedy = solve_greedy(problem)
            milp = solve_milp(problem)
            sa = solve_simulated_annealing(problem, n_sweeps=1500, seed=seed)
            opt_ref = exact.objective if exact is not None else milp.objective

            for r in [greedy, milp, sa] + ([exact] if exact is not None else []):
                records.append(dict(topology=topo, n_nodes=TOPOLOGY_SIZES[topo], seed=seed,
                                     n_qubits=n_qubits, solver=r.solver, objective=r.objective,
                                     runtime_s=r.runtime_s, violations=r.constraint_violations,
                                     approximation_ratio=r.objective / opt_ref if opt_ref > 0 else np.nan))

            if run_qaoa and n_qubits <= qaoa_max_qubits:
                qr = run_qaoa_aer(problem, reps=2, shots=1024, maxiter=60, seed=seed,
                                   optimal_energy=opt_ref)
                records.append(dict(topology=topo, n_nodes=TOPOLOGY_SIZES[topo], seed=seed,
                                     n_qubits=n_qubits, solver="QAOA(p=2)", objective=qr.objective,
                                     runtime_s=qr.runtime_s, violations=qr.constraint_violations,
                                     approximation_ratio=qr.approximation_ratio))
        print(f"topology {topo} ({TOPOLOGY_SIZES[topo]} nodes) done, n_qubits={n_qubits}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e5_topology_generalization.csv"), index=False)

    summary = df.groupby(["topology", "n_nodes", "solver"]).agg(
        mean_AR=("approximation_ratio", "mean"), std_AR=("approximation_ratio", "std"),
        mean_runtime_ms=("runtime_s", lambda x: 1000 * x.mean()),
        mean_violations=("violations", "mean"), mean_n_qubits=("n_qubits", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(outdir_tables, "e5_topology_generalization_summary.csv"), index=False)
    print(summary.to_string(index=False))

    _plot(df, summary, outdir_figs)
    return df, summary


def _plot(df: pd.DataFrame, summary: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig, PALETTE

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))

    ax1 = axes[0]
    for solver in summary["solver"].unique():
        sub = summary[summary.solver == solver].sort_values("n_nodes")
        ax1.errorbar(sub.n_nodes, sub.mean_AR, yerr=sub.std_AR, marker="o",
                     label=solver, color=PALETTE.get(solver, None), capsize=3)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    ax1.set_xlabel("Network size (nodes)")
    ax1.set_ylabel("Approximation ratio")
    ax1.set_title("(a) Solution quality vs. network size")
    ax1.legend(fontsize=7, loc="upper left")

    ax2 = axes[1]
    for solver in summary["solver"].unique():
        sub = summary[summary.solver == solver].sort_values("n_nodes")
        ax2.plot(sub.n_nodes, sub.mean_runtime_ms, marker="o", label=solver,
                 color=PALETTE.get(solver, None))
    ax2.set_yscale("log")
    ax2.set_xlabel("Network size (nodes)")
    ax2.set_ylabel("Mean runtime (ms, log scale)")
    ax2.set_title("(b) Runtime scaling vs. network size")
    ax2.legend(fontsize=7, loc="upper left")

    fig.suptitle("Generalization across topologies A (5) -> E (20 nodes)", fontsize=10, y=1.03)
    savefig(fig, os.path.join(outdir_figs, "e5_topology_generalization"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-requests", type=int, default=2)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topologies", nargs="+", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--no-qaoa", action="store_true", help="Skip QAOA (faster, classical-only scaling check)")
    ap.add_argument("--qaoa-max-qubits", type=int, default=14)
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    run(args.n_seeds, args.n_requests, args.k_paths, args.topologies,
        args.outdir_tables, args.outdir_figs, run_qaoa=not args.no_qaoa,
        qaoa_max_qubits=args.qaoa_max_qubits)
