"""
Experiment 7 (E2 + E3 combined): exact small-instance validation and
QAOA depth sweep, with non-parametric significance testing (Section
52: Mann-Whitney U for pairwise comparisons rather than assuming
normality).

Part A (E2): for instances with EXACTLY n in {3,6,9} qubits (achieved
by varying n_requests/k_paths -- these are the smallest achievable
sizes in this formulation, since every candidate path contributes
exactly 3 mode-options; see _find_config_for_n_qubits docstring),
compare QAOA(p=2) against brute-force exact across N seeds; report
the optimality gap distribution.

Part B (E3): fix one n=6 instance family, sweep QAOA depth p=1,2,3,4,
report approximation ratio, transpiled circuit depth, and number of
circuit evaluations (COBYLA iterations) vs p. A Mann-Whitney U test
checks whether consecutive depths' AR distributions differ
significantly.

Produces:
    tables/e7a_exact_validation.csv
    tables/e7b_depth_sweep.csv
    tables/e7b_depth_sweep_mannwhitney.csv
    figures/e7a_exact_validation.pdf/png
    figures/e7b_depth_sweep.pdf/png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from qopt.network.topology import get_topology
from qopt.network.traffic import sample_requests
from qopt.optimization.qubo import build_options, build_qubo
from qopt.optimization.classical_baselines import solve_exact
from qopt.optimization.qaoa_qiskit import run_qaoa_aer, build_qaoa_circuit
from qopt.optimization.qubo_to_ising import qubo_to_ising, ising_to_sparse_pauli_op


def _find_config_for_n_qubits(target_n: int):
    """Search small (n_requests, k_paths) combos on topology A for one that
    yields exactly `target_n` options (qubits).

    STRUCTURAL NOTE (discovered while building this experiment, kept here
    rather than silently worked around): in the path+mode QUBO formulation,
    every candidate path always contributes exactly 3 mode-options (QKD,
    PQC, HYBRID). Therefore n_qubits = (number of distinct candidate paths
    found, summed across requests) * 3 -- it is ALWAYS a multiple of 3.
    n_qubits in {4, 5} (as in the original design-notes phrasing "n=4,5,6")
    are structurally unreachable in this formulation. We therefore validate
    at the nearest achievable analogous sizes: n in {3, 6, 9} (one path,
    two paths, three paths' worth of options), which is the correct
    honest reading of "exact validation at small qubit counts" here.
    """
    G = get_topology("A", seed=0)
    for n_requests in range(1, 5):
        for k_paths in range(1, 4):
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=0)
            link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
            options = build_options(G, reqs, link_state, k_paths=k_paths)
            if len(options) == target_n:
                return n_requests, k_paths
    return None


def run_part_a(n_seeds: int, outdir_tables: str, outdir_figs: str):
    records = []
    for target_n in [3, 6, 9]:  # see _find_config_for_n_qubits docstring: always a multiple of 3
        cfg = _find_config_for_n_qubits(target_n)
        if cfg is None:
            print(f"WARNING: could not find a config with exactly n_qubits={target_n} on topology A; skipping.")
            continue
        n_requests, k_paths = cfg
        G = get_topology("A", seed=0)
        for seed in range(n_seeds):
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            rng = np.random.default_rng(seed + 5000)
            link_state = {frozenset(e): dict(qber=float(rng.uniform(0.01, 0.05)),
                                              risk=float(rng.uniform(0.05, 0.4)),
                                              key_pool_frac=float(rng.uniform(0.3, 0.95)))
                           for e in G.edges()}
            options = build_options(G, reqs, link_state, k_paths=k_paths)
            if len(options) != target_n:
                continue  # link-state randomness doesn't change n_qubits, but guard anyway
            problem = build_qubo(options, reqs)
            exact = solve_exact(problem, max_n=10)
            qaoa = run_qaoa_aer(problem, reps=2, shots=2048, maxiter=100, seed=seed,
                                 optimal_energy=exact.objective)
            gap = (qaoa.objective - exact.objective) / abs(exact.objective) if exact.objective != 0 else 0.0
            records.append(dict(n_qubits=target_n, seed=seed, exact_obj=exact.objective,
                                 qaoa_obj=qaoa.objective, optimality_gap=gap,
                                 approximation_ratio=qaoa.approximation_ratio))
        print(f"n_qubits={target_n}: config n_requests={n_requests}, k_paths={k_paths} -> {n_seeds} seeds done")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e7a_exact_validation.csv"), index=False)
    summary = df.groupby("n_qubits").agg(
        mean_gap=("optimality_gap", "mean"), std_gap=("optimality_gap", "std"),
        mean_AR=("approximation_ratio", "mean"), std_AR=("approximation_ratio", "std"),
    ).reset_index()
    print(summary.to_string(index=False))

    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig
    set_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ns = sorted(df["n_qubits"].unique())
    data = [df[df.n_qubits == n]["optimality_gap"].values for n in ns]
    ax.boxplot(data, tick_labels=[str(n) for n in ns], showmeans=True, meanline=True)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Number of qubits (n)")
    ax.set_ylabel("QAOA(p=2) optimality gap vs. exact")
    ax.set_title("Exact validation: QAOA gap at n=3,6,9 qubits\n(smallest achievable sizes in the path+mode formulation)")
    savefig(fig, os.path.join(outdir_figs, "e7a_exact_validation"))
    plt.close(fig)
    return df, summary


def run_part_b(n_seeds: int, n_requests: int, k_paths: int, topology: str,
                outdir_tables: str, outdir_figs: str):
    G = get_topology(topology, seed=0)
    records = []
    depth_records = []
    for p in [1, 2, 3, 4]:
        for seed in range(n_seeds):
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            rng = np.random.default_rng(seed + 6000)
            link_state = {frozenset(e): dict(qber=float(rng.uniform(0.01, 0.05)),
                                              risk=float(rng.uniform(0.05, 0.4)),
                                              key_pool_frac=float(rng.uniform(0.3, 0.95)))
                           for e in G.edges()}
            options = build_options(G, reqs, link_state, k_paths=k_paths)
            problem = build_qubo(options, reqs)
            exact = solve_exact(problem, max_n=20)
            qr = run_qaoa_aer(problem, reps=p, shots=1024, maxiter=80, seed=seed,
                               optimal_energy=exact.objective)

            model = qubo_to_ising(problem.Q, problem.offset)
            cost_op = ising_to_sparse_pauli_op(model)
            circ = build_qaoa_circuit(cost_op, reps=p)
            depth = circ.decompose(reps=2).depth()

            records.append(dict(p=p, seed=seed, approximation_ratio=qr.approximation_ratio,
                                 runtime_s=qr.runtime_s, n_circuit_evals=qr.n_circuit_evals,
                                 circuit_depth=depth))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e7b_depth_sweep.csv"), index=False)

    summary = df.groupby("p").agg(
        mean_AR=("approximation_ratio", "mean"), std_AR=("approximation_ratio", "std"),
        mean_depth=("circuit_depth", "mean"), mean_evals=("n_circuit_evals", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))

    mw_records = []
    ps = sorted(df["p"].unique())
    for i in range(len(ps) - 1):
        a = df[df.p == ps[i]]["approximation_ratio"].values
        b = df[df.p == ps[i + 1]]["approximation_ratio"].values
        try:
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            u_stat, p_val = np.nan, np.nan
        mw_records.append(dict(p_a=ps[i], p_b=ps[i + 1], u_statistic=u_stat, p_value=p_val,
                                significant_at_0_05=bool(p_val < 0.05) if not np.isnan(p_val) else None))
    mw_df = pd.DataFrame(mw_records)
    mw_df.to_csv(os.path.join(outdir_tables, "e7b_depth_sweep_mannwhitney.csv"), index=False)
    print("\nMann-Whitney U tests between consecutive depths (approximation ratio):")
    print(mw_df.to_string(index=False))

    _plot_depth(summary, outdir_figs)
    return df, summary, mw_df


def _plot_depth(summary: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig

    set_style()
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))
    ax1.plot(summary.p, summary.mean_AR, marker="o", color="#2a6fdb", label="Approximation ratio")
    ax1.fill_between(summary.p, summary.mean_AR - summary.std_AR, summary.mean_AR + summary.std_AR,
                      color="#2a6fdb", alpha=0.15)
    ax1.set_xlabel("QAOA depth p")
    ax1.set_ylabel("Approximation ratio", color="#2a6fdb")
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=0.7)
    ax1.tick_params(axis="y", labelcolor="#2a6fdb")

    ax2 = ax1.twinx()
    ax2.plot(summary.p, summary.mean_depth, marker="s", color="#d1495b", label="Circuit depth")
    ax2.set_ylabel("Transpiled circuit depth", color="#d1495b")
    ax2.tick_params(axis="y", labelcolor="#d1495b")
    ax2.grid(False)

    ax1.set_title("QAOA depth sweep: solution quality vs. circuit cost")
    savefig(fig, os.path.join(outdir_figs, "e7b_depth_sweep"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--n-requests", type=int, default=2)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="B")
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    ap.add_argument("--skip-part-a", action="store_true")
    ap.add_argument("--skip-part-b", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir_tables, exist_ok=True)
    os.makedirs(args.outdir_figs, exist_ok=True)

    if not args.skip_part_a:
        print("=== Part A: exact validation at n=4,5,6 ===")
        run_part_a(args.n_seeds, args.outdir_tables, args.outdir_figs)
    if not args.skip_part_b:
        print("\n=== Part B: QAOA depth sweep p=1..4 ===")
        run_part_b(args.n_seeds, args.n_requests, args.k_paths, args.topology,
                   args.outdir_tables, args.outdir_figs)
