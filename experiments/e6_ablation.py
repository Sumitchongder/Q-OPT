"""
Experiment 6 (Section 40 ablation study): remove one QUBO component at
a time and measure the resulting change in the FULL objective
(evaluated with the ORIGINAL/full weights, so ablated solutions are
scored fairly against the full model) to show which components matter.

Ablations:
    Full        - all terms (baseline)
    NoRisk      - w_R = 0 during solving (risk term ignored by the optimizer)
    NoKeyPool   - w_K = 0 (key-pool depletion term ignored)
    NoComputeCost - w_C = 0
    NoDisruption  - w_D = 0
    NoSecurityFloor - security-floor penalty disabled entirely

For every ablation, the optimizer solves with the ABLATED weights, but
we then re-score the resulting bitstring under the FULL (original)
objective -- this isolates "what does removing this term from the
DECISION cost" rather than just reporting a smaller number because we
deleted a positive term from the score.

Produces:
    tables/e6_ablation.csv
    tables/e6_ablation_summary.csv
    figures/e6_ablation.pdf/png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from qopt.network.topology import get_topology
from qopt.network.traffic import sample_requests
from qopt.optimization.qubo import build_options, build_qubo, evaluate_bitstring, DEFAULT_WEIGHTS
from qopt.optimization.classical_baselines import solve_exact

ABLATIONS = {
    "Full": {},
    "NoRisk (w_R=0)": {"w_R": 0.0},
    "NoKeyPool (w_K=0)": {"w_K": 0.0},
    "NoComputeCost (w_C=0)": {"w_C": 0.0},
    "NoDisruption (w_D=0)": {"w_D": 0.0},
}


def run(n_seeds: int, n_requests: int, k_paths: int, topology: str,
        outdir_tables: str, outdir_figs: str):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    records = []
    for seed in range(n_seeds):
        G = get_topology(topology, seed=0)
        reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
        rng = np.random.default_rng(seed + 4000)
        link_state = {
            frozenset(e): dict(qber=float(rng.uniform(0.01, 0.05)),
                                risk=float(rng.uniform(0.05, 0.4)),
                                key_pool_frac=float(rng.uniform(0.3, 0.95)))
            for e in G.edges()
        }
        options = build_options(G, reqs, link_state, k_paths=k_paths)

        # Full-objective problem, used as the fixed scoring rod for every ablation.
        full_problem = build_qubo(options, reqs, weights=DEFAULT_WEIGHTS)
        full_result = solve_exact(full_problem, max_n=30)
        full_score = full_result.objective

        for name, override in ABLATIONS.items():
            if name == "Full":
                records.append(dict(seed=seed, ablation=name,
                                     full_objective_of_ablated_solution=full_score,
                                     delta_vs_full=0.0))
                continue
            weights = dict(DEFAULT_WEIGHTS)
            weights.update(override)
            ablated_problem = build_qubo(options, reqs, weights=weights)
            ablated_result = solve_exact(ablated_problem, max_n=30)
            # Re-score the ablated solution's bitstring under the FULL objective.
            rescored = evaluate_bitstring(full_problem, ablated_result.y)
            records.append(dict(seed=seed, ablation=name,
                                 full_objective_of_ablated_solution=rescored["qubo_value"],
                                 delta_vs_full=rescored["qubo_value"] - full_score))

        # Security-floor ablation is structurally different (penalty removed,
        # not a weight zeroed), handled separately with penalty_security=0.
        no_floor_problem = build_qubo(options, reqs, weights=DEFAULT_WEIGHTS, penalty_security=0.0)
        no_floor_result = solve_exact(no_floor_problem, max_n=30)
        rescored = evaluate_bitstring(full_problem, no_floor_result.y)
        records.append(dict(seed=seed, ablation="NoSecurityFloor (penalty=0)",
                             full_objective_of_ablated_solution=rescored["qubo_value"],
                             delta_vs_full=rescored["qubo_value"] - full_score))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e6_ablation.csv"), index=False)

    summary = df.groupby("ablation").agg(
        mean_delta=("delta_vs_full", "mean"), std_delta=("delta_vs_full", "std"),
        mean_full_obj=("full_objective_of_ablated_solution", "mean"),
    ).reset_index().sort_values("mean_delta")
    summary.to_csv(os.path.join(outdir_tables, "e6_ablation_summary.csv"), index=False)
    print(summary.to_string(index=False))

    _plot(summary, outdir_figs)
    return df, summary


def _plot(summary: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig

    set_style()
    order = summary.sort_values("mean_delta", ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colors = ["#8c8c8c" if v == "Full" else "#d1495b" for v in order["ablation"]]
    ax.barh(order["ablation"], order["mean_delta"], xerr=order["std_delta"],
            color=colors, alpha=0.85, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta objective vs. full model (evaluated under full weights)\n"
                  "Higher = removing this component hurts solution quality more")
    ax.set_title("Ablation study: contribution of each QUBO component")
    savefig(fig, os.path.join(outdir_figs, "e6_ablation"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-requests", type=int, default=4)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="C")
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    run(args.n_seeds, args.n_requests, args.k_paths, args.topology, args.outdir_tables, args.outdir_figs)
