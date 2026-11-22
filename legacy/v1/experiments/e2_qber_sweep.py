"""
Experiment 2 (E8): sweep QBER from baseline to near the BB84 abort
threshold and record how Q-OPT's chosen crypto-mode mix (QKD / PQC /
HYBRID) shifts, using the exact solver as the decision-maker (exact
is used here, not QAOA, because this experiment is about the
*optimization formulation's* behaviour, not about the quantum solver;
QAOA's job is to reproduce this decision surface, which Experiment 1
already establishes it does at AR~1.0 on small instances).

Produces:
    tables/e2_qber_sweep.csv
    figures/e2_qber_mode_selection.pdf/png

Run:
    python experiments/e2_qber_sweep.py
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


def run(n_seeds: int, n_requests: int, k_paths: int, topology: str,
        outdir_tables: str, outdir_figs: str):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    qber_values = np.linspace(0.01, 0.10, 10)  # up to near the ~0.11 GLLP abort bound
    records = []

    for qber in qber_values:
        for seed in range(n_seeds):
            G = get_topology(topology, seed=0)
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            # risk scales with QBER via the same normalization used in the risk engine
            risk = float(np.clip(qber / 0.11, 0, 1)) * 0.9 + 0.05
            link_state = {frozenset(e): dict(qber=float(qber), risk=risk, key_pool_frac=0.7)
                           for e in G.edges()}
            options = build_options(G, reqs, link_state, k_paths=k_paths)
            problem = build_qubo(options, reqs)
            result = solve_exact(problem, max_n=30)
            for opt in result.y.nonzero()[0]:
                mode = problem.options[opt].mode
                records.append(dict(qber=qber, seed=seed, mode=mode))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e2_qber_sweep.csv"), index=False)

    mix = (df.groupby(["qber", "mode"]).size().unstack(fill_value=0))
    mix = mix.div(mix.sum(axis=1), axis=0)  # fraction of selections per QBER level
    mix.to_csv(os.path.join(outdir_tables, "e2_qber_mode_mix.csv"))
    print(mix.to_string())

    _plot(mix, outdir_figs)
    return df, mix


def _plot(mix: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig

    set_style()
    colors = {"QKD": "#2a6fdb", "PQC": "#f0a35a", "HYBRID": "#3a6b35"}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bottom = np.zeros(len(mix))
    for mode in ["QKD", "HYBRID", "PQC"]:
        if mode not in mix.columns:
            continue
        ax.bar(mix.index, mix[mode], bottom=bottom, label=mode, color=colors[mode],
               width=0.007, alpha=0.9)
        bottom += mix[mode].values
    ax.set_xlabel("Channel QBER")
    ax.set_ylabel("Fraction of requests selecting mode")
    ax.set_title("Crypto-mode selection shift under increasing QBER")
    ax.axvline(0.11, color="black", linestyle=":", linewidth=1, label="BB84 abort bound (~0.11)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    savefig(fig, os.path.join(outdir_figs, "e2_qber_mode_selection"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-requests", type=int, default=3)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="B")
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    run(args.n_seeds, args.n_requests, args.k_paths, args.topology, args.outdir_tables, args.outdir_figs)
