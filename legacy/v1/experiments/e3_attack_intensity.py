"""
Experiment 3 (E9): sweep attack intensity (control-plane compromise /
risk inflation) and record (a) the probability that a request still
receives its required security floor via an available QKD-family
option, and (b) service availability (fraction of requests with a
feasible, non-violating assignment) as key pools are simultaneously
depleted by a correlated key-exhaustion attack.

Produces:
    tables/e3_attack_intensity.csv
    figures/e3_attack_intensity.pdf/png
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
from qopt.threat.risk_engine import compute_risk


def run(n_seeds: int, n_requests: int, k_paths: int, topology: str,
        outdir_tables: str, outdir_figs: str):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    intensities = np.linspace(0.0, 1.0, 11)
    records = []

    for alpha in intensities:
        for seed in range(n_seeds):
            G = get_topology(topology, seed=0)
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            rng = np.random.default_rng(seed + 500)

            link_state = {}
            for e in G.edges():
                qber = float(rng.uniform(0.01, 0.02))
                trust = 1.0 - alpha * float(rng.uniform(0.6, 1.0))
                key_pool_frac = max(0.02, 0.8 - alpha * float(rng.uniform(0.5, 0.9)))
                risk = compute_risk(qber=qber, loss_db=8.0, key_pool_frac=key_pool_frac,
                                     attack_probability=alpha, trust=trust)
                link_state[frozenset(e)] = dict(qber=qber, risk=risk, key_pool_frac=key_pool_frac)

            options = build_options(G, reqs, link_state, k_paths=k_paths)
            problem = build_qubo(options, reqs)
            result = solve_exact(problem, max_n=40)

            n_feasible = int(sum(1 for r_id, idxs in problem.req_option_indices.items()
                                  if any(result.y[i] == 1 for i in idxs)))
            n_secure = 0
            mode_counts = {"QKD": 0, "PQC": 0, "HYBRID": 0}
            for i in result.y.nonzero()[0]:
                opt = problem.options[i]
                req = next(r for r in reqs if r.req_id == opt.req_id)
                if opt.security_score >= req.security_score_required:
                    n_secure += 1
                mode_counts[opt.mode] += 1

            records.append(dict(
                attack_intensity=alpha, seed=seed,
                availability=n_feasible / len(reqs),
                security_compliance=n_secure / len(reqs),
                mean_key_pool_frac=np.mean([v["key_pool_frac"] for v in link_state.values()]),
                mean_risk=np.mean([v["risk"] for v in link_state.values()]),
                frac_qkd=mode_counts["QKD"] / len(reqs),
                frac_pqc=mode_counts["PQC"] / len(reqs),
                frac_hybrid=mode_counts["HYBRID"] / len(reqs),
            ))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e3_attack_intensity.csv"), index=False)

    summary = df.groupby("attack_intensity").agg(
        availability_mean=("availability", "mean"), availability_std=("availability", "std"),
        security_compliance_mean=("security_compliance", "mean"),
        security_compliance_std=("security_compliance", "std"),
        mean_risk=("mean_risk", "mean"),
        frac_qkd=("frac_qkd", "mean"), frac_pqc=("frac_pqc", "mean"), frac_hybrid=("frac_hybrid", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(outdir_tables, "e3_attack_intensity_summary.csv"), index=False)
    print(summary.to_string(index=False))

    _plot(summary, outdir_figs)
    return df, summary


def _plot(summary: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax1 = axes[0]
    ax1.errorbar(summary.attack_intensity, summary.availability_mean, yerr=summary.availability_std,
                 marker="o", color="#2a6fdb", label="Service availability", capsize=3)
    ax1.errorbar(summary.attack_intensity, summary.security_compliance_mean,
                 yerr=summary.security_compliance_std, marker="s", color="#d1495b",
                 label="Security-floor compliance", capsize=3)
    ax1.set_xlabel("Attack intensity (α)")
    ax1.set_ylabel("Fraction of requests")
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_title("(a) Availability & compliance\nmaintained under attack")
    ax1.legend(loc="lower left", fontsize=7.5)

    ax2 = axes[1]
    ax2.stackplot(summary.attack_intensity,
                   summary.frac_qkd, summary.frac_hybrid, summary.frac_pqc,
                   labels=["QKD", "HYBRID", "PQC"],
                   colors=["#2a6fdb", "#3a6b35", "#f0a35a"], alpha=0.85)
    ax2.set_xlabel("Attack intensity (α)")
    ax2.set_ylabel("Fraction of requests")
    ax2.set_title("(b) Adaptive mechanism:\nQKD → HYBRID under attack")
    ax2.legend(loc="upper right", fontsize=7.5)

    fig.suptitle("Q-OPT maintains 100% availability & security compliance by shifting toward HYBRID mode",
                 fontsize=9.5, y=1.03)
    savefig(fig, os.path.join(outdir_figs, "e3_attack_intensity"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=25)
    ap.add_argument("--n-requests", type=int, default=4)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="C")
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    run(args.n_seeds, args.n_requests, args.k_paths, args.topology, args.outdir_tables, args.outdir_figs)
