"""
Experiment 11 (Section 17 of the design notes): penalty-sensitivity
sweep.

IMPORTANT METHODOLOGICAL NOTE (found while building this): none of
`solve_exact`, `solve_greedy`, `solve_simulated_annealing`, or
`solve_milp` in this repo can ever report a one-hot constraint
violation, because each of them is DELIBERATELY implemented to search
only within the feasible (one-hot-respecting) subspace directly --
they never touch the penalty term at all when deciding what to try
next. That's correct design for a fair classical-baseline comparison,
but it means none of them can test what Section 17 actually asks:
"does minimizing the full penalized QUBO objective over the ENTIRE
2^n bitstring space naturally still prefer a feasible solution, or
does a too-weak penalty let the optimizer 'cheat' by picking an
infeasible (e.g. all-zero, or two-options-per-request) bitstring
because it's cheaper on the raw objective terms?"

This experiment therefore adds a genuine RAW BRUTE-FORCE search over
the full, unrestricted 2^n bitstring space (only tractable for small
n -- this is exactly why the design notes ask for a small-n penalty
study in the first place) and reports whether the GLOBAL minimum of
the penalized QUBO is feasible, for a sweep of penalty strengths
P in {0.5, 1, 2, 5, 10, 25, 50, 100} (applied to both
`penalty_onehot` and, separately, `penalty_security`, holding the
other fixed at a safe default).

Produces:
    tables/e12_penalty_sensitivity_onehot.csv
    tables/e12_penalty_sensitivity_security.csv
    figures/e12_penalty_sensitivity.pdf/png

Run:
    python experiments/e12_penalty_sensitivity.py --n-seeds 15
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from qopt.network.topology import get_topology
from qopt.network.traffic import sample_requests
from qopt.optimization.qubo import build_options, build_qubo, evaluate_bitstring, QUBOProblem

PENALTY_VALUES = [0.5, 1, 2, 5, 10, 25, 50, 100]


def raw_brute_force(problem: QUBOProblem, max_n: int = 14) -> dict:
    """
    Search the ENTIRE 2^n bitstring space (no one-hot restriction) for
    the bitstring that minimizes the full penalized QUBO objective, and
    report whether that global minimum happens to be feasible. This is
    the search regime QAOA/annealing actually operate in, unlike the
    structured classical baselines in this repo.
    """
    n = len(problem.options)
    if n > max_n:
        raise ValueError(f"n={n} too large for raw brute force (limit {max_n})")
    best_y, best_val = None, np.inf
    for bits in range(2 ** n):
        y = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        val = float(y @ problem.Q @ y + problem.offset)
        if val < best_val:
            best_val, best_y = val, y
    res = evaluate_bitstring(problem, best_y)
    return dict(global_min_value=best_val, constraint_violations=res["constraint_violations"],
                n_chosen=len(res["chosen"]))


def run(n_seeds: int, n_requests: int, k_paths: int, topology: str,
        outdir_tables: str, outdir_figs: str):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    G = get_topology(topology, seed=0)

    # --- Sweep 1: penalty_onehot, security penalty held at a safe fixed value ---
    records_onehot = []
    for P in PENALTY_VALUES:
        for seed in range(n_seeds):
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            rng = np.random.default_rng(seed + 10000)
            link_state = {frozenset(e): dict(qber=float(rng.uniform(0.01, 0.05)),
                                              risk=float(rng.uniform(0.05, 0.4)),
                                              key_pool_frac=float(rng.uniform(0.3, 0.95)))
                           for e in G.edges()}
            options = build_options(G, reqs, link_state, k_paths=k_paths)
            problem = build_qubo(options, reqs, penalty_onehot=P, penalty_security=50.0)
            if len(options) > 14:
                continue
            result = raw_brute_force(problem)
            records_onehot.append(dict(penalty_value=P, seed=seed, n_qubits=len(options), **result))

    df_onehot = pd.DataFrame(records_onehot)
    df_onehot.to_csv(os.path.join(outdir_tables, "e12_penalty_sensitivity_onehot.csv"), index=False)
    summary_onehot = df_onehot.groupby("penalty_value").agg(
        frac_feasible=("constraint_violations", lambda x: (x == 0).mean()),
        mean_violations=("constraint_violations", "mean"),
    ).reset_index()
    print("=== Penalty sweep: penalty_onehot (P_security fixed at 50) ===")
    print(summary_onehot.to_string(index=False))

    # --- Sweep 2: penalty_security, one-hot penalty held at a safe fixed value ---
    records_sec = []
    for P in PENALTY_VALUES:
        for seed in range(n_seeds):
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            rng = np.random.default_rng(seed + 11000)
            link_state = {frozenset(e): dict(qber=float(rng.uniform(0.01, 0.05)),
                                              risk=float(rng.uniform(0.05, 0.4)),
                                              key_pool_frac=float(rng.uniform(0.3, 0.95)))
                           for e in G.edges()}
            options = build_options(G, reqs, link_state, k_paths=k_paths)
            problem = build_qubo(options, reqs, penalty_onehot=50.0, penalty_security=P)
            if len(options) > 14:
                continue
            result = raw_brute_force(problem)

            # Also check SECURITY-floor compliance of the global minimum specifically
            # (separate from the one-hot feasibility check above): does a weak
            # security penalty let the optimizer drop below the required score?
            req_by_id = {r.req_id: r for r in reqs}
            n = len(options)
            best_y, best_val = None, np.inf
            for bits in range(2 ** n):
                y = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
                val = float(y @ problem.Q @ y + problem.offset)
                if val < best_val:
                    best_val, best_y = val, y
            ev = evaluate_bitstring(problem, best_y)
            n_secure = sum(1 for o in ev["chosen"] if o.security_score >= req_by_id[o.req_id].security_score_required)
            security_compliance = n_secure / len(reqs) if ev["chosen"] else 0.0

            records_sec.append(dict(penalty_value=P, seed=seed, n_qubits=len(options),
                                     **result, security_compliance=security_compliance))

    df_sec = pd.DataFrame(records_sec)
    df_sec.to_csv(os.path.join(outdir_tables, "e12_penalty_sensitivity_security.csv"), index=False)
    summary_sec = df_sec.groupby("penalty_value").agg(
        frac_feasible=("constraint_violations", lambda x: (x == 0).mean()),
        mean_security_compliance=("security_compliance", "mean"),
    ).reset_index()
    print("\n=== Penalty sweep: penalty_security (P_onehot fixed at 50) ===")
    print(summary_sec.to_string(index=False))

    _plot(summary_onehot, summary_sec, outdir_figs)
    return summary_onehot, summary_sec


def _plot(summary_onehot: pd.DataFrame, summary_sec: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    ax1 = axes[0]
    ax1.plot(summary_onehot.penalty_value, summary_onehot.frac_feasible, marker="o", color="#2a6fdb")
    ax1.set_xscale("log")
    ax1.set_xlabel("One-hot penalty strength P (log scale)")
    ax1.set_ylabel("Fraction of global optima that are feasible")
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_title("(a) One-hot penalty sensitivity")

    ax2 = axes[1]
    ax2.plot(summary_sec.penalty_value, summary_sec.mean_security_compliance, marker="s", color="#d1495b",
             label="Security-floor compliance")
    ax2.plot(summary_sec.penalty_value, summary_sec.frac_feasible, marker="o", color="#3a6b35",
             label="One-hot feasibility")
    ax2.set_xscale("log")
    ax2.set_xlabel("Security-floor penalty strength P (log scale)")
    ax2.set_ylabel("Fraction")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_title("(b) Security-floor penalty sensitivity")
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("Penalty-sensitivity sweep: global QUBO minimum stays feasible across P",
                 fontsize=10, y=1.03)
    savefig(fig, os.path.join(outdir_figs, "e12_penalty_sensitivity"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--n-requests", type=int, default=3)
    ap.add_argument("--k-paths", type=int, default=1)
    ap.add_argument("--topology", default="A")
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    run(args.n_seeds, args.n_requests, args.k_paths, args.topology, args.outdir_tables, args.outdir_figs)
