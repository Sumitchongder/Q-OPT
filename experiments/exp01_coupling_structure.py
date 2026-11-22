"""
Experiment 1: Why the formulation had to change.

Part A  (v0.1 QUBO)  Measures the fraction of inter-request couplings in the
        v0.1 QUBO and the rate at which per-request greedy equals the exact
        optimum. A block-separable QUBO has zero inter-request couplings and
        greedy is exact by construction.
Part B  (v2 coupled) Optimality gap of every heuristic against the exact
        optimum of the coupled problem, versus number of requests and load.

Outputs: tables/exp01_v01_separability.csv, tables/exp01_heuristic_gaps.csv,
         tables/exp01_heuristic_gaps_summary.csv, figures/exp01_heuristic_gaps.{pdf,png}
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import COLORS, TAB, manifest, mean_ci, savefig, style


def part_a(n_seeds):
    from qopt.network.topology import get_topology
    from qopt.network.traffic import sample_requests
    from qopt.optimization.classical_baselines import solve_exact, solve_greedy
    from qopt.optimization.qubo import build_options, build_qubo
    rows = []
    for topo, R, k in [("A", 2, 2), ("B", 3, 1), ("C", 4, 1)]:
        for s in range(n_seeds):
            G = get_topology(topo, seed=s)
            reqs = sample_requests(list(G.nodes()), n_requests=R, seed=s)
            rng = np.random.default_rng(s)
            ls = {frozenset(e): dict(qber=float(rng.uniform(0.01, 0.08)), risk=float(rng.uniform(0.05, 0.5)),
                                     key_pool_frac=float(rng.uniform(0.1, 0.9))) for e in G.edges()}
            prob = build_qubo(build_options(G, reqs, ls, k_paths=k), reqs)
            blk = np.empty(len(prob.options), int)
            for rid, idx in prob.req_option_indices.items():
                blk[idx] = rid
            iu = np.triu_indices(len(blk), 1)
            nz = np.abs(prob.Q[iu]) > 1e-12
            inter = int((nz & (blk[iu[0]] != blk[iu[1]])).sum())
            ex, gr = solve_exact(prob), solve_greedy(prob)
            rows.append(dict(topology=topo, n_requests=R, seed=s, n_qubits=len(blk),
                             inter_request_couplings=inter, greedy_equals_exact=bool(abs(gr.objective - ex.objective) < 1e-9)))
    return pd.DataFrame(rows)


def part_b(sizes, loads, n_seeds, k_paths):
    from qopt.agility.model import random_instance
    from qopt.agility.solvers import ALL_HEURISTICS, feasible_spectrum, solve_exact
    rows = []
    for R in sizes:
        for load in loads:
            for s in range(n_seeds):
                prob = random_instance("C", n_requests=R, k_paths=k_paths, seed=1000 * R + s, load=load)
                ex = solve_exact(prob)
                E = feasible_spectrum(prob)
                span = float(E.max() - E.min())
                for name, fn in ALL_HEURISTICS.items():
                    sol = fn(prob)
                    rows.append(dict(n_requests=R, load=load, seed=s, solver=name, n_vars=prob.n,
                                     n_feasible=prob.n_feasible,
                                     gap_rel_span=(sol.energy - ex.energy) / span,
                                     optimal=bool(sol.energy <= ex.energy + 1e-9 * span),
                                     runtime_ms=1e3 * sol.runtime_s))
                rows.append(dict(n_requests=R, load=load, seed=s, solver="Exact", n_vars=prob.n,
                                 n_feasible=prob.n_feasible, gap_rel_span=0.0, optimal=True,
                                 runtime_ms=1e3 * ex.runtime_s))
    return pd.DataFrame(rows)


def plot(df):
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    solvers = ["Separable (v0.1)", "Rule-based", "Greedy (coupled)", "Tabu search", "Simulated annealing"]
    for ax, metric, lab in [(axes[0], "gap_rel_span", "Optimality gap / spectral span"),
                            (axes[1], "optimal", "P(global optimum)")]:
        for s in solvers:
            sub = df[df.solver == s]
            g = sub.groupby("n_requests")[metric]
            xs = sorted(sub.n_requests.unique())
            m = [mean_ci(g.get_group(x).astype(float))[0] for x in xs]
            lo = [mean_ci(g.get_group(x).astype(float))[1] for x in xs]
            hi = [mean_ci(g.get_group(x).astype(float))[2] for x in xs]
            ax.plot(xs, m, marker="o", color=COLORS[s], label=s)
            ax.fill_between(xs, lo, hi, color=COLORS[s], alpha=0.15, lw=0)
        ax.set_xlabel("Requests per epoch R")
        ax.set_ylabel(lab)
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(loc="center left", ncol=1, fontsize=6)
    fig.tight_layout()
    savefig(fig, "exp01_heuristic_gaps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=40)
    ap.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--loads", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="re-plot from tables/exp01_heuristic_gaps.csv")
    ap.add_argument("--merge", action="store_true", help="merge these sizes into an existing gaps CSV")
    args = ap.parse_args()
    t0 = time.time()
    if args.plot_only:
        plot(pd.read_csv(TAB / "exp01_heuristic_gaps.csv"))
        return
    if not args.skip_a:
        a = part_a(min(args.n_seeds, 30))
        a.to_csv(TAB / "exp01_v01_separability.csv", index=False)
        print("[A] v0.1: inter-request couplings (max) =", a.inter_request_couplings.max(),
              "| greedy == exact in", f"{100 * a.greedy_equals_exact.mean():.1f}% of", len(a), "instances")
    b = part_b(args.sizes, args.loads, args.n_seeds, args.k_paths)
    path = TAB / "exp01_heuristic_gaps.csv"
    if args.merge and path.exists():
        old = pd.read_csv(path)
        b = pd.concat([old[~old.n_requests.isin(args.sizes)], b], ignore_index=True)
    b.to_csv(path, index=False)
    summ = []
    for (R, s), g in b.groupby(["n_requests", "solver"]):
        m, lo, hi = mean_ci(g.gap_rel_span)
        summ.append(dict(n_requests=R, solver=s, n_feasible=int(g.n_feasible.iloc[0]),
                         gap_mean=m, gap_ci_lo=lo, gap_ci_hi=hi, p_opt=g.optimal.mean(),
                         runtime_ms_median=g.runtime_ms.median()))
    summ = pd.DataFrame(summ)
    summ.to_csv(TAB / "exp01_heuristic_gaps_summary.csv", index=False)
    print(summ.pivot(index="solver", columns="n_requests", values="p_opt").round(3))
    plot(b)
    manifest("exp01_coupling_structure", args, t0)


if __name__ == "__main__":
    main()
