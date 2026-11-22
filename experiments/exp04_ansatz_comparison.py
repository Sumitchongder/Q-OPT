"""
Experiment 4: Constraint-preserving XY-QAOA vs. textbook penalty QAOA
(exact noiseless simulation, held-out instances).

Instance families (topology C, load 1.5):  R in {2, 3, 4} with three modes
(6/9/12 qubits) and R=2 with two paths x three modes (12 qubits, block size 6).

  XY-QAOA   p = 1..3, line mixer, W-state initial state
  Penalty   p = 1..2, X mixer; penalty P in {0.25, 0.5, 1, 2} x spectral span.
            Reported both at the fixed P = 0.5 span used on hardware and at
            the per-instance *oracle* best P (generous to the baseline).

Outputs: tables/exp04_ansatz.csv, tables/exp04_ansatz_summary.csv,
         figures/exp04_ansatz.{pdf,png}
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import legend_cols, COLORS, TAB, manifest, mean_ci, savefig, style
from qopt.agility.model import random_instance
from qopt.quantum.fastsim import PenaltyQAOASim, XYQAOASim
from qopt.quantum.train import metrics_from_feasible_dist, train_depth_sweep

FAMILIES = [("R2-k3", 2, 1), ("R3-k3", 3, 1), ("R4-k3", 4, 1), ("R2-k6", 2, 2)]
PENALTIES = [0.25, 0.5, 1.0, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--p-xy", type=int, default=3)
    ap.add_argument("--p-pen", type=int, default=2)
    ap.add_argument("--plot-only", action="store_true", help="re-plot from tables/exp04_ansatz.csv")
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    if args.plot_only:
        rows = pd.read_csv(TAB / "exp04_ansatz.csv").to_dict("records")
    for fam, R, k in (FAMILIES if not args.plot_only else []):
        for s in range(args.n_seeds):
            prob = random_instance("C", n_requests=R, k_paths=k, seed=2000 + 100 * R + 10 * k + s, load=1.5)
            xy = XYQAOASim(prob)
            E = xy.E
            span = float(E.max() - E.min())
            base = dict(family=fam, seed=s, n_qubits=prob.n, n_feasible=prob.n_feasible)
            tr = train_depth_sweep(xy.expectation, args.p_xy, seed=s)
            for p, (x, _) in tr.items():
                rows.append(dict(**base, ansatz="XY", p=p, penalty=np.nan,
                                 **metrics_from_feasible_dist(E, xy.probs(x), 1.0)))
            for P in PENALTIES:
                pen = PenaltyQAOASim(prob, penalty=P * span, scale=xy.scale)
                trp = train_depth_sweep(pen.expectation, args.p_pen, seed=s)
                for p, (x, _) in trp.items():
                    pf, dist = pen.feasible_distribution(x)
                    rows.append(dict(**base, ansatz="PEN", p=p, penalty=P, **metrics_from_feasible_dist(E, dist, pf)))
        print(f"  {fam} done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    if not args.plot_only:
        df.to_csv(TAB / "exp04_ansatz.csv", index=False)

    # penalty variants: fixed P=0.5 and per-instance oracle best r_eff
    pen = df[df.ansatz == "PEN"]
    oracle = pen.loc[pen.groupby(["family", "seed", "p"]).r_eff.idxmax()].assign(variant="PEN (oracle P)")
    fixed = pen[pen.penalty == 0.5].assign(variant="PEN (P=0.5)")
    xyv = df[df.ansatz == "XY"].assign(variant="XY")
    allv = pd.concat([xyv, fixed, oracle])
    summ = []
    for (fam, var, p), g in allv.groupby(["family", "variant", "p"]):
        r = dict(family=fam, variant=var, p=p, n_qubits=int(g.n_qubits.iloc[0]))
        for m in ("p_feas", "r_feas", "r_eff", "p_opt", "pctl", "r_rand"):
            mu, lo, hi = mean_ci(g[m])
            r.update({m: mu, f"{m}_lo": lo, f"{m}_hi": hi})
        summ.append(r)
    summ = pd.DataFrame(summ)
    if not args.plot_only:
        summ.to_csv(TAB / "exp04_ansatz_summary.csv", index=False)
    print(summ[["family", "variant", "p", "p_feas", "r_feas", "r_eff", "p_opt"]].round(3).to_string(index=False))

    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    fams = [f for f, _, _ in FAMILIES]
    x = np.arange(len(fams))
    variants = [("XY", 1, COLORS["XY"], "XY p=1"), ("XY", 2, "#56B4E9", "XY p=2"),
                ("PEN (oracle P)", 1, COLORS["PEN"], "Penalty p=1 (oracle P)"),
                ("PEN (oracle P)", 2, "#F0C080", "Penalty p=2 (oracle P)")]
    w = 0.2
    for ax, m, lab in [(axes[0], "p_feas", "P(feasible)"), (axes[1], "r_eff", "Effective ratio r_eff"),
                       (axes[2], "p_opt", "P(optimum)")]:
        for j, (var, p, col, lb) in enumerate(variants):
            d = summ[(summ.variant == var) & (summ.p == p)].set_index("family").reindex(fams)
            if d[m].isna().all():      # depth not run (e.g. CI smoke test with --p-xy 1 --p-pen 1)
                continue
            ax.bar(x + (j - 1.5) * w, d[m], w, color=col, label=lb,
                   yerr=[d[m] - d[f"{m}_lo"], d[f"{m}_hi"] - d[m]], error_kw=dict(lw=0.6, capsize=1.5))
        if m == "r_eff":
            rr = summ[(summ.variant == "XY") & (summ.p == 1)].set_index("family").reindex(fams).r_rand
            ax.scatter(x, rr, marker="_", s=180, color="k", zorder=5, label="uniform-feasible floor")
        ax.set_xticks(x, fams, rotation=0, fontsize=7)
        ax.set_title(lab, fontsize=8.5)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=legend_cols(5), fontsize=6.3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92 if legend_cols(5) == 5 else 0.84))
    savefig(fig, "exp04_ansatz")
    manifest("exp04_ansatz_comparison", args, t0)


if __name__ == "__main__":
    main()
