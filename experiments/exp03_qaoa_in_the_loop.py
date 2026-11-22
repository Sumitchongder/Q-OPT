"""
Experiment 3: XY-QAOA as the per-epoch controller inside the digital twin.

Scale: R=4 requests, three modes, one path (12 qubits, 81 feasible states).
At this size best-of-many sampling is trivially optimal, so the question is
*sample efficiency*: with a budget of S shots per epoch, how close does each
sampler come to exact control?

  XY-QAOA (fixed angles)  p=1, angles transferred from 10 development
                          instances (seeds 500-509), no training at run time
  Random feasible         S uniform samples from the feasible set
  references              Exact, coupled Greedy, Separable (v0.1)

Sampling is from the exact noiseless distribution (shot noise only).
Outputs: tables/exp03_in_loop.csv, tables/exp03_in_loop_summary.csv,
         tables/exp03_transfer_angles.csv, figures/exp03_in_loop.{pdf,png}
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import COLORS, TAB, manifest, mean_ci, savefig, style
from qopt.agility import solvers as S
from qopt.agility.model import random_instance
from qopt.agility.twin import TwinConfig, run_twin
from qopt.quantum.fastsim import XYQAOASim
from qopt.quantum.train import train_depth_sweep


def transfer_angles(n_dev=10):
    rows = []
    for s in range(n_dev):
        p = random_instance("C", n_requests=4, k_paths=1, seed=500 + s, load=1.5)
        x = train_depth_sweep(XYQAOASim(p).expectation, 1, seed=s)[1][0]
        if x[0] > 0:            # canonicalise the (gamma, beta) -> (-gamma, -beta) symmetry
            x = -x
        rows.append(dict(seed=500 + s, gamma=x[0], beta=x[1]))
    df = pd.DataFrame(rows)
    return np.array([df.gamma.median(), df.beta.median()]), df


def transfer_check(angles, n_seeds=30):
    """Fixed transferred angles vs instance-trained angles on held-out instances (seeds 900+)."""
    from qopt.quantum.train import metrics_from_feasible_dist
    rows = []
    for s in range(n_seeds):
        for R, k in [(4, 1), (5, 1), (3, 2)]:
            p = random_instance("C", n_requests=R, k_paths=k, seed=900 + s, load=1.5)
            sim = XYQAOASim(p)
            xt = train_depth_sweep(sim.expectation, 1, seed=s)[1][0]
            mt = metrics_from_feasible_dist(sim.E, sim.probs(xt), 1.0)
            mf = metrics_from_feasible_dist(sim.E, sim.probs(angles), 1.0)
            rows.append(dict(seed=900 + s, family=f"R{R}-k{3 * k}", n_qubits=p.n, r_trained=mt["r_feas"],
                             r_fixed=mf["r_feas"], r_rand=mt["r_rand"], p_opt_trained=mt["p_opt"],
                             p_opt_fixed=mf["p_opt"], p_opt_rand=1.0 / p.n_feasible))
    df = pd.DataFrame(rows)
    df["retained_gain"] = (df.r_fixed - df.r_rand) / (df.r_trained - df.r_rand)
    return df


def best_of(prob, idx_samples, E):
    k = idx_samples[np.argmin(E[idx_samples])]
    sizes = np.array(prob.block_sizes)
    radix = np.concatenate([[1], np.cumprod(sizes[:-1])])
    return list(((k // radix) % sizes).astype(int))


def make_sampler_policy(kind, shots, angles, seed):
    rng = np.random.default_rng([seed, shots, 0 if kind == "xy" else 1])

    def policy(prob):
        sim = XYQAOASim(prob)
        pr = sim.probs(angles) if kind == "xy" else np.full(sim.E.size, 1.0 / sim.E.size)
        smp = rng.choice(sim.E.size, size=shots, p=pr / pr.sum())
        return best_of(prob, smp, sim.E)
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--transfer-check", action="store_true",
                    help="only evaluate fixed vs trained angles on 90 held-out instances")
    args = ap.parse_args()
    t0 = time.time()
    angles, dev = transfer_angles()
    dev.to_csv(TAB / "exp03_transfer_angles.csv", index=False)
    print("transferred p=1 angles (gamma, beta):", angles.round(4))
    if args.transfer_check:
        tc = transfer_check(angles)
        tc.to_csv(TAB / "exp03_transfer_check.csv", index=False)
        print(tc[["r_trained", "r_fixed", "r_rand", "p_opt_trained", "p_opt_fixed", "p_opt_rand"]].mean().round(4).to_dict(),
              "retained gain (mean):", round(float(tc.retained_gain.mean()), 4), "n =", len(tc))
        print(tc.groupby("family")[["r_trained", "r_fixed", "r_rand", "retained_gain"]].mean().round(4))
        return
    rows = []
    for seed in range(200, 200 + args.n_seeds):
        cfg = TwinConfig(n_requests=4, k_paths=1, load=1.5, attack_intensity=0.6, n_epochs=args.n_epochs, seed=seed)
        refs = {"Q-OPT v2 (exact)": lambda p: S.solve_exact(p).choice,
                "Greedy (coupled)": lambda p: S.solve_greedy(p).choice,
                "Separable (v0.1)": lambda p: S.solve_separable(p).choice}
        for name, fn in refs.items():
            rows.append(dict(seed=seed, policy=name, shots=np.nan, **run_twin(fn, cfg)))
        for sh in args.shots:
            for kind, label in [("xy", "XY-QAOA p=1 (fixed angles)"), ("rand", "Random feasible")]:
                rows.append(dict(seed=seed, policy=label, shots=sh,
                                 **run_twin(make_sampler_policy(kind, sh, angles, seed), cfg)))
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "exp03_in_loop.csv", index=False)
    summ = []
    for (pol, sh), g in df.groupby(["policy", "shots"], dropna=False):
        r = dict(policy=pol, shots=sh)
        for m in ("compliance", "mean_objective", "expected_compromises_per_1k", "downgrade_rate"):
            mu, lo, hi = mean_ci(g[m])
            r.update({m: mu, f"{m}_lo": lo, f"{m}_hi": hi})
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(TAB / "exp03_in_loop_summary.csv", index=False)
    print(summ[["policy", "shots", "compliance", "mean_objective"]].round(4).to_string(index=False))

    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, m, lab in [(axes[0], "mean_objective", "Mean objective per epoch"),
                       (axes[1], "compliance", "Compliance")]:
        for label, col in [("XY-QAOA p=1 (fixed angles)", COLORS["XY"]), ("Random feasible", "#999999")]:
            d = summ[summ.policy == label].sort_values("shots")
            ax.plot(d.shots, d[m], marker="o", color=col, label=label)
            ax.fill_between(d.shots, d[f"{m}_lo"], d[f"{m}_hi"], color=col, alpha=0.15, lw=0)
        for ref, ls in [("Q-OPT v2 (exact)", "-"), ("Greedy (coupled)", "--"), ("Separable (v0.1)", ":")]:
            v = summ[summ.policy == ref][m].iloc[0]
            ax.axhline(v, color=COLORS[ref], ls=ls, lw=1.1, label=ref)
        ax.set_xscale("log", base=2)
        if m == "mean_objective":
            from matplotlib.ticker import NullFormatter, ScalarFormatter
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Shots per control epoch S")
        ax.set_title(lab, fontsize=8.5)
    axes[0].legend(fontsize=5.5, loc="upper right")
    fig.tight_layout()
    savefig(fig, "exp03_in_loop")
    manifest("exp03_qaoa_in_the_loop", args, t0, extra=dict(angles=angles.tolist()))


if __name__ == "__main__":
    main()
