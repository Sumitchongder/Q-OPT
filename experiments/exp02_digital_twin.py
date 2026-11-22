"""
Experiment 2: Network-level value of coupled crypto-agility (headline).

Seven policies run on the multi-epoch digital twin (hard key budgets,
Markov attack process, link denials; common random numbers across
policies). Two sweeps:
   (i)  attack intensity in {0, 0.3, 0.6, 0.9} at load 1.5
   (ii) offered load in {0.5, 1, 1.5, 2, 3}  at attack intensity 0.6
Paired Wilcoxon signed-rank tests (Q-OPT v2 vs each baseline, per seed),
Holm-corrected across baselines within each sweep point.

Weights are the a-priori DEFAULT_WEIGHTS in qopt.agility.model; they were
not tuned on these seeds.

Outputs: tables/exp02_twin_{runs,summary,tests}.csv, tables/exp02_trace.csv,
         figures/exp02_twin_sweeps.{pdf,png}, figures/exp02_trace.{pdf,png}
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from _common import legend_cols, COLORS, TAB, holm, manifest, mean_ci, paired_test, savefig, style
from qopt.agility import solvers as S
from qopt.agility.twin import TwinConfig, run_twin

POLICIES = {
    "Static-QKD": lambda p: S.solve_static(p, "QKD").choice,
    "Static-PQC": lambda p: S.solve_static(p, "PQC").choice,
    "Static-HYB": lambda p: S.solve_static(p, "HYB").choice,
    "Rule-based": lambda p: S.solve_rule_based(p).choice,
    "Separable (v0.1)": lambda p: S.solve_separable(p).choice,
    "Greedy (coupled)": lambda p: S.solve_greedy(p).choice,
    "Q-OPT v2 (exact)": lambda p: S.solve_exact(p).choice,
}
METRICS = ["compliance", "critical_compliance", "downgrade_rate", "expected_compromises_per_1k",
           "sla_violation_rate", "entropy_shortfall_rate", "key_utilisation"]
REF = "Q-OPT v2 (exact)"


def sweep(name, values, make_cfg, seeds):
    rows = []
    for v in values:
        for s in seeds:
            cfg = make_cfg(v, s)
            for pol, fn in POLICIES.items():
                out = run_twin(fn, cfg)
                rows.append(dict(sweep=name, value=v, seed=s, policy=pol, **out))
        print(f"  {name}={v} done")
    return rows


def summarise(df):
    summ, tests = [], []
    for (sw, v), g in df.groupby(["sweep", "value"]):
        for pol, gp in g.groupby("policy"):
            row = dict(sweep=sw, value=v, policy=pol)
            for m in METRICS:
                mu, lo, hi = mean_ci(gp[m])
                row.update({m: mu, f"{m}_lo": lo, f"{m}_hi": hi})
            summ.append(row)
        ref = g[g.policy == REF].sort_values("seed")
        block = []
        for pol in POLICIES:
            if pol == REF:
                continue
            other = g[g.policy == pol].sort_values("seed")
            for m in ("compliance", "expected_compromises_per_1k", "downgrade_rate"):
                d, p = paired_test(ref[m].values, other[m].values)
                block.append(dict(sweep=sw, value=v, baseline=pol, metric=m, median_diff_ref_minus_base=d, p=p))
        adj = holm([b["p"] for b in block])
        for b, a in zip(block, adj):
            b["p_holm"] = a
        tests += block
    return pd.DataFrame(summ), pd.DataFrame(tests)


def plot(summ, trace):
    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    panels = [("compliance", "Compliance"), ("expected_compromises_per_1k", "Compromises per 1000"),
              ("downgrade_rate", "Downgrade rate")]
    show = ["Static-QKD", "Static-HYB", "Rule-based", "Separable (v0.1)", "Greedy (coupled)", "Q-OPT v2 (exact)"]
    for row, (sw, xl) in enumerate([("attack", "Attack intensity α"), ("load", "Offered load")]):
        sub = summ[summ.sweep == sw]
        for col, (m, lab) in enumerate(panels):
            ax = axes[row, col]
            for pol in show:
                d = sub[sub.policy == pol].sort_values("value")
                ax.plot(d.value, d[m], marker="o", color=COLORS[pol], label=pol,
                        lw=2.0 if pol == REF else 1.2, zorder=3 if pol == REF else 2)
                ax.fill_between(d.value, d[f"{m}_lo"], d[f"{m}_hi"], color=COLORS[pol], alpha=0.12, lw=0)
            ax.set_xlabel(xl)
            if row == 0:
                ax.set_title(lab)
            if m == "expected_compromises_per_1k":
                ax.set_yscale("log")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=legend_cols(6), fontsize=6.5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96 if legend_cols(6) == 6 else 0.91))
    savefig(fig, "exp02_twin_sweeps")

    fig, ax = plt.subplots(2, 1, figsize=(4.6, 3.6), sharex=True)
    for pol in ["Separable (v0.1)", "Greedy (coupled)", "Q-OPT v2 (exact)"]:
        d = trace[trace.policy == pol]
        ax[0].plot(d.epoch, d.mean_pool_frac, color=COLORS[pol], label=pol)
        ax[1].plot(d.epoch, d.compliance.rolling(3, min_periods=1).mean(), color=COLORS[pol])
    att = trace[trace.policy == REF]
    ax0b = ax[0].twinx()
    ax0b.bar(att.epoch, att.n_attacked, color="#cc0000", alpha=0.15, width=1.0)
    ax0b.set_ylabel("links under attack", color="#cc0000", fontsize=7)
    ax0b.grid(False)
    ax[0].set_ylabel("mean pool fill")
    ax[1].set_ylabel("compliance (3-ep. avg)")
    ax[1].set_xlabel("epoch")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig(fig, "exp02_trace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--n-epochs", type=int, default=40)
    ap.add_argument("--n-requests", type=int, default=6)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="C")
    ap.add_argument("--reuse", action="store_true", help="re-plot from tables/exp02_twin_runs.csv")
    args = ap.parse_args()
    t0 = time.time()
    seeds = list(range(100, 100 + args.n_seeds))     # evaluation seeds disjoint from development seeds 0-9
    base = dict(topology=args.topology, n_epochs=args.n_epochs, n_requests=args.n_requests, k_paths=args.k_paths)
    if args.reuse and (TAB / "exp02_twin_runs.csv").exists():
        df = pd.read_csv(TAB / "exp02_twin_runs.csv")
    else:
        df = None
    rows = [] if df is not None else sweep("attack", [0.0, 0.3, 0.6, 0.9],
                 lambda v, s: TwinConfig(**base, load=1.5, attack_intensity=v, seed=s), seeds)
    if df is None:
        rows += sweep("load", [0.5, 1.0, 1.5, 2.0, 3.0],
                      lambda v, s: TwinConfig(**base, load=v, attack_intensity=0.6, seed=s), seeds)
        df = pd.DataFrame(rows)
        df.to_csv(TAB / "exp02_twin_runs.csv", index=False)
    summ, tests = summarise(df)
    summ.to_csv(TAB / "exp02_twin_summary.csv", index=False)
    tests.to_csv(TAB / "exp02_twin_tests.csv", index=False)
    # one illustrative trace
    tr = []
    cfg = TwinConfig(**{**base, "n_epochs": 60}, load=1.5, attack_intensity=0.6, seed=seeds[0])
    for pol in POLICIES:
        out = run_twin(POLICIES[pol], cfg, record_epochs=True)
        for r in out["trace"]:
            tr.append(dict(policy=pol, **r))
    trace = pd.DataFrame(tr)
    trace.to_csv(TAB / "exp02_trace.csv", index=False)
    plot(summ, trace)
    key = summ[(summ.sweep == "attack") & (summ.value == 0.6)].set_index("policy")
    print(key[["compliance", "critical_compliance", "downgrade_rate", "expected_compromises_per_1k"]].round(4))
    manifest("exp02_digital_twin", args, t0)


if __name__ == "__main__":
    main()
