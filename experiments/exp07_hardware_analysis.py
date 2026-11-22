"""
Experiment 7: Analysis of a hardware (or emulated) run produced by
notebooks/qpu_run.ipynb.

For every circuit it reports, with 95% multinomial-bootstrap CIs:
  * raw P(feasible)
  * r_feas after symmetry verification (post-selection on block one-hot)
  * r_eff (infeasible shots scored as the worst feasible assignment)
  * P(optimum), mean percentile
  * r after greedy repair of infeasible shots (classical post-processing,
    reported in its own column, never merged with the raw numbers)
next to the noiseless ideal and the uniform-feasible floor, plus two
hypothesis tests per instance (one-sided, bootstrap):
  H1: r_feas(XY p=1) > r_rand          (signal above random on hardware)
  H2: r_eff(XY p=1)  > r_eff(PEN p=1)  (constraint preservation helps on hardware)

Usage:
  python experiments/exp07_hardware_analysis.py --run-dir results/hardware/run_qpu_ibm_xxx_YYYYMMDD-HHMMSS
Outputs (tag = run directory name):
  tables/exp07_<tag>.csv, tables/exp07_<tag>_tests.csv,
  figures/exp07_<tag>.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import legend_cols, COLORS, ROOT, TAB, manifest, savefig, style
from qopt.quantum.hardware import problem_from_plan
from qopt.quantum.train import bootstrap_metrics, metrics_from_feasible_dist, repair_counts


def boot_samples(prob, counts, E, n_boot, seed):
    keys = list(counts)
    vals = np.array([counts[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    out = []
    from qopt.quantum.train import counts_to_feasible
    for _ in range(n_boot):
        cv = rng.multinomial(int(vals.sum()), vals / vals.sum())
        tot, feas, vec = counts_to_feasible(prob, {k: int(v) for k, v in zip(keys, cv) if v})
        out.append(metrics_from_feasible_dist(E, vec / tot, feas / tot))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--plan-dir", default=str(ROOT / "results/hardware/plan"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    t0 = time.time()
    run_dir = Path(args.run_dir)
    raw = json.loads((run_dir / "raw.json").read_text())
    plan = json.loads((Path(args.plan_dir) / "plan.json").read_text())
    inst = {i["name"]: i for i in plan["instances"]}
    tag = run_dir.name
    tstats = {s["name"]: s for s in raw["transpile"]}
    rows, boots = [], {}
    for k, pb in enumerate(raw["pubs"]):
        I = inst[pb["instance"]]
        prob = problem_from_plan(I)
        E = np.array(I["spectrum"])
        b = bootstrap_metrics(prob, pb["counts"], E, n_boot=args.n_boot, seed=k)
        rep = repair_counts(prob, pb["counts"], E)
        mr = metrics_from_feasible_dist(E, rep / rep.sum(), 1.0)
        name = f"{pb['instance']}|{pb['ansatz']}|p{pb['p']}"
        ts = tstats.get(name, {})
        row = dict(instance=pb["instance"], ansatz=pb["ansatz"], p=pb["p"], n_qubits=I["n_qubits"],
                   cz=ts.get("twoq"), cz_depth=ts.get("twoq_depth"), shots=b["shots"])
        for m in ("p_feas", "r_feas", "r_eff", "p_opt", "pctl"):
            row[m] = b["point"][m]
            row[f"{m}_lo"], row[f"{m}_hi"] = b["ci"][m]
            row[f"{m}_ideal"] = pb["ideal"][m]
        row["r_repaired"] = mr["r_feas"]
        row["p_opt_repaired"] = mr["p_opt"]
        row["r_rand"] = b["point"]["r_rand"]
        rows.append(row)
        boots[(pb["instance"], pb["ansatz"], pb["p"])] = boot_samples(prob, pb["counts"], E, 1000, 10_000 + k)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / f"exp07_{tag}.csv", index=False)

    tests = []
    for name in inst:
        xy = boots.get((name, "XY", 1))
        pen = boots.get((name, "PEN", 1))
        rr = float(df[(df.instance == name)].r_rand.iloc[0])
        if xy is not None:
            tests.append(dict(instance=name, hypothesis="r_feas(XY p1) > r_rand",
                              estimate=float(xy.r_feas.mean() - rr), p_one_sided=float(((xy.r_feas <= rr).sum() + 1) / (len(xy) + 1))))
        if xy is not None and pen is not None:
            diff = xy.r_eff.values - pen.r_eff.values
            tests.append(dict(instance=name, hypothesis="r_eff(XY p1) > r_eff(PEN p1)",
                              estimate=float(diff.mean()), p_one_sided=float(((diff <= 0).sum() + 1) / (diff.size + 1))))
    tests = pd.DataFrame(tests)
    tests.to_csv(TAB / f"exp07_{tag}_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"run {tag}: backend={raw['backend']} fake={raw.get('fake_backend')} shots/circuit={raw['shots_per_circuit']} "
          f"billed={raw.get('billed_quantum_seconds')}s")
    print(df[["instance", "ansatz", "p", "cz", "p_feas", "r_feas", "r_feas_lo", "r_feas_hi", "r_feas_ideal",
              "r_eff", "r_repaired", "p_opt", "r_rand"]].round(3).to_string(index=False))
    print(tests.round(4).to_string(index=False))

    import matplotlib.pyplot as plt
    style()
    names = list(inst)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    variants = [("W", 0, COLORS["W"], "W state (p=0)"), ("XY", 1, COLORS["XY"], "XY-QAOA p=1"),
                ("XY", 2, "#56B4E9", "XY-QAOA p=2"), ("PEN", 1, COLORS["PEN"], "Penalty QAOA p=1")]
    x = np.arange(len(names))
    w = 0.2
    for ax, m, lab in [(axes[0], "p_feas", "P(feasible), raw"),
                       (axes[1], "r_feas", "r after symmetry verification")]:
        for j, (a, p, col, lb) in enumerate(variants):
            d = df[(df.ansatz == a) & (df.p == p)].set_index("instance").reindex(names)
            if d[m].isna().all():
                continue
            ax.bar(x + (j - 1.5) * w, d[m], w, color=col, label=lb,
                   yerr=[d[m] - d[f"{m}_lo"], d[f"{m}_hi"] - d[m]], error_kw=dict(lw=0.6, capsize=1.5))
            ax.scatter(x + (j - 1.5) * w, d[f"{m}_ideal"], marker="_", s=60, color="k", zorder=5,
                       label="noiseless ideal" if j == 0 else None)
        if m == "r_feas":
            rr = df.groupby("instance").r_rand.first().reindex(names)
            for i, v in enumerate(rr):
                ax.hlines(v, i - 0.45, i + 0.45, color="#cc0000", lw=0.9, ls="--",
                          label="uniform-feasible floor" if i == 0 else None)
            ax.set_ylim(0.4, 1.02)
        ax.set_xticks(x, names, fontsize=7)
        ax.set_title(lab, fontsize=8.5)
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", ncol=legend_cols(6), fontsize=6.3, bbox_to_anchor=(0.5, 1.02))
    src = "emulated (FakeMarrakesh)" if raw.get("fake_backend") else raw["backend"]
    fig.text(0.5, -0.02, f"{src}, {raw['shots_per_circuit']} shots/circuit; error bars: 95% bootstrap CI",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if legend_cols(6) == 6 else 0.86))
    savefig(fig, f"exp07_{tag}")
    manifest(f"exp07_{tag}", args, t0)


if __name__ == "__main__":
    main()
