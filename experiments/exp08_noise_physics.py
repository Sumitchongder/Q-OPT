"""
Experiment 8: Noise physics of the hardware run, and supporting physics figures.

(a) Symmetry-sector survival.  For a circuit whose noiseless output lies in the
    block-one-hot sector with probability p_ideal, the measured feasible
    fraction is p_meas. We model the survival S = p_meas / p_ideal as

        S_pred = (1 - e_ro)^n * exp(-kappa * eps_2q * N_CZ),   kappa = 12/15,

    i.e. a two-qubit depolarising channel of strength eps_2q after every CZ, of
    whose 15 non-identity Paulis 12 contain an X or Y and therefore change the
    Hamming weight of at least one block, plus independent readout flips. The
    model is evaluated with the *median calibrated* errors of the qubits used,
    then compared with the hardware and with the FakeMarrakesh emulation; an
    effective per-CZ rate kappa_eff * eps is fitted to each data set.
(b) Undetected-error fraction.  If the post-selected distribution is
    (1-q) * ideal + q * uniform-feasible, then
        q = (r_ideal - r_post) / (r_ideal - r_rand).
(c) Cumulative distributions of the normalised energy of feasible samples on
    hardware versus the noiseless ideal, per ansatz.
(d) LMC05 key rate versus distance and adversarial QBER shift.

Outputs: tables/exp08_survival.csv, tables/exp08_fits.csv,
         figures/exp08_survival.{pdf,png}, figures/exp08_energy_cdf.{pdf,png},
         figures/exp08_keyrate.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from _common import COLORS, ROOT, TAB, manifest, savefig, style
from qopt.quantum.hardware import problem_from_plan
from qopt.quantum.train import counts_to_feasible

KAPPA = 12.0 / 15.0


def load_run(run_dir, plan):
    raw = json.loads((run_dir / "raw.json").read_text())
    inst = {i["name"]: i for i in plan["instances"]}
    ts = {s["name"]: s for s in raw["transpile"]}
    cal = raw["calibration"]
    rows = []
    for pb in raw["pubs"]:
        I = inst[pb["instance"]]
        prob = problem_from_plan(I)
        E = np.array(I["spectrum"])
        tot, feas, vec = counts_to_feasible(prob, pb["counts"])
        p_meas = feas / tot
        r_post = float(vec @ E / feas) if feas else np.nan
        span = E.max() - E.min()
        r_post = (E.max() - r_post) / span
        name = f"{pb['instance']}|{pb['ansatz']}|p{pb['p']}"
        n = I["n_qubits"]
        ncz = ts[name]["twoq"]
        s_pred = (1 - cal["median_readout_error"]) ** n * np.exp(-KAPPA * cal["median_2q_error"] * ncz)
        ideal = pb["ideal"]
        q = np.nan
        if pb["ansatz"] != "W" and ideal["r_feas"] - ideal["r_rand"] > 1e-6:
            q = (ideal["r_feas"] - r_post) / (ideal["r_feas"] - ideal["r_rand"])
        rows.append(dict(source="emulation" if raw.get("fake_backend") else "hardware", backend=raw["backend"],
                         instance=pb["instance"], ansatz=pb["ansatz"], p=pb["p"], n_qubits=n, n_cz=ncz,
                         p_feas_ideal=ideal["p_feas"], p_feas=p_meas, survival=p_meas / ideal["p_feas"],
                         survival_pred=s_pred, r_post=r_post, r_ideal=ideal["r_feas"], r_rand=ideal["r_rand"],
                         q_undetected=q, eps2q=cal["median_2q_error"], e_ro=cal["median_readout_error"]))
    return pd.DataFrame(rows), raw


def fit_rate(d):
    """ln S + n*ln(1/(1-e_ro)) = ln A - lambda * N_CZ ; least squares for (A, lambda)."""
    y = np.log(d.survival.values) - d.n_qubits.values * np.log(1 - d.e_ro.values)
    X = np.c_[np.ones(len(d)), -d.n_cz.values]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(d) - 2, 1)
    cov = np.linalg.inv(X.T @ X) * (resid @ resid) / dof
    return float(np.exp(coef[0])), float(coef[1]), float(np.sqrt(cov[1, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpu-run", default=None, help="run_qpu_* directory (default: newest)")
    ap.add_argument("--emu-run", default=str(ROOT / "results/hardware/run_fake_emulation_FakeMarrakesh"))
    ap.add_argument("--cdf-instance", default="R4_s2")
    args = ap.parse_args()
    t0 = time.time()
    from pathlib import Path
    plan = json.loads((ROOT / "results/hardware/plan/plan.json").read_text())
    qpu_dir = Path(args.qpu_run) if args.qpu_run else sorted((ROOT / "results/hardware").glob("run_qpu_*"))[-1]
    hw, raw_hw = load_run(qpu_dir, plan)
    em, _ = load_run(Path(args.emu_run), plan)
    df = pd.concat([hw, em], ignore_index=True)
    # (e) circuit durations with ibm_fez gate timings, for the amplitude-damping term
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qopt.quantum.hardware import load_plan
    fez = FakeFez()
    _, circs = load_plan(ROOT / "results/hardware/plan")
    seeds = {s["name"]: s["seed_transpiler"] for s in raw_hw["transpile"]}
    dur = {}
    for qc in circs:
        t = generate_preset_pass_manager(optimization_level=3, backend=fez, seed_transpiler=seeds[qc.name]).run(qc)
        dur[qc.name] = t.estimate_duration(fez.target, unit="s")
    t1 = raw_hw["calibration"]["median_t1_us"] * 1e-6
    names = df.instance + "|" + df.ansatz + "|p" + df.p.astype(str)
    df["duration_us"] = [dur[k] * 1e6 for k in names]
    df["n_blocks"] = df.n_qubits // 3
    # one excitation per block in the ideal state; T1 decay of any of them leaves the sector
    df["survival_pred_t1"] = df.survival_pred * np.exp(-df.n_blocks * df.duration_us * 1e-6 / t1)
    df.to_csv(TAB / "exp08_survival.csv", index=False)

    fits = []
    for src, d in df.groupby("source"):
        sub = d[d.ansatz != "PEN"]          # sector preserved by construction
        A, lam, se = fit_rate(sub)
        eps = float(d.eps2q.iloc[0])
        fits.append(dict(mean_abs_err_gate_ro=float((d.survival - d.survival_pred).abs().mean()),
                         mean_abs_err_with_t1=float((d.survival - d.survival_pred_t1).abs().mean()),
                         median_duration_us=float(d.duration_us.median()),
                         source=src, backend=d.backend.iloc[0], n_circuits=len(sub), intercept_A=A,
                         lambda_per_cz=lam, lambda_se=se, eps2q_median=eps, kappa_eff=lam / eps,
                         kappa_model=KAPPA, mean_q_xy_p1=float(d[(d.ansatz == "XY") & (d.p == 1)].q_undetected.mean()),
                         mean_q_xy_p2=float(d[(d.ansatz == "XY") & (d.p == 2)].q_undetected.mean()),
                         mean_q_pen=float(d[d.ansatz == "PEN"].q_undetected.mean())))
    fits = pd.DataFrame(fits)
    fits.to_csv(TAB / "exp08_fits.csv", index=False)
    print(fits.round(5).to_string(index=False))

    import matplotlib.pyplot as plt
    style()
    # ---------------- (a)+(b) survival and undetected-error fraction --------------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    mk = {"W": "s", "XY": "o", "PEN": "^"}
    for src, col in [("hardware", "#D55E00"), ("emulation", "#0072B2")]:
        d = df[df.source == src]
        for a in ("W", "XY", "PEN"):
            s = d[d.ansatz == a]
            axes[0].scatter(s.n_cz, s.survival, marker=mk[a], s=16, color=col, alpha=0.85,
                            label=f"{src}, {a}")
        f = fits[fits.source == src].iloc[0]
        xs = np.linspace(0, 160, 50)
        n_mid = 10.5
        axes[0].plot(xs, f.intercept_A * (1 - d.e_ro.iloc[0]) ** n_mid * np.exp(-f.lambda_per_cz * xs),
                     color=col, lw=1.0, ls="-")
    d = df[df.source == "hardware"]
    xs = np.linspace(0, 160, 50)
    axes[0].plot(xs, (1 - d.e_ro.iloc[0]) ** 10.5 * np.exp(-KAPPA * d.eps2q.iloc[0] * xs), color="k", lw=0.9,
                 ls="--", label="calibration model")
    axes[0].set_xlabel("CZ gates"); axes[0].set_ylabel("sector survival $S$")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=4.8, loc="lower left", ncol=2, handletextpad=0.2, columnspacing=0.6)
    for src, col in [("hardware", "#D55E00"), ("emulation", "#0072B2")]:
        s = df[df.source == src]
        axes[1].scatter(s.survival_pred, s.survival, s=14, color=col, label=f"{src}: gate+readout")
        if src == "hardware":
            axes[1].scatter(s.survival_pred_t1, s.survival, s=14, facecolors="none", edgecolors=col,
                            label="hardware: + T$_1$ decay")
    axes[1].plot([0, 1], [0, 1], color="k", lw=0.7, ls=":")
    axes[1].set_xlabel("predicted survival"); axes[1].set_ylabel("measured survival")
    axes[1].set_xlim(0.4, 1.0); axes[1].set_ylim(0.2, 1.05); axes[1].legend(fontsize=4.6, loc="upper left", handletextpad=0.2, borderaxespad=0.2)
    for src, col in [("hardware", "#D55E00"), ("emulation", "#0072B2")]:
        s = df[(df.source == src) & (df.ansatz != "W")]
        for a in ("XY", "PEN"):
            t = s[s.ansatz == a]
            axes[2].scatter(t.n_cz, t.q_undetected, marker=mk[a], s=16, color=col, label=f"{src}, {a}")
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_xlabel("CZ gates"); axes[2].set_ylabel("undetected fraction $q$")
    axes[2].legend(fontsize=5, loc="upper left")
    fig.tight_layout()
    savefig(fig, "exp08_survival")

    # ---------------- (c) energy CDFs on hardware ----------------------------------
    inst = {i["name"]: i for i in plan["instances"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, name in zip(axes, [args.cdf_instance, "R3_s0"]):
        I = inst[name]
        prob = problem_from_plan(I)
        E = np.array(I["spectrum"])
        eps_n = (E - E.min()) / (E.max() - E.min())
        order = np.argsort(eps_n)
        from qopt.quantum.fastsim import XYQAOASim
        from qopt.quantum.circuits import qubo_to_ising  # noqa: F401  (kept for clarity of provenance)
        for pb in raw_hw["pubs"]:
            if pb["instance"] != name:
                continue
            tot, feas, vec = counts_to_feasible(prob, pb["counts"])
            lab = {"W": "W state", "XY": f"XY p={pb['p']}", "PEN": "penalty p=1"}[pb["ansatz"]]
            col = {"W": COLORS["W"], "XY": COLORS["XY"] if pb["p"] == 1 else "#56B4E9", "PEN": COLORS["PEN"]}[pb["ansatz"]]
            cdf = np.cumsum(vec[order]) / max(feas, 1)
            ax.step(eps_n[order], cdf, where="post", color=col, lw=1.3, label=lab + " (QPU)")
        # noiseless ideal for XY p=1 for reference
        sim = XYQAOASim(prob, scale=I["scale"])
        x1 = next(pb["params"] for pb in plan["pubs"] if pb["instance"] == name and pb["ansatz"] == "XY" and pb["p"] == 1)
        pr = sim.probs(np.array(x1))
        ax.step(eps_n[order], np.cumsum(pr[order]), where="post", color=COLORS["XY"], ls="--", lw=1.0,
                label="XY p=1 (noiseless)")
        ax.set_title(f"{name}: {I['n_qubits']} qubits, {I['n_feasible']} feasible", fontsize=8)
        ax.set_xlabel("normalised energy")
    axes[0].set_ylabel("cumulative probability")
    axes[1].legend(fontsize=5.5, loc="center right")
    fig.tight_layout()
    savefig(fig, "exp08_energy_cdf")

    # ---------------- (d) key-rate physics -------------------------------------------
    from qopt.qkd.channel import ChannelParams, key_generation_rate_bps
    p = ChannelParams()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    L = np.linspace(1, 160, 300)
    for dq, col in zip([0.0, 0.01, 0.02, 0.03, 0.04], ["#000000", "#0072B2", "#009E73", "#E69F00", "#D55E00"]):
        R = np.array([key_generation_rate_bps(x, p, qber_shift=dq)[0] for x in L])
        axes[0].semilogy(L, np.where(R > 0, R, np.nan), color=col, label=fr"$\delta$ = {dq:.2f}")
    axes[0].set_xlabel("fibre length [km]"); axes[0].set_ylabel("asymptotic secret-key rate [bit/s]")
    axes[0].legend(fontsize=6)
    dq = np.linspace(0, 0.07, 300)
    for Lk, col in zip([10, 30, 60], ["#0072B2", "#009E73", "#D55E00"]):
        R = np.array([key_generation_rate_bps(Lk, p, qber_shift=x)[0] for x in dq])
        axes[1].plot(dq, R / R[0], color=col, label=f"L = {Lk} km")
    axes[1].set_xlabel(r"adversarial QBER shift $\delta$"); axes[1].set_ylabel("rate relative to $\\delta=0$")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    savefig(fig, "exp08_keyrate")
    manifest("exp08_noise_physics", args, t0, extra=dict(qpu_run=str(qpu_dir.name)))


if __name__ == "__main__":
    main()
