"""
Experiment 6: Scaling and resource estimates.

Part A  Classical: runtime and gap vs R (two paths x three modes per request).
        Exact enumeration while feasible (R <= 7); beyond that the reference
        is the best energy found by any heuristic (SA with 4x the default
        restarts included), and the gap is reported against that reference.
Part B  Quantum resources on a heavy-hex Heron device model (FakeMarrakesh):
        logical qubits, cost-Hamiltonian ZZ terms and transpiled CZ counts /
        two-qubit depth for XY-QAOA p=1 with and without the feasibility
        compression, and for penalty QAOA p=1, versus R (three modes).

Outputs: tables/exp06_classical_scaling.csv, tables/exp06_quantum_resources.csv,
         figures/exp06_scaling.{pdf,png}
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from _common import COLORS, TAB, manifest, savefig, style
from qopt.agility import solvers as S
from qopt.agility.model import random_instance


def part_a(sizes, n_seeds):
    rows = []
    for R in sizes:
        for s in range(n_seeds):
            prob = random_instance("C", n_requests=R, k_paths=2, seed=3000 + 100 * R + s, load=1.5)
            sols = {name: fn(prob) for name, fn in S.ALL_HEURISTICS.items()}
            sols["SA (4x restarts)"] = S.solve_simulated_annealing(prob, restarts=16, seed=1)
            if prob.n_feasible <= 300_000:
                sols["Exact"] = S.solve_exact(prob)
                ref, ref_kind = sols["Exact"].energy, "exact"
            else:
                ref, ref_kind = min(v.energy for v in sols.values()), "best-found"
            scale = abs(ref) if ref != 0 else 1.0
            for name, sol in sols.items():
                rows.append(dict(n_requests=R, seed=s, n_vars=prob.n, log10_feasible=np.log10(float(prob.n_feasible)),
                                 solver=name, runtime_ms=1e3 * sol.runtime_s, reference=ref_kind,
                                 rel_gap=(sol.energy - ref) / scale, hits_reference=bool(sol.energy <= ref + 1e-9 * scale)))
        print(f"  [A] R={R} done")
    return pd.DataFrame(rows)


def part_b(sizes, n_seeds, n_tseeds):
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    from qopt.quantum.circuits import bind, penalty_qaoa_circuit, xy_qaoa_circuit
    be = FakeMarrakesh()
    rows = []
    for R in sizes:
        for s in range(n_seeds):
            prob = random_instance("C", n_requests=R, k_paths=1, seed=4000 + 100 * R + s, load=1.5)
            Qc, _ = prob.qubo_feasible_compressed()
            Qf, _ = prob.qubo_penalized(0.0)
            Qp, _ = prob.qubo_penalized(1.0)
            nz = lambda Q: int((np.abs(np.triu(Q, 1)) > 1e-12).sum())
            variants = {"XY p=1 (compressed)": (xy_qaoa_circuit(prob, 1, 1.0), nz(Qc)),
                        "XY p=1 (uncompressed)": (xy_qaoa_circuit(prob, 1, 1.0, compress=False), nz(Qf)),
                        "Penalty p=1": (penalty_qaoa_circuit(prob, 1, 1.0, 1.0), nz(Qp))}
            for name, (qc, nzz) in variants.items():
                qc = bind(qc, [0.37, 0.41])
                best = None
                for ts in range(n_tseeds):
                    t = generate_preset_pass_manager(optimization_level=3, backend=be, seed_transpiler=ts).run(qc)
                    c = t.count_ops().get("cz", 0)
                    d = t.depth(lambda g: g.operation.num_qubits == 2)
                    if best is None or (c, d) < best:
                        best = (c, d)
                rows.append(dict(n_requests=R, seed=s, n_qubits=prob.n, variant=name, zz_terms=nzz,
                                 cz=best[0], cz_depth=best[1]))
        print(f"  [B] R={R} done")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-a", type=int, nargs="+", default=[4, 6, 7, 10, 15, 20, 30])
    ap.add_argument("--sizes-b", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--n-tseeds", type=int, default=4)
    ap.add_argument("--part", choices=["a", "b", "plot", "all"], default="all",
                    help="run one part (results are merged into the CSVs), or only re-plot")
    args = ap.parse_args()
    t0 = time.time()
    fa, fb = TAB / "exp06_classical_scaling.csv", TAB / "exp06_quantum_resources.csv"

    def merge(new, path, key):
        if path.exists():
            old = pd.read_csv(path)
            old = old[~old[key].isin(new[key].unique())]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values([key, "seed"]).to_csv(path, index=False)
        return new

    if args.part in ("a", "all"):
        merge(part_a(args.sizes_a, args.n_seeds), fa, "n_requests")
    if args.part in ("b", "all"):
        merge(part_b(args.sizes_b, max(3, args.n_seeds // 2), args.n_tseeds), fb, "n_requests")
    if args.part not in ("plot", "all"):
        manifest(f"exp06_scaling_resources_part{args.part}", args, t0)
        return
    a, b = pd.read_csv(fa), pd.read_csv(fb)
    print(a.groupby(["n_requests", "solver"]).agg(hit=("hits_reference", "mean"), gap=("rel_gap", "mean"),
                                                   ms=("runtime_ms", "median")).round(4).unstack("solver")["hit"])
    print(b.groupby(["n_requests", "variant"])[["zz_terms", "cz", "cz_depth"]].mean().round(1).unstack("variant"))

    import matplotlib.pyplot as plt
    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    cols = {"Exact": COLORS["Q-OPT v2 (exact)"], "Greedy (coupled)": COLORS["Greedy (coupled)"],
            "Tabu search": "#B8A800", "Simulated annealing": COLORS["Simulated annealing"],
            "Separable (v0.1)": COLORS["Separable (v0.1)"]}
    for name, col in cols.items():
        g = a[a.solver == name].groupby("n_requests")
        axes[0].plot(g.runtime_ms.median().index, g.runtime_ms.median().values, marker="o", color=col, label=name)
        if name != "Exact":
            axes[1].plot(g.hits_reference.mean().index, g.hits_reference.mean().values, marker="o", color=col, label=name)
    axes[0].set_yscale("log"); axes[0].set_xlabel("R"); axes[0].set_title("Median runtime [ms]")
    axes[1].set_xlabel("R"); axes[1].set_title("P(reaches reference)")
    axes[1].axvline(7.5, color="k", lw=0.6, ls=":"); axes[1].text(7.7, 0.05, "best-found\nreference", fontsize=6)
    h0, l0 = axes[0].get_legend_handles_labels()
    for name, col in [("XY p=1 (compressed)", COLORS["XY"]), ("XY p=1 (uncompressed)", "#56B4E9"),
                      ("Penalty p=1", COLORS["PEN"])]:
        g = b[b.variant == name].groupby("n_qubits").cz
        m = g.mean()
        axes[2].errorbar(m.index, m.values, yerr=g.std().values, marker="o", color=col, label=name, capsize=2, lw=1.2)
    axes[2].set_xlabel("logical qubits"); axes[2].set_title("CZ count (heavy-hex)")
    axes[2].legend(fontsize=5.5, loc="upper left")
    fig.legend(h0, l0, loc="lower center", ncol=5, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    savefig(fig, "exp06_scaling")
    manifest("exp06_scaling_resources", args, t0)


if __name__ == "__main__":
    main()
