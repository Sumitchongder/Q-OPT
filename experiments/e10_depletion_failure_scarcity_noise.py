"""
Experiment 10: four lighter design-spec experiments bundled into one
script (each is independently runnable via --only):

  E7  key-depletion sweep     -> service failures / mode switches as
                                  key_pool_frac alone is driven down
                                  (isolated from QBER/trust, unlike E9)
  E10 link failure/rerouting  -> zero out generation on a specific edge,
                                  confirm the optimizer reroutes via an
                                  alternate candidate path
  E11 QRNG scarcity            -> HYBRID options penalized when QRNG
                                  entropy at the path's nodes is scarce
  E15 noise-aware QAOA sweep   -> sweep depolarizing-noise strength
                                  (not just the single FakeMarrakesh
                                  snapshot) and plot AR degradation

Run:
    python experiments/e10_depletion_failure_scarcity_noise.py --only e7
    python experiments/e10_depletion_failure_scarcity_noise.py --only e10
    python experiments/e10_depletion_failure_scarcity_noise.py --only e11
    python experiments/e10_depletion_failure_scarcity_noise.py --only e15
    python experiments/e10_depletion_failure_scarcity_noise.py            # all four
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from qopt.network.topology import get_topology
from qopt.network.traffic import sample_requests
from qopt.optimization.qubo import build_options, build_qubo, evaluate_bitstring, Option
from qopt.optimization.classical_baselines import solve_exact


# ---------------------------------------------------------------------------
# E7: key-depletion sweep
# ---------------------------------------------------------------------------
def run_e7(n_seeds, n_requests, k_paths, topology, outdir_tables, outdir_figs):
    pool_fracs = np.linspace(0.9, 0.02, 10)
    records = []
    for frac in pool_fracs:
        for seed in range(n_seeds):
            G = get_topology(topology, seed=0)
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            link_state = {frozenset(e): dict(qber=0.015, risk=0.1, key_pool_frac=float(frac))
                           for e in G.edges()}
            options = build_options(G, reqs, link_state, k_paths=k_paths)
            problem = build_qubo(options, reqs)
            result = solve_exact(problem, max_n=30)
            ev = evaluate_bitstring(problem, result.y)
            mode_counts = {"QKD": 0, "PQC": 0, "HYBRID": 0}
            for o in ev["chosen"]:
                mode_counts[o.mode] += 1
            records.append(dict(pool_frac=frac, seed=seed, n_served=len(ev["chosen"]),
                                 n_requests=len(reqs), frac_pqc=mode_counts["PQC"] / len(reqs)))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e7_key_depletion.csv"), index=False)
    summary = df.groupby("pool_frac").agg(
        availability=("n_served", lambda x: x.mean() / df["n_requests"].iloc[0]),
        frac_pqc=("frac_pqc", "mean")).reset_index()
    print("=== E7: key depletion ===")
    print(summary.to_string(index=False))

    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig
    set_style()
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))
    ax1.plot(summary.pool_frac, summary.frac_pqc, marker="o", color="#f0a35a", label="Fraction using PQC")
    ax1.invert_xaxis()
    ax1.set_xlabel("Remaining QKD key-pool fraction (depleting -->)")
    ax1.set_ylabel("Fraction of requests using PQC")
    ax1.set_title("E7: migration to PQC as QKD key pools deplete")
    savefig(fig, os.path.join(outdir_figs, "e7_key_depletion"))
    plt.close(fig)
    return df, summary


# ---------------------------------------------------------------------------
# E10: link failure / rerouting
# ---------------------------------------------------------------------------
def run_e10(n_seeds, n_requests, k_paths, topology, outdir_tables, outdir_figs):
    """
    Compares topologies of differing redundancy (A: line, no redundancy;
    C: mesh, redundant), since single-edge failure on a well-connected
    mesh nearly always reroutes successfully (making a single-topology
    result uninformative) -- the reportable finding is the CONTRAST:
    topology redundancy determines whether a link failure is survivable
    at all. `--topology` is honored as an additional comparison point
    alongside A/C.
    """
    import networkx as nx
    topologies_to_test = sorted(set(["A", "C", topology]))
    records = []
    for topo in topologies_to_test:
        for seed in range(n_seeds):
            G = get_topology(topo, seed=0)
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            link_state = {frozenset(e): dict(qber=0.015, risk=0.1, key_pool_frac=0.8) for e in G.edges()}

            options_before = build_options(G, reqs, link_state, k_paths=k_paths)
            problem_before = build_qubo(options_before, reqs)
            result_before = solve_exact(problem_before, max_n=30)
            ev_before = evaluate_bitstring(problem_before, result_before.y)
            used_edges_before = set()
            for o in ev_before["chosen"]:
                used_edges_before.update(o.edges)

            if not used_edges_before:
                continue
            failed_edge = next(iter(used_edges_before))
            u, v = tuple(failed_edge)
            G_after = G.copy()
            if G_after.has_edge(u, v):
                G_after.remove_edge(u, v)
            still_connected = nx.is_connected(G_after)

            link_state_after = {frozenset(e): dict(qber=0.015, risk=0.1, key_pool_frac=0.8)
                                 for e in G_after.edges()}
            options_after = build_options(G_after, reqs, link_state_after, k_paths=k_paths)
            if options_after:
                problem_after = build_qubo(options_after, reqs)
                result_after = solve_exact(problem_after, max_n=30)
                ev_after = evaluate_bitstring(problem_after, result_after.y)
                n_served_after = len(ev_after["chosen"])
            else:
                n_served_after = 0

            records.append(dict(topology=topo, seed=seed, still_connected=still_connected,
                                 n_served_before=len(ev_before["chosen"]), n_requests=len(reqs),
                                 n_served_after=n_served_after,
                                 rerouting_success=n_served_after / max(1, len(ev_before["chosen"]))))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e10_link_failure.csv"), index=False)
    print("=== E10: link failure / rerouting (by topology redundancy) ===")
    summary = df.groupby("topology").agg(
        mean_rerouting_success=("rerouting_success", "mean"),
        std_rerouting_success=("rerouting_success", "std"),
        frac_disconnected_by_failure=("still_connected", lambda x: 1 - x.mean()),
    ).reset_index()
    print(summary.to_string(index=False))

    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig
    set_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar(summary["topology"], summary["mean_rerouting_success"],
           yerr=summary["std_rerouting_success"], color="#2a6fdb", alpha=0.85, capsize=4)
    ax.set_ylabel("Mean rerouting success rate")
    ax.set_xlabel("Topology (A = line, no redundancy; C = mesh, redundant)")
    ax.set_title("E10: link-failure survivability depends on topology redundancy")
    ax.set_ylim(0, 1.15)
    savefig(fig, os.path.join(outdir_figs, "e10_link_failure"))
    plt.close(fig)
    return df, summary


# ---------------------------------------------------------------------------
# E11: QRNG scarcity
# ---------------------------------------------------------------------------
def run_e11(n_seeds, n_requests, k_paths, topology, outdir_tables, outdir_figs):
    """
    HYBRID's combiner requires fresh entropy for its KDF context/nonce
    (qopt.pqc.hybrid_kdf), so it is modeled here as QRNG-dependent: when
    network-wide QRNG availability is scarce, HYBRID options get a
    latency/risk penalty proportional to scarcity (a node waiting on
    entropy replenishment), while QKD and PQC (which do not depend on
    the QRNG resource in this design) are unaffected.
    """
    scarcity_levels = np.linspace(0.0, 1.0, 8)  # 0 = plentiful, 1 = fully scarce
    records = []
    for scarcity in scarcity_levels:
        for seed in range(n_seeds):
            G = get_topology(topology, seed=0)
            reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
            # QBER/risk set high enough that HYBRID is the scarcity=0 baseline
            # choice for security-critical traffic (per the E2 QBER-sweep
            # crossover point) -- otherwise QKD dominates regardless of
            # scarcity and this experiment has nothing to show.
            link_state = {frozenset(e): dict(qber=0.08, risk=0.75, key_pool_frac=0.7) for e in G.edges()}
            options = build_options(G, reqs, link_state, k_paths=k_paths)

            # Penalize HYBRID options' effective cost by inflating their
            # key_pool_pressure term proportional to QRNG scarcity (a stand-in
            # for "waiting on entropy" -- reuses the existing pool-pressure
            # channel rather than adding a new QUBO term, since it already has
            # the right normalization and weight).
            adjusted = []
            for o in options:
                if o.mode == "HYBRID":
                    o = Option(o.req_id, o.path, o.mode, o.edges, o.risk, o.latency_ms,
                               min(1.0, o.key_pool_pressure + 0.6 * scarcity), o.comp_cost,
                               o.disruption_risk, o.security_score)
                adjusted.append(o)
            problem = build_qubo(adjusted, reqs)
            result = solve_exact(problem, max_n=30)
            ev = evaluate_bitstring(problem, result.y)
            mode_counts = {"QKD": 0, "PQC": 0, "HYBRID": 0}
            for o in ev["chosen"]:
                mode_counts[o.mode] += 1
            records.append(dict(scarcity=scarcity, seed=seed, n_requests=len(reqs), **mode_counts))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e11_qrng_scarcity.csv"), index=False)
    summary = df.groupby("scarcity").agg(
        frac_qkd=("QKD", lambda x: x.sum() / df["n_requests"].iloc[0] / n_seeds),
        frac_pqc=("PQC", lambda x: x.sum() / df["n_requests"].iloc[0] / n_seeds),
        frac_hybrid=("HYBRID", lambda x: x.sum() / df["n_requests"].iloc[0] / n_seeds),
    ).reset_index()
    print("=== E11: QRNG scarcity ===")
    print(summary.to_string(index=False))

    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig
    set_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.stackplot(summary.scarcity, summary.frac_qkd, summary.frac_hybrid, summary.frac_pqc,
                 labels=["QKD", "HYBRID", "PQC"], colors=["#2a6fdb", "#3a6b35", "#f0a35a"], alpha=0.85)
    ax.set_xlabel("Network-wide QRNG scarcity")
    ax.set_ylabel("Fraction of requests")
    ax.set_title("E11: mode selection shift away from HYBRID under QRNG scarcity")
    ax.legend(loc="upper right", fontsize=8)
    savefig(fig, os.path.join(outdir_figs, "e11_qrng_scarcity"))
    plt.close(fig)
    return df, summary


# ---------------------------------------------------------------------------
# E15: noise-aware QAOA sweep
# ---------------------------------------------------------------------------
def run_e15(n_seeds, n_requests, k_paths, topology, outdir_tables, outdir_figs):
    """
    Isolates the effect of hardware noise on SAMPLING a fixed, already-
    optimized QAOA circuit, rather than re-running the full COBYLA
    optimization loop at every noise level. An earlier version of this
    experiment re-optimized at each noise level, which mixed optimizer
    variance (COBYLA landing in different local optima run to run) with
    the actual noise-degradation signal and produced a non-monotonic,
    uninterpretable curve. The corrected protocol:

      1. For each seed, build one problem instance and optimize QAOA(p=1)
         parameters ONCE against the noiseless simulator (this is the
         "you tuned your circuit on a clean simulator" scenario, which is
         also exactly the real workflow used before submitting to
         ibm_marrakesh in this repo).
      2. Bind those SAME fixed parameters and sample under each noise
         level -- no re-optimization -- so the only thing that varies
         across the sweep is execution noise, isolating its effect
         cleanly.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qopt.optimization.qaoa_qiskit import run_qaoa_aer, build_qaoa_circuit, _ising_energy_from_bitstring
    from qopt.optimization.qubo_to_ising import qubo_to_ising, ising_to_sparse_pauli_op

    def make_noise_model(p1, p2):
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "rx", "h", "x", "sx"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "cz", "rzz"])
        return nm

    noise_strengths = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25]
    records = []

    for seed in range(n_seeds):
        G = get_topology(topology, seed=0)
        reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
        link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
        options = build_options(G, reqs, link_state, k_paths=k_paths)
        problem = build_qubo(options, reqs)
        exact = solve_exact(problem, max_n=20)

        # Step 1: optimize ONCE, noiseless. Use p=3 (not p=1) so the circuit
        # is deep enough that depolarizing noise actually has visible effect
        # within realistic error-rate ranges (a p=1 circuit on this small a
        # QUBO is shallow enough to be largely noise-insensitive below ~25%
        # error, which was tried first and found uninformative).
        opt_run = run_qaoa_aer(problem, reps=3, shots=1024, maxiter=100, seed=seed,
                                noise_model=None, optimal_energy=exact.objective)
        fixed_params = opt_run.optimal_params

        model = qubo_to_ising(problem.Q, problem.offset)
        cost_op = ising_to_sparse_pauli_op(model)
        circuit = build_qaoa_circuit(cost_op, reps=3)
        n_qubits = len(problem.options)
        optimal_bitstring = "".join(str(int(exact.y[n_qubits - 1 - i])) for i in range(n_qubits))

        # Step 2: sample the SAME fixed circuit under each noise level.
        for p_err in noise_strengths:
            nm = make_noise_model(p_err, p_err * 2) if p_err > 0 else None
            backend = AerSimulator(noise_model=nm)
            pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
            isa_circuit = pm.run(circuit)
            bound = isa_circuit.assign_parameters(fixed_params)
            sampler = AerSampler.from_backend(backend)
            result = sampler.run([bound], shots=2048).result()
            counts = result[0].data.meas.get_counts()
            total = sum(counts.values())
            # Standard NISQ metric: probability of sampling the TRUE optimal
            # bitstring (identical in spirit to the "count/total" confidence
            # figures reported for the real ibm_marrakesh run in this repo's
            # history -- e.g. 92/2000 at p=1, decreasing with circuit depth).
            # This is far more sensitive to noise than best-of-shots energy
            # (which stays near-optimal as long as the optimal bitstring
            # appears even once in thousands of shots) and more stable than
            # raw distribution-mean energy (which is dominated by QAOA's
            # inherent p=1 spread, not noise).
            clean_counts = {bs.replace(" ", ""): c for bs, c in counts.items()}
            success_prob = clean_counts.get(optimal_bitstring, 0) / total
            best_energy = min(_ising_energy_from_bitstring(model, bs) for bs in clean_counts)
            ar = best_energy / exact.objective if exact.objective != 0 else np.nan
            records.append(dict(noise_strength=p_err, seed=seed, best_sampled_energy=best_energy,
                                 approximation_ratio=ar, success_probability=success_prob))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e15_noise_sweep.csv"), index=False)
    summary = df.groupby("noise_strength").agg(
        mean_AR=("approximation_ratio", "mean"), std_AR=("approximation_ratio", "std"),
        mean_success_prob=("success_probability", "mean"), std_success_prob=("success_probability", "std"),
    ).reset_index()
    print("=== E15: noise-aware QAOA sweep (fixed pre-optimized params, sampling noise only) ===")
    print(summary.to_string(index=False))

    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.errorbar(summary.noise_strength, summary.mean_success_prob, yerr=summary.std_success_prob,
                marker="o", color="#d1495b", capsize=3)
    ax.set_xlabel("Single-/two-qubit depolarizing error rate")
    ax.set_ylabel("P(sampling the true optimal bitstring)")
    ax.set_title("E15: QAOA(p=1) degradation under increasing gate noise")
    savefig(fig, os.path.join(outdir_figs, "e15_noise_sweep"))
    plt.close(fig)
    return df, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["e7", "e10", "e11", "e15"], default=None)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-requests", type=int, default=3)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="C")
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir_tables, exist_ok=True)
    os.makedirs(args.outdir_figs, exist_ok=True)

    fns = {"e7": run_e7, "e10": run_e10, "e11": run_e11, "e15": run_e15}
    to_run = [args.only] if args.only else list(fns.keys())
    for name in to_run:
        print(f"\n{'='*20} Running {name} {'='*20}")
        fns[name](args.n_seeds, args.n_requests, args.k_paths, args.topology,
                   args.outdir_tables, args.outdir_figs)
