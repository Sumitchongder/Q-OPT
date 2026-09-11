"""
Prepare a hardware-ready QAOA job for ibm_marrakesh: WITHOUT touching
any real QPU. Builds a small Q-OPT QUBO (kept small deliberately -- a
handful of qubits -- both because that's what fits an 8-minute
allocation and because it's the honest laptop-feasible regime per the
design notes), converts it to Ising form, runs the full COBYLA
parameter optimization loop against Aer with the FakeMarrakesh noise
model, and dumps:

    results/qaoa_hw_problem.json         (n_qubits, h, J, C0)
    results/qaoa_hw_optimal_params.json  (optimal_params, aer_energy)

`scripts/run_on_ibm_marrakesh.py` consumes these two files and does
nothing else except bind the parameters and submit ONE circuit.

Run:
    python experiments/e_hw_prepare_marrakesh_job.py --n-requests 2 --k-paths 2 --reps 1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from qopt.network.topology import get_topology
from qopt.network.traffic import sample_requests
from qopt.optimization.qubo import build_options, build_qubo
from qopt.optimization.qubo_to_ising import qubo_to_ising
from qopt.optimization.qaoa_qiskit import run_qaoa_aer
from qopt.optimization.classical_baselines import solve_exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="A")
    ap.add_argument("--n-requests", type=int, default=2)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--use-marrakesh-noise", action="store_true",
                     help="Optimize params against the FakeMarrakesh noise model "
                          "instead of noiseless Aer, so angles are pre-adapted to "
                          "the real device's noise profile.")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    G = get_topology(args.topology, seed=0)
    reqs = sample_requests(list(G.nodes()), n_requests=args.n_requests, seed=1)
    link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
    options = build_options(G, reqs, link_state, k_paths=args.k_paths)
    problem = build_qubo(options, reqs)
    n_qubits = len(options)
    print(f"Built QUBO with n_qubits={n_qubits} (requests={args.n_requests}, k_paths={args.k_paths})")
    if n_qubits > 20:
        print("WARNING: n_qubits > 20 -- consider reducing --n-requests/--k-paths for an "
              "8-minute hardware budget (shallower/fewer-qubit circuits run faster).")

    exact = solve_exact(problem, max_n=30)
    print(f"Classical exact optimum: {exact.objective:.4f}")

    noise_model = None
    if args.use_marrakesh_noise:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        from qiskit_aer.noise import NoiseModel
        noise_model = NoiseModel.from_backend(FakeMarrakesh())
        print("Optimizing QAOA angles against FakeMarrakesh noise model (recommended before real hardware).")

    result = run_qaoa_aer(problem, reps=args.reps, shots=args.shots, maxiter=args.maxiter,
                           seed=0, noise_model=noise_model, optimal_energy=exact.objective)
    print(f"QAOA (Aer) obj={result.objective:.4f}  AR={result.approximation_ratio:.3f}  "
          f"violations={result.constraint_violations}")

    model = qubo_to_ising(problem.Q, problem.offset)
    problem_path = os.path.join(args.outdir, "qaoa_hw_problem.json")
    params_path = os.path.join(args.outdir, "qaoa_hw_optimal_params.json")

    with open(problem_path, "w") as f:
        json.dump({"n_qubits": n_qubits, "h": model.h.tolist(), "J": model.J.tolist(), "C0": model.C0}, f, indent=2)
    with open(params_path, "w") as f:
        json.dump({"optimal_params": result.optimal_params.tolist(),
                    "reps": args.reps, "aer_energy": result.best_energy}, f, indent=2)

    print(f"\nSaved: {problem_path}")
    print(f"Saved: {params_path}")
    print(f"\nNext (spends real QPU time -- run yourself, deliberately):")
    print(f"  python scripts/run_on_ibm_marrakesh.py --problem-json {problem_path} "
          f"--params-json {params_path} --reps {args.reps} --shots 2000 --dry-run")
    print(f"  (drop --dry-run once the time estimate looks right)")


if __name__ == "__main__":
    main()
