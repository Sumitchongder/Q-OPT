#!/usr/bin/env python3
"""
Q-OPT: real-hardware QAOA submission to IBM `ibm_marrakesh`.

THIS SCRIPT IS INTENTIONALLY ISOLATED from the rest of the pipeline.
Nothing in `qopt/` or in `experiments/` imports or calls this file, so
none of the reproducible classical/simulated experiments can
accidentally spend your QPU allocation. Only run this file yourself,
deliberately, when you intend to spend real `ibm_marrakesh` time.

--------------------------------------------------------------------
HARD TIME BUDGET
--------------------------------------------------------------------
You reported ~8 minutes of total QPU time. This script enforces that
budget in two independent ways:

  1. It estimates the wall-clock QPU time for the job BEFORE
     submitting (via `backend.target` gate durations x shots x circuits)
     and refuses to submit if the estimate exceeds `MAX_QPU_SECONDS`.
  2. It submits a SINGLE Sampler job containing only ONE circuit
     (the QAOA ansatz at your chosen p, bound at fixed, pre-optimized
     parameters) with a conservative shot count, rather than doing the
     parameter optimization loop on hardware (that loop is run
     entirely on `qopt.optimization.qaoa_qiskit.run_qaoa_aer` /
     FakeMarrakesh first -- see step 5 in the README run order).
     Hardware is used ONLY for one confirmatory sampling shot budget
     at the already-optimized angles, which is standard practice given
     a small QPU time allocation and is what you should report in the
     paper as "hardware validation" (Table 6 row: platform=ibm_marrakesh).

If you want a p-sweep on real hardware, run this script multiple
times with different `--reps`, watching your remaining allocation
between runs -- do NOT loop it automatically.

--------------------------------------------------------------------
USAGE
--------------------------------------------------------------------
    conda activate qgss
    python scripts/run_on_ibm_marrakesh.py \
        --problem-json results/qaoa_hw_problem.json \
        --params-json  results/qaoa_hw_optimal_params.json \
        --reps 1 --shots 2000

Both JSON inputs are produced automatically by
`experiments/e_hw_prepare_marrakesh_job.py`, which builds the QUBO,
runs the full COBYLA optimization loop on Aer/FakeMarrakesh (free),
and dumps the optimized circuit parameters + Ising model + expected
Aer energy, so this script only has to bind parameters and submit.

Credentials: this script expects you have already run, once:

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        token="my_api_key", instance="my_crn", overwrite=True, set_as_default=True,
    )

and does not embed or request any credentials itself.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

MAX_QPU_SECONDS = 8 * 60  # hard cap: 8 minutes, per user allocation
SAFETY_MARGIN = 0.75      # only ever plan to use 75% of the stated budget


def estimate_runtime_seconds(backend, n_qubits: int, depth: int, shots: int) -> float:
    """
    Conservative estimate: (per-circuit execution time) x shots, using the
    backend's reported reference gate/readout durations where available,
    else a documented conservative fallback (500 microseconds per shot for
    a depth-~dozens QAOA circuit on a 150+ qubit superconducting device,
    consistent with IBM's published typical QPU throughput figures).
    """
    try:
        dt = backend.target.dt or 1e-9
        # Reference two-qubit gate duration (ECR/CZ) as depth-dominant term.
        two_q_gate_names = [g for g in backend.target.operation_names if g.lower() in ("ecr", "cz", "cx")]
        gate_time_s = 300e-9  # fallback: ~300ns, typical IBM 2Q gate time
        if two_q_gate_names:
            try:
                durations = [backend.target[g][None].duration for g in two_q_gate_names
                             if backend.target[g].get(None) is not None]
                if durations:
                    gate_time_s = float(np.mean(durations))
            except Exception:
                pass
        readout_time_s = 1.2e-6  # typical IBM readout ~1-1.5us
        per_shot_time_s = depth * gate_time_s + readout_time_s
    except Exception:
        per_shot_time_s = 5e-4  # conservative documented fallback

    return per_shot_time_s * shots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem-json", required=True,
                     help="Path to Ising model JSON produced by experiments/e_hw_prepare_marrakesh_job.py")
    ap.add_argument("--params-json", required=True,
                     help="Path to optimized QAOA parameters JSON (from the free Aer/FakeMarrakesh run)")
    ap.add_argument("--reps", type=int, default=1, help="QAOA depth p used to build the params (must match)")
    ap.add_argument("--shots", type=int, default=2000, help="Shots for the single confirmatory hardware run")
    ap.add_argument("--backend-name", default="ibm_marrakesh")
    ap.add_argument("--dry-run", action="store_true",
                     help="Estimate time and print the circuit WITHOUT submitting anything.")
    args = ap.parse_args()

    with open(args.problem_json) as f:
        problem_data = json.load(f)
    with open(args.params_json) as f:
        params_data = json.load(f)

    n_qubits = problem_data["n_qubits"]
    h = np.array(problem_data["h"])
    J = np.array(problem_data["J"])
    C0 = problem_data["C0"]
    optimal_params = np.array(params_data["optimal_params"])
    expected_aer_energy = params_data.get("aer_energy")

    if len(optimal_params) != 2 * args.reps:
        print(f"ERROR: --reps={args.reps} implies {2*args.reps} parameters, "
              f"but params-json has {len(optimal_params)}. Aborting (no submission).")
        sys.exit(1)

    # --- Rebuild the identical circuit-construction path used everywhere else ---
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qopt.optimization.qaoa_qiskit import build_qaoa_circuit
    from qopt.optimization.qubo_to_ising import IsingModel, ising_to_sparse_pauli_op

    model = IsingModel(h=h, J=J, C0=C0, n=n_qubits)
    cost_op = ising_to_sparse_pauli_op(model)
    circuit = build_qaoa_circuit(cost_op, reps=args.reps)

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    backend = service.backend(args.backend_name)
    print(f"Backend: {backend.name}  n_qubits={backend.num_qubits}  "
          f"operational={backend.status().operational}  "
          f"queue={backend.status().pending_jobs}")

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    isa_circuit = pm.run(circuit)
    bound_circuit = isa_circuit.assign_parameters(optimal_params)

    est_seconds = estimate_runtime_seconds(backend, n_qubits, bound_circuit.depth(), args.shots)
    budget = MAX_QPU_SECONDS * SAFETY_MARGIN
    print(f"Transpiled circuit depth: {bound_circuit.depth()}")
    print(f"Estimated QPU execution time: {est_seconds:.1f} s "
          f"(budget: {budget:.0f} s of an {MAX_QPU_SECONDS}s / 8min allocation)")

    if est_seconds > budget:
        print("REFUSING TO SUBMIT: estimated runtime exceeds the safety-margined "
              "8-minute budget. Reduce --shots and retry. No job was submitted.")
        sys.exit(1)

    if args.dry_run:
        print("Dry run only (--dry-run passed). No job submitted. "
              f"Would submit {args.shots} shots at reps={args.reps}.")
        return

    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)
    t0 = time.time()
    job = sampler.run([bound_circuit], shots=args.shots)
    print(f"Submitted job id: {job.job_id()}  (waiting for result -- "
          f"do not launch another job until this returns)")
    result = job.result()
    wall_s = time.time() - t0
    print(f"Job completed in {wall_s:.1f} s wall-clock (includes queue wait).")

    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())
    energies = {}
    from qopt.optimization.qaoa_qiskit import _ising_energy_from_bitstring
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        e = _ising_energy_from_bitstring(model, clean)
        energies[clean] = (e, count)

    best_bitstring = min(energies, key=lambda k: energies[k][0])
    best_energy, best_count = energies[best_bitstring]
    mean_energy = sum(e * c for e, c in energies.values()) / total

    print("\n--- ibm_marrakesh hardware result ---")
    print(f"Best bitstring observed : {best_bitstring}  (count {best_count}/{total})")
    print(f"Best energy observed    : {best_energy:.4f}")
    print(f"Mean sampled energy     : {mean_energy:.4f}")
    if expected_aer_energy is not None:
        print(f"Aer (noiseless) energy  : {expected_aer_energy:.4f}")
        print(f"Hardware degradation    : {mean_energy - expected_aer_energy:+.4f}")

    out_path = args.problem_json.replace(".json", "_hw_result.json")
    with open(out_path, "w") as f:
        json.dump({
            "job_id": job.job_id(), "backend": backend.name, "shots": args.shots,
            "reps": args.reps, "wall_seconds": wall_s,
            "best_bitstring": best_bitstring, "best_energy": best_energy,
            "mean_energy": mean_energy, "expected_aer_energy": expected_aer_energy,
            "counts": {k: v[1] for k, v in energies.items()},
        }, f, indent=2)
    print(f"\nSaved hardware result to {out_path}")


if __name__ == "__main__":
    main()
