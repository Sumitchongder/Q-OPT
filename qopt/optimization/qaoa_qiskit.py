"""
QAOA solver for the Q-OPT QUBO, built with Qiskit (Section 18-19 of
the design notes).

Two execution paths, sharing 100% of the circuit-construction code:

  1. `run_qaoa_aer(...)`      -- local noiseless/noisy simulation via
     qiskit-aer. This is the path used for all development, CI and the
     experiment sweeps in this repository (E1-E3, E12, E13, E15).

  2. `scripts/run_on_ibm_marrakesh.py` (separate file, NOT imported by
     default) -- submits the *same* circuits built here to the real
     `ibm_marrakesh` QPU via qiskit-ibm-runtime, with a hard shot/time
     budget so it stays within an ~8 minute allocation. It is kept in
     its own script specifically so nothing in the reproducible
     experiment pipeline can accidentally consume QPU time.

QAOA here follows the textbook alternating-operator ansatz:

    |psi(gamma, beta)> = prod_p exp(-i*beta_p*H_B) exp(-i*gamma_p*H_C) |+>^n
    H_B = sum_i X_i
    H_C = Ising cost Hamiltonian from qubo_to_ising

The outer-loop classical optimizer is COBYLA (gradient-free, robust to
sampling noise -- a standard choice for NISQ-era QAOA).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize

from qopt.optimization.qubo import QUBOProblem, evaluate_bitstring
from qopt.optimization.qubo_to_ising import qubo_to_ising, ising_to_sparse_pauli_op, IsingModel


@dataclass
class QAOAResult:
    solver: str
    y: np.ndarray
    objective: float
    constraint_violations: int
    runtime_s: float
    p: int
    best_energy: float
    optimal_params: np.ndarray
    n_qubits: int
    n_circuit_evals: int
    approximation_ratio: float | None = None


def build_qaoa_circuit(cost_op, reps: int):
    ansatz = QAOAAnsatz(cost_operator=cost_op, reps=reps)
    ansatz.measure_all()
    return ansatz


def run_qaoa_aer(problem: QUBOProblem, reps: int = 1, shots: int = 2048,
                  maxiter: int = 100, seed: int = 0,
                  noise_model=None, optimal_energy: float | None = None) -> QAOAResult:
    """
    Run QAOA on the Q-OPT QUBO using Aer's statevector-based sampler.
    `noise_model` (a qiskit_aer.noise.NoiseModel) can be passed for the
    noise-aware robustness study (Experiment E15); left None this is a
    noiseless simulation, which is also exactly what you get by
    swapping `AerSimulator()` for `FakeMarrakesh()`-style fake backends
    for pre-QPU validation.
    """
    t0 = time.perf_counter()
    n = len(problem.options)
    model = qubo_to_ising(problem.Q, problem.offset)
    cost_op = ising_to_sparse_pauli_op(model)

    circuit = build_qaoa_circuit(cost_op, reps=reps)

    backend = AerSimulator(noise_model=noise_model)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuit = pm.run(circuit)
    isa_cost_op = cost_op.apply_layout(isa_circuit.layout)

    # BUG FIX (discovered while building the E15 noise-sweep experiment):
    # `AerSampler()` constructed with no arguments uses its OWN internal
    # default simulator and completely IGNORES the `backend` object built
    # above -- `qiskit_aer.primitives.SamplerV2` has no `backend` constructor
    # parameter. This silently meant every noise_model passed to this
    # function (including every call with FakeMarrakesh's calibrated noise
    # in tests/test_qaoa_fake_marrakesh.py) had ZERO effect: results were
    # always computed noiselessly regardless of what was passed in. The
    # correct way to wire a backend's (and therefore its noise model's)
    # execution semantics into SamplerV2 is `SamplerV2.from_backend(...)`,
    # or equivalently passing `options={"backend_options": {...}}`. Fixed
    # here via `from_backend`, which also means results for noisy runs
    # will now differ from before this fix -- see the errata note in
    # README.md documenting exactly what this changes.
    sampler = AerSampler.from_backend(backend)
    rng = np.random.default_rng(seed)
    n_evals = [0]

    def objective_fn(params: np.ndarray) -> float:
        n_evals[0] += 1
        bound = isa_circuit.assign_parameters(params)
        result = sampler.run([bound], shots=shots).result()
        counts = result[0].data.meas.get_counts()
        energy = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            clean = bitstring.replace(" ", "")
            e = _ising_energy_from_bitstring(model, clean)
            energy += e * (count / total)
        return energy

    x0 = rng.uniform(0, np.pi, size=circuit.num_parameters)
    opt_result = minimize(objective_fn, x0, method="COBYLA", options={"maxiter": maxiter})

    # Final sampling with the optimized parameters to extract the best bitstring
    bound_final = isa_circuit.assign_parameters(opt_result.x)
    final_result = sampler.run([bound_final], shots=max(shots, 4096)).result()
    counts = final_result[0].data.meas.get_counts()

    best_bitstring, best_energy = None, np.inf
    for bitstring, count in counts.items():
        clean = bitstring.replace(" ", "")
        e = _ising_energy_from_bitstring(model, clean)
        if e < best_energy:
            best_energy, best_bitstring = e, clean

    y = np.array([int(best_bitstring[n - 1 - i]) for i in range(n)])
    res = evaluate_bitstring(problem, y)
    runtime = time.perf_counter() - t0

    ar = None
    if optimal_energy is not None and optimal_energy != 0:
        ar = float(res["qubo_value"] / optimal_energy)

    return QAOAResult(
        solver=f"QAOA(p={reps})", y=y, objective=res["qubo_value"],
        constraint_violations=res["constraint_violations"], runtime_s=runtime,
        p=reps, best_energy=best_energy, optimal_params=opt_result.x,
        n_qubits=n, n_circuit_evals=n_evals[0], approximation_ratio=ar,
    )


def _ising_energy_from_bitstring(model: IsingModel, bitstring: str) -> float:
    n = model.n
    if len(bitstring) != n:
        bitstring = bitstring.zfill(n)
    z = np.array([1 - 2 * int(bitstring[n - 1 - i]) for i in range(n)])
    e = model.C0 + float(model.h @ z)
    if model.J.any():
        e += float(z @ np.triu(model.J, k=1) @ z)
    return e


if __name__ == "__main__":
    from qopt.network.topology import get_topology
    from qopt.network.traffic import sample_requests
    from qopt.optimization.qubo import build_options, build_qubo
    from qopt.optimization.classical_baselines import solve_exact

    G = get_topology("A", seed=0)
    reqs = sample_requests(list(G.nodes()), n_requests=2, seed=1)
    link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
    options = build_options(G, reqs, link_state, k_paths=2)
    problem = build_qubo(options, reqs)
    print(f"n_qubits (options) = {len(options)}")

    exact = solve_exact(problem)
    print(f"Exact optimum: {exact.objective:.4f}")

    for p in [1, 2, 3]:
        r = run_qaoa_aer(problem, reps=p, shots=1024, maxiter=60, seed=0,
                          optimal_energy=exact.objective)
        print(f"QAOA p={p}: obj={r.objective:.4f}  AR={r.approximation_ratio:.3f}  "
              f"violations={r.constraint_violations}  n_evals={r.n_circuit_evals}  "
              f"runtime={r.runtime_s:.2f}s")
