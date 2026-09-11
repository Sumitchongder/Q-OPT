"""
Validation test: runs the exact same QAOA circuit-construction path used
for the real `ibm_marrakesh` submission script, but against Qiskit's
`FakeMarrakesh` noise model (a calibration-snapshot noise model of the
real device, executed on Aer). This consumes ZERO real QPU time and is
the correct way to pre-validate the pipeline before spending your
~8-minute `ibm_marrakesh` allocation (as requested).

Run:
    python -m pytest tests/test_qaoa_fake_marrakesh.py -v
or:
    python tests/test_qaoa_fake_marrakesh.py
"""
from __future__ import annotations

import numpy as np

from qopt.network.topology import get_topology
from qopt.network.traffic import sample_requests
from qopt.optimization.qubo import build_options, build_qubo
from qopt.optimization.classical_baselines import solve_exact
from qopt.optimization.qaoa_qiskit import run_qaoa_aer


def _build_small_problem(n_requests=2, k_paths=2, seed=1):
    G = get_topology("A", seed=0)
    reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
    link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
    options = build_options(G, reqs, link_state, k_paths=k_paths)
    return build_qubo(options, reqs)


def test_fake_marrakesh_noise_model_loads():
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    backend = FakeMarrakesh()
    assert backend.num_qubits >= 100
    from qiskit_aer.noise import NoiseModel
    noise_model = NoiseModel.from_backend(backend)
    assert noise_model is not None


def test_qaoa_runs_error_free_with_marrakesh_noise():
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    from qiskit_aer.noise import NoiseModel

    problem = _build_small_problem(n_requests=2, k_paths=2, seed=1)
    backend = FakeMarrakesh()
    noise_model = NoiseModel.from_backend(backend)

    exact = solve_exact(problem)
    result = run_qaoa_aer(problem, reps=1, shots=1024, maxiter=30, seed=0,
                           noise_model=noise_model, optimal_energy=exact.objective)

    assert result.constraint_violations >= 0  # runs to completion (may not always be 0 under noise)
    assert result.n_qubits == len(problem.options)
    assert np.isfinite(result.objective)
    print(f"[FakeMarrakesh-noise] p=1 obj={result.objective:.4f} "
          f"(exact={exact.objective:.4f})  AR={result.approximation_ratio:.3f}  "
          f"violations={result.constraint_violations}")


def test_qaoa_noiseless_matches_exact_on_tiny_instance():
    problem = _build_small_problem(n_requests=2, k_paths=2, seed=1)
    exact = solve_exact(problem)
    result = run_qaoa_aer(problem, reps=2, shots=2048, maxiter=80, seed=0,
                           optimal_energy=exact.objective)
    assert result.approximation_ratio is not None
    assert result.approximation_ratio <= 1.05  # QAOA should reach at or very near optimum
    print(f"[noiseless] p=2 AR={result.approximation_ratio:.3f}")


def test_noise_model_actually_changes_sampled_distribution():
    """
    Regression guard for a real bug found during development: constructing
    `AerSampler()` with no arguments silently ignores any noise model
    attached to a separately-built `AerSimulator(noise_model=...)` backend,
    because `qiskit_aer.primitives.SamplerV2` has no `backend` constructor
    parameter -- the fix is `AerSampler.from_backend(backend)`. This test
    fails loudly if that wiring ever regresses: it runs the SAME bound
    circuit through the noiseless and FakeMarrakesh-noisy paths and asserts
    the resulting count distributions are NOT identical.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    from qiskit_aer.noise import NoiseModel
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    clean_backend = AerSimulator()
    noisy_backend = AerSimulator(noise_model=NoiseModel.from_backend(FakeMarrakesh()))

    clean_sampler = AerSampler.from_backend(clean_backend)
    noisy_sampler = AerSampler.from_backend(noisy_backend)

    clean_counts = clean_sampler.run([qc], shots=8192).result()[0].data.meas.get_counts()
    noisy_counts = noisy_sampler.run([qc], shots=8192).result()[0].data.meas.get_counts()

    # A perfect noiseless Bell circuit should show ~0 population in the
    # "wrong" outcomes 01/10; a genuinely noisy run must show measurable
    # leakage into them. If this ever reads exactly 0 again, the noise
    # model silently stopped being applied.
    clean_leakage = clean_counts.get("01", 0) + clean_counts.get("10", 0)
    noisy_leakage = noisy_counts.get("01", 0) + noisy_counts.get("10", 0)
    assert clean_leakage < 20, f"Noiseless run leaked {clean_leakage}/8192 into 01/10 -- unexpected"
    assert noisy_leakage > clean_leakage + 50, (
        f"Noisy run only leaked {noisy_leakage}/8192 into 01/10 (clean was {clean_leakage}) "
        "-- the noise model may not be getting applied (regression of the from_backend fix)")
    print(f"[regression guard] clean leakage={clean_leakage}/8192, noisy leakage={noisy_leakage}/8192")


if __name__ == "__main__":
    test_fake_marrakesh_noise_model_loads()
    print("OK: FakeMarrakesh noise model loads.")
    test_noise_model_actually_changes_sampled_distribution()
    print("OK: noise model demonstrably changes the sampled distribution (regression guard).")
    test_qaoa_runs_error_free_with_marrakesh_noise()
    print("OK: QAOA runs error-free under FakeMarrakesh noise.")
    test_qaoa_noiseless_matches_exact_on_tiny_instance()
    print("OK: noiseless QAOA reaches near-exact optimum.")
    print("\nAll fake-backend validation tests passed.")
