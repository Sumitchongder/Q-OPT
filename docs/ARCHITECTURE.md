# Architecture Overview

## Pipeline

```
                         DYNAMIC QUANTUM-SAFE NETWORK
                                      |
              +-----------------------+-----------------------+
              |                       |                       |
             QKD                     PQC                     QRNG
    (qopt/qkd/)              (qopt/pqc/)              (qopt/qrng/)
              |                       |                       |
        QBER / loss              ML-KEM / ML-DSA         entropy rate
        key generation           latency / cost           availability
        key pools                bandwidth
              |                       |                       |
              +-----------------------+-----------------------+
                                      |
                              THREAT INTELLIGENCE
                            (qopt/threat/risk_engine.py)
                                      |
                                      v
                         SECURITY-CONSTRAINED QUBO
                            (qopt/optimization/qubo.py)
                                      |
                       +--------------+--------------+
                       |                             |
                  Classical                     Quantum
             (exact / greedy / SA /               QAOA
              MILP / Q-learning)          (qopt/optimization/qaoa_qiskit.py)
                       |                             |
                       +--------------+--------------+
                                      v
                              Q-OPT DECISION
                                      |
                  +-------------------+-------------------+
                  v                   v                   v
                 QKD                  PQC               HYBRID
                  |                   |                   |
                  +-------------------+-------------------+
                                      v
                         RESILIENT SECURE SERVICE
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `qopt/network/` | Topology generators (A–E), traffic sampler, per-timestep scenario simulator |
| `qopt/qkd/` | Decoy-state BB84 channel physics, key-pool leaky-bucket dynamics, 5 attack models |
| `qopt/pqc/` | ML-KEM/ML-DSA resource profiles (literature reference or `liboqs`-measured), hybrid KDF combiner |
| `qopt/qrng/` | Entropy resource abstraction (leaky-bucket, same structure as QKD key pools) |
| `qopt/threat/` | Risk engine: maps physical/operational signals (or external Q-TRACE output) to per-link risk ρ_ij |
| `qopt/optimization/` | QUBO builder, QUBO→Ising conversion, all solvers (Exact, Greedy, SA, MILP, Q-learning, QAOA) |
| `qopt/viz/` | Shared matplotlib styling for all generated figures |
| `experiments/` | One script per experiment (E1–E15 + ablation + penalty sweep), each independently runnable |
| `scripts/` | Real-hardware submission (isolated) and PQC benchmark aggregation |
| `tests/` | Unit tests + fake-hardware (FakeMarrakesh) validation + regression guards |

## Design decisions worth knowing before reading the code

1. **Path-based QUBO, not full flow-conservation.** Routing decisions
   are restricted to a small set of K classically-precomputed
   candidate paths per request; the QUBO then selects one
   (path, crypto-mode) pair per request. This keeps the qubit count
   tractable for QAOA (documented in `qopt/optimization/qubo.py`).
   A structural consequence: qubit count is driven by **traffic**
   (`n_requests × k_paths × 3 modes`), not by network size — found
   while building the topology-generalization experiment (E13).

2. **Every candidate path contributes exactly 3 mode-options**
   (QKD / PQC / HYBRID), so the QUBO's qubit count is always a
   multiple of 3. n=4 or n=5 qubits are structurally unreachable in
   this formulation (found while building the exact-validation
   experiment, E2) — validated at n=3, 6, 9 instead.

3. **The one-hot and security-floor constraints are soft penalties,
   not hard constraints.** `experiments/e12_penalty_sensitivity.py`
   exists specifically to show these penalties are set well within a
   safe margin (see `docs/ERRATA.md`-adjacent discussion in
   `README.md` Section 7).

4. **QAOA solver correctness is layered**: `qopt/optimization/qubo_to_ising.py`
   is unit-tested exact on all 2^n bitstrings before it's ever used
   inside `qaoa_qiskit.py`, so a QAOA result that looks wrong is a
   solver/optimization issue, not a Hamiltonian-construction bug.

See `README.md` Section 7 ("Design notes") for pointers to the exact
equations, and `docs/experiment_specs.md` for what each experiment
tests and why.
