# Q-OPT: Quantum-Optimized Crypto-Agility for Attack-Resilient QKD–PQC–QRNG Networks

**A security-constrained QUBO framework for adaptive routing and key management across hybrid quantum-key-distribution (QKD), post-quantum cryptography (PQC), and quantum-random-number-generation (QRNG) networks, solved with QAOA (Qiskit) and benchmarked against exact, MILP, greedy, and simulated-annealing baselines.**

[![CI](https://github.com/Sumitchongder/Q-OPT/actions/workflows/ci.yml/badge.svg)](https://github.com/Sumitchongder/Q-OPT/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Qiskit](https://img.shields.io/badge/qiskit-2.x-6929c4)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-16%20passing-brightgreen)]()

> **Note:** replace `Sumitchongder/Q-OPT` in the CI badge URL above with your actual
> GitHub username/repository once pushed (see Section 0 of the setup
> instructions in this repo's companion setup guide, or just find-and-replace
> `Sumitchongder/Q-OPT` throughout this file).

## Table of Contents

- [Errata](#errata)
- [1. What this repository is](#1-what-this-repository-is)
- [2. Repository layout](#2-repository-layout)
- [3. Installation](#3-installation)
- [4. Reproducing the results](#4-reproducing-the-results)
- [5. Running QAOA on real `ibm_marrakesh` hardware](#5-running-qaoa-on-real-ibm_marrakesh-hardware-your-8-minute-budget)
- [6. Regenerating the PQC benchmark table](#6-regenerating-the-pqc-benchmark-table-with-real-measurements)
- [7. Design notes / where the equations live](#7-design-notes--where-the-equations-live)
- [8. Known limitations](#8-known-limitations-be-upfront-about-these-in-your-methods-section)
- [9. Citation](#9-citation)
- [10. License](#10-license)
- [11. Project Governance](#11-project-governance)

---

## Errata

Two real bugs were found and fixed during this project's own development
(a silently-ignored noise model in the QAOA solver, and a matplotlib
backend crash on headless Linux/WSL) — both are documented in full,
including exactly which results were and were not affected, in
**[docs/ERRATA.md](docs/ERRATA.md)**. Read it before citing any
noise-model result from an early version of this repository.

---

## 1. What this repository is

This is the reference implementation accompanying the Q-OPT research program: a
joint security-constrained optimization of QKD key resources, PQC fallback,
QRNG entropy, and routing under dynamic attacks and physical-network
degradation. It formulates the problem as a QUBO, solves it with QAOA on
Qiskit (validated on `FakeMarrakesh` and runnable on real `ibm_marrakesh`
hardware), and compares against classical baselines (exact/brute-force,
MILP, greedy, simulated annealing).

**Companion project:** Q-TRACE (attack-probability estimation from QKD
telemetry) is not shipped in this repo; `qopt/threat/risk_engine.py`
exposes `from_q_trace_output()` as the integration point so its output can
be fed straight into Q-OPT's risk term rho_ij (closes the two-paper loop).

### Honesty notice (please read before citing results)

This repository is a **working, tested reference implementation**, not a
finished Q1 manuscript's full experimental campaign. Concretely:

- Every module listed below has been executed and its output checked
  (unit tests, `__main__` self-checks, and the experiment scripts have all
  been run at least once during development).
- The QUBO formulation uses **path-selection** (K-shortest candidate paths
  per request, chosen classically) rather than full edge-indicator flow
  routing. This is a deliberate, documented reduction (see
  `qopt/optimization/qubo.py` docstring) -- full flow-conservation routing
  is combinatorially far beyond what fits in 12 qubits / an 8-minute QPU
  budget. State this design choice explicitly in your methods section.
- The QKD channel model is a standard simplified GLLP-type decoy-state BB84
  rate formula (see `qopt/qkd/channel.py`), used as an **operational
  resource model**, not a new physical or security result.
- The PQC resource table defaults to literature-consistent reference
  figures (`qopt/pqc/profiles.py`); regenerate Table 2 of your paper with
  `--benchmark` (requires `liboqs-python`) on your target hardware before
  submission.
- The included experiments now cover 12 of the 15 original design-notes
  experiments (E1/E4/E5, E2, E3, E6, E7, E8, E9, E10, E11, E12, E13, E14,
  E15 -- everything except a full penalty-sensitivity sweep and dataset
  generation at scale, see Section 8). Each is independently runnable; see
  Section 4 for the full command list.
- The RL baseline (Q-learning/PPO) mentioned in the roadmap is **not**
  implemented in this drop; `qopt/optimization/classical_baselines.py` is
  the place to add it (its `SolverResult` interface is solver-agnostic).

Run the numbers yourself before trusting them in a submission -- that's the
entire point of shipping the code.

---

## 2. Repository layout

See also [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a pipeline
diagram and the design decisions worth knowing before reading the code.

```
.github/workflows/ci.yml             CI: runs the full test suite + notebook execution on every push/PR
.github/ISSUE_TEMPLATE/, PULL_REQUEST_TEMPLATE.md
CITATION.cff                         Machine-readable citation metadata (GitHub "Cite this repository" button)
CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CHANGELOG.md
Makefile                             make test / make quick / make full / make benchmark-pqc / ...
qopt/                          Installable Python package (pip install -e .)
├── network/
│   ├── topology.py            5 topology generators (A: line, B: ring, C: mesh, D: RGG, E: scale-free)
│   ├── traffic.py             Service-request sampler (source, dest, demand, security class, latency)
│   └── simulator.py           Per-timestep scenario simulator (Q-OPT-NET dataset schema)
├── qkd/
│   ├── channel.py             Decoy-state BB84 loss/QBER/key-rate model
│   ├── key_pool.py            Leaky-bucket per-link key pool dynamics
│   └── attacks.py             5 attack models (QBER injection, PNS, link denial, key exhaustion, control-plane)
├── pqc/
│   ├── profiles.py            ML-KEM / ML-DSA resource table (+ optional live liboqs benchmarking)
│   └── hybrid_kdf.py          HKDF-based hybrid QKD+PQC key combiner
├── qrng/
│   └── resource_model.py      QRNG entropy leaky-bucket resource
├── threat/
│   └── risk_engine.py         rho_ij risk function + Q-TRACE output adapter
└── optimization/
    ├── qubo.py                  Path+mode QUBO builder (objective + one-hot + security-floor penalties)
    ├── qubo_to_ising.py          Explicit QUBO -> Ising Hamiltonian conversion (Qiskit SparsePauliOp)
    ├── classical_baselines.py    Exact (brute force), Greedy, Simulated Annealing
    ├── milp_baseline.py          MILP baseline (scipy.optimize.milp)
    ├── rl_baseline.py            Minimal tabular Q-learning baseline (E6)
    └── qaoa_qiskit.py            QAOA solver (Qiskit + Aer), noiseless or with a supplied NoiseModel

experiments/
├── e1_solver_comparison.py           E1/E4/E5: Exact/Greedy/MILP/SA/QAOA(p=1,2,3) quality & runtime
├── e2_qber_sweep.py                  E8: crypto-mode selection shift vs channel QBER (QKD -> HYBRID)
├── e3_attack_intensity.py            E9: availability & security compliance vs attack intensity
├── e4_pareto_front.py                E12: multi-objective Pareto front (risk vs. latency)
├── e5_topology_generalization.py     E13: solver behavior across topologies A (5 nodes) -> E (20 nodes)
├── e6_ablation.py                    Ablation study (Section 40): contribution of each QUBO term
├── e7_qaoa_depth_and_exact_validation.py
│                                     E2: exact validation at n=3,6,9 qubits
│                                     E3: QAOA depth sweep p=1-4 with Mann-Whitney U tests
├── e8_qtrace_closed_loop.py          E14 (flagship): simulated Q-TRACE -> Q-OPT closed loop
├── e9_qaoa_vs_rl.py                  E6: QAOA vs. a tabular Q-learning baseline
├── e10_depletion_failure_scarcity_noise.py
│                                     E7 (key depletion), E10 (link failure/rerouting),
│                                     E11 (QRNG scarcity), E15 (noise-aware QAOA sweep) -- run any
│                                     of the four independently via --only {e7,e10,e11,e15}
├── e12_penalty_sensitivity.py        Section 17: penalty-sensitivity sweep (one-hot + security floor)
└── e_hw_prepare_marrakesh_job.py    FREE: builds + optimizes a QAOA job for real hardware (Aer only)

scripts/
└── run_on_ibm_marrakesh.py          ISOLATED, spends real ibm_marrakesh QPU time -- run manually only

tests/
├── test_core.py                     11 fast unit tests (topology, channel, QUBO<->Ising, solvers, ...)
└── test_qaoa_fake_marrakesh.py      QAOA validated against the real FakeMarrakesh noise model (0 QPU time)

figures/, tables/, results/          Generated outputs (gitignored; regenerate, don't commit)
docs/experiment_specs.md             Equations/spec for the not-yet-scripted experiments
docs/ARCHITECTURE.md                 Pipeline diagram and key design decisions
docs/ERRATA.md                       Full detail on the two documented bugs and their fixes
pyproject.toml, requirements.txt     Packaging + pinned dependencies
```

---

## 3. Installation

```bash
conda create -n qgss python=3.11 -y
conda activate qgss

git clone https://github.com/Sumitchongder/Q-OPT.git qopt
cd qopt
pip install -r requirements.txt
pip install -e .          # makes `import qopt` work from anywhere, incl. Jupyter
```

Or, once cloned: `make install` does the same two `pip install` steps.

Verify the install:

```bash
python -m pytest tests/ -v          # or: make test
```

Expected: **16 passed**, zero warnings.

---

## 4. Reproducing the results

Every experiment script is deterministic given its `--n-seeds`/topology
arguments (all randomness flows through explicit `numpy.random.default_rng`
seeds -- no hidden global RNG state). Figures are written as both `.pdf`
(vector, for LaTeX) and `.png` (raster, for quick viewing).

```bash
# Module self-checks (fast, no pytest needed) -- good smoke test after any edit
python -m qopt.network.topology
python -m qopt.qkd.channel
python -m qopt.pqc.hybrid_kdf
python -m qopt.threat.risk_engine
python -m qopt.optimization.qubo
python -m qopt.optimization.qubo_to_ising
python -m qopt.optimization.classical_baselines
python -m qopt.optimization.milp_baseline
python -m qopt.optimization.qaoa_qiskit
python -m qopt.optimization.rl_baseline

# Full unit + fake-hardware test suite (includes a noise-model regression guard)
python -m pytest tests/ -v

# Experiment 1 (E1/E4/E5): solver comparison
python experiments/e1_solver_comparison.py --n-seeds 30 --n-requests 2 --k-paths 2 --topology A

# Experiment 2 (E8): QBER sweep / crypto-mode selection
python experiments/e2_qber_sweep.py --n-seeds 25 --n-requests 4 --topology B

# Experiment 3 (E9): attack-intensity sweep / resilience
python experiments/e3_attack_intensity.py --n-seeds 25 --n-requests 4 --topology C

# Experiment 4 (E12): Pareto front (risk vs. latency)
python experiments/e4_pareto_front.py --n-seeds 15 --n-requests 4 --topology B

# Experiment 5 (E13): topology generalization A (5 nodes) -> E (20 nodes)
python experiments/e5_topology_generalization.py --n-seeds 10 --n-requests 2 --k-paths 2

# Experiment 6: ablation study (Section 40) -- which QUBO term matters most
python experiments/e6_ablation.py --n-seeds 20 --n-requests 4 --topology C

# Experiment 7, Part A (E2): exact validation at n=3,6,9 qubits
python experiments/e7_qaoa_depth_and_exact_validation.py --n-seeds 15 --skip-part-b

# Experiment 7, Part B (E3): QAOA depth sweep p=1-4 with Mann-Whitney U tests
python experiments/e7_qaoa_depth_and_exact_validation.py --n-seeds 15 --skip-part-a \
    --n-requests 2 --k-paths 2 --topology B

# Experiment 8 (E14, flagship): simulated Q-TRACE -> Q-OPT closed loop
python experiments/e8_qtrace_closed_loop.py --n-seeds 15 --n-requests 4 --topology C

# Experiment 9 (E6): QAOA vs. RL (Q-learning) vs. classical baselines
python experiments/e9_qaoa_vs_rl.py --n-seeds 15 --n-requests 3 --n-episodes 400

# Experiment 10: four lighter experiments, run independently or all at once
python experiments/e10_depletion_failure_scarcity_noise.py --only e7  --n-seeds 10 --n-requests 3 --topology C
python experiments/e10_depletion_failure_scarcity_noise.py --only e10 --n-seeds 10 --n-requests 3 --topology C
python experiments/e10_depletion_failure_scarcity_noise.py --only e11 --n-seeds 10 --n-requests 3 --topology C
python experiments/e10_depletion_failure_scarcity_noise.py --only e15 --n-seeds 10 --n-requests 2 --k-paths 2 --topology A

# Experiment 11 (Section 17): penalty-sensitivity sweep
python experiments/e12_penalty_sensitivity.py --n-seeds 15 --n-requests 3 --k-paths 1 --topology A
```

Outputs land in `tables/*.csv` and `figures/*.{pdf,png}`. The full list of
files each script produces is in the table at the end of this section.

Increase `--n-seeds` for the final paper run (30+ recommended); the values
above are chosen to finish in reasonable time on a laptop for iteration.
Note: after the noise-model bugfix (see Errata), E15 and the FakeMarrakesh
tests genuinely simulate device noise now, which is much slower than before
the fix -- budget a few minutes for `--only e15` alone, and note that
`bash run_all.sh` may need to run E15 in a separate invocation if your
environment has a wall-clock limit on a single command (see `run_all.sh`
itself, which runs everything else first).

### What each script produces

| Script | Figures | Tables |
|---|---|---|
| `e1_solver_comparison.py` | `solver_comparison_objective`, `solver_comparison_runtime` | `solver_comparison(.csv, _summary.csv)` |
| `e2_qber_sweep.py` | `e2_qber_mode_selection` | `e2_qber_sweep.csv`, `e2_qber_mode_mix.csv` |
| `e3_attack_intensity.py` | `e3_attack_intensity` | `e3_attack_intensity(.csv, _summary.csv)` |
| `e4_pareto_front.py` | `e4_pareto_front` | `e4_pareto_grid.csv`, `e4_pareto_front.csv` |
| `e5_topology_generalization.py` | `e5_topology_generalization` | `e5_topology_generalization(.csv, _summary.csv)` |
| `e6_ablation.py` | `e6_ablation` | `e6_ablation(.csv, _summary.csv)` |
| `e7_..._validation.py` (Part A) | `e7a_exact_validation` | `e7a_exact_validation.csv` |
| `e7_..._validation.py` (Part B) | `e7b_depth_sweep` | `e7b_depth_sweep.csv`, `e7b_depth_sweep_mannwhitney.csv` |
| `e8_qtrace_closed_loop.py` | `e8_qtrace_closed_loop` | `e8_qtrace_closed_loop(.csv, _summary.csv)` |
| `e9_qaoa_vs_rl.py` | `e9_qaoa_vs_rl` | `e9_qaoa_vs_rl.csv` |
| `e10_..._noise.py --only e7` | `e7_key_depletion` | `e7_key_depletion.csv` |
| `e10_..._noise.py --only e10` | `e10_link_failure` | `e10_link_failure.csv` |
| `e10_..._noise.py --only e11` | `e11_qrng_scarcity` | `e11_qrng_scarcity.csv` |
| `e10_..._noise.py --only e15` | `e15_noise_sweep` | `e15_noise_sweep.csv` |
| `e12_penalty_sensitivity.py` | `e12_penalty_sensitivity` | `e12_penalty_sensitivity_onehot.csv`, `e12_penalty_sensitivity_security.csv` |

---

## 5. Running QAOA on real `ibm_marrakesh` hardware (your 8-minute budget)

**Nothing above touches real quantum hardware.** All QAOA results in
Experiments 1-3 and in the test suite run on `qiskit-aer`, optionally with
the real device's calibrated noise model via
`qiskit_ibm_runtime.fake_provider.FakeMarrakesh` -- this is free, unlimited,
and is what you should use for all development and debugging.

When you're ready to spend real QPU time, follow these steps **exactly**:

### Step 1 -- Save your IBM credentials (once)

In a Jupyter cell or Python shell:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token="my_api_key",
    instance="my_crn",
    overwrite=True,
    set_as_default=True,
)

service = QiskitRuntimeService()
backends = service.backends()
print(f"Account OK. {len(backends)} backend(s) available:")
for b in backends[:5]:
    print(f"  {b.name} ({b.num_qubits} qubits)")
```

### Step 2 -- Prepare the job for FREE (Aer only, no QPU time spent)

```bash
python experiments/e_hw_prepare_marrakesh_job.py \
    --topology A --n-requests 2 --k-paths 2 --reps 1 \
    --use-marrakesh-noise
```

This builds a small (roughly 6-8 qubit) QUBO, converts it to Ising form, and
runs the **entire COBYLA parameter-optimization loop** against Aer with the
`FakeMarrakesh` noise model baked in -- so the angles you submit to real
hardware are already noise-adapted. It writes:

- `results/qaoa_hw_problem.json`
- `results/qaoa_hw_optimal_params.json`

### Step 3 -- Dry-run the time estimate (still free)

```bash
python scripts/run_on_ibm_marrakesh.py \
    --problem-json results/qaoa_hw_problem.json \
    --params-json  results/qaoa_hw_optimal_params.json \
    --reps 1 --shots 2000 --dry-run
```

This authenticates, fetches the real `ibm_marrakesh` backend's calibration
data, transpiles the actual circuit you would submit, and prints an
estimated execution time. **It refuses to proceed to Step 4 if the estimate
exceeds a hard-coded, safety-margined 8-minute budget.**

### Step 4 -- Submit for real (spends QPU time -- run deliberately)

```bash
python scripts/run_on_ibm_marrakesh.py \
    --problem-json results/qaoa_hw_problem.json \
    --params-json  results/qaoa_hw_optimal_params.json \
    --reps 1 --shots 2000
```

This submits **exactly one circuit** (the QAOA ansatz at your pre-optimized
angles) as a single `SamplerV2` job -- no on-hardware optimization loop, by
design, since that would burn your budget fast. Results are saved to
`results/qaoa_hw_problem_hw_result.json` and printed, including the
hardware-vs-Aer energy gap (report this as your "hardware validation" row).

If you want results at `p=2` or `p=3` too, repeat Steps 2-4 with
`--reps 2` / `--reps 3`, watching your remaining allocation between runs.
**Do not** script a loop over `p` that calls Step 4 automatically.

---

## 6. Regenerating the PQC benchmark table with real measurements

The shipped `qopt/pqc/profiles.py::REFERENCE_PROFILES` are literature-scale
placeholder figures. For a submission-grade table:

```bash
pip install liboqs-python
python -m qopt.pqc.profiles --benchmark
```

This measures real ML-KEM keygen/encapsulation/decapsulation latency AND
real ML-DSA keygen/sign/verify latency on your machine (both algorithm
families are fully measured as of this version -- confirmed stable across
n_iters=200 vs n_iters=2000 runs, varying by <20%, consistent with genuine
sub-millisecond timing noise rather than measurement instability). If
`liboqs-python` needs to compile liboqs from source on your machine and
fails with `cmake: not found`, install build tools first:

```bash
conda install -c conda-forge cmake ninja gxx_linux-64 gcc_linux-64 -y
```

For the ML-DSA numbers, note the schema reuses the KEM field names for
convenience: `encapsulation_latency_ms` holds the **sign** latency and
`decapsulation_latency_ms` holds the **verify** latency for signature
algorithms (documented in `qopt/pqc/profiles.py`, not a bug). State this
explicitly in your Table 2 caption to avoid confusing a reviewer.

---

## 7. Design notes / where the equations live

- **QUBO objective** (risk, latency, key-pool depletion, compute cost,
  service disruption): `qopt/optimization/qubo.py::build_qubo`, weights in
  `DEFAULT_WEIGHTS`. A **penalty-sensitivity sweep** is one line to add:
  loop `penalty_onehot`/`penalty_security` over `{1,5,10,25,50,100} *
  obj_scale` and re-solve.
- **QUBO -> Ising**: `qopt/optimization/qubo_to_ising.py`, unit-tested exact
  on all 2^n bitstrings for random instances up to n=5.
- **QAOA ansatz**: `qopt/optimization/qaoa_qiskit.py::build_qaoa_circuit`,
  uses `qiskit.circuit.library.QAOAAnsatz`, COBYLA outer loop.
- **Hybrid key combiner**: `qopt/pqc/hybrid_kdf.py`, HKDF-based dual-PRF
  combiner (K_H = HKDF(KDF(K_QKD) concat KDF(K_PQC), ctx)).
- **Attack models**: `qopt/qkd/attacks.py`, five categories (QBER
  injection, PNS degradation, link denial, key exhaustion, control-plane
  compromise).

See `docs/experiment_specs.md` for the equations and expected
figures/tables for every experiment in the original design, including the
ones not yet scripted.

---

## 8. Known limitations (be upfront about these in your methods section)

1. Path-based (not full flow-conservation) routing QUBO -- documented above.
   A structural consequence (found while building E13): the qubit count in
   this formulation is driven by traffic (n_requests x k_paths x 3 modes),
   NOT by network size -- a 20-node network and a 5-node network can need
   the exact same number of qubits if traffic volume is held fixed. Report
   this explicitly if you claim "generalization to larger networks."
2. The RL baseline (`qopt/optimization/rl_baseline.py`) is a minimal
   tabular Q-learning agent treating each request as an independent
   contextual bandit -- not a sophisticated deep-RL (PPO) agent. Treat E6
   results as a floor, not a ceiling, for what RL can achieve here.
3. PQC/QRNG are resource abstractions (rate/latency/availability), not
   physical simulations -- intentional, per the design spec.
4. `qopt/network/simulator.py`'s default parameters (key-pool capacity vs.
   generation rate) saturate pools quickly in the demo config -- tune
   `pool_capacity_bits` relative to `key_generation_rate_bps` for your
   target link lengths before using it for a depletion-focused experiment.
   (Experiment 10's `--only e7` works around this by sweeping
   `key_pool_frac` directly rather than through the dynamical simulator.)
5. n_qubits in {4, 5} are structurally UNREACHABLE in the path+mode QUBO
   formulation (every candidate path always contributes exactly 3
   mode-options, so n_qubits is always a multiple of 3) -- found while
   building Experiment 7 Part A; validated at n=3,6,9 instead.
6. See the Errata section at the top of this file for a real noise-model
   wiring bug found and fixed during development, and exactly which
   results it did and didn't affect.
7. Not yet done: Q-OPT-NET dataset generation at scale (Section 30). This
   is a straightforward extension of `qopt/network/simulator.py` but has
   not been run and verified here. (The penalty-sensitivity sweep, Section
   17, item formerly listed here, is now done -- see
   `experiments/e12_penalty_sensitivity.py`. It found a clean, sharp
   feasibility transition: the global QUBO minimum is one-hot-INFEASIBLE
   60% of the time at penalty P=0.5, but perfectly feasible (0 violations)
   for every P>=1 tested up to P=100 -- a wide, comfortable safety margin
   around the repo's actual default penalty magnitudes.)
8. PQC benchmark table (`qopt/pqc/profiles.py`) still defaults to
   literature reference figures pending `liboqs-python` on your machine
   (Section 6).

---

## 9. Citation

This repository includes a [`CITATION.cff`](CITATION.cff) file — on GitHub,
use the **"Cite this repository"** button in the sidebar to get a
formatted citation, or use the BibTeX below directly (fill in your name
and repository URL in both places):

```bibtex
@misc{qopt2026,
  title  = {Q-OPT: Quantum-Optimized Crypto-Agility for Attack-Resilient QKD-PQC-QRNG Networks},
  author = {<your name>},
  year   = {2026},
  note   = {Code: https://github.com/Sumitchongder/Q-OPT},
  url    = {https://github.com/Sumitchongder/Q-OPT}
}
```

## 10. License

MIT -- see [`LICENSE`](LICENSE).

## 11. Project Governance

- [`CONTRIBUTING.md`](CONTRIBUTING.md) -- how to propose changes, coding
  conventions, and how to add a new experiment script.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) -- community standards
  (Contributor Covenant v2.1).
- [`SECURITY.md`](SECURITY.md) -- how to report a security issue,
  including this project's stance on credentials (never committed;
  see also [`docs/ERRATA.md`](docs/ERRATA.md)).
- [`CHANGELOG.md`](CHANGELOG.md) -- notable changes, including both
  documented bug fixes.
