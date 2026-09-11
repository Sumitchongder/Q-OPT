# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-09

### Added
- Core package: network topology generators (A–E), QKD channel physics
  model, key-pool dynamics, PQC resource profiles, QRNG resource model,
  risk engine, hybrid KDF combiner.
- QUBO builder (path+mode selection formulation) and QUBO→Ising
  conversion, verified exact on all 2^n bitstrings for test instances.
- Classical baselines: Exact (brute force), Greedy, Simulated Annealing,
  MILP (`scipy.optimize.milp`), tabular Q-learning.
- QAOA solver on Qiskit/Aer, with real `ibm_marrakesh` hardware
  validation at p=1, 2, 3 (see `results/`).
- 15 experiment scripts covering the full original design-notes
  experiment list (E1–E15) plus an ablation study and a penalty-
  sensitivity sweep (Section 17).
- Full test suite (16 tests): unit tests for every core module plus a
  noise-model regression guard.
- `scripts/run_on_ibm_marrakesh.py`: isolated real-hardware submission
  script with a pre-submission time-budget check.
- `scripts/summarize_pqc_benchmark.py`: multi-run mean±std PQC
  benchmark aggregator.

### Fixed
- **Noise-model wiring bug** in `qopt/optimization/qaoa_qiskit.py`:
  `AerSampler()` was constructed with no backend reference, silently
  ignoring any `NoiseModel` passed to `run_qaoa_aer(...)`. This meant
  every noise-model-aware test/experiment (including the original
  "FakeMarrakesh noise" test suite) ran noiselessly regardless of what
  noise model was supplied. Fixed via `AerSampler.from_backend(backend)`.
  A regression-guard test
  (`test_noise_model_actually_changes_sampled_distribution`) was added
  so this class of bug cannot silently reappear. See `README.md`
  Errata section for full details and exactly which results were and
  were not affected.
- **Matplotlib backend crash on headless Linux/WSL**: several plotting
  scripts crashed at interpreter exit with `RuntimeError: main thread
  is not in main loop` because matplotlib auto-selected the interactive
  `TkAgg` backend whenever Tkinter was importable, even with no display
  server running. Fixed by forcing `matplotlib.use("Agg")` as the first
  statement executed on `import qopt` (`qopt/__init__.py`), before any
  script can import `matplotlib.pyplot`.
- Qiskit `NLocal`/`BlueprintCircuit` deprecation warnings (upstream,
  harmless) now suppressed via `[tool.pytest.ini_options]` in
  `pyproject.toml` so `pytest -v` output stays clean.

### Changed
- `qopt/pqc/profiles.py::benchmark_live()` extended to measure ML-DSA
  keygen/sign/verify latency (previously ML-KEM only, with ML-DSA
  stubbed to literature reference values). Both families now report
  `source="measured"` when `liboqs-python` is installed and built.

### Known limitations (see `README.md` Section 8 for full list)
- Path-based (not full flow-conservation) routing QUBO.
- RL baseline is a minimal tabular Q-learning agent, not PPO.
- Q-OPT-NET dataset generation at scale (Section 30 of the design
  notes) is not implemented.
