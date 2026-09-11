# Experiment specifications (design-notes numbering E1-E15)

**Update: all of E2, E3, E6, E7, E9, E10, E11, E12, E13, E14, E15, and the
Section 17 penalty-sensitivity sweep now have working, tested
implementations** (see the table below). Only Q-OPT-NET dataset
generation at scale (Section 30) remains unimplemented.

| Experiment | Script | Notes |
|---|---|---|
| E1/E4/E5 | `experiments/e1_solver_comparison.py` | |
| E2 | `experiments/e7_qaoa_depth_and_exact_validation.py` (Part A) | n=4,5 structurally impossible; uses n=3,6,9 |
| E3 | `experiments/e7_qaoa_depth_and_exact_validation.py` (Part B) | includes Mann-Whitney U tests |
| E6 | `experiments/e9_qaoa_vs_rl.py` + `qopt/optimization/rl_baseline.py` | minimal tabular Q-learning |
| E7 | `experiments/e10_depletion_failure_scarcity_noise.py --only e7` | |
| E8 | `experiments/e2_qber_sweep.py` | |
| E9 | `experiments/e3_attack_intensity.py` | |
| E10 | `experiments/e10_depletion_failure_scarcity_noise.py --only e10` | compares topology A vs. C redundancy |
| E11 | `experiments/e10_depletion_failure_scarcity_noise.py --only e11` | |
| E12 | `experiments/e4_pareto_front.py` | |
| E13 | `experiments/e5_topology_generalization.py` | found qubit count is traffic-driven, not size-driven |
| E14 | `experiments/e8_qtrace_closed_loop.py` | flagship; simulated Q-TRACE noise stand-in |
| E15 | `experiments/e10_depletion_failure_scarcity_noise.py --only e15` | see Errata in README re: noise-model bugfix |
| Section 17 (penalty sensitivity) | `experiments/e12_penalty_sensitivity.py` | sharp feasibility transition found at P=1 |

Below is the original equations/rationale reference, kept as-is.

## E2 — Exact small-network validation
Compare QAOA against brute force for n=4,5,6 qubits.
- Build: `qopt.optimization.qubo.build_qubo` with `n_requests` small enough
  that `len(options) in {4,5,6}` (tune `k_paths`/`n_requests`).
- Solve: `classical_baselines.solve_exact` + `qaoa_qiskit.run_qaoa_aer`.
- Metric: optimality gap `(QAOA_obj - exact_obj) / |exact_obj|`.
- This is essentially `tests/test_qaoa_fake_marrakesh.py` generalized across
  n and seeds, with results tabulated instead of asserted.

## E3 — QAOA depth (p=1,2,3,4)
- Loop `reps` in `run_qaoa_aer`, record `approximation_ratio`,
  `n_circuit_evals`, transpiled circuit depth
  (`build_qaoa_circuit(...).decompose().depth()`), and `runtime_s`.
- Plot: approximation ratio and depth vs p (twin y-axes).

## E6 — QAOA vs RL
- Requires an RL baseline (Q-learning or PPO) — **not implemented**.
  Suggested minimal version: treat each request's option choice as an
  independent contextual bandit; tabular Q-learning over
  `(request_features_bucketed) -> option_index` is enough for a first
  result and keeps the state space small.
- Once implemented, it plugs into `SolverResult` exactly like the other
  baselines in `classical_baselines.py`.

## E7 — Key depletion sweep
- Use `qopt.network.simulator.run_scenario` with an
  `AttackEvent(attack_type=KEY_EXHAUSTION, intensity=alpha)` sweep, or
  directly sweep `link_state[...]["key_pool_frac"]` downward in
  `build_options` calls (as `e3_attack_intensity.py` already does for the
  `key_pool_frac` channel — E7 is the same idea isolated to *only* the
  key-pool term, holding QBER/trust fixed, to isolate this one failure mode).
- Metrics: service failures (`n_requests - n_feasible`), remaining key
  reserve (`mean_key_pool_frac`), mode-switch counts across consecutive
  alpha steps (diff the per-seed mode assignment).

## E10 — Link failure / rerouting
- Set `key_generation_rate_bps -> 0` on specific edges
  (`qopt.qkd.attacks.apply_link_denial`), which should make
  `k_shortest_paths` fall back to alternate paths (or return fewer paths if
  the network is disconnected by removing the edge — check with
  `networkx.is_connected` after simulating removal).
- Metrics: recovery time (time steps until a request is reassigned to
  a working path in a stepped simulation), rerouting success rate,
  service availability.

## E11 — QRNG scarcity
- Sweep `qopt.qrng.resource_model.QRNGResource.rate_bps` downward /
  pre-deplete `current_bits`; wire `qr.is_scarce()` into the QUBO as a
  soft penalty on HYBRID options at scarce nodes (HYBRID needs QRNG-backed
  entropy for its combiner nonce in the current design — extend `Option`
  with a `qrng_dependency` flag if you want this to bind tightly to
  specific requests rather than a network-wide toggle).

## E12 — Multi-objective sensitivity / Pareto fronts
- Sweep `w_R, w_L, w_K, w_C` in `DEFAULT_WEIGHTS` over a grid (e.g. each in
  `{0.2, 0.5, 1.0, 2.0}`), solve exactly at each point, record
  `(mean_risk_of_solution, mean_latency_of_solution)` pairs, and plot the
  non-dominated front. `evaluate_bitstring(problem, y)["chosen"]` gives you
  the `Option` objects to recompute raw (non-normalized) risk/latency for
  the Pareto axes.

## E13 — Unseen topology generalization
- Calibrate QAOA angles (COBYLA-optimized `optimal_params`) on topologies
  A/B/C (5-10 nodes), then **reuse those same fixed angles** (no
  re-optimization) on topologies D/E (15-20 nodes) by rebuilding the QUBO
  at the new size and padding/truncating the parameter vector — or, more
  honestly, report that QAOA angles don't transfer across qubit counts
  directly and instead compare *architecture* transfer (same weights,
  same solver pipeline, fresh COBYLA run) rather than literal angle reuse.

## E14 — End-to-end Q-TRACE -> Q-OPT (flagship experiment)
- Since Q-TRACE isn't shipped here, simulate its output: replace
  `attack_probability` in `e3_attack_intensity.py`'s link_state
  construction with a noisy, *delayed/imperfect* estimate of the true
  attack intensity (e.g. `p_hat = clip(true_alpha + noise, 0, 1)` with
  noise variance as a knob), via
  `qopt.threat.risk_engine.from_q_trace_output`. Compare:
  - static (no risk adaptation, fixed mode assignment)
  - trust-only fallback (mode switches only on a hard trust threshold)
  - Q-OPT + true attack probability (oracle upper bound)
  - Q-OPT + simulated Q-TRACE estimate (realistic)
  on availability/compliance/mean-risk, same style as `e3`'s plot.

## E15 — Noise-aware QAOA
- Sweep depolarizing/readout noise strength directly via
  `qiskit_aer.noise.NoiseModel` (not just the fixed `FakeMarrakesh`
  snapshot): `qiskit_aer.noise.depolarizing_error` on 1q/2q gates, pass as
  `noise_model` to `run_qaoa_aer`. Plot approximation ratio vs noise
  strength. `tests/test_qaoa_fake_marrakesh.py` already demonstrates the
  plumbing for a single noise model; E15 is that same call in a sweep.

---

### Statistical methodology (Section 52 of the original design notes)
For every experiment above: run `N >= 30` seeds where compute allows,
report mean +/- std (or median/IQR for runtime, which tends to be
right-skewed), and use a non-parametric test (e.g.
`scipy.stats.mannwhitneyu`) for pairwise solver comparisons rather than
assuming normality.
