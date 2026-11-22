# Architecture

```
            physical layer                      decision layer                     evaluation
 ┌─────────────────────────────┐   ┌───────────────────────────────────┐   ┌──────────────────────────┐
 │ qkd.channel  LMC05 rate/QBER│──▶│ agility.model                      │──▶│ agility.twin              │
 │ qkd.attacks  QBER/PNS/denial│   │  options (path x mode), c, W       │   │  hard budgets, Markov    │
 │ network      topology/paths │   │  block-one-hot quadratic program   │   │  attacks, CRN, metrics   │
 │ pqc / qrng   resource costs │   └───────────────┬───────────────────┘   └──────────────────────────┘
 └─────────────────────────────┘                   │
                                   ┌───────────────┴───────────────────┐
                                   │ agility.solvers   quantum.*         │
                                   │  exact, greedy,   circuits (XY/PEN) │
                                   │  SA, tabu, rule,  fastsim, train    │
                                   │  separable        hardware (budget) │
                                   └───────────────────────────────────┘
```

Design decisions worth knowing before reading the code:

1. **One objective, many solvers.** Every solver, classical or quantum, is scored by
   `AgilityProblem.energy_choice`, so quality comparisons are exact.
2. **Surrogate vs. metric are separated.** The optimiser sees quadratic congestion; the
   twin enforces hard budgets. Gains in E2 are measured, not assumed.
3. **Two energy views.** `qubo_penalized(P)` (full space, for the penalty baseline) and
   `qubo_feasible_compressed()` (one-hot subspace, intra-block couplings removed). Both
   are tested to reproduce `E(y)` on every feasible assignment.
4. **Simulator-circuit equivalence is a test, not an assumption.** The exact subspace
   simulator reproduces the Trotterised gate order of the circuit; agreement with Qiskit
   `Statevector` is asserted to 1e-12.
5. **Hardware isolation.** Nothing in `qopt/` or `experiments/` submits jobs. QPU time is
   spent only from `notebooks/qpu_run.ipynb`, behind pilots, a budget planner and an
   explicit `CONFIRM_SUBMIT` flag. The plan is serialised (`plan.json`, `circuits.qpy`)
   so the circuits executed are byte-identical to those analysed.
6. **Provenance.** Each experiment writes `results/manifest/<name>.json` (arguments,
   versions, git commit, runtime); `docs/RESULTS.md` is generated from `tables/`.
