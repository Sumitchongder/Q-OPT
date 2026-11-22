# Running the hardware experiment

This is the only step that spends IBM Quantum time. Everything it needs is
already in the repository: `results/hardware/plan/plan.json` and
`results/hardware/plan/circuits.qpy` (16 circuits, built and trained for free).

## What will be executed

| Instance | Qubits | Feasible states | Circuits |
|---|---|---|---|
| R3_s0, R3_s1 | 9 | 27 | W state (p=0), XY-QAOA p=1, XY-QAOA p=2, penalty QAOA p=1 |
| R4_s2, R4_s11 | 12 | 81 | same four |

Transpiled on a Heron heavy-hex model: 12-156 CZ gates per circuit (best of 16 layouts).
The W-state circuits are the control that measures how well the feasible subspace
survives state preparation alone; the penalty circuits are the textbook baseline at a
comparable gate count.

## Steps

```bash
conda activate qopt
cd notebooks && jupyter notebook qpu_run.ipynb
```

1. **Rehearse (free).** Leave `USE_FAKE_BACKEND = True` and *Run All*. It must finish
   with a results table. This exercises transpilation, pilots, the budget planner and
   the result writer on `FakeMarrakesh`.
2. **Credentials (once).** In the credentials cell set `SAVE_ACCOUNT = True`, paste your
   token and CRN, run it, then set it back to `False` and **delete the token from the
   cell**. Never commit a notebook containing a token. If a token was ever pasted into a
   chat, an issue or a commit, revoke it in the IBM Quantum dashboard and create a new one.
3. **Configure.** `USE_FAKE_BACKEND = False`; optionally `BACKEND_NAME = "ibm_marrakesh"`
   (default: least-busy device with >= 127 qubits); `BUDGET_SECONDS = 420` keeps a
   one-minute reserve inside an 8-minute allocation.
4. **Run cells up to the pilot cell.** The pilots are two single-circuit jobs
   (512 and 4 096 shots, a few seconds each). The cell prints the fitted usage model and
   the shots per circuit the main job will use.
5. **Submit.** If the prediction is acceptable, set `CONFIRM_SUBMIT = True` and run the
   main-job cell. Without it the cell refuses to submit. Expected cost ≈ 1-1.5 min.
6. **Save.** The next cell writes `results/hardware/run_qpu_<backend>_<timestamp>/raw.json`
   (all counts, job ids, layouts, calibration snapshot, options, usage). Do this before
   anything else.
7. **Analyse.**
   ```bash
   python experiments/exp07_hardware_analysis.py --run-dir results/hardware/run_qpu_<...>
   python scripts/summarize_results.py
   ```

Send back the whole `run_qpu_*` folder (and the executed notebook, outputs included).

## Error suppression and post-processing

* Sampler options: dynamical decoupling (XpXm) and Pauli twirling of gates and
  measurements. What was actually applied is recorded in `raw.json`.
* Symmetry verification: shots violating block one-hot are discarded (error detection,
  no extra circuits). Raw P(feasible) is always reported alongside.
* Greedy repair of infeasible shots is reported in its own column as classical
  post-processing.

## If something goes wrong

* *Pilot billed seconds are `None`*: the planner falls back to a pessimistic 0.6 ms/shot
  + 4 s overhead model, which only lowers the shot count.
* *Queue is long*: nothing is lost; `job.result()` waits. Do not resubmit, check the
  job id in the IBM Quantum dashboard first.
* *Kernel died after submission*: retrieve the result with
  `QiskitRuntimeService().job("<job_id>").result()` and run the save cell.
* *Budget assertion fails*: lower `SHOTS_REQUESTED` or raise `BUDGET_SECONDS`; never
  bypass the assertion.
