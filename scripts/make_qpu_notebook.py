"""Generates notebooks/qpu_run.ipynb (kept as code so the notebook is reviewable in diffs)."""
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Q-OPT v2: IBM Quantum hardware run (budget-guarded)

This notebook executes the pre-registered hardware experiment in
`results/hardware/plan/` (4 instances × {W-state control, XY-QAOA p=1, XY-QAOA p=2,
penalty QAOA p=1} = 16 circuits) as **one** Sampler job, after two short pilot jobs
that measure the real billed QPU time on the chosen backend.

**Budget logic.** Set `BUDGET_SECONDS` to the QPU time you are willing to spend.
The pilots cost a few seconds; the main job's shots per circuit are then chosen
so that the *extrapolated* usage stays below 75 % of the remaining budget.
Nothing is submitted without an explicit `CONFIRM_SUBMIT = True`.

**Rehearsal.** With `USE_FAKE_BACKEND = True` the identical code path runs locally
on `FakeMarrakesh` (Aer, free). Run it that way once before spending QPU time.

All raw counts, job ids, layouts and the calibration snapshot are written to
`results/hardware/run_<backend>_<timestamp>/raw.json` before any analysis.
""")

code(r"""
# ---- user settings --------------------------------------------------------
USE_FAKE_BACKEND = True        # True: free local rehearsal; False: real IBM QPU
BACKEND_NAME     = None        # e.g. "ibm_marrakesh"; None = least busy Heron-class device
BUDGET_SECONDS   = 420         # QPU seconds you allow this notebook to spend (you have ~480)
SHOTS_REQUESTED  = 8192        # per circuit, before budget scaling
CONFIRM_SUBMIT   = False       # must be True to submit the MAIN job to real hardware
""")

code(r"""
# ---- make the repository importable from this kernel ----------------------
# Works even if `pip install -e .` was run in a different environment than the
# one this Jupyter kernel uses (the usual cause of "No module named 'qopt'").
import sys
from pathlib import Path
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "qopt" / "quantum" / "hardware.py").exists()), None)
if ROOT is None:
    raise RuntimeError("Open this notebook from inside the Q-OPT repository (notebooks/ folder).")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import qopt
assert qopt.__version__.startswith("2."), (
    f"Found an old qopt {qopt.__version__} at {qopt.__file__}; run `pip uninstall qopt` and restart the kernel.")
print("kernel python :", sys.executable)
print("qopt          :", qopt.__version__, "from", Path(qopt.__file__).parent)
PLAN_DIR = str(ROOT / "results" / "hardware" / "plan")
RESULTS_DIR = ROOT / "results" / "hardware"
""")

code(r"""
import json, time, warnings, hashlib, platform
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

import qiskit, qiskit_ibm_runtime
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qopt.quantum import hardware as hw
from qopt.quantum.train import bootstrap_metrics, repair_counts, metrics_from_feasible_dist
print("qiskit", qiskit.__version__, "| qiskit-ibm-runtime", qiskit_ibm_runtime.__version__,
      "| python", platform.python_version())
""")

md(r"""
### One-time credentials
Run the cell below **once** with your token and CRN, then set `SAVE_ACCOUNT = False`
and delete the token from the cell. Never commit a notebook that contains a token.
""")

code(r"""
SAVE_ACCOUNT = False
if SAVE_ACCOUNT:
    QiskitRuntimeService.save_account(
        token="my_api_key",
        instance="my_crn_id",
        overwrite=True,
        set_as_default=True,
    )
if not USE_FAKE_BACKEND:
    service = QiskitRuntimeService()
    backends = service.backends()
    print(f"Account OK. {len(backends)} backend(s) available:")
    for b in backends[:8]:
        print(f"  {b.name} ({b.num_qubits} qubits)")
    try:
        print("Plan usage:", service.usage())
    except Exception as exc:
        print("usage() not available on this plan:", exc)
""")

code(r"""
plan, circuits = hw.load_plan(PLAN_DIR)
plan_hash = hashlib.sha256(Path(PLAN_DIR, "plan.json").read_bytes()).hexdigest()[:12]
print(f"plan {plan_hash}: {len(circuits)} circuits, created {plan['created']}")
for pb in plan["pubs"]:
    print(f"  [{pb['index']:2d}] {pb['instance']:7s} {pb['ansatz']:3s} p={pb['p']}  "
          f"ideal r_feas={pb['ideal']['r_feas']:.3f}  p_feas={pb['ideal']['p_feas']:.3f}")
""")

code(r"""
if USE_FAKE_BACKEND:
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    backend = FakeMarrakesh()
else:
    if BACKEND_NAME:
        backend = service.backend(BACKEND_NAME)
    else:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
    st = backend.status()
    print(f"{backend.name}: operational={st.operational} pending_jobs={st.pending_jobs}")
print("Backend:", backend.name, backend.num_qubits, "qubits")

t0 = time.time()
isa, tstats = hw.transpile_best(circuits, backend, n_seeds=16)
print(f"transpiled in {time.time()-t0:.1f}s")
for s in tstats:
    print(f"  {s['name']:18s} 2q={s['twoq']:4d}  2q-depth={s['twoq_depth']:4d}  layout={s['layout']}")
calib = hw.calibration_snapshot(backend, [s["layout"] for s in tstats])
print(json.dumps({k: v for k, v in calib.items() if k != "physical_qubits"}, indent=1))
""")

md(r"""
### Pilot jobs (≈ 4-8 QPU seconds)
Two single-circuit jobs with different shot counts give a linear model of billed
seconds = overhead + rate × shots on *this* backend today.
""")

code(r"""
sampler = Sampler(mode=backend)
applied = hw.configure_sampler(sampler, dd=True, twirl=True)
print("error suppression:", applied)

pilot_idx = int(np.argmax([s["twoq"] for s in tstats]))   # deepest circuit = worst case
pilot_points, pilot_jobs = [], []
for shots in (512, 4096):
    job = sampler.run([isa[pilot_idx]], shots=shots)
    res = job.result()
    qs = hw.quantum_seconds(job)
    pilot_points.append((shots, qs))
    pilot_jobs.append(dict(job_id=job.job_id(), shots=shots, quantum_seconds=qs))
    print(f"pilot {shots:5d} shots -> job {job.job_id()}  billed={qs}")
per_shot, overhead, how = hw.fit_usage(pilot_points)
spent = sum(q for _, q in pilot_points if q) if not USE_FAKE_BACKEND else 0.0
remaining = BUDGET_SECONDS - spent
shots = hw.plan_shots(per_shot, overhead, len(isa), SHOTS_REQUESTED, remaining)
pred = overhead + per_shot * shots * len(isa)
print(f"usage model ({how}): overhead={overhead:.2f}s  rate={per_shot*1e3:.3f} ms/shot")
print(f"pilots spent {spent:.1f}s; remaining budget {remaining:.1f}s")
print(f"main job: {len(isa)} circuits x {shots} shots  ->  predicted {pred:.1f}s")
assert shots >= 1024, "Budget too small for a meaningful main job; stop here."
""")

code(r"""
if not USE_FAKE_BACKEND and not CONFIRM_SUBMIT:
    raise SystemExit("Review the prediction above, then set CONFIRM_SUBMIT = True and re-run this cell.")
t_submit = time.time()
job = sampler.run(isa, shots=shots)
print("main job submitted:", job.job_id())
result = job.result()
qs_main = hw.quantum_seconds(job)
print(f"done in {time.time()-t_submit:.0f}s wall; billed QPU seconds = {qs_main}")
""")

code(r"""
counts = [hw.counts_from_pub_result(r) for r in result]
stamp = time.strftime("%Y%m%d-%H%M%S")
tag = "fake" if USE_FAKE_BACKEND else "qpu"
run_dir = RESULTS_DIR / f"run_{tag}_{backend.name}_{stamp}"
run_dir.mkdir(parents=True, exist_ok=True)
raw = dict(plan_hash=plan_hash, backend=backend.name, fake_backend=USE_FAKE_BACKEND,
           submitted=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_submit)),
           job_id=job.job_id(), shots_per_circuit=shots, billed_quantum_seconds=qs_main,
           pilot_jobs=pilot_jobs, usage_model=dict(per_shot_s=per_shot, overhead_s=overhead, method=how),
           error_suppression=applied, calibration=calib, transpile=tstats,
           versions=dict(qiskit=qiskit.__version__, runtime=qiskit_ibm_runtime.__version__),
           pubs=[dict(**pb, counts=c) for pb, c in zip(plan["pubs"], counts)])
(run_dir / "raw.json").write_text(json.dumps(raw, indent=1))
print("saved", run_dir / "raw.json")
""")

md("### Quick look (full analysis: `python experiments/exp07_hardware_analysis.py --run-dir <run_dir>`)")

code(r"""
from qopt.quantum.hardware import problem_from_plan
inst = {i["name"]: i for i in plan["instances"]}
print(f"{'circuit':18s} {'p_feas':>14s} {'r_feas (post-sel.)':>22s} {'ideal':>7s} {'random':>7s}")
for pb, c in zip(plan["pubs"], counts):
    I = inst[pb["instance"]]; prob = problem_from_plan(I); E = np.array(I["spectrum"])
    b = bootstrap_metrics(prob, c, E, n_boot=300)
    pt, ci = b["point"], b["ci"]
    print(f"{pb['instance']+'|'+pb['ansatz']+str(pb['p']):18s} "
          f"{pt['p_feas']:.3f} [{ci['p_feas'][0]:.3f},{ci['p_feas'][1]:.3f}] "
          f"{pt['r_feas']:.3f} [{ci['r_feas'][0]:.3f},{ci['r_feas'][1]:.3f}]   "
          f"{pb['ideal']['r_feas']:.3f}  {pt['r_rand']:.3f}")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/qpu_run.ipynb")
print("wrote notebooks/qpu_run.ipynb")
