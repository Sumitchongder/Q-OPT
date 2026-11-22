"""
Hardware experiment plan and budget-guarded execution on IBM Quantum.

Workflow
--------
1. ``build_plan`` (free, local): selects the hardware instance set with
   pre-registered criteria, trains angles on the exact simulator, and
   writes ``plan.json`` (instances, spectra, angles) + ``circuits.qpy``.
2. ``execute_plan`` (spends QPU time, called from notebooks/qpu_run.ipynb):
   transpiles every circuit with several layout seeds and keeps the one
   with the fewest two-qubit gates, runs a short *pilot* job to measure
   real QPU usage on this backend, extrapolates, scales shots down if the
   main job would exceed the remaining budget, and only then submits the
   main job. Every raw count, job id, layout and calibration snapshot is
   written to disk before any analysis is done.

Instance selection (pre-registered; disclose verbatim in the paper)
--------------------------------------------------------------------
Topology C, k_paths=1 (three modes per request), load=1.5, seeds scanned
upward from 0; an instance is accepted iff
  (a) it has at least one inter-request coupling term (non-separable), and
  (b) it has at most 3R inter-request coupling terms (hardware sparsity:
      denser instances need >200 two-qubit gates on heavy-hex at p=1).
The first two accepted seeds for R=3 and for R=4 form the set. Whether
coupled greedy finds the optimum is recorded per instance, not filtered
on (at these sizes greedy is optimal on most instances; see exp01).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from qopt.agility.model import random_instance
from qopt.agility.solvers import solve_exact, solve_greedy

PLAN_DEFAULT = dict(topology="C", k_paths=1, load=1.5, sizes=(3, 4), per_size=2,
                    p_xy=(1, 2), p_pen=(1,), penalty_factor=0.5, seed_scan=200)


def select_instances(topology="C", k_paths=1, load=1.5, sizes=(3, 4), per_size=2, seed_scan=200,
                     max_inter_per_request=3):
    chosen = []
    for R in sizes:
        got = 0
        for seed in range(seed_scan):
            prob = random_instance(topology, n_requests=R, k_paths=k_paths, seed=seed, load=load)
            Q, _ = prob.qubo_feasible_compressed()
            n_inter = int((np.abs(np.triu(Q, 1)) > 1e-12).sum())
            if n_inter == 0 or n_inter > max_inter_per_request * R:
                continue
            ex, gr = solve_exact(prob), solve_greedy(prob)
            chosen.append(dict(name=f"R{R}_s{seed}", R=R, seed=seed, n_inter=n_inter, prob=prob,
                               exact=ex, greedy=gr))
            got += 1
            if got == per_size:
                break
    return chosen


def build_plan(out_dir: str | Path, cfg: dict | None = None, verbose: bool = True) -> dict:
    from qiskit import qpy

    from qopt.quantum.circuits import bind, penalty_qaoa_circuit, w_state_circuit, xy_qaoa_circuit
    from qopt.quantum.fastsim import PenaltyQAOASim, XYQAOASim
    from qopt.quantum.train import metrics_from_feasible_dist, train_depth_sweep

    cfg = {**PLAN_DEFAULT, **(cfg or {})}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inst = select_instances(cfg["topology"], cfg["k_paths"], cfg["load"], cfg["sizes"],
                            cfg["per_size"], cfg["seed_scan"])
    plan = dict(config={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), instances=[], pubs=[])
    circuits = []
    for it in inst:
        prob = it["prob"]
        xy = XYQAOASim(prob)
        E = xy.E
        span = float(E.max() - E.min())
        tr = train_depth_sweep(xy.expectation, max(cfg["p_xy"]), seed=it["seed"])
        pen_P = cfg["penalty_factor"] * span
        pen = PenaltyQAOASim(prob, penalty=pen_P, scale=xy.scale)
        trp = train_depth_sweep(pen.expectation, max(cfg["p_pen"]), seed=it["seed"])
        plan["instances"].append(dict(
            name=it["name"], R=it["R"], seed=it["seed"], n_qubits=prob.n, n_feasible=prob.n_feasible,
            n_inter_couplings=it["n_inter"], scale=xy.scale, penalty=pen_P,
            exact_choice=it["exact"].choice, exact_energy=it["exact"].energy,
            greedy_energy=it["greedy"].energy, spectrum=E.tolist(),
            blocks=prob.blocks, c=prob.c.tolist(), W=prob.W.tolist(),
            modes=[o.mode for o in prob.options],
            classes=[r.service_class for r in prob.requests]))
        specs = [("W", 0, None, w_state_circuit(prob))]
        for p in cfg["p_xy"]:
            specs.append(("XY", p, tr[p][0], bind(xy_qaoa_circuit(prob, p, xy.scale), tr[p][0])))
        for p in cfg["p_pen"]:
            specs.append(("PEN", p, trp[p][0], bind(penalty_qaoa_circuit(prob, p, xy.scale, pen_P), trp[p][0])))
        for ans, p, params, qc in specs:
            if ans == "W":
                pr, pf = np.full(E.size, 1.0 / E.size), 1.0
            elif ans == "XY":
                pr, pf = xy.probs(params), 1.0
            else:
                pf, pr = pen.feasible_distribution(params)
            ideal = metrics_from_feasible_dist(E, pr, pf)
            qc.name = f"{it['name']}|{ans}|p{p}"
            plan["pubs"].append(dict(index=len(circuits), instance=it["name"], ansatz=ans, p=p,
                                     params=None if params is None else list(map(float, params)),
                                     ideal=ideal))
            circuits.append(qc)
        if verbose:
            print(f"[plan] {it['name']}: n={prob.n} D={prob.n_feasible} inter-ZZ={it['n_inter']} "
                  f"greedy gap={(it['greedy'].energy - it['exact'].energy) / span:.3f} span")
    with open(out / "circuits.qpy", "wb") as f:
        qpy.dump(circuits, f)
    with open(out / "plan.json", "w") as f:
        json.dump(plan, f, indent=1)
    return plan


# --------------------------------------------------------------------------- #
# Execution helpers (used by the notebook)
# --------------------------------------------------------------------------- #
def load_plan(plan_dir: str | Path):
    from qiskit import qpy
    plan_dir = Path(plan_dir)
    with open(plan_dir / "plan.json") as f:
        plan = json.load(f)
    with open(plan_dir / "circuits.qpy", "rb") as f:
        circuits = qpy.load(f)
    return plan, circuits


def transpile_best(circuits, backend, n_seeds: int = 16, optimization_level: int = 3):
    """For each circuit keep the transpilation (over layout seeds) with fewest 2q gates."""
    from qiskit.transpiler import generate_preset_pass_manager
    twoq = {"cz", "ecr", "cx", "rzz"}
    out, stats = [], []
    for qc in circuits:
        best = None
        for s in range(n_seeds):
            pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                              seed_transpiler=s)
            t = pm.run(qc)
            n2 = sum(v for k, v in t.count_ops().items() if k in twoq)
            d2 = t.depth(lambda g: g.operation.num_qubits == 2)
            if best is None or (n2, d2) < (best[1], best[2]):
                best = (t, n2, d2, s)
        t, n2, d2, s = best
        layout = t.layout.final_index_layout() if t.layout is not None else list(range(qc.num_qubits))
        out.append(t)
        stats.append(dict(name=qc.name, n_qubits=qc.num_qubits, twoq=int(n2), twoq_depth=int(d2),
                          depth=int(t.depth()), seed_transpiler=int(s), layout=list(map(int, layout))))
    return out, stats


def calibration_snapshot(backend, layouts) -> dict:
    """Median gate/readout errors on the physical qubits actually used."""
    snap = dict(backend=backend.name, retrieved=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        tgt = backend.target
        used = sorted({q for lay in layouts for q in lay})
        ro = [tgt["measure"][(q,)].error for q in used if tgt["measure"].get((q,)) is not None]
        twoq_name = next((g for g in ("cz", "ecr", "cx") if g in tgt.operation_names), None)
        e2 = []
        if twoq_name:
            for qargs, props in tgt[twoq_name].items():
                if props is not None and props.error is not None and set(qargs) <= set(used):
                    e2.append(props.error)
        t1 = [backend.qubit_properties(q).t1 for q in used]
        t2 = [backend.qubit_properties(q).t2 for q in used]
        snap.update(physical_qubits=used, median_readout_error=float(np.median(ro)) if ro else None,
                    median_2q_error=float(np.median(e2)) if e2 else None, twoq_gate=twoq_name,
                    median_t1_us=float(np.median([x for x in t1 if x]) * 1e6) if any(t1) else None,
                    median_t2_us=float(np.median([x for x in t2 if x]) * 1e6) if any(t2) else None)
    except Exception as exc:  # calibration data are informative, never blocking
        snap["error"] = repr(exc)
    return snap


def configure_sampler(sampler, dd: bool = True, twirl: bool = True) -> dict:
    """Error suppression: dynamical decoupling + Pauli gate/measurement twirling."""
    applied = {}
    try:
        if dd:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XpXm"
            applied["dynamical_decoupling"] = "XpXm"
    except Exception as exc:
        applied["dynamical_decoupling_error"] = repr(exc)
    try:
        if twirl:
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
            sampler.options.twirling.num_randomizations = "auto"
            sampler.options.twirling.shots_per_randomization = "auto"
            applied["twirling"] = "gates+measure, auto"
    except Exception as exc:
        applied["twirling_error"] = repr(exc)
    return applied


def counts_from_pub_result(pub_result) -> dict:
    data = pub_result.data
    reg = next(iter(data.keys())) if hasattr(data, "keys") else "meas"
    return getattr(data, reg).get_counts()


def quantum_seconds(job) -> float | None:
    """Billed QPU seconds of a finished runtime job (None if unavailable)."""
    for getter in (lambda j: j.usage(), lambda j: j.metrics()["usage"]["quantum_seconds"],
                   lambda j: j.metrics()["usage"]["seconds"]):
        try:
            v = getter(job)
            if v is not None:
                return float(v)
        except Exception:
            continue
    return None


def fit_usage(points, fallback_per_shot_s: float = 6e-4, fallback_overhead_s: float = 4.0):
    """
    Linear model  seconds = overhead + per_shot * total_shots  from pilot
    jobs [(total_shots, billed_seconds), ...]. Billed seconds are coarse
    (often whole seconds), so the fit is clipped to conservative bounds and
    falls back to pessimistic Heron figures when it is ill-conditioned.
    """
    pts = [(s, t) for s, t in points if t is not None]
    if len(pts) >= 2 and pts[-1][0] != pts[0][0]:
        (s1, t1), (s2, t2) = pts[0], pts[-1]
        slope = (t2 - t1) / (s2 - s1)
        if slope > 0:
            per_shot = max(slope, 1e-4)
            overhead = max(t1 - per_shot * s1, 1.0)
            return per_shot, overhead, "fit"
    if pts:   # single point: attribute everything above a 2 s floor to shots, pessimistically
        s, t = max(pts)
        return max((t - 2.0) / s, fallback_per_shot_s), max(2.0, fallback_overhead_s), "single-point"
    return fallback_per_shot_s, fallback_overhead_s, "fallback"


def plan_shots(per_shot_s: float, overhead_s: float, n_pubs: int, shots_req: int,
               budget_s: float, margin: float = 0.75) -> int:
    """Largest shots/PUB (<= shots_req, multiple of 256) predicted to fit margin*budget."""
    allowed = (margin * budget_s - overhead_s) / max(per_shot_s * n_pubs, 1e-12)
    return max(int(min(shots_req, allowed) // 256 * 256), 0)


def problem_from_plan(inst: dict):
    """Rebuild the scoring view (blocks, c, W) of a plan instance without the network objects."""
    from types import SimpleNamespace

    from qopt.agility.model import AgilityProblem
    opts = [SimpleNamespace(mode=m) for m in inst["modes"]]
    return AgilityProblem(requests=[], options=opts, blocks=[list(b) for b in inst["blocks"]],
                          c=np.array(inst["c"]), W=np.array(inst["W"]), links={}, params=None,
                          weights={})
