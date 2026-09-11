"""
Classical baselines for Q-OPT: exact (brute force, small n only), greedy,
and simulated annealing over the same QUBO problem representation, so
every solver is compared on an identical objective (Experiment E4/E5).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from qopt.optimization.qubo import QUBOProblem, evaluate_bitstring


@dataclass
class SolverResult:
    solver: str
    y: np.ndarray
    objective: float
    constraint_violations: int
    runtime_s: float


def solve_exact(problem: QUBOProblem, max_n: int = 20) -> SolverResult:
    """Brute force over one-hot-respecting assignments only (exact, feasible-only search)."""
    t0 = time.perf_counter()
    n = len(problem.options)
    if n > max_n:
        raise ValueError(f"n={n} too large for brute force (limit {max_n})")
    req_ids = list(problem.req_option_indices.keys())
    best_y, best_val = None, np.inf

    def rec(k, y):
        nonlocal best_y, best_val
        if k == len(req_ids):
            res = evaluate_bitstring(problem, y)
            if res["qubo_value"] < best_val:
                best_val, best_y = res["qubo_value"], y.copy()
            return
        for idx in problem.req_option_indices[req_ids[k]]:
            y[idx] = 1
            rec(k + 1, y)
            y[idx] = 0

    rec(0, np.zeros(n))
    runtime = time.perf_counter() - t0
    res = evaluate_bitstring(problem, best_y)
    return SolverResult("Exact", best_y, res["qubo_value"], res["constraint_violations"], runtime)


def solve_greedy(problem: QUBOProblem) -> SolverResult:
    """Per-request greedy: independently pick the option with lowest diagonal cost."""
    t0 = time.perf_counter()
    n = len(problem.options)
    y = np.zeros(n)
    diag = np.diag(problem.Q)
    for req_id, idxs in problem.req_option_indices.items():
        best_idx = min(idxs, key=lambda i: diag[i])
        y[best_idx] = 1
    runtime = time.perf_counter() - t0
    res = evaluate_bitstring(problem, y)
    return SolverResult("Greedy", y, res["qubo_value"], res["constraint_violations"], runtime)


def solve_simulated_annealing(problem: QUBOProblem, n_sweeps: int = 2000, seed: int = 0,
                               t0_temp: float = 5.0, t_min: float = 1e-3) -> SolverResult:
    """
    Simulated annealing directly on the one-hot-structured decision space:
    moves are "reassign request r's chosen option" (structured neighborhood),
    which respects feasibility by construction and lets SA be compared
    fairly against QAOA/MILP on solution *quality*, not on how well it
    discovers the one-hot constraint.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    req_ids = list(problem.req_option_indices.keys())
    n = len(problem.options)

    y = np.zeros(n)
    for req_id in req_ids:
        idx = rng.choice(problem.req_option_indices[req_id])
        y[idx] = 1

    def energy(y):
        return float(y @ problem.Q @ y + problem.offset)

    cur_e = energy(y)
    best_y, best_e = y.copy(), cur_e
    cooling = (t_min / t0_temp) ** (1.0 / n_sweeps)
    T = t0_temp

    for step in range(n_sweeps):
        req_id = req_ids[rng.integers(len(req_ids))]
        idxs = problem.req_option_indices[req_id]
        cur_choice = [i for i in idxs if y[i] == 1][0]
        new_choice = idxs[rng.integers(len(idxs))]
        if new_choice == cur_choice:
            T *= cooling
            continue
        y_new = y.copy()
        y_new[cur_choice] = 0
        y_new[new_choice] = 1
        new_e = energy(y_new)
        delta = new_e - cur_e
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            y, cur_e = y_new, new_e
            if cur_e < best_e:
                best_y, best_e = y.copy(), cur_e
        T *= cooling

    runtime = time.perf_counter() - t0
    res = evaluate_bitstring(problem, best_y)
    return SolverResult("SimulatedAnnealing", best_y, res["qubo_value"], res["constraint_violations"], runtime)


if __name__ == "__main__":
    from qopt.network.topology import get_topology
    from qopt.network.traffic import sample_requests
    from qopt.optimization.qubo import build_options, build_qubo

    G = get_topology("A", seed=0)
    reqs = sample_requests(list(G.nodes()), n_requests=3, seed=1)
    link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
    options = build_options(G, reqs, link_state, k_paths=2)
    problem = build_qubo(options, reqs)

    for solver_fn in [solve_exact, solve_greedy, solve_simulated_annealing]:
        r = solver_fn(problem)
        print(f"{r.solver:20s} obj={r.objective:8.4f}  violations={r.constraint_violations}  "
              f"runtime={r.runtime_s*1000:7.2f} ms")
