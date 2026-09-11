"""
MILP baseline: solves the *original* constrained formulation (one-hot
per request) natively via linear equality constraints, reusing the
same linear objective coefficients as the QUBO diagonal. This gives a
constraint-native exact baseline distinct from `solve_exact`'s pure
brute force (Section 22/23 of the design notes).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from qopt.optimization.qubo import QUBOProblem, evaluate_bitstring
from qopt.optimization.classical_baselines import SolverResult


def _pure_objective_coeffs(problem: QUBOProblem) -> np.ndarray:
    """Same linear objective as the QUBO diagonal, MINUS the one-hot
    penalty contribution (which MILP enforces natively via A@y==1
    instead), but PLUS the soft security-floor penalty, so this stays
    an apples-to-apples comparison against solve_exact/solve_greedy/SA,
    which all evaluate the full penalized QUBO objective on a feasible
    (zero-violation) bitstring -- for any feasible y the one-hot penalty
    evaluates to exactly its offset-cancelling constant, so the two
    objectives coincide exactly on the feasible set.
    """
    def _fixed_scale(vals, ref_max):
        vals = np.asarray(vals, dtype=float)
        return np.clip(vals / ref_max, 0.0, 1.0)

    w = problem.weights
    risk_n = _fixed_scale([o.risk for o in problem.options], ref_max=1.0)
    lat_n = _fixed_scale([o.latency_ms for o in problem.options], ref_max=50.0)
    pool_n = _fixed_scale([o.key_pool_pressure ** 2 for o in problem.options], ref_max=1.0)
    comp_n = _fixed_scale([o.comp_cost for o in problem.options], ref_max=10.0)
    disr_n = _fixed_scale([o.disruption_risk for o in problem.options], ref_max=1.0)
    c = (w["w_R"] * risk_n + w["w_L"] * lat_n + w["w_K"] * pool_n
         + w["w_C"] * comp_n + w["w_D"] * disr_n)

    # Re-derive the security-floor penalty exactly as build_qubo does, so it's
    # included here too (it is a diagonal-only term, so it transfers exactly).
    from qopt.optimization.qubo import SECURITY_SCORES  # noqa: F401  (kept for clarity)
    req_by_id = {}
    for opt in problem.options:
        req_by_id.setdefault(opt.req_id, opt)
    # security penalty magnitude was folded into problem.Q's diagonal already;
    # recover it as (full diagonal) - (pure objective) - (one-hot diagonal term).
    P_onehot = problem.penalty_onehot
    full_diag = np.diag(problem.Q)
    onehot_diag_term = -P_onehot  # every option gets exactly one '-P' one-hot diagonal contribution
    security_term = full_diag - c - onehot_diag_term
    return c + security_term


def solve_milp(problem: QUBOProblem) -> SolverResult:
    t0 = time.perf_counter()
    n = len(problem.options)
    c = _pure_objective_coeffs(problem)

    A_rows = []
    for req_id, idxs in problem.req_option_indices.items():
        row = np.zeros(n)
        row[idxs] = 1
        A_rows.append(row)
    A = np.array(A_rows)
    constraint = LinearConstraint(A, lb=1, ub=1)
    bounds = Bounds(0, 1)
    integrality = np.ones(n)

    result = milp(c=c, constraints=[constraint], bounds=bounds, integrality=integrality)
    runtime = time.perf_counter() - t0
    y = np.round(result.x).astype(int) if result.success else np.zeros(n, dtype=int)
    res = evaluate_bitstring(problem, y)
    return SolverResult("MILP", y, res["qubo_value"], res["constraint_violations"], runtime)


if __name__ == "__main__":
    from qopt.network.topology import get_topology
    from qopt.network.traffic import sample_requests
    from qopt.optimization.qubo import build_options, build_qubo
    from qopt.optimization.classical_baselines import solve_exact

    G = get_topology("B", seed=0)
    reqs = sample_requests(list(G.nodes()), n_requests=4, seed=2)
    link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
    options = build_options(G, reqs, link_state, k_paths=2)
    problem = build_qubo(options, reqs)

    r_milp = solve_milp(problem)
    r_exact = solve_exact(problem, max_n=30)
    print(f"MILP  obj={r_milp.objective:.4f} violations={r_milp.constraint_violations} runtime={r_milp.runtime_s*1e3:.2f} ms")
    print(f"Exact obj={r_exact.objective:.4f} violations={r_exact.constraint_violations} runtime={r_exact.runtime_s*1e3:.2f} ms")
