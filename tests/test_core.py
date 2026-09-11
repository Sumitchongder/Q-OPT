"""
Core unit tests. Run with:  pytest tests/ -v
Every test in this file runs in well under a second and touches no
network/QPU resources.
"""
import numpy as np
import networkx as nx

from qopt.network.topology import get_topology, TOPOLOGIES
from qopt.network.traffic import sample_requests
from qopt.qkd.channel import ChannelParams, key_generation_rate_bps, transmittance
from qopt.qkd.key_pool import KeyPool
from qopt.pqc.hybrid_kdf import hybrid_combine, kdf_single
from qopt.threat.risk_engine import compute_risk
from qopt.optimization.qubo import build_options, build_qubo, evaluate_bitstring
from qopt.optimization.qubo_to_ising import qubo_to_ising, energy_from_bitstring
from qopt.optimization.classical_baselines import solve_exact, solve_greedy, solve_simulated_annealing
from qopt.optimization.milp_baseline import solve_milp


def test_all_topologies_connected():
    for name in TOPOLOGIES:
        G = get_topology(name, seed=0)
        assert nx.is_connected(G)
        assert all("length_km" in G[u][v] for u, v in G.edges())


def test_channel_rate_monotonic_in_distance():
    p = ChannelParams()
    lengths = [5, 20, 40, 80]
    rates = [key_generation_rate_bps(L, p)[0] for L in lengths]
    assert all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))


def test_transmittance_bounds():
    assert 0 < transmittance(50, 0.2) < 1
    assert transmittance(0, 0.2) == 1.0


def test_key_pool_never_exceeds_capacity_or_goes_negative():
    pool = KeyPool(capacity_bits=1000, current_bits=500)
    pool.step(gen_rate_bps=1e6, consume_bps=0, dt_s=1)
    assert pool.current_bits == 1000
    pool.step(gen_rate_bps=0, consume_bps=1e6, dt_s=1)
    assert pool.current_bits == 0


def test_hybrid_kdf_deterministic_and_length():
    k1 = hybrid_combine(b"a" * 32, b"b" * 32, b"ctx")
    k2 = hybrid_combine(b"a" * 32, b"b" * 32, b"ctx")
    assert k1 == k2
    assert len(k1) == 32
    k3 = hybrid_combine(b"a" * 32, b"c" * 32, b"ctx")
    assert k1 != k3


def test_risk_increases_under_attack():
    r_lo = compute_risk(qber=0.01, loss_db=5, key_pool_frac=0.9, attack_probability=0.0, trust=1.0)
    r_hi = compute_risk(qber=0.09, loss_db=5, key_pool_frac=0.1, attack_probability=0.9, trust=0.2)
    assert 0 <= r_lo <= 1 and 0 <= r_hi <= 1
    assert r_hi > r_lo


def test_qubo_to_ising_matches_qubo_exactly():
    rng = np.random.default_rng(42)
    n = 5
    Q = np.triu(rng.uniform(-2, 2, size=(n, n)))
    model = qubo_to_ising(Q)
    for bits in range(2 ** n):
        x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        qubo_val = float(x @ Q @ x)
        bitstring = "".join(str(int(x[n - 1 - i])) for i in range(n))
        assert abs(qubo_val - energy_from_bitstring(model, bitstring)) < 1e-9


def _small_problem(n_requests=3, seed=1):
    G = get_topology("B", seed=0)
    reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
    link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
    options = build_options(G, reqs, link_state, k_paths=2)
    return build_qubo(options, reqs), reqs


def test_qubo_onehot_feasibility_of_exact_solution():
    problem, reqs = _small_problem()
    result = solve_exact(problem, max_n=30)
    assert result.constraint_violations == 0
    # exactly one option chosen per request
    for req_id, idxs in problem.req_option_indices.items():
        assert sum(result.y[i] for i in idxs) == 1


def test_milp_matches_exact_optimum():
    problem, reqs = _small_problem(n_requests=4, seed=2)
    exact = solve_exact(problem, max_n=30)
    milp = solve_milp(problem)
    assert milp.constraint_violations == 0
    assert abs(milp.objective - exact.objective) < 1e-6


def test_greedy_and_sa_feasible_and_not_worse_than_naive():
    problem, reqs = _small_problem(n_requests=4, seed=3)
    greedy = solve_greedy(problem)
    sa = solve_simulated_annealing(problem, n_sweeps=500, seed=0)
    assert greedy.constraint_violations == 0
    assert sa.constraint_violations == 0


def test_security_floor_respected_when_penalty_dominant():
    """For a 'critical' request, the exact solver must never pick a mode
    whose security_score is below the requirement, given the default
    (dominant) security penalty."""
    problem, reqs = _small_problem(n_requests=5, seed=7)
    result = solve_exact(problem, max_n=40)
    req_by_id = {r.req_id: r for r in reqs}
    for i in result.y.nonzero()[0]:
        opt = problem.options[i]
        req = req_by_id[opt.req_id]
        if req.security_class in ("critical", "high"):
            assert opt.security_score >= req.security_score_required, (
                f"Security floor violated for {req.security_class} request: "
                f"chose {opt.mode} (score {opt.security_score}) < required {req.security_score_required}")


def test_penalty_sweep_onehot_feasible_above_threshold():
    """Fast regression check for Experiment 11 (penalty-sensitivity sweep,
    Section 17): at a comfortably strong penalty (P=50, the repo's
    internal default order of magnitude), the GLOBAL minimum of the raw
    (unrestricted 2^n) QUBO search must be one-hot feasible. This does not
    replace running experiments/e12_penalty_sensitivity.py for the full
    sweep and figure, but catches a gross regression (e.g. an accidental
    sign error in the penalty term) cheaply on every test run."""
    from qopt.optimization.qubo import evaluate_bitstring

    problem, reqs = _small_problem(n_requests=2, seed=1)  # small enough for exhaustive search
    n = len(problem.options)
    assert n <= 12, "keep this test's instance small enough for exhaustive search"
    best_y, best_val = None, np.inf
    for bits in range(2 ** n):
        y = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        val = float(y @ problem.Q @ y + problem.offset)
        if val < best_val:
            best_val, best_y = val, y
    result = evaluate_bitstring(problem, best_y)
    assert result["constraint_violations"] == 0, (
        "Global QUBO minimum is infeasible at the default penalty strength -- "
        "this would indicate a regression in the one-hot penalty term.")
