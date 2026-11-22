"""Tests for the v2 coupled model, solvers, simulators, circuits and hardware helpers."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qopt.agility import solvers as S
from qopt.agility.model import random_instance
from qopt.agility.twin import TwinConfig, run_twin
from qopt.qkd.channel import ChannelParams, key_generation_rate_bps
from qopt.quantum.circuits import bind, ising_energy_bits, penalty_qaoa_circuit, qubo_to_ising, w_state_circuit, xy_qaoa_circuit
from qopt.quantum.fastsim import PenaltyQAOASim, XYQAOASim
from qopt.quantum.hardware import fit_usage, plan_shots, problem_from_plan
from qopt.quantum.train import counts_to_feasible, interp_init, metrics_from_feasible_dist, repair_counts


@pytest.fixture(scope="module")
def small():
    return random_instance("C", n_requests=3, k_paths=1, seed=3, load=1.5)


# ---------------------------------------------------------------- physics
def test_lmc05_rate_decreases_with_distance_and_aborts_under_attack():
    p = ChannelParams()
    rates = [key_generation_rate_bps(L, p)[0] for L in (5, 20, 50, 100)]
    assert all(a > b > 0 for a, b in zip(rates, rates[1:]))
    assert key_generation_rate_bps(30, p, qber_shift=0.08)[0] == 0.0


# ---------------------------------------------------------------- model
def test_energy_views_agree_on_feasible_set(small):
    Qp, offp = small.qubo_penalized(7.0)
    Qc, _ = small.qubo_feasible_compressed()
    h, J, C = qubo_to_ising(Qc)
    for ch, e in S.enumerate_feasible(small):
        for c, ee in zip(ch[:20], e[:20]):
            y = small.choice_to_y(c)
            assert np.isclose(small.energy_y(y), ee)
            assert np.isclose(y @ Qp @ y + offp, ee)
            assert np.isclose(y @ Qc @ y, ee)
            assert np.isclose(ising_energy_bits(h, J, C, y), ee)


def test_penalty_ground_state_is_feasible(small):
    Qp, off = small.qubo_penalized(10.0 * np.ptp(S.feasible_spectrum(small)))
    n = small.n
    best = min(range(2 ** n), key=lambda b: (lambda y: y @ Qp @ y)(np.array([(b >> i) & 1 for i in range(n)], float)))
    assert small.y_to_choice([(best >> i) & 1 for i in range(n)]) is not None


def test_coupled_model_is_not_separable():
    p = random_instance("C", n_requests=5, k_paths=2, seed=11, load=2.0)
    blk = np.concatenate([[r] * len(b) for r, b in enumerate(p.blocks)])
    iu = np.triu_indices(p.n, 1)
    assert np.any((np.abs(p.W[iu]) > 0) & (blk[iu[0]] != blk[iu[1]]))


def test_hybrid_risk_is_product_of_components(small):
    rho_p = small.params.rho_pqc
    for blk in small.blocks:
        opts = {small.options[i].mode: small.options[i] for i in blk}
        assert np.isclose(opts["HYB"].risk, opts["QKD"].risk * rho_p)
        assert np.isclose(opts["PQC"].risk, rho_p)


# ---------------------------------------------------------------- solvers
def test_exact_is_lower_bound_and_sa_finds_it():
    for seed in range(5):
        p = random_instance("C", n_requests=5, k_paths=2, seed=seed, load=1.5)
        ex = S.solve_exact(p)
        assert np.isclose(ex.energy, S.feasible_spectrum(p).min())
        for fn in S.ALL_HEURISTICS.values():
            assert fn(p).energy >= ex.energy - 1e-9
        assert np.isclose(S.solve_simulated_annealing(p).energy, ex.energy)


def test_incremental_deltas_match_brute_force(small):
    ch = [1, 0, 2]
    lf = S._LocalField(small, ch)
    for r in range(3):
        for k, d in enumerate(lf.deltas(r)):
            c2 = list(ch)
            c2[r] = k
            assert np.isclose(small.energy_choice(c2) - small.energy_choice(ch), d)


# ---------------------------------------------------------------- simulators vs Qiskit
@pytest.mark.parametrize("R,k", [(2, 1), (3, 1), (2, 2)])
def test_xy_fastsim_matches_qiskit_and_preserves_symmetry(R, k):
    p = random_instance("C", n_requests=R, k_paths=k, seed=5, load=1.5)
    sim = XYQAOASim(p)
    x = np.array([0.6, -0.2, 0.3, 0.9])
    sv = Statevector(bind(xy_qaoa_circuit(p, 2, sim.scale, measure=False), x)).probabilities()
    sizes = np.array(p.block_sizes)
    radix = np.concatenate([[1], np.cumprod(sizes[:-1])])
    pf, leak = np.zeros(p.n_feasible), 0.0
    for idx, pr in enumerate(sv):
        ch = p.y_to_choice([(idx >> q) & 1 for q in range(p.n)])
        if ch is None:
            leak += pr
        else:
            pf[int(np.dot(ch, radix))] += pr
    assert leak < 1e-12
    assert np.allclose(pf, sim.probs(x), atol=1e-12)


def test_penalty_fastsim_matches_qiskit(small):
    sim = XYQAOASim(small)
    pen = PenaltyQAOASim(small, penalty=2.0, scale=sim.scale)
    x = np.array([0.4, 1.1, -0.7, 0.2])
    sv = Statevector(bind(penalty_qaoa_circuit(small, 2, sim.scale, 2.0, measure=False), x)).probabilities()
    assert np.allclose(sv, pen.probs(x), atol=1e-12)


def test_w_state_is_uniform_over_feasible(small):
    qc = w_state_circuit(small)
    qc.remove_final_measurements()
    pr = Statevector(qc).probabilities()
    nz = pr[pr > 1e-12]
    assert len(nz) == small.n_feasible and np.allclose(nz, 1.0 / small.n_feasible)


def test_compression_removes_intra_block_terms(small):
    Qc, _ = small.qubo_feasible_compressed()
    for b in small.blocks:
        assert np.allclose(Qc[np.ix_(b, b)] - np.diag(np.diag(Qc[np.ix_(b, b)])), 0.0)


# ---------------------------------------------------------------- metrics & hardware helpers
def test_metrics_limits():
    E = np.array([0.0, 1.0, 2.0, 3.0])
    m = metrics_from_feasible_dist(E, np.array([1.0, 0, 0, 0]), 1.0)
    assert m["r_feas"] == 1.0 and m["p_opt"] == 1.0
    u = metrics_from_feasible_dist(E, np.full(4, 0.25), 1.0)
    assert np.isclose(u["r_feas"], u["r_rand"])


def test_counts_parsing_and_repair(small):
    n = small.n
    y = small.choice_to_y(S.solve_exact(small).choice)
    good = "".join(str(int(y[i])) for i in reversed(range(n)))
    bad = "1" * n
    tot, feas, vec = counts_to_feasible(small, {good: 30, bad: 10})
    assert (tot, feas, vec.sum()) == (40, 30, 30)
    rep = repair_counts(small, {good: 30, bad: 10}, S.feasible_spectrum(small))
    assert rep.sum() == 40


def test_plan_roundtrip_view(small):
    inst = dict(modes=[o.mode for o in small.options], blocks=small.blocks, c=small.c.tolist(), W=small.W.tolist())
    v = problem_from_plan(inst)
    ch = [0, 1, 2]
    assert np.isclose(v.energy_choice(ch), small.energy_choice(ch))


def test_budget_guard_is_conservative():
    per, oh, how = fit_usage([(512, 2.0), (4096, 3.0)])
    assert how == "fit" and per > 0
    assert plan_shots(per, oh, 16, 8192, budget_s=10.0) * 16 * per + oh <= 0.75 * 10.0 + 1e-9
    assert fit_usage([])[2] == "fallback"


def test_interp_preserves_length():
    assert len(interp_init(np.array([0.1, 0.2, 0.3, 0.4]))) == 6


# ---------------------------------------------------------------- twin
def test_twin_runs_and_is_deterministic():
    cfg = TwinConfig(n_epochs=4, n_requests=3, k_paths=1, seed=1)
    a = run_twin(lambda p: S.solve_greedy(p).choice, cfg)
    b = run_twin(lambda p: S.solve_greedy(p).choice, cfg)
    assert a == b and 0.0 <= a["compliance"] <= 1.0
