"""
Classical solvers for the block-one-hot crypto-agility problem.

Every solver returns a ``Solution`` holding the local choice per block and
its energy under the *same* coupled objective, so quality comparisons are
apples-to-apples.

  exact            vectorised enumeration of all prod_r k_r feasible points
  separable        option-local optimum, coupling ignored (== Q-OPT v0.1)
  greedy           sequential marginal-cost greedy on the coupled objective
  rule_based       tiered threshold controller (QBER / pool / class), the
                   kind of policy found in deployed adaptive QKD-PQC stacks
  simulated_annealing   block-move SA with geometric schedule, restarts
  tabu             block-move tabu search with aspiration
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from qopt.agility.model import AgilityProblem


@dataclass
class Solution:
    solver: str
    choice: list
    energy: float
    runtime_s: float
    evaluations: int = 0


def _done(name, prob, choice, t0, evals=0):
    return Solution(name, list(map(int, choice)), prob.energy_choice(choice), time.perf_counter() - t0, evals)


def enumerate_feasible(prob: AgilityProblem, chunk: int = 200_000):
    """Yield (choices, energies) chunks over the full feasible set (mixed-radix order)."""
    sizes = np.array(prob.block_sizes)
    total = prob.n_feasible
    radix = np.concatenate([[1], np.cumprod(sizes[:-1])])
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        ch = (idx[:, None] // radix[None, :]) % sizes[None, :]
        yield ch, prob.energies_batch(ch)


def solve_exact(prob: AgilityProblem, max_feasible: int = 5_000_000) -> Solution:
    if prob.n_feasible > max_feasible:
        raise ValueError(f"{prob.n_feasible} feasible points exceeds enumeration limit {max_feasible}")
    t0 = time.perf_counter()
    best_e, best_c = np.inf, None
    for ch, e in enumerate_feasible(prob):
        k = int(np.argmin(e))
        if e[k] < best_e:
            best_e, best_c = float(e[k]), ch[k]
    return _done("Exact", prob, best_c, t0, prob.n_feasible)


def feasible_spectrum(prob: AgilityProblem) -> np.ndarray:
    """All feasible energies in mixed-radix order (small instances only)."""
    return np.concatenate([e for _, e in enumerate_feasible(prob)])


def solve_separable(prob: AgilityProblem) -> Solution:
    """Q-OPT v0.1 behaviour: minimise the option-local cost, blind to shared resources."""
    t0 = time.perf_counter()
    choice = [int(np.argmin(prob.c[b])) for b in prob.blocks]
    return _done("Separable (v0.1)", prob, choice, t0, prob.n)


def solve_greedy(prob: AgilityProblem, order=None) -> Solution:
    """
    Sequential greedy: requests are committed one at a time (critical
    classes first, then by demand) and each takes the option with the lowest
    marginal coupled cost given the requests already committed.
    """
    t0 = time.perf_counter()
    R = len(prob.blocks)
    if order is None:
        rank = {"critical": 0, "high": 1, "normal": 2, "best_effort": 3}
        order = sorted(range(R), key=lambda r: (rank[prob.requests[r].service_class],
                                                 -prob.requests[r].demand_bits))
    committed = []
    choice = [0] * R
    evals = 0
    for r in order:
        best, best_k = np.inf, 0
        for k, i in enumerate(prob.blocks[r]):
            m = prob.c[i] + prob.W[i, i] + 2.0 * sum(prob.W[i, j] for j in committed)
            evals += 1
            if m < best:
                best, best_k = m, k
        choice[r] = best_k
        committed.append(prob.blocks[r][best_k])
    return _done("Greedy (coupled)", prob, choice, t0, evals)


def solve_rule_based(prob: AgilityProblem, qber_ok: float = 0.03, qber_max: float = 0.055,
                     pool_reserve: float = 0.25) -> Solution:
    """
    Tiered threshold controller (no optimisation): for each request look at
    the worst link on its shortest candidate path.
      * worst QBER <= qber_ok and budget headroom  -> QKD (HYB for critical/high)
      * worst QBER <= qber_max and headroom        -> HYB
      * otherwise                                   -> PQC
    Budget headroom is tracked greedily as requests are admitted.
    """
    t0 = time.perf_counter()
    remaining = {e: ls.budget_bits for e, ls in prob.links.items()}
    choice = []
    for r, blk in enumerate(prob.blocks):
        req = prob.requests[r]
        opts = [prob.options[i] for i in blk]
        path0 = opts[0].path
        by_mode = {o.mode: k for k, o in enumerate(opts) if o.path == path0}
        edges = opts[0].edges
        worst_q = max(prob.links[e].qber for e in edges)
        head = all(remaining[e] - req.demand_bits >= pool_reserve * prob.links[e].budget_bits for e in edges)
        if worst_q <= qber_ok and head:
            mode = "HYB" if req.service_class in ("critical", "high") else "QKD"
        elif worst_q <= qber_max and head:
            mode = "HYB"
        else:
            mode = "PQC"
        if mode != "PQC":
            for e in edges:
                remaining[e] -= req.demand_bits
        choice.append(by_mode[mode])
    return _done("Rule-based", prob, choice, t0, prob.n)


def solve_static(prob: AgilityProblem, mode: str) -> Solution:
    t0 = time.perf_counter()
    choice = []
    for blk in prob.blocks:
        opts = [prob.options[i] for i in blk]
        choice.append(next(k for k, o in enumerate(opts) if o.mode == mode and o.path == opts[0].path))
    return _done(f"Static-{mode}", prob, choice, t0, 0)


def _delta_table(prob, choice):  # kept for reference / tests
    """Energy change for moving block r to local option k, for all (r, k)."""
    g = [prob.blocks[r][k] for r, k in enumerate(choice)]
    field = prob.W[:, g].sum(axis=1)            # sum_j W[i, j] over current picks
    out = []
    for r, blk in enumerate(prob.blocks):
        cur = g[r]
        base = prob.c[cur] + 2.0 * (field[cur] - prob.W[cur, cur]) + prob.W[cur, cur]
        cand = prob.c[blk] + 2.0 * (field[blk] - prob.W[blk, cur]) + prob.W[blk, blk]
        out.append(cand - base)
    return out


class _LocalField:
    """Incremental bookkeeping: f_i = sum_r W[i, g_r] over the current picks g."""

    def __init__(self, prob, choice):
        self.p = prob
        self.g = [prob.blocks[r][k] for r, k in enumerate(choice)]
        self.f = prob.W[:, self.g].sum(axis=1)

    def deltas(self, r):
        p, cur = self.p, self.g[r]
        blk = p.blocks[r]
        return (p.c[blk] - p.c[cur] + 2.0 * (self.f[blk] - p.W[blk, cur]) - 2.0 * (self.f[cur] - p.W[cur, cur])
                + p.W[blk, blk] - p.W[cur, cur])

    def move(self, r, k):
        new, cur = self.p.blocks[r][k], self.g[r]
        self.f += self.p.W[:, new] - self.p.W[:, cur]
        self.g[r] = new


def solve_simulated_annealing(prob: AgilityProblem, n_sweeps: int = 400, restarts: int = 4,
                              t_hi: float | None = None, t_lo: float = 1e-3, seed: int = 0) -> Solution:
    """Block-move SA (geometric schedule, random restarts, final steepest descent)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    R = len(prob.blocks)
    sizes = prob.block_sizes
    if t_hi is None:
        t_hi = max(float(np.std(prob.c)) * 2.0, 1e-2)
    best_c, best_e, evals = None, np.inf, 0
    for _ in range(restarts):
        choice = [int(rng.integers(s)) for s in sizes]
        lf = _LocalField(prob, choice)
        e = prob.energy_choice(choice)
        for T in np.geomspace(t_hi, t_lo, n_sweeps):
            for r in rng.permutation(R):
                k = int(rng.integers(sizes[r]))
                if k == choice[r]:
                    continue
                d = float(lf.deltas(r)[k])
                evals += 1
                if d <= 0 or rng.random() < np.exp(-d / T):
                    lf.move(r, k)
                    choice[r] = k
                    e += d
            if e < best_e - 1e-12:
                best_e, best_c = e, list(choice)
        improved = True
        while improved:
            improved = False
            for r in range(R):
                d = lf.deltas(r)
                k = int(np.argmin(d))
                if d[k] < -1e-12:
                    lf.move(r, k)
                    choice[r] = k
                    improved = True
        e = prob.energy_choice(choice)
        if e < best_e:
            best_e, best_c = e, list(choice)
    return _done("Simulated annealing", prob, best_c, t0, evals)


def solve_tabu(prob: AgilityProblem, iters: int = 300, tenure: int | None = None, seed: int = 0) -> Solution:
    """Best-improvement block-move tabu search from the greedy start, with aspiration."""
    t0 = time.perf_counter()
    R = len(prob.blocks)
    tenure = tenure or max(2, R // 3)
    choice = solve_greedy(prob).choice
    lf = _LocalField(prob, choice)
    e = prob.energy_choice(choice)
    best_c, best_e = list(choice), e
    tabu, evals = {}, 0
    for it in range(iters):
        mv, mv_d = None, np.inf
        for r in range(R):
            d = lf.deltas(r)
            for k in range(len(d)):
                if k == choice[r]:
                    continue
                evals += 1
                if (tabu.get((r, k), -1) <= it or e + d[k] < best_e - 1e-12) and d[k] < mv_d:
                    mv, mv_d = (r, k), float(d[k])
        if mv is None:
            break
        r, k = mv
        tabu[(r, choice[r])] = it + tenure
        lf.move(r, k)
        choice[r] = k
        e += mv_d
        if e < best_e - 1e-12:
            best_e, best_c = e, list(choice)
    return _done("Tabu search", prob, best_c, t0, evals)


ALL_HEURISTICS = {
    "Separable (v0.1)": solve_separable,
    "Rule-based": solve_rule_based,
    "Greedy (coupled)": solve_greedy,
    "Simulated annealing": solve_simulated_annealing,
    "Tabu search": solve_tabu,
}
