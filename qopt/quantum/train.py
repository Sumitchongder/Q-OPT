"""
Angle training on the exact simulators, plus the evaluation metrics used
for simulation *and* hardware data, so both are scored identically.

Training: p=1 is initialised from a coarse (gamma, beta) grid; depth p+1
is initialised by INTERP (Zhou et al., PRX 10, 021067 (2020)) from the
depth-p optimum, plus a few perturbed restarts; all starts are refined with
L-BFGS-B on the exact expectation value.

Metrics (minimisation, feasible energies E in [E_min, E_max])
  p_feas      probability mass on block-one-hot bitstrings
  r_feas      normalised approximation ratio of the feasible-conditioned
              mean:  (E_max - <E>_feas) / (E_max - E_min); 1 = optimal,
              and the uniform-feasible value r_rand is the natural floor
  r_eff       p_feas * r_feas (infeasible samples scored as the worst
              feasible assignment, the most conservative convention)
  p_opt       probability of sampling a global optimum
  pctl        mean percentile: fraction of feasible assignments that are
              strictly worse than the sampled one, averaged over samples
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _refine(fun, x0, bounds):
    res = minimize(fun, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400})
    return res.x, float(res.fun)


def interp_init(x_prev: np.ndarray) -> np.ndarray:
    p = len(x_prev) // 2
    def _interp(v):
        out = np.zeros(p + 1)
        for i in range(p + 1):
            a = v[i - 1] if i - 1 >= 0 else 0.0
            b = v[i] if i < p else 0.0
            out[i] = (i / p) * a + ((p - i) / p) * b
        return out
    return np.concatenate([_interp(x_prev[:p]), _interp(x_prev[p:])])


def train_depth_sweep(expect_fn, p_max: int, seed: int = 0, n_grid: int = 24,
                      n_restarts: int = 3, gamma_max: float = 2.0 * np.pi) -> dict:
    """Returns {p: (params, energy)} for p = 1..p_max."""
    rng = np.random.default_rng(seed)
    out = {}
    gs = np.linspace(-gamma_max / 2, gamma_max / 2, n_grid)
    bs = np.linspace(-np.pi / 2, np.pi / 2, n_grid)
    grid = [(expect_fn(np.array([g, b])), g, b) for g in gs for b in bs]
    grid.sort(key=lambda t: t[0])
    best_x, best_f = None, np.inf
    for _, g, b in grid[:4]:
        x, f = _refine(expect_fn, np.array([g, b]), [(-gamma_max, gamma_max), (-np.pi, np.pi)])
        if f < best_f:
            best_x, best_f = x, f
    out[1] = (best_x, best_f)
    for p in range(2, p_max + 1):
        bounds = [(-gamma_max, gamma_max)] * p + [(-np.pi, np.pi)] * p
        starts = [interp_init(out[p - 1][0])]
        for _ in range(n_restarts):
            starts.append(starts[0] + rng.normal(0, 0.15, size=2 * p))
        best_x, best_f = None, np.inf
        for x0 in starts:
            x, f = _refine(expect_fn, x0, bounds)
            if f < best_f:
                best_x, best_f = x, f
        if best_f > out[p - 1][1]:          # depth p can always reproduce depth p-1
            x = np.concatenate([out[p - 1][0][: p - 1], [0.0], out[p - 1][0][p - 1:], [0.0]])
            best_x, best_f = x, out[p - 1][1]
        out[p] = (best_x, best_f)
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics_from_feasible_dist(E: np.ndarray, dist_feas: np.ndarray, p_feas: float) -> dict:
    """
    E: feasible energies (mixed-radix order); dist_feas: probability mass on
    each feasible point (sums to p_feas).
    """
    Emin, Emax = float(E.min()), float(E.max())
    span = max(Emax - Emin, 1e-12)
    order = np.sort(E)
    worse = 1.0 - np.searchsorted(order, E, side="right") / E.size
    out = dict(p_feas=float(p_feas))
    if p_feas > 0:
        cond = dist_feas / p_feas
        mean_e = float(cond @ E)
        out["r_feas"] = (Emax - mean_e) / span
        out["pctl"] = float(cond @ worse)
    else:
        out["r_feas"] = np.nan
        out["pctl"] = np.nan
    out["r_eff"] = out["p_feas"] * (out["r_feas"] if p_feas > 0 else 0.0)
    out["p_opt"] = float(dist_feas[np.isclose(E, Emin, rtol=0, atol=1e-9 * span)].sum())
    out["r_rand"] = (Emax - float(E.mean())) / span
    return out


def counts_to_feasible(prob, counts: dict):
    """Hardware/sampler counts -> (n_shots, n_feasible_shots, per-feasible-index counts)."""
    sizes = np.array(prob.block_sizes)
    radix = np.concatenate([[1], np.cumprod(sizes[:-1])])
    vec = np.zeros(prob.n_feasible)
    total, feas = 0, 0
    n = prob.n
    for bs, c in counts.items():
        bs = bs.replace(" ", "")[-n:].zfill(n)
        y = [int(bs[-1 - i]) for i in range(n)]
        total += c
        ch = prob.y_to_choice(y)
        if ch is None:
            continue
        feas += c
        vec[int(np.dot(ch, radix))] += c
    return total, feas, vec


def repair_counts(prob, counts: dict, E: np.ndarray):
    """
    Classical post-processing (reported separately, never mixed with raw
    numbers): each infeasible block is repaired to the option of minimum
    marginal coupled energy given the other blocks' measured bits.
    """
    sizes = np.array(prob.block_sizes)
    radix = np.concatenate([[1], np.cumprod(sizes[:-1])])
    vec = np.zeros(prob.n_feasible)
    n = prob.n
    for bs, c in counts.items():
        bs = bs.replace(" ", "")[-n:].zfill(n)
        y = np.array([int(bs[-1 - i]) for i in range(n)], dtype=float)
        fixed = [r for r, b in enumerate(prob.blocks) if y[b].sum() == 1]
        bad = [r for r in range(len(prob.blocks)) if r not in fixed]
        for r in bad:
            y[prob.blocks[r]] = 0.0
        for r in bad:
            best, bk = np.inf, 0
            for k, i in enumerate(prob.blocks[r]):
                y[i] = 1.0
                e = prob.energy_y(y)
                y[i] = 0.0
                if e < best:
                    best, bk = e, k
            y[prob.blocks[r][bk]] = 1.0
        ch = prob.y_to_choice(y)
        vec[int(np.dot(ch, radix))] += c
    return vec


def bootstrap_metrics(prob, counts: dict, E: np.ndarray, n_boot: int = 1000, seed: int = 0,
                      post_select: bool = True) -> dict:
    """Point estimates + 95% bootstrap CIs (multinomial resampling of shots)."""
    keys = list(counts)
    vals = np.array([counts[k] for k in keys], dtype=float)
    shots = int(vals.sum())
    rng = np.random.default_rng(seed)

    def _m(cvec):
        cd = {k: int(v) for k, v in zip(keys, cvec) if v > 0}
        tot, feas, vec = counts_to_feasible(prob, cd)
        pf = feas / tot
        return metrics_from_feasible_dist(E, vec / tot, pf)

    point = _m(vals)
    boots = [_m(rng.multinomial(shots, vals / shots)) for _ in range(n_boot)]
    ci = {}
    for k in point:
        arr = np.array([b[k] for b in boots], dtype=float)
        arr = arr[np.isfinite(arr)]
        ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if arr.size else (np.nan, np.nan)
    return dict(point=point, ci=ci, shots=shots)
