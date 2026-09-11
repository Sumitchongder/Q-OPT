"""
QUBO builder for Q-OPT.

Design choice (documented, not hidden): full edge-indicator flow-
conservation routing (Section 11 of the design notes) blows up the
qubit count far beyond what is laptop/NISQ feasible (item 20: n<=12
qubits for exact QAOA simulation, and only ~8 minutes of real QPU time
on `ibm_marrakesh`). We therefore use the standard reduction used
throughout QKD/optical network optimization literature: **restrict
routing to a small set of K candidate paths per request** (computed
classically via k-shortest-paths on the physical topology), and let
the QUBO choose one (path, crypto-mode) *option* per request. This is
mathematically a one-hot combinatorial selection problem, which is
exactly the QUBO-friendly regime, while still being a genuine
multi-resource joint optimization (risk + latency + key depletion +
compute cost + disruption), not a toy.

    For request r, options are indexed o = (path_idx, mode) where
    mode in {QKD, PQC, HYBRID}.

    y_{r,o} in {0,1},  sum_o y_{r,o} = 1  for every r   (one-hot / mode+route select)

Objective terms (Section 15) are all expressed as linear + quadratic
functions of y and folded into Q via the standard binary penalty
method (Section 16-17).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice

import networkx as nx
import numpy as np

MODES = ["QKD", "PQC", "HYBRID"]


@dataclass
class Option:
    req_id: int
    path: tuple            # tuple of node ids
    mode: str               # one of MODES
    edges: tuple            # tuple of frozenset({u,v}) edges used
    risk: float
    latency_ms: float
    key_pool_pressure: float   # normalized [0,1] contribution to key depletion
    comp_cost: float
    disruption_risk: float     # 1 - availability estimate for this option
    security_score: int        # S_Q / S_P / S_H


@dataclass
class QUBOProblem:
    options: list          # list[Option], flattened across all requests
    req_option_indices: dict  # req_id -> list[index into options]
    Q: np.ndarray
    offset: float
    weights: dict
    penalty_onehot: float


def k_shortest_paths(G: nx.Graph, source: int, target: int, k: int = 2,
                      weight: str = "length_km") -> list[tuple]:
    try:
        gen = nx.shortest_simple_paths(G, source, target, weight=weight)
        paths = list(islice(gen, k))
    except nx.NetworkXNoPath:
        paths = []
    return [tuple(p) for p in paths]


def path_edges(path: tuple) -> tuple:
    return tuple(frozenset((path[i], path[i + 1])) for i in range(len(path) - 1))


SECURITY_SCORES = {"QKD": 5, "PQC": 3, "HYBRID": 5}  # S_Q, S_P, S_H (normalized 1-5 scale)
MODE_COMP_COST = {"QKD": 0.2, "PQC": 1.0, "HYBRID": 1.2}          # relative CPU units
MODE_LATENCY_MS = {"QKD": 0.5, "PQC": 0.05, "HYBRID": 0.55}        # PQC handshake is fast; QKD needs sync


def build_options(G: nx.Graph, requests: list, link_state: dict, k_paths: int = 2) -> list[Option]:
    """
    link_state: {frozenset({u,v}): dict(qber=..., risk=..., key_pool_frac=...)}
    """
    options = []
    for req in requests:
        paths = k_shortest_paths(G, req.source, req.dest, k=k_paths)
        for path in paths:
            edges = path_edges(path)
            if not edges:
                continue
            risks = [link_state[e]["risk"] for e in edges if e in link_state]
            pool_press = [1 - link_state[e]["key_pool_frac"] for e in edges if e in link_state]
            base_risk = float(np.mean(risks)) if risks else 0.5
            base_pool_pressure = float(np.mean(pool_press)) if pool_press else 0.5
            base_latency = sum(2.0 for _ in edges)  # simplistic per-hop propagation ms
            for mode in MODES:
                # PQC/hybrid are largely immune to QKD-specific channel risk (QBER, key
                # depletion) but hybrid still requires a QKD leg to combine, so it
                # inherits a fraction of the QKD risk/pool pressure.
                if mode == "QKD":
                    risk = base_risk
                    pool_pressure = base_pool_pressure
                elif mode == "PQC":
                    risk = base_risk * 0.15   # PQC unaffected by quantum-channel attacks
                    pool_pressure = 0.0
                else:  # HYBRID
                    # Hybrid still requires a QKD leg to combine, so it is
                    # not risk-immune like PQC, only modestly better than
                    # pure QKD (mirrors real designs: hybrid protects
                    # against a PQC-side break but the QKD leg still carries
                    # most of the channel risk). Kept close to QKD (0.85x,
                    # not 0.5x) deliberately: an earlier, more optimistic
                    # 0.5x factor made HYBRID unconditionally dominate QKD
                    # regardless of channel conditions, which produced a
                    # QBER-invariant mode mix (caught via Experiment E8) --
                    # with the risk benefit this modest, HYBRID only
                    # becomes worth its extra compute cost once QBER-driven
                    # risk is high enough, giving a genuine crossover.
                    risk = base_risk * 0.85
                    pool_pressure = base_pool_pressure * 0.8

                latency = base_latency + MODE_LATENCY_MS[mode] * len(edges)
                comp_cost = MODE_COMP_COST[mode] * len(edges)
                disruption = risk * 0.5 + pool_pressure * 0.3  # heuristic composite

                options.append(Option(
                    req_id=req.req_id, path=path, mode=mode, edges=edges,
                    risk=risk, latency_ms=latency, key_pool_pressure=pool_pressure,
                    comp_cost=comp_cost, disruption_risk=disruption,
                    security_score=SECURITY_SCORES[mode],
                ))
    return options


DEFAULT_WEIGHTS = dict(w_R=1.0, w_L=0.4, w_K=0.8, w_C=1.1, w_D=0.6, w_S=0.5)


def build_qubo(options: list, requests: list, weights: dict = None,
               s_min: dict = None, penalty_onehot: float = None,
               penalty_security: float = None) -> QUBOProblem:
    """
    Build the QUBO matrix Q (n x n, n = len(options)) such that
        objective(y) = y^T Q y + offset
    encodes:
      - J(x): weighted sum of risk, latency, key depletion, comp cost, disruption
      - one-hot constraint per request (exactly one option chosen)
      - soft security-floor constraint S_r >= S_r^min (penalized if violated)
    """
    weights = weights or DEFAULT_WEIGHTS
    n = len(options)
    req_option_indices: dict = {}
    for idx, opt in enumerate(options):
        req_option_indices.setdefault(opt.req_id, []).append(idx)

    # Objective scale normalization. IMPORTANT: we use FIXED reference scales
    # (not per-instance min-max) for terms that are already meaningful in
    # absolute units (risk, key-pool pressure are already in [0,1] by
    # construction). Per-instance min-max normalization was tried initially
    # and found to erase genuine across-scenario signal (e.g. two QUBO
    # instances built at QBER=0.01 vs QBER=0.09 would both get rescaled to
    # the same [0,1] spread, making the optimizer's decisions QBER-invariant
    # except through the hard security floor) -- this was caught via
    # Experiment E8 (QBER sweep) showing a flat mode-mix and is documented
    # here so the fix doesn't get silently reverted.
    def _fixed_scale(vals, ref_max):
        vals = np.asarray(vals, dtype=float)
        return np.clip(vals / ref_max, 0.0, 1.0)

    risk_n = _fixed_scale([o.risk for o in options], ref_max=1.0)              # already in [0,1]
    lat_n = _fixed_scale([o.latency_ms for o in options], ref_max=50.0)        # ms, generous upper bound
    pool_n = _fixed_scale([o.key_pool_pressure ** 2 for o in options], ref_max=1.0)
    comp_n = _fixed_scale([o.comp_cost for o in options], ref_max=10.0)        # relative CPU units
    disr_n = _fixed_scale([o.disruption_risk for o in options], ref_max=1.0)

    linear = (weights["w_R"] * risk_n + weights["w_L"] * lat_n + weights["w_K"] * pool_n
              + weights["w_C"] * comp_n + weights["w_D"] * disr_n)

    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] += linear[i]

    # Auto-scale penalties relative to the objective's dynamic range so the
    # one-hot constraint dominates (Section 17: penalty sensitivity sweep).
    obj_scale = max(float(np.sum(np.abs(linear))), 1.0)
    penalty_onehot = penalty_onehot if penalty_onehot is not None else 3.0 * obj_scale
    penalty_security = penalty_security if penalty_security is not None else 2.0 * obj_scale

    offset = 0.0
    # One-hot: P * (sum_o y_o - 1)^2 = P*(sum y_o^2 + 2*sum_{o<o'} y_o y_o' - 2 sum y_o + 1)
    # y_o^2 = y_o for binary -> combine into diagonal.
    for req_id, idxs in req_option_indices.items():
        P = penalty_onehot
        for i in idxs:
            Q[i, i] += P * (1 - 2)  # + P*y_i^2 - 2P*y_i  -> diagonal contribution (1-2)=-1 times P, plus quad below
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                Q[i, j] += P  # cross term coefficient for y_i*y_j (2*P/2 folded since Q is used as y^T Q y with symmetric read as upper-triangular sum)
        offset += P * 1.0

    # Soft security floor: penalize options whose security_score < required, since
    # a hard combinatorial constraint per request is already enforced by one-hot,
    # so we bias the objective against infeasible-quality picks rather than forbid
    # them outright (keeps the QUBO always feasible for QAOA sampling).
    s_min = s_min or {}
    req_by_id = {r.req_id: r for r in requests}
    for idx, opt in enumerate(options):
        req = req_by_id[opt.req_id]
        required = s_min.get(opt.req_id, req.security_score_required)
        if opt.security_score < required:
            Q[idx, idx] += penalty_security * (required - opt.security_score) / 5.0

    return QUBOProblem(options=options, req_option_indices=req_option_indices, Q=Q,
                        offset=offset, weights=weights, penalty_onehot=penalty_onehot)


def evaluate_bitstring(problem: QUBOProblem, y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    qubo_value = float(y @ problem.Q @ y + problem.offset)
    violations = 0
    for req_id, idxs in problem.req_option_indices.items():
        s = sum(y[i] for i in idxs)
        if s != 1:
            violations += 1
    chosen = []
    for req_id, idxs in problem.req_option_indices.items():
        picked = [i for i in idxs if y[i] > 0.5]
        if len(picked) == 1:
            chosen.append(problem.options[picked[0]])
    return dict(qubo_value=qubo_value, constraint_violations=violations, chosen=chosen,
                n_requests=len(problem.req_option_indices))


if __name__ == "__main__":
    from qopt.network.topology import get_topology
    from qopt.network.traffic import sample_requests

    G = get_topology("A", seed=0)
    reqs = sample_requests(list(G.nodes()), n_requests=2, seed=1)
    link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
    options = build_options(G, reqs, link_state, k_paths=2)
    problem = build_qubo(options, reqs)
    print(f"n_requests={len(reqs)}  n_options={len(options)}  QUBO shape={problem.Q.shape}")

    # brute force check on this tiny instance
    n = len(options)
    best = None
    for bits in range(2 ** n):
        y = np.array([(bits >> i) & 1 for i in range(n)])
        res = evaluate_bitstring(problem, y)
        if res["constraint_violations"] == 0:
            if best is None or res["qubo_value"] < best["qubo_value"]:
                best = res
    print("Brute-force optimum:", best["qubo_value"], "modes:", [(o.req_id, o.mode) for o in best["chosen"]])
