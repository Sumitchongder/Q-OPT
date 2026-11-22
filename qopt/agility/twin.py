"""
Multi-epoch network digital twin used to evaluate crypto-agility policies.

The optimiser sees a smooth quadratic surrogate of resource contention.
The twin, by contrast, enforces *hard* budgets and records operational
outcomes, so a policy is judged on what actually happens to traffic:

  * QKD / HYB requests draw key from every link pool on their path; if any
    pool is short the request is downgraded to PQC (key-exhaustion event).
  * PQC / HYB handshakes draw CPU and QRNG entropy at both end nodes; CPU
    overload adds queueing delay, entropy shortfall is logged.
  * Pools replenish from the LMC05 key rate, which collapses under attack
    (QBER above the decoy-state threshold aborts key generation).

Common random numbers: the attack process and the traffic of every epoch
are drawn from streams keyed only by (seed, epoch), so every policy faces
the identical sequence of attacks and requests; only the pool trajectories
differ, because they depend on the policy's own past decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qopt.agility.model import (ScenarioParams, build_problem, make_links, nominal_pool_capacity,
                                refresh_links, sample_requests)


@dataclass
class TwinConfig:
    topology: str = "C"
    n_epochs: int = 40
    n_requests: int = 6
    k_paths: int = 2
    load: float = 1.0
    attack_intensity: float = 0.6
    p_attack_on: float = 0.08          # per link per epoch
    p_attack_off: float = 0.25
    p_link_denial: float = 0.01        # per link per epoch, lasts one epoch
    hotspot: bool = True
    seed: int = 0
    params: ScenarioParams = field(default_factory=ScenarioParams)
    weights: dict | None = None


def _attack_trajectory(G, cfg: TwinConfig):
    """Pre-draw the attack/denial process (identical for every policy)."""
    rng = np.random.default_rng([cfg.seed, 991])
    edges = [frozenset(e) for e in G.edges()]
    on = {e: False for e in edges}
    sev = {e: rng.uniform(0.5, 1.0) for e in edges}
    traj = []
    for t in range(cfg.n_epochs):
        step = {}
        for e in edges:
            if on[e]:
                on[e] = rng.random() > cfg.p_attack_off
            else:
                on[e] = rng.random() < cfg.p_attack_on
            denial = rng.random() < cfg.p_link_denial
            step[e] = (on[e], sev[e], denial)
        traj.append(step)
    return traj


def run_twin(policy, cfg: TwinConfig, record_epochs: bool = False) -> dict:
    """
    policy(problem) -> list of local choices (one per block).
    Returns aggregate metrics (and optionally a per-epoch trace).
    """
    from qopt.network.topology import get_topology

    P = cfg.params
    G = get_topology(cfg.topology, seed=cfg.seed)
    links = make_links(G, P, np.random.default_rng([cfg.seed, 7]), 0.0, 0.0, pool_frac=(0.5, 0.5))
    caps = {e: nominal_pool_capacity(ls, P) for e, ls in links.items()}
    traj = _attack_trajectory(G, cfg)
    hs_rng = np.random.default_rng([cfg.seed, 13])
    u, v = list(G.edges())[int(hs_rng.integers(G.number_of_edges()))]
    hotspot = (int(u), int(v)) if cfg.hotspot else None

    agg = dict(n=0, compliant=0, crit_n=0, crit_ok=0, downgrades=0, risk_sum=0.0, sla_viol=0,
               lat_sum=0.0, rng_short=0, key_used=0.0, key_generated=0.0, objective_sum=0.0)
    per_epoch = []
    for t in range(cfg.n_epochs):
        # ---- update physical layer --------------------------------------------
        for e, ls in links.items():
            att, sev, den = traj[t][e]
            ls.qber_shift = 0.05 * cfg.attack_intensity * sev if att else 0.0
            ls.attack_prob = cfg.attack_intensity * sev if att else 0.0
            ls.trust = 1.0 - 0.5 * cfg.attack_intensity if att else 1.0
        refresh_links(G, links, P)
        for e, ls in links.items():
            if traj[t][e][2]:  # link denial: no generation this epoch
                ls.rate_bps = 0.0
                ls.budget_bits = max(ls.pool_bits, P.min_budget_bits)
        # ---- traffic (common random numbers) -----------------------------------
        trng = np.random.default_rng([cfg.seed, 31, t])
        reqs = sample_requests(G, cfg.n_requests, trng, links, P, load=cfg.load, hotspot=hotspot)
        # demand is drawn against the *nominal* network supply so that it does
        # not depend on the policy-dependent pool state
        nominal = np.mean([caps[e] / P.pool_capacity_s for e in links])
        current = np.mean([ls.budget_bits for ls in links.values()])
        for r in reqs:
            r.demand_bits *= nominal / max(current, 1e-9)
        prob = build_problem(G, reqs, links, P, cfg.weights, k_paths=cfg.k_paths)
        choice = policy(prob)
        agg["objective_sum"] += prob.energy_choice(choice)

        # ---- execute with hard budgets ----------------------------------------
        avail = {e: ls.budget_bits for e, ls in links.items()}
        cpu = {n: 0.0 for n in G.nodes()}
        ent = {n: 0.0 for n in G.nodes()}
        rank = {"critical": 0, "high": 1, "normal": 2, "best_effort": 3}
        order = sorted(range(len(reqs)), key=lambda r: rank[reqs[r].service_class])
        ep = dict(n=0, ok=0, down=0, risk=0.0)
        executed = {}
        for r in order:
            req = reqs[r]
            opt = prob.options[prob.blocks[r][choice[r]]]
            mode = opt.mode
            if mode in ("QKD", "HYB"):
                if all(avail[e] >= req.demand_bits for e in opt.edges):
                    for e in opt.edges:
                        avail[e] -= req.demand_bits
                    agg["key_used"] += req.demand_bits * len(opt.edges)
                else:
                    agg["downgrades"] += 1
                    ep["down"] += 1
                    mode = "PQC"
            executed[r] = (opt, mode)
            if mode in ("PQC", "HYB"):
                for n in (req.source, req.dest):
                    cpu[n] += req.handshakes * P.cpu_per_handshake
                    ent[n] += req.handshakes * P.rng_bits_pqc
        for r in order:
            req = reqs[r]
            opt, mode = executed[r]
            if mode == opt.mode:
                risk, lat = opt.risk, opt.latency_ms
            else:  # downgraded to PQC on the same path
                risk = P.rho_pqc
                lat = opt.latency_ms - P.kms_relay_ms * len(opt.edges) + P.pqc_handshake_ms
            if mode in ("PQC", "HYB"):
                over = max(cpu[req.source], cpu[req.dest]) / P.node_cpu_capacity
                if over > 1.0:
                    lat += P.pqc_handshake_ms * (over - 1.0) * 4.0   # queueing delay
                if max(ent[req.source], ent[req.dest]) > P.node_rng_capacity:
                    agg["rng_short"] += 1
                    risk = min(1.0, risk + 0.5 * P.rho_pqc)        # DRBG fallback weakens keys
            sla_ok = lat <= req.latency_sla_ms
            sec_ok = risk <= req.tolerance
            agg["n"] += 1
            agg["risk_sum"] += risk
            agg["lat_sum"] += lat
            agg["sla_viol"] += int(not sla_ok)
            agg["compliant"] += int(sla_ok and sec_ok)
            if req.service_class == "critical":
                agg["crit_n"] += 1
                agg["crit_ok"] += int(sla_ok and sec_ok)
            ep["n"] += 1
            ep["ok"] += int(sla_ok and sec_ok)
            ep["risk"] += risk
        # ---- pool dynamics ------------------------------------------------------
        for e, ls in links.items():
            agg["key_generated"] += ls.rate_bps * P.epoch_s
            ls.pool_bits = float(np.clip(avail[e], 0.0, caps[e]))
        if record_epochs:
            per_epoch.append(dict(epoch=t, compliance=ep["ok"] / ep["n"], downgrades=ep["down"],
                                  mean_risk=ep["risk"] / ep["n"],
                                  mean_pool_frac=float(np.mean([links[e].pool_bits / caps[e] for e in links])),
                                  n_attacked=sum(traj[t][e][0] for e in links)))
    n = max(agg["n"], 1)
    out = dict(compliance=agg["compliant"] / n,
               critical_compliance=agg["crit_ok"] / max(agg["crit_n"], 1),
               downgrade_rate=agg["downgrades"] / n,
               mean_risk=agg["risk_sum"] / n,
               expected_compromises_per_1k=1000.0 * agg["risk_sum"] / n,
               sla_violation_rate=agg["sla_viol"] / n,
               mean_latency_ms=agg["lat_sum"] / n,
               entropy_shortfall_rate=agg["rng_short"] / n,
               key_utilisation=agg["key_used"] / max(agg["key_generated"], 1e-9),
               mean_objective=agg["objective_sum"] / cfg.n_epochs)
    if record_epochs:
        out["trace"] = per_epoch
    return out
