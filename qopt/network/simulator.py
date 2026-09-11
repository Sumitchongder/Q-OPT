"""
Q-OPT-NET scenario simulator: composes topology + QKD channel physics +
key pools + QRNG resources + attacks + risk engine into per-timestep
link/node state, matching the dataset schema in Section 30 of the
design notes (a reduced but faithful subset -- every column that is
actually consumed by the QUBO/optimizers is present; purely descriptive
columns from the full spec that don't affect optimization, e.g. raw
per-photon detector counts, are omitted for tractability).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from qopt.qkd.channel import ChannelParams, key_generation_rate_bps
from qopt.qkd.key_pool import NetworkKeyPools
from qopt.qrng.resource_model import QRNGResource
from qopt.qkd import attacks as atk
from qopt.threat.risk_engine import compute_risk, RiskWeights


@dataclass
class ScenarioConfig:
    topology: str = "A"
    seed: int = 0
    n_timesteps: int = 50
    dt_s: float = 1.0
    pool_capacity_bits: float = 5e6
    qrng_capacity_bits: float = 1e5
    qrng_rate_bps: float = 5e4
    base_consume_bps: float = 2e3
    attack: atk.AttackEvent | None = None
    channel_params: ChannelParams = field(default_factory=ChannelParams)


def run_scenario(cfg: ScenarioConfig) -> "pandas.DataFrame":
    import pandas as pd
    from qopt.network.topology import get_topology

    G = get_topology(cfg.topology, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    pools = NetworkKeyPools()
    for u, v in G.edges():
        pools.get_or_create(u, v, cfg.pool_capacity_bits)

    qrngs = {n: QRNGResource(node_id=n, max_entropy_bits=cfg.qrng_capacity_bits,
                              rate_bps=cfg.qrng_rate_bps) for n in G.nodes()}

    rows = []
    for t in range(cfg.n_timesteps):
        active_attack = (cfg.attack is not None and cfg.attack.start_t <= t < cfg.attack.end_t)
        for u, v in G.edges():
            edge = frozenset((u, v))
            length_km = G[u][v]["length_km"]

            qber_shift = 0.0
            active_this_edge = active_attack and edge in {frozenset(e) for e in cfg.attack.target_edges}

            gen_rate, Q_mu, qber = key_generation_rate_bps(length_km, cfg.channel_params, qber_shift)

            if active_this_edge:
                if cfg.attack.attack_type == atk.AttackType.QBER_INJECTION:
                    qber = atk.apply_qber_injection(qber, cfg.attack.intensity)
                elif cfg.attack.attack_type == atk.AttackType.PNS_DEGRADATION:
                    gen_rate = atk.apply_pns_degradation(gen_rate, cfg.attack.intensity)
                elif cfg.attack.attack_type == atk.AttackType.LINK_DENIAL:
                    gen_rate = atk.apply_link_denial(gen_rate, True)

            consume = cfg.base_consume_bps * (1 + 0.2 * rng.standard_normal())
            consume = max(0.0, consume)
            if active_this_edge and cfg.attack.attack_type == atk.AttackType.KEY_EXHAUSTION:
                consume = atk.apply_key_exhaustion(consume, cfg.attack.intensity)

            pool = pools[(u, v)]
            pool.step(gen_rate, consume, cfg.dt_s)

            trust = 1.0
            attack_prob = 0.0
            if active_this_edge and cfg.attack.attack_type == atk.AttackType.CONTROL_PLANE:
                attack_prob = cfg.attack.intensity
                risk_cp, trust = atk.apply_control_plane_compromise(0.1, 1.0, cfg.attack.intensity)
            loss_db = cfg.channel_params.fiber_loss_db_per_km * length_km

            risk = compute_risk(qber, loss_db, pool.utilization, attack_prob, trust)

            rows.append(dict(
                t=t, edge=f"{u}-{v}", u=u, v=v, length_km=length_km,
                qber=qber, loss_db=loss_db, key_gen_rate_bps=gen_rate,
                key_consume_bps=consume, key_pool_bits=pool.current_bits,
                key_pool_capacity_bits=pool.capacity_bits, key_pool_frac=pool.utilization,
                trust=trust, attack_probability=attack_prob, risk=risk,
                attack_active=int(active_this_edge),
                attack_type=(cfg.attack.attack_type.value if active_this_edge else "none"),
            ))

        for n in G.nodes():
            qr = qrngs[n]
            qr.step(consume_bps=cfg.base_consume_bps * 0.1, dt_s=cfg.dt_s)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = ScenarioConfig(topology="B", n_timesteps=20, seed=0)
    df = run_scenario(cfg)
    print(df.shape)
    print(df[["t", "edge", "qber", "key_pool_frac", "risk"]].head(10))
