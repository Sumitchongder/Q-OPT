"""
Risk engine: rho_ij = f(QBER, loss, key_pool, attack_probability, trust)

This module plays two roles:

1. A standalone calibrated risk function (used when Q-TRACE output is
   not available), combining normalized signal channels with tunable
   weights.
2. An adapter (`from_q_trace_output`) that ingests an externally
   supplied attack-probability time series (e.g. from a Q-TRACE-style
   anomaly classifier) and folds it into the same rho_ij used by the
   Q-OPT QUBO, closing the Q-TRACE -> Q-OPT loop (Experiment E14).

Since this repository does not ship the companion Q-TRACE classifier,
`from_q_trace_output` accepts any callable/array of per-link
P(attack | x_t) so it can be swapped in without touching the
optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class RiskWeights:
    w_qber: float = 0.35
    w_loss: float = 0.15
    w_pool: float = 0.20
    w_attack: float = 0.20
    w_trust: float = 0.10

    def normalized(self) -> "RiskWeights":
        total = self.w_qber + self.w_loss + self.w_pool + self.w_attack + self.w_trust
        return RiskWeights(self.w_qber / total, self.w_loss / total,
                            self.w_pool / total, self.w_attack / total, self.w_trust / total)


def compute_risk(qber: float, loss_db: float, key_pool_frac: float,
                  attack_probability: float, trust: float,
                  weights: RiskWeights = RiskWeights(),
                  qber_ref_max: float = 0.11, loss_db_ref_max: float = 25.0) -> float:
    """
    rho_ij in [0, 1]. Higher = riskier link.

    - qber: current QBER (higher -> riskier; normalized against the
      0.11 GLLP-type security threshold commonly used as an abort
      bound for BB84).
    - loss_db: current path loss (higher -> riskier, proxy for
      exposure/attack surface and lower key rate margin).
    - key_pool_frac: current_bits / capacity_bits (LOWER pool -> riskier).
    - attack_probability: P(attack | telemetry), e.g. from Q-TRACE.
    - trust: node/link trust score in [0, 1] (LOWER trust -> riskier).
    """
    w = weights.normalized()
    n_qber = _normalize(qber, 0.0, qber_ref_max)
    n_loss = _normalize(loss_db, 0.0, loss_db_ref_max)
    n_pool_risk = 1.0 - float(np.clip(key_pool_frac, 0.0, 1.0))
    n_attack = float(np.clip(attack_probability, 0.0, 1.0))
    n_trust_risk = 1.0 - float(np.clip(trust, 0.0, 1.0))

    rho = (w.w_qber * n_qber + w.w_loss * n_loss + w.w_pool * n_pool_risk
           + w.w_attack * n_attack + w.w_trust * n_trust_risk)
    return float(np.clip(rho, 0.0, 1.0))


def from_q_trace_output(p_attack_per_link: dict, base_signals: dict,
                         weights: RiskWeights = RiskWeights()) -> dict:
    """
    Fold externally-supplied Q-TRACE attack probabilities into rho_ij.

    p_attack_per_link : {edge: P(attack|x_t)}
    base_signals       : {edge: dict(qber=..., loss_db=..., key_pool_frac=..., trust=...)}
    returns             {edge: rho_ij}
    """
    out = {}
    for edge, sig in base_signals.items():
        p_atk = p_attack_per_link.get(edge, 0.0)
        out[edge] = compute_risk(sig["qber"], sig["loss_db"], sig["key_pool_frac"],
                                  p_atk, sig.get("trust", 1.0), weights)
    return out


if __name__ == "__main__":
    rho_normal = compute_risk(qber=0.01, loss_db=8, key_pool_frac=0.8, attack_probability=0.02, trust=0.95)
    rho_attack = compute_risk(qber=0.08, loss_db=8, key_pool_frac=0.2, attack_probability=0.85, trust=0.3)
    print(f"rho (normal conditions)  = {rho_normal:.3f}")
    print(f"rho (under attack)       = {rho_attack:.3f}")
    assert rho_attack > rho_normal
