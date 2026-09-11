"""
Attack / degradation event models for the Q-OPT digital twin.

Five categories, matching the project specification:
    A: QBER injection (intercept-resend style)
    B: PNS-induced yield degradation
    C: link denial (generation rate -> 0)
    D: key-pool exhaustion (consumption spike)
    E: control-plane compromise (trust collapse / risk inflation)

Each attack is expressed as a pure function that perturbs the relevant
scenario fields; the network simulator composes them per-timestep.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttackType(str, Enum):
    NONE = "none"
    QBER_INJECTION = "qber_injection"
    PNS_DEGRADATION = "pns_degradation"
    LINK_DENIAL = "link_denial"
    KEY_EXHAUSTION = "key_exhaustion"
    CONTROL_PLANE = "control_plane"


@dataclass
class AttackEvent:
    attack_type: AttackType
    target_edges: tuple  # tuple of (u, v) edges affected
    intensity: float = 0.5  # alpha in [0, 1]
    start_t: int = 0
    end_t: int = 10


def apply_qber_injection(qber_intrinsic: float, intensity: float,
                          qber_eve: float = 0.25) -> float:
    """QBER' = (1 - alpha) * QBER + alpha * QBER_eve (intercept-resend ~ 0.25 asymptotic)."""
    return (1 - intensity) * qber_intrinsic + intensity * qber_eve


def apply_pns_degradation(gen_rate_bps: float, intensity: float,
                           max_yield_loss_frac: float = 0.6) -> float:
    """PNS attack: photon-number-splitting reduces effective single-photon yield."""
    return gen_rate_bps * (1 - intensity * max_yield_loss_frac)


def apply_link_denial(gen_rate_bps: float, active: bool) -> float:
    return 0.0 if active else gen_rate_bps


def apply_key_exhaustion(consume_bps: float, intensity: float,
                          surge_factor: float = 4.0) -> float:
    """Malicious/artificial demand surge on a link's key pool."""
    return consume_bps * (1 + intensity * (surge_factor - 1))


def apply_control_plane_compromise(risk: float, trust: float, intensity: float) -> tuple[float, float]:
    """Inflate risk and collapse trust proportionally to attack intensity."""
    risk_out = min(1.0, risk + intensity * (1.0 - risk))
    trust_out = max(0.0, trust * (1 - intensity))
    return risk_out, trust_out
