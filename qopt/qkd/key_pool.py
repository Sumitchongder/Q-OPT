"""
QKD key-pool dynamics (per-link finite buffer of generated secret key).

    q_ij(t + dt) = clip( q_ij(t) + g_ij(t)*dt - c_ij(t)*dt , 0, Q_max_ij )

This is the standard leaky-bucket formulation used in adaptive QKD
network resource-allocation literature; we use it here as the baseline
dynamical model (Section 4 of the design notes), not as a novel
contribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KeyPool:
    capacity_bits: float
    current_bits: float = 0.0

    def step(self, gen_rate_bps: float, consume_bps: float, dt_s: float) -> float:
        self.current_bits = min(
            self.capacity_bits,
            max(0.0, self.current_bits + gen_rate_bps * dt_s - consume_bps * dt_s),
        )
        return self.current_bits

    @property
    def utilization(self) -> float:
        return 0.0 if self.capacity_bits <= 0 else self.current_bits / self.capacity_bits

    @property
    def depletion(self) -> float:
        return 1.0 - self.utilization


@dataclass
class NetworkKeyPools:
    """Container mapping edge -> KeyPool, keyed by frozenset({u, v})."""
    pools: dict = field(default_factory=dict)

    def get_or_create(self, u: int, v: int, capacity_bits: float) -> KeyPool:
        key = frozenset((u, v))
        if key not in self.pools:
            self.pools[key] = KeyPool(capacity_bits=capacity_bits, current_bits=capacity_bits * 0.5)
        return self.pools[key]

    def __getitem__(self, edge):
        u, v = edge
        return self.pools[frozenset((u, v))]
