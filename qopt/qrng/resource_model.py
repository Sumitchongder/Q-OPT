"""
QRNG resource abstraction (per node): entropy is modeled as a
leaky-bucket resource identical in structure to the QKD key pool,
per item 6 / 50 of the design notes -- we do not simulate a physical
QRNG, only its rate/availability/latency envelope.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QRNGResource:
    node_id: int
    max_entropy_bits: float
    rate_bps: float                 # entropy generation rate
    latency_ms: float = 0.05        # per-request latency
    current_bits: float = 0.0

    def __post_init__(self):
        if self.current_bits == 0.0:
            self.current_bits = 0.5 * self.max_entropy_bits

    def step(self, consume_bps: float, dt_s: float) -> float:
        self.current_bits = min(
            self.max_entropy_bits,
            max(0.0, self.current_bits + self.rate_bps * dt_s - consume_bps * dt_s),
        )
        return self.current_bits

    @property
    def availability(self) -> float:
        return 0.0 if self.max_entropy_bits <= 0 else self.current_bits / self.max_entropy_bits

    def is_scarce(self, r_min_frac: float = 0.15) -> bool:
        return self.availability < r_min_frac
