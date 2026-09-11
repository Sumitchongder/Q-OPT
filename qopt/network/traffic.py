"""
Traffic / service-request model. r = (s, d, D, S, tau)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SECURITY_CLASSES = {"critical": 5, "high": 4, "normal": 3, "best_effort": 1}


@dataclass
class ServiceRequest:
    req_id: int
    source: int
    dest: int
    key_demand_bps: float     # D
    security_class: str       # S
    latency_req_ms: float     # tau

    @property
    def security_score_required(self) -> int:
        return SECURITY_CLASSES[self.security_class]


def sample_requests(nodes: list[int], n_requests: int, seed: int = 0) -> list[ServiceRequest]:
    rng = np.random.default_rng(seed)
    classes = list(SECURITY_CLASSES.keys())
    class_probs = [0.15, 0.25, 0.40, 0.20]  # critical, high, normal, best_effort
    reqs = []
    for i in range(n_requests):
        s, d = rng.choice(nodes, size=2, replace=False)
        sec_class = rng.choice(classes, p=class_probs)
        D = float(rng.uniform(1e3, 5e4))          # bits/s demand
        tau = float(rng.uniform(5, 200))           # ms
        reqs.append(ServiceRequest(i, int(s), int(d), D, sec_class, tau))
    return reqs
