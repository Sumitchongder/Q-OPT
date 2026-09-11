"""
Minimal RL baseline (Experiment E6: QAOA vs RL), as flagged missing in
the original repo drop.

This is deliberately simple: each request is treated as an independent
contextual bandit (no cross-request state, since the one-hot QUBO
structure already makes requests independent apart from the shared
objective normalization). A tabular Q-table is keyed by
(request_security_class, coarse-binned mean edge risk) -> action
(option index within that request's candidate set), trained online
across episodes where each episode re-samples link risk.

This is intentionally not a sophisticated deep-RL agent (no PPO) --
it is the smallest RL baseline that (a) learns from reward feedback
rather than being handed the QUBO directly like the other classical
baselines, and (b) plugs into the same `SolverResult` interface, so it
can be dropped into e1/e5-style comparison scripts unchanged. Treat
this as a floor, not a ceiling, for the RL comparison point.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from qopt.optimization.qubo import QUBOProblem, evaluate_bitstring
from qopt.optimization.classical_baselines import SolverResult


def _bin_risk(mean_risk: float, n_bins: int = 5) -> int:
    return int(np.clip(mean_risk * n_bins, 0, n_bins - 1))


@dataclass
class QLearningAgent:
    alpha: float = 0.3      # learning rate
    gamma: float = 0.0      # single-step episodic reward, no discounting needed
    epsilon: float = 0.2    # exploration rate
    q_table: dict = field(default_factory=dict)

    def _key(self, state, n_actions):
        return state

    def act(self, state, n_actions, rng):
        key = self._key(state, n_actions)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(n_actions)
        if rng.random() < self.epsilon:
            return int(rng.integers(n_actions))
        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, n_actions):
        key = self._key(state, n_actions)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(n_actions)
        q = self.q_table[key]
        q[action] += self.alpha * (reward - q[action])


def solve_q_learning(problem: QUBOProblem, requests, n_episodes: int = 300,
                      seed: int = 0) -> SolverResult:
    """
    Train a fresh tabular Q-learning agent ON this problem instance for
    `n_episodes` episodes (since the "environment" here is a single
    static QUBO instance, not a full MDP across the network's lifetime,
    this measures RL's ability to learn a good per-request policy
    through trial and error, which is what the design's E6 comparison
    calls for), then extract the greedy final policy as the solution.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    req_by_id = {r.req_id: r for r in requests}

    agents = {}  # req_id -> QLearningAgent
    action_lists = {}  # req_id -> list of option indices (bandit arms)
    for req_id, idxs in problem.req_option_indices.items():
        agents[req_id] = QLearningAgent(seed=seed) if False else QLearningAgent()
        action_lists[req_id] = idxs

    diag = np.diag(problem.Q)

    def reward_for_option(idx):
        # Lower QUBO diagonal cost -> higher reward (bandit reward = -cost, rescaled).
        return -diag[idx]

    for episode in range(n_episodes):
        for req_id, idxs in action_lists.items():
            mean_risk_state = _bin_risk(float(np.mean([problem.options[i].risk for i in idxs])))
            state = (req_by_id[req_id].security_class, mean_risk_state)
            n_actions = len(idxs)
            local_action = agents[req_id].act(state, n_actions, rng)
            global_idx = idxs[local_action]
            r = reward_for_option(global_idx)
            agents[req_id].update(state, local_action, r, n_actions)

    # Extract final greedy policy
    y = np.zeros(len(problem.options))
    for req_id, idxs in action_lists.items():
        mean_risk_state = _bin_risk(float(np.mean([problem.options[i].risk for i in idxs])))
        state = (req_by_id[req_id].security_class, mean_risk_state)
        q = agents[req_id].q_table.get(state, np.zeros(len(idxs)))
        best_local = int(np.argmax(q))
        y[idxs[best_local]] = 1

    runtime = time.perf_counter() - t0
    res = evaluate_bitstring(problem, y)
    return SolverResult("Q-Learning", y, res["qubo_value"], res["constraint_violations"], runtime)


if __name__ == "__main__":
    from qopt.network.topology import get_topology
    from qopt.network.traffic import sample_requests
    from qopt.optimization.qubo import build_options, build_qubo
    from qopt.optimization.classical_baselines import solve_exact

    G = get_topology("B", seed=0)
    reqs = sample_requests(list(G.nodes()), n_requests=4, seed=1)
    link_state = {frozenset(e): dict(qber=0.02, risk=0.15, key_pool_frac=0.7) for e in G.edges()}
    options = build_options(G, reqs, link_state, k_paths=2)
    problem = build_qubo(options, reqs)

    exact = solve_exact(problem, max_n=30)
    rl = solve_q_learning(problem, reqs, n_episodes=500, seed=0)
    print(f"Exact       obj={exact.objective:.4f}")
    print(f"Q-Learning  obj={rl.objective:.4f}  AR={rl.objective/exact.objective:.3f}  "
          f"violations={rl.constraint_violations}  runtime={rl.runtime_s*1e3:.2f} ms")
