"""
Experiment 8 (E14, flagship): Q-TRACE -> Q-OPT closed loop.

Q-TRACE (the companion attack-probability classifier) is not shipped
in this repository (see README). To evaluate the closed-loop
architecture end-to-end anyway, this script SIMULATES a Q-TRACE-style
classifier output as a noisy, imperfect estimate of the true attack
intensity:

    p_hat_attack = clip(true_alpha + N(0, sigma), 0, 1)

and feeds it into qopt.threat.risk_engine.from_q_trace_output, which
is the actual integration point Q-TRACE would use. This isolates the
QUESTION E14 asks -- "does imperfect threat intelligence still help,
compared to no threat intelligence at all?" -- without requiring the
real classifier.

Four conditions are compared at each true attack intensity:
    1. Static        - mode assignment fixed at the alpha=0 solution, never updated
    2. TrustOnly      - switches to HYBRID for ALL requests once trust drops
                        below a hard threshold (0.5), no continuous risk signal
    3. QOPT_Oracle    - Q-OPT with the TRUE attack probability (upper bound)
    4. QOPT_QTRACE    - Q-OPT with the SIMULATED noisy Q-TRACE estimate (realistic)

Produces:
    tables/e8_qtrace_closed_loop.csv
    tables/e8_qtrace_closed_loop_summary.csv
    figures/e8_qtrace_closed_loop.pdf/png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from qopt.network.topology import get_topology
from qopt.network.traffic import sample_requests
from qopt.optimization.qubo import build_options, build_qubo, evaluate_bitstring
from qopt.optimization.classical_baselines import solve_exact
from qopt.threat.risk_engine import compute_risk, from_q_trace_output, RiskWeights


def _link_state_for_alpha(G, alpha, rng):
    base_signals = {}
    for e in G.edges():
        qber = float(rng.uniform(0.01, 0.02))
        trust = 1.0 - alpha * float(rng.uniform(0.6, 1.0))
        key_pool_frac = max(0.02, 0.8 - alpha * float(rng.uniform(0.5, 0.9)))
        base_signals[frozenset(e)] = dict(qber=qber, loss_db=8.0, key_pool_frac=key_pool_frac, trust=trust)
    return base_signals


def _build_problem_with_risk(G, reqs, base_signals, p_attack_per_link, k_paths):
    rho = from_q_trace_output(p_attack_per_link, base_signals, weights=RiskWeights())
    link_state = {edge: dict(qber=sig["qber"], risk=rho[edge], key_pool_frac=sig["key_pool_frac"])
                   for edge, sig in base_signals.items()}
    options = build_options(G, reqs, link_state, k_paths=k_paths)
    return build_qubo(options, reqs)


def run(n_seeds: int, n_requests: int, k_paths: int, topology: str, qtrace_noise_sigma: float,
        outdir_tables: str, outdir_figs: str):
    os.makedirs(outdir_tables, exist_ok=True)
    os.makedirs(outdir_figs, exist_ok=True)

    intensities = np.linspace(0.0, 1.0, 9)
    records = []

    for seed in range(n_seeds):
        G = get_topology(topology, seed=0)
        reqs = sample_requests(list(G.nodes()), n_requests=n_requests, seed=seed)
        rng = np.random.default_rng(seed + 7000)

        # --- Static baseline: solve once at alpha=0, freeze that assignment ---
        base_signals_0 = _link_state_for_alpha(G, 0.0, rng)
        p_zero = {e: 0.0 for e in base_signals_0}
        static_problem = _build_problem_with_risk(G, reqs, base_signals_0, p_zero, k_paths)
        static_result = solve_exact(static_problem, max_n=30)
        static_res_eval = evaluate_bitstring(static_problem, static_result.y)
        static_chosen = static_res_eval["chosen"]

        for alpha in intensities:
            rng_a = np.random.default_rng(seed * 1000 + int(alpha * 100) + 8000)
            base_signals = _link_state_for_alpha(G, alpha, rng_a)

            # true attack probability per edge (ground truth, oracle)
            p_true = {e: alpha for e in base_signals}
            # simulated Q-TRACE estimate: noisy version of the truth
            noise = rng_a.normal(0, qtrace_noise_sigma, size=len(base_signals))
            p_qtrace = {e: float(np.clip(alpha + n, 0, 1)) for e, n in zip(base_signals, noise)}

            # --- Condition 1: Static (frozen assignment re-scored under current risk) ---
            cur_problem = _build_problem_with_risk(G, reqs, base_signals, p_true, k_paths)
            static_y = np.zeros(len(cur_problem.options))
            # Map frozen (req_id, mode) choices onto the current option indexing. Multiple
            # candidate paths can share the same mode label, so pick only the FIRST matching
            # option per request -- otherwise more than one option gets set per request and
            # the one-hot constraint is (correctly) flagged as violated, zeroing out
            # "availability" for a condition that should always be trivially available.
            static_modes = {o.req_id: o.mode for o in static_chosen}
            already_set = set()
            for idx, opt in enumerate(cur_problem.options):
                if opt.req_id in already_set:
                    continue
                if static_modes.get(opt.req_id) == opt.mode:
                    static_y[idx] = 1
                    already_set.add(opt.req_id)
            static_eval = evaluate_bitstring(cur_problem, static_y)

            # --- Condition 2: TrustOnly (hard threshold, no continuous signal) ---
            mean_trust = np.mean([s["trust"] for s in base_signals.values()])
            trust_forces_hybrid = mean_trust < 0.5
            trustonly_y = np.zeros(len(cur_problem.options))
            for req_id, idxs in cur_problem.req_option_indices.items():
                candidates = [i for i in idxs if cur_problem.options[i].mode ==
                              ("HYBRID" if trust_forces_hybrid else "PQC")]
                pick = candidates[0] if candidates else idxs[0]
                trustonly_y[pick] = 1
            trustonly_eval = evaluate_bitstring(cur_problem, trustonly_y)

            # --- Condition 3: Q-OPT + oracle (true alpha) ---
            oracle_problem = _build_problem_with_risk(G, reqs, base_signals, p_true, k_paths)
            oracle_result = solve_exact(oracle_problem, max_n=30)
            oracle_eval = evaluate_bitstring(oracle_problem, oracle_result.y)

            # --- Condition 4: Q-OPT + simulated Q-TRACE estimate ---
            qtrace_problem = _build_problem_with_risk(G, reqs, base_signals, p_qtrace, k_paths)
            qtrace_result = solve_exact(qtrace_problem, max_n=30)
            qtrace_eval = evaluate_bitstring(qtrace_problem, qtrace_result.y)

            for name, ev, prob in [("Static", static_eval, cur_problem),
                                     ("TrustOnly", trustonly_eval, cur_problem),
                                     ("QOPT_Oracle", oracle_eval, oracle_problem),
                                     ("QOPT_QTRACE", qtrace_eval, qtrace_problem)]:
                req_by_id = {r.req_id: r for r in reqs}
                n_secure = sum(1 for o in ev["chosen"] if o.security_score >= req_by_id[o.req_id].security_score_required)
                records.append(dict(
                    seed=seed, attack_intensity=alpha, condition=name,
                    availability=len(ev["chosen"]) / len(reqs),
                    security_compliance=n_secure / len(reqs),
                    mean_risk_of_solution=float(np.mean([o.risk for o in ev["chosen"]])) if ev["chosen"] else np.nan,
                ))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(outdir_tables, "e8_qtrace_closed_loop.csv"), index=False)

    summary = df.groupby(["condition", "attack_intensity"]).agg(
        availability_mean=("availability", "mean"),
        security_compliance_mean=("security_compliance", "mean"),
        mean_risk_of_solution=("mean_risk_of_solution", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(outdir_tables, "e8_qtrace_closed_loop_summary.csv"), index=False)
    print(summary.to_string(index=False))

    _plot(summary, outdir_figs)
    return df, summary


def _plot(summary: pd.DataFrame, outdir_figs: str):
    import matplotlib.pyplot as plt
    from qopt.viz.style import set_style, savefig

    set_style()
    colors = {"Static": "#8c8c8c", "TrustOnly": "#f0a35a", "QOPT_Oracle": "#3a6b35", "QOPT_QTRACE": "#2a6fdb"}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    ax1 = axes[0]
    for cond in ["Static", "TrustOnly", "QOPT_Oracle", "QOPT_QTRACE"]:
        sub = summary[summary.condition == cond].sort_values("attack_intensity")
        ax1.plot(sub.attack_intensity, sub.security_compliance_mean, marker="o",
                 label=cond, color=colors[cond])
    ax1.set_xlabel("True attack intensity (alpha)")
    ax1.set_ylabel("Security-floor compliance")
    ax1.set_title("(a) Compliance: Static/TrustOnly vs. Q-OPT")
    ax1.legend(fontsize=7.5, loc="lower left")

    ax2 = axes[1]
    for cond in ["Static", "TrustOnly", "QOPT_Oracle", "QOPT_QTRACE"]:
        sub = summary[summary.condition == cond].sort_values("attack_intensity")
        ax2.plot(sub.attack_intensity, sub.mean_risk_of_solution, marker="o",
                 label=cond, color=colors[cond])
    ax2.set_xlabel("True attack intensity (alpha)")
    ax2.set_ylabel("Mean risk of chosen solution")
    ax2.set_title("(b) Residual risk: oracle vs. noisy Q-TRACE estimate")
    ax2.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Q-TRACE -> Q-OPT closed loop: imperfect threat intelligence still helps",
                 fontsize=10, y=1.03)
    savefig(fig, os.path.join(outdir_figs, "e8_qtrace_closed_loop"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--n-requests", type=int, default=4)
    ap.add_argument("--k-paths", type=int, default=2)
    ap.add_argument("--topology", default="C")
    ap.add_argument("--qtrace-noise-sigma", type=float, default=0.2,
                     help="Std dev of simulated Q-TRACE estimation error")
    ap.add_argument("--outdir-tables", default="tables")
    ap.add_argument("--outdir-figs", default="figures")
    args = ap.parse_args()
    run(args.n_seeds, args.n_requests, args.k_paths, args.topology, args.qtrace_noise_sigma,
        args.outdir_tables, args.outdir_figs)
