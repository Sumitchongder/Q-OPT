#!/usr/bin/env bash
# Reproduce every free (non-QPU) result in the repository end-to-end.
# Usage: bash run_all.sh [quick|full]
#   quick (default): small --n-seeds, finishes in ~15-25 minutes
#   full            : paper-grade --n-seeds=30, takes considerably longer
#                     (the noise-model tests/experiments in particular are
#                     slow post-bugfix, since they now genuinely simulate
#                     device noise -- see README "Errata" section)
set -euo pipefail

MODE="${1:-quick}"
if [ "$MODE" = "full" ]; then
  SEEDS=30
  RL_EPISODES=800
else
  SEEDS=10
  RL_EPISODES=300
fi

echo "== Module self-checks =="
python -m qopt.network.topology
python -m qopt.qkd.channel
python -m qopt.pqc.hybrid_kdf
python -m qopt.threat.risk_engine
python -m qopt.optimization.qubo
python -m qopt.optimization.qubo_to_ising
python -m qopt.optimization.classical_baselines
python -m qopt.optimization.milp_baseline
python -m qopt.optimization.qaoa_qiskit
python -m qopt.optimization.rl_baseline

echo "== Unit + fake-hardware test suite (includes noise-model regression guard) =="
python -m pytest tests/ -v

echo "== Experiment 1 (E1/E4/E5): solver comparison (seeds=$SEEDS) =="
python experiments/e1_solver_comparison.py --n-seeds "$SEEDS" --n-requests 2 --k-paths 2 --topology A

echo "== Experiment 2 (E8): QBER sweep (seeds=$SEEDS) =="
python experiments/e2_qber_sweep.py --n-seeds "$SEEDS" --n-requests 4 --topology B

echo "== Experiment 3 (E9): attack-intensity sweep (seeds=$SEEDS) =="
python experiments/e3_attack_intensity.py --n-seeds "$SEEDS" --n-requests 4 --topology C

echo "== Experiment 4 (E12): Pareto front (seeds=$SEEDS) =="
python experiments/e4_pareto_front.py --n-seeds "$SEEDS" --n-requests 4 --topology B

echo "== Experiment 5 (E13): topology generalization A-E (seeds=$SEEDS) =="
python experiments/e5_topology_generalization.py --n-seeds "$SEEDS" --n-requests 2 --k-paths 2

echo "== Experiment 6: ablation study (Section 40) (seeds=$SEEDS) =="
python experiments/e6_ablation.py --n-seeds "$SEEDS" --n-requests 4 --topology C

echo "== Experiment 7 (E2/E3): exact validation n=3,6,9 + QAOA depth sweep p=1-4 (seeds=$SEEDS) =="
python experiments/e7_qaoa_depth_and_exact_validation.py --n-seeds "$SEEDS" --n-requests 2 --k-paths 2 --topology B

echo "== Experiment 8 (E14, flagship): simulated Q-TRACE -> Q-OPT closed loop (seeds=$SEEDS) =="
python experiments/e8_qtrace_closed_loop.py --n-seeds "$SEEDS" --n-requests 4 --topology C

echo "== Experiment 9 (E6): QAOA vs RL (Q-learning) (seeds=$SEEDS) =="
python experiments/e9_qaoa_vs_rl.py --n-seeds "$SEEDS" --n-requests 3 --n-episodes "$RL_EPISODES"

echo "== Experiment 10 (E7/E10/E11/E15): key depletion, link failure, QRNG scarcity, noise sweep (seeds=$SEEDS) =="
python experiments/e10_depletion_failure_scarcity_noise.py --only e7  --n-seeds "$SEEDS" --n-requests 3 --topology C
python experiments/e10_depletion_failure_scarcity_noise.py --only e10 --n-seeds "$SEEDS" --n-requests 3 --topology C
python experiments/e10_depletion_failure_scarcity_noise.py --only e11 --n-seeds "$SEEDS" --n-requests 3 --topology C
python experiments/e10_depletion_failure_scarcity_noise.py --only e15 --n-seeds "$SEEDS" --n-requests 2 --k-paths 2 --topology A

echo "== Experiment 11 (Section 17): penalty-sensitivity sweep (seeds=$SEEDS) =="
python experiments/e12_penalty_sensitivity.py --n-seeds "$SEEDS" --n-requests 3 --k-paths 1 --topology A

echo "== Done. See figures/ and tables/ for outputs. =="
echo "(Real ibm_marrakesh QPU submission is a separate, manual step -- see README section 5.)"
