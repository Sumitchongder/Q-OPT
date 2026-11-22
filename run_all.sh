#!/usr/bin/env bash
# Reproduce every v2 table and figure (CPU only, no QPU time). ~25 min on one core.
# Each step is independent; long steps are split so no single command runs > 5 min.
set -euo pipefail
cd "$(dirname "$0")/experiments"
python exp01_coupling_structure.py --sizes 3 4 5 6
python exp01_coupling_structure.py --sizes 7 --skip-a --merge
python exp02_digital_twin.py
python exp03_qaoa_in_the_loop.py
python exp03_qaoa_in_the_loop.py --transfer-check
python exp04_ansatz_comparison.py
python exp06_scaling_resources.py --part a --sizes-a 4 6 7 10 15
python exp06_scaling_resources.py --part a --sizes-a 20 30
python exp06_scaling_resources.py --part b --sizes-b 2 3 4 5
python exp06_scaling_resources.py --part b --sizes-b 6 8
python exp06_scaling_resources.py --part plot
for d in ../results/hardware/run_*/; do
  [ -f "$d/raw.json" ] && python exp07_hardware_analysis.py --run-dir "$d"
done
python exp08_noise_physics.py
cd .. && python scripts/summarize_results.py
# Journal-size figures (written to figures/paper/): re-run the plotting steps with QOPT_FIGSCALE=0.85
