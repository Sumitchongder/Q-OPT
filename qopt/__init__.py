"""
Q-OPT: Quantum-Optimized Crypto-Agility for Attack-Resilient QKD-PQC-QRNG Networks
====================================================================================

Top-level package. See README.md at the repository root for the full
project description, roadmap and reproducibility instructions.

Subpackages
-----------
network        Graph topology generators and traffic model
qkd            QKD channel physics (QBER, loss, key rate), key pools, attacks
pqc            NIST ML-KEM / ML-DSA resource profiles (benchmarked, not re-implemented)
qrng           QRNG resource abstraction
threat         Risk engine (Q-TRACE-style threat probability -> rho_ij)
optimization   QUBO builder, MILP / SA / greedy / RL baselines, QAOA (Qiskit)
"""

# --- Force a non-interactive matplotlib backend BEFORE anything else in this
# package (or any script that imports qopt) can import matplotlib.pyplot. ---
#
# Found in the wild: running experiments/e7_qaoa_depth_and_exact_validation.py
# (and other plotting scripts) via `bash run_all.sh` on Linux/WSL crashed with
#     RuntimeError: main thread is not in main loop
#     Tcl_AsyncDelete: async handler deleted by the wrong thread
# at interpreter exit. Root cause: matplotlib auto-selects the interactive
# TkAgg backend whenever Tk/Tkinter is importable, even in a headless
# terminal session with no display server actually running a Tk main loop.
# Every experiment script imports a `qopt.*` module before it ever imports
# matplotlib.pyplot (matplotlib is only imported lazily inside each script's
# _plot()/_make_figures() function), so setting the backend here -- the
# first line of code that runs anywhere in this package -- guarantees
# matplotlib.pyplot is never imported anywhere in this codebase before the
# backend is pinned to the safe, non-interactive "Agg" renderer. Agg still
# writes .png/.pdf files perfectly; it just never opens a GUI window, which
# is exactly what a headless script that only calls savefig() needs.
import matplotlib
matplotlib.use("Agg")

# --- Silence known-harmless, upstream-only deprecation warnings ---
#
# QAOAAnsatz (qiskit.circuit.library) internally builds on NLocal/
# BlueprintCircuit, both deprecated as of Qiskit 2.1 for removal in Qiskit
# 3.0. This is entirely inside Qiskit's own implementation -- nothing in
# this repository calls NLocal or BlueprintCircuit directly, so there is
# no local code change that removes the warning; it will disappear on its
# own once Qiskit ships a non-deprecated QAOAAnsatz internals or this repo
# upgrades past whichever qiskit release removes the old implementation.
# Filtered here (not globally via -W flags) so it's scoped to importing
# this package rather than silencing deprecation warnings project-wide.
import warnings
warnings.filterwarnings(
    "ignore", category=DeprecationWarning,
    message=r".*NLocal.*deprecated.*|.*BlueprintCircuit.*deprecated.*",
)

__version__ = "0.1.0"
