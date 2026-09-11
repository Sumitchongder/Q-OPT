# Errata

This page documents real bugs found and fixed during development of this
repository, and exactly which results they did and did not affect. We keep
this history visible rather than silently rewriting it, because a
reproducibility-focused research repository should show its work — including
the mistakes and how they were caught.


**A real bug was found and fixed during development of Experiment E15 (noise-aware QAOA sweep): `qopt/optimization/qaoa_qiskit.py::run_qaoa_aer` constructed `AerSampler()` with no backend reference, which silently ignores any `NoiseModel` passed in — `qiskit_aer.primitives.SamplerV2` has no `backend` constructor argument.** Every call site that passed an explicit `noise_model` (rather than relying on the default `None`, i.e. noiseless) was affected:

1. `tests/test_qaoa_fake_marrakesh.py` — the tests claimed to validate QAOA "under FakeMarrakesh noise." They ran to completion and passed, but were **silently running noiselessly** the entire time. The tests still pass after the fix (this specific tiny instance is easy enough that noise doesn't change the outcome at 2048+ shots), but a new regression-guard test (`test_noise_model_actually_changes_sampled_distribution`) was added that fails loudly if this ever happens again.
2. `experiments/e_hw_prepare_marrakesh_job.py --use-marrakesh-noise` — the flag's stated purpose ("optimize QAOA angles against FakeMarrakesh noise before submitting to real hardware") **did not actually happen**; the angles were optimized noiselessly regardless of the flag. This does not invalidate the real `ibm_marrakesh` hardware results reported earlier (those ran on real hardware regardless of how the angles were tuned), but the "pre-adapted to device noise" framing for that step was incorrect.
3. `experiments/e10_depletion_failure_scarcity_noise.py` (E15) — this experiment could not be built correctly until the bug was found; see its docstring for the full story.

**Fix**: `sampler = AerSampler.from_backend(backend)` instead of `AerSampler()`. Verified by constructing a Bell-state circuit and confirming the noiseless path shows ~0 leakage into the "wrong" `01`/`10` outcomes while the FakeMarrakesh-noisy path shows measurable leakage (see `test_noise_model_actually_changes_sampled_distribution`).

**Everything else in this repository was unaffected**: every other experiment (E1, E2/E3 Part A validation, E4–E9, E10, E11, the ablation study, the solver comparisons) calls `run_qaoa_aer` with the default `noise_model=None`, which was already correct (a `None`-noise `AerSimulator` and a bare `AerSampler()` are computationally equivalent). Only code paths that explicitly requested noise were silently broken.

---

**A second real bug was found and fixed after a user reported a crash running `bash run_all.sh` on Linux/WSL**: several plotting scripts crashed at interpreter exit with `RuntimeError: main thread is not in main loop` / `Tcl_AsyncDelete: async handler deleted by the wrong thread`. Root cause: matplotlib auto-selects the interactive `TkAgg` backend whenever Tk/Tkinter is importable, even in a headless terminal session with no display server actually running a Tk event loop — figures got created against a "GUI" backend that was never actually driven by a GUI, and cleanup at exit crashed. **Fixed** by forcing the non-interactive `Agg` backend (`matplotlib.use("Agg")`) as the first thing that happens when `qopt` is imported, in `qopt/__init__.py` — since every experiment script imports a `qopt.*` module before it ever imports `matplotlib.pyplot`, this guarantees the backend is pinned before any plotting code runs anywhere in this codebase. `Agg` still writes `.png`/`.pdf` files exactly as before; it just never opens a (nonexistent) GUI window. Re-ran the exact command that crashed (`experiments/e7_qaoa_depth_and_exact_validation.py --skip-part-a`) after the fix with no error.

The same update also silences the two harmless upstream Qiskit `DeprecationWarning`s (`NLocal`, `BlueprintCircuit`) that appeared in every prior test run — added via `[tool.pytest.ini_options] filterwarnings` in `pyproject.toml` (pytest maintains its own internal warning-capture context that overrides plain `warnings.filterwarnings()` calls in application code, so both a package-level filter *and* a pytest-config filter were needed). `pytest -v` now reports `16 passed` with **zero warnings**.

**Practical note**: `run_all.sh` runs 10 experiment scripts back-to-back and can take 20-40+ minutes depending on `--n-seeds`; if your terminal/CI environment enforces a wall-clock limit on a single command, run the slower experiments (`e5_topology_generalization.py`, `e7_..._validation.py`, `e9_qaoa_vs_rl.py`, and `e10_..._noise.py --only e15`) as separate invocations rather than through the combined script — every experiment script is fully independent and re-runnable on its own.

---
