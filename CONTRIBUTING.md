# Contributing to Q-OPT

Thanks for your interest in this project. This repository accompanies an
academic research program, so contributions are especially welcome in
these areas:

- Filling in the not-yet-implemented experiments listed in
  `docs/experiment_specs.md` (E2 full sweep variants, dataset generation
  at scale, PPO-based RL baseline, etc.)
- Extending `qopt/pqc/profiles.py::benchmark_live()` with additional
  measured algorithms
- Improving numerical stability, adding new topologies, or adding new
  attack models
- Bug reports and fixes (see `README.md`'s Errata section for the kind
  of issue we take seriously — two real bugs were found and fixed
  during this project's own development, and we'd rather find the next
  one via a clear bug report than by accident)

## Getting Started

```bash
git clone https://github.com/<your-username>/qopt.git
cd qopt
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m pytest tests/ -v
```

All 16 tests should pass with zero warnings before you start making
changes. If they don't, please open an issue rather than working around
it locally.

## Development Workflow

1. **Fork** the repository and create a feature branch:
   `git checkout -b feature/my-change`
2. **Make your change.** Keep functions documented the way the existing
   codebase is — every non-trivial design decision in this repo has a
   docstring explaining *why*, not just *what* (see any file under
   `qopt/optimization/` for the style).
3. **Add or update tests.** `tests/test_core.py` for anything that
   doesn't touch a noise model; `tests/test_qaoa_fake_marrakesh.py` for
   anything that does. Both must keep passing.
4. **Run the full test suite** before opening a PR:
   ```bash
   python -m pytest tests/ -v
   ```
5. **If you add a new experiment script**, follow the existing pattern
   in `experiments/`: a CLI with `argparse`, seeded randomness via
   `numpy.random.default_rng`, output written to `tables/*.csv` and
   `figures/*.{png,pdf}`, and a one-paragraph module docstring stating
   what the script demonstrates and any known caveats.
6. **Open a pull request** with a clear description of what changed and
   why. Link any relevant issue.

## Reporting Bugs

Open a GitHub issue with:
- The exact command you ran
- The full traceback or unexpected output
- Your environment (`python --version`, `pip freeze | grep qiskit`)

If the bug involves quantum hardware execution, please do **not** include
your IBM Quantum token, CRN, or job IDs tied to your account in the issue.

## Code Style

- Standard PEP 8, no strict formatter is enforced yet (see
  `requirements-dev.txt` for optional `black`/`ruff` if you want to run
  them locally).
- Prefer explicit, documented trade-offs over silent workarounds — this
  codebase's docstrings explain several deliberate simplifications
  (e.g. path-based rather than full flow-conservation routing in
  `qopt/optimization/qubo.py`); please keep that standard.

## Questions

Open a GitHub Discussion or issue — see `README.md` for the project
overview and `docs/experiment_specs.md` for what's implemented vs. not.
