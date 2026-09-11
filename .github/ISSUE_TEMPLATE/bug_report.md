---
name: Bug report
about: Report something that doesn't work as documented
title: "[BUG] "
labels: bug
assignees: ""
---

**Describe the bug**
A clear description of what went wrong.

**Command run**
```bash
# the exact command, e.g.
python experiments/e1_solver_comparison.py --n-seeds 10 --n-requests 2
```

**Expected behavior**
What you expected to happen (e.g. "16 passed", a specific table shape).

**Actual behavior / traceback**
```
paste the full traceback or unexpected output here
```

**Environment**
- OS: [e.g. Ubuntu 22.04 / WSL2, Windows 11]
- Python version: `python --version`
- Qiskit version: `pip show qiskit | grep Version`
- Installed via: `pip install -r requirements.txt` / other

**Additional context**
Anything else relevant — e.g. did this happen on `quick` or `full` mode,
did it happen after pulling a specific commit, etc.

**Security note:** if this bug involves real quantum hardware
(`scripts/run_on_ibm_marrakesh.py`), please do **not** paste your IBM
Quantum token, CRN, or account-linked job IDs into this issue.
