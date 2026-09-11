# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this repository (e.g. a way the code
could leak credentials, an unsafe dependency, or an issue that could
cause `scripts/run_on_ibm_marrakesh.py` to submit unintended jobs to real
quantum hardware), please **do not open a public issue**. Instead:

1. Open a [GitHub Security Advisory](../../security/advisories/new) on
   this repository, or
2. Email the maintainer directly (see repository owner profile).

Please include:
- A description of the issue and its potential impact
- Steps to reproduce
- Any suggested fix, if you have one

We aim to acknowledge reports within 5 business days.

## Scope Notes Specific to This Project

- **No credentials are stored in this repository.** IBM Quantum
  API tokens and CRNs are never committed; every code path that
  needs them (`scripts/run_on_ibm_marrakesh.py`) reads them from your
  local `QiskitRuntimeService` account configuration, which you set up
  yourself and which lives outside this repository (see
  `README.md` Section 5). If you ever find a real token or CRN
  committed anywhere in this repository's history, please report it
  immediately via the channel above — it should never happen, and if
  it does, treat it as a valid security report.
- `scripts/run_on_ibm_marrakesh.py` is intentionally isolated from the
  rest of the pipeline and includes a pre-submission time-budget
  check specifically so that no other script in this repository can
  accidentally submit a job to real quantum hardware. If you find a
  code path that bypasses this isolation, please report it.
