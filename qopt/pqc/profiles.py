"""
NIST-standardized PQC resource profiles: ML-KEM (FIPS 203) for key
establishment and ML-DSA (FIPS 204) for signatures/authentication.

Per item 49 of the project design ("don't simulate PQC cryptography
unnecessarily"), Q-OPT does NOT re-implement lattice cryptography.
Instead it treats PQC as a *resource* characterised by measured
operation latency, CPU cost, bandwidth (key/ciphertext/signature
sizes) and NIST security category, exactly as the optimizer needs.

Two ways to populate this table:

1. `benchmark_live()` -- if the optional `liboqs-python` /
   `cryptography` (>=42, which ships ML-KEM support) package is
   installed, this measures real keygen/encaps/decaps/sign/verify
   timings on the local machine and returns a fresh profile table.
2. `REFERENCE_PROFILES` -- a fallback table populated with
   order-of-magnitude figures consistent with published NIST PQC
   round-3/round-4 benchmarking reports (e.g. Kannwischer et al.,
   pqm4/PQClean benchmarking suites; Open Quantum Safe project
   benchmarks). These are clearly marked as reference figures: for a
   Q1 submission you should regenerate Table 2 of the paper with
   `benchmark_live()` on your target hardware and cite that instead.

Both paths return the same schema so the rest of the pipeline
(optimization / QUBO cost terms) never has to care which was used.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict


@dataclass
class PQCProfile:
    algorithm: str
    security_category: int          # NIST PQC category (1, 3, 5)
    keygen_latency_ms: float
    encapsulation_latency_ms: float    # or "sign" for signature schemes
    decapsulation_latency_ms: float    # or "verify" for signature schemes
    public_key_bytes: int
    ciphertext_or_sig_bytes: int
    secret_key_bytes: int
    cpu_cycles_estimate: float       # relative CPU cost unit used by the optimizer
    source: str                      # "measured" or "reference"


# ---------------------------------------------------------------------------
# Reference fallback table (order-of-magnitude, literature-consistent).
# ---------------------------------------------------------------------------
REFERENCE_PROFILES: dict[str, PQCProfile] = {
    "ML-KEM-512": PQCProfile("ML-KEM-512", 1, 0.015, 0.018, 0.017, 800, 768, 1632, 1.0, "reference"),
    "ML-KEM-768": PQCProfile("ML-KEM-768", 3, 0.024, 0.028, 0.027, 1184, 1088, 2400, 1.5, "reference"),
    "ML-KEM-1024": PQCProfile("ML-KEM-1024", 5, 0.035, 0.041, 0.039, 1568, 1568, 3168, 2.1, "reference"),
    "ML-DSA-44": PQCProfile("ML-DSA-44", 2, 0.03, 0.09, 0.03, 1312, 2420, 2560, 1.8, "reference"),
    "ML-DSA-65": PQCProfile("ML-DSA-65", 3, 0.05, 0.15, 0.05, 1952, 3309, 4032, 2.6, "reference"),
    "ML-DSA-87": PQCProfile("ML-DSA-87", 5, 0.07, 0.21, 0.07, 2592, 4627, 4896, 3.4, "reference"),
}


def benchmark_live(n_iters: int = 200) -> dict[str, PQCProfile]:
    """
    Benchmark ML-KEM (keygen/encapsulation/decapsulation) AND ML-DSA
    (keygen/sign/verify) on the local machine using `liboqs-python`, if
    installed and its underlying C library successfully built. Falls
    back to REFERENCE_PROFILES with a printed warning if liboqs is not
    available -- this keeps the pipeline runnable out-of-the-box while
    still giving you a one-line upgrade path for the real paper.

    For ML-DSA, the schema's KEM-shaped field names are reused for
    convenience rather than adding signature-specific fields:
    `encapsulation_latency_ms` = sign latency,
    `decapsulation_latency_ms` = verify latency. State this explicitly
    in any table caption built from this data.
    """
    try:
        import oqs  # liboqs-python
    except ImportError:
        oqs = None

    if oqs is None:
        print("[pqc.profiles] liboqs-python not installed -- using REFERENCE_PROFILES. "
              "For Q1-grade Table 2, `pip install liboqs-python` and rerun "
              "`python -m qopt.pqc.profiles --benchmark`.")
        return REFERENCE_PROFILES

    out = {}
    for alg in ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]:
        t_kg, t_enc, t_dec = [], [], []
        for _ in range(n_iters):
            with oqs.KeyEncapsulation(alg) as kem:
                t0 = time.perf_counter()
                pk = kem.generate_keypair()
                t_kg.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                ct, ss = kem.encap_secret(pk)
                t_enc.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                kem.decap_secret(ct)
                t_dec.append(time.perf_counter() - t0)
        ref = REFERENCE_PROFILES[alg]
        out[alg] = PQCProfile(
            algorithm=alg, security_category=ref.security_category,
            keygen_latency_ms=1000 * sum(t_kg) / n_iters,
            encapsulation_latency_ms=1000 * sum(t_enc) / n_iters,
            decapsulation_latency_ms=1000 * sum(t_dec) / n_iters,
            public_key_bytes=ref.public_key_bytes,
            ciphertext_or_sig_bytes=ref.ciphertext_or_sig_bytes,
            secret_key_bytes=ref.secret_key_bytes,
            cpu_cycles_estimate=ref.cpu_cycles_estimate,
            source="measured",
        )
    for alg in ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]:
        ref = REFERENCE_PROFILES[alg]
        t_kg, t_sign, t_verify = [], [], []
        message = b"Q-OPT PQC benchmark message payload, fixed length for comparability."
        for _ in range(n_iters):
            with oqs.Signature(alg) as signer:
                t0 = time.perf_counter()
                pk = signer.generate_keypair()
                t_kg.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                sig = signer.sign(message)
                t_sign.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                # Verification can be done with a fresh Signature object (no
                # secret key needed) or the same one; liboqs-python supports
                # verify() on the signer instance directly.
                signer.verify(message, sig, pk)
                t_verify.append(time.perf_counter() - t0)
        out[alg] = PQCProfile(
            algorithm=alg, security_category=ref.security_category,
            keygen_latency_ms=1000 * sum(t_kg) / n_iters,
            encapsulation_latency_ms=1000 * sum(t_sign) / n_iters,   # "sign" for signature schemes
            decapsulation_latency_ms=1000 * sum(t_verify) / n_iters,  # "verify" for signature schemes
            public_key_bytes=ref.public_key_bytes,
            ciphertext_or_sig_bytes=ref.ciphertext_or_sig_bytes,
            secret_key_bytes=ref.secret_key_bytes,
            cpu_cycles_estimate=ref.cpu_cycles_estimate,
            source="measured",
        )
    return out


def profiles_to_dataframe(profiles: dict[str, PQCProfile]):
    import pandas as pd
    return pd.DataFrame([asdict(p) for p in profiles.values()])


if __name__ == "__main__":
    import sys
    profiles = benchmark_live() if "--benchmark" in sys.argv else REFERENCE_PROFILES
    df = profiles_to_dataframe(profiles)
    print(df.to_string(index=False))
