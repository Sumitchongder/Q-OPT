"""
Hybrid key construction: combine a QKD-derived key and a PQC-derived
key into a single session key via HKDF, following the dual-PRF
hybrid-KEM combiner pattern used in RFC 9180 / draft hybrid-KEM
constructions (X-Wing-style: concatenate then HKDF-extract-and-expand
over both secrets and their context/transcript binding).

We use Python's `cryptography` package's HKDF (RFC 5869) for the
actual primitive; the *cryptographic* claim made in the paper should
be limited to "the combiner does not reduce security below either
individual scheme, given a secure HKDF and a binding context string"
per the cited combiner literature -- Q-OPT does not attempt to prove
a new security theorem, it uses an established combiner mechanism.
"""
from __future__ import annotations

import hashlib
import hmac


def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if not salt:
        salt = bytes([0] * hashlib.sha256().digest_size)
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    t = b""
    okm = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:length]


def hkdf(ikm: bytes, salt: bytes = b"", info: bytes = b"", length: int = 32) -> bytes:
    prk = _hkdf_extract(salt, ikm)
    return _hkdf_expand(prk, info, length)


def kdf_single(secret: bytes, ctx: bytes, length: int = 32) -> bytes:
    """K_Q = KDF(K_QKD, ctx) or K_P = KDF(K_PQC, ctx)."""
    return hkdf(secret, salt=b"", info=b"Q-OPT-single|" + ctx, length=length)


def hybrid_combine(k_qkd: bytes, k_pqc: bytes, ctx: bytes, length: int = 32) -> bytes:
    """
    K_H = HKDF( K_Q || K_P , ctx )

    Both inputs are first independently derived (`kdf_single`) so that
    a weakness in the raw QKD or PQC secret cannot leak structure into
    the combiner input, then concatenated and run through a second
    HKDF stage bound to the session context (transcript hash, session
    id, algorithm identifiers) to prevent cross-protocol / algorithm
    confusion attacks. This mirrors the "dual-PRF combiner" pattern
    recommended in hybrid-KEM IETF drafts.
    """
    k_q = kdf_single(k_qkd, ctx, length)
    k_p = kdf_single(k_pqc, ctx, length)
    return hkdf(k_q + k_p, salt=b"", info=b"Q-OPT-hybrid|" + ctx, length=length)


if __name__ == "__main__":
    import os
    k_qkd = os.urandom(32)
    k_pqc = os.urandom(32)
    ctx = b"session-42|ML-KEM-768|link(3,7)"
    k_h = hybrid_combine(k_qkd, k_pqc, ctx)
    print("K_H:", k_h.hex())
    assert len(k_h) == 32
    # determinism check
    assert hybrid_combine(k_qkd, k_pqc, ctx) == k_h
    print("Deterministic combiner OK.")
