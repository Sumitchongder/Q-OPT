"""
Explicit QUBO -> Ising conversion.

    x_i in {0,1},  x_i = (1 - Z_i) / 2,  Z_i in {-1,+1}

    C(x) = x^T Q x  =  sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + C_0

This module derives (h, J, C_0) directly from the QUBO matrix Q so the
mapping is fully transparent and auditable (Section 18 of the design
notes), then exposes the result as a Qiskit `SparsePauliOp` cost
Hamiltonian for QAOA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass
class IsingModel:
    h: np.ndarray          # linear fields, shape (n,)
    J: np.ndarray          # coupling matrix (upper triangular populated), shape (n, n)
    C0: float               # constant offset
    n: int


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0) -> IsingModel:
    n = Q.shape[0]
    Qs = 0.5 * (Q + Q.T)  # symmetrize (Q may be given as upper-triangular-only)

    h = np.zeros(n)
    J = np.zeros((n, n))
    C0 = offset

    for i in range(n):
        C0 += Qs[i, i] / 2.0
        h[i] += -Qs[i, i] / 2.0
        for j in range(n):
            if j == i:
                continue
            C0 += Qs[i, j] / 4.0
            h[i] += -Qs[i, j] / 2.0
            if j > i:
                J[i, j] += Qs[i, j] / 2.0  # combines the (i,j) and (j,i) contribution once

    return IsingModel(h=h, J=J, C0=C0, n=n)


def ising_to_sparse_pauli_op(model: IsingModel) -> SparsePauliOp:
    """Build H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j as a SparsePauliOp."""
    n = model.n
    terms = []
    coeffs = []

    for i in range(n):
        if abs(model.h[i]) < 1e-12:
            continue
        label = ["I"] * n
        label[n - 1 - i] = "Z"  # Qiskit little-endian qubit ordering
        terms.append("".join(label))
        coeffs.append(model.h[i])

    for i in range(n):
        for j in range(i + 1, n):
            if abs(model.J[i, j]) < 1e-12:
                continue
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            label[n - 1 - j] = "Z"
            terms.append("".join(label))
            coeffs.append(model.J[i, j])

    if not terms:  # trivial all-identity fallback (degenerate tiny instance)
        terms, coeffs = ["I" * n], [0.0]

    return SparsePauliOp(terms, coeffs=np.array(coeffs, dtype=complex))


def energy_from_bitstring(model: IsingModel, bitstring: str) -> float:
    """bitstring in Qiskit order (bit 0 = rightmost char = qubit 0)."""
    n = model.n
    z = np.array([1 - 2 * int(bitstring[n - 1 - i]) for i in range(n)])  # x_i=0->Z=+1, x_i=1->Z=-1
    e = model.C0 + float(model.h @ z)
    for i in range(n):
        for j in range(i + 1, n):
            e += model.J[i, j] * z[i] * z[j]
    return e


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 4
    Q = rng.uniform(-1, 1, size=(n, n))
    Q = np.triu(Q)  # upper triangular as produced by build_qubo
    model = qubo_to_ising(Q)
    # sanity check: brute-force QUBO value should equal Ising energy for every bitstring
    for bits in range(2 ** n):
        x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        qubo_val = float(x @ Q @ x)
        bitstring = "".join(str(int(x[n - 1 - i])) for i in range(n))
        ising_val = energy_from_bitstring(model, bitstring)
        assert abs(qubo_val - ising_val) < 1e-9, (qubo_val, ising_val)
    print("QUBO <-> Ising conversion verified exactly on all 2^4 bitstrings.")
