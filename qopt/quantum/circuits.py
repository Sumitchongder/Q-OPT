"""
Qiskit circuit construction for the two QAOA variants.

Qubit i <-> option i (blocks are contiguous). Measured bitstrings follow
Qiskit's little-endian convention: y_i = int(bitstring[-1 - i]).

XY-QAOA
  * initial state: product over blocks of a uniform-phase W state, prepared
    on a line with k-1 Givens (XX+YY) rotations and virtual phase fixes;
  * cost layer: RZ / RZZ from the *feasibility-compressed* Ising model
    (intra-block ZZ terms are identically linear on the one-hot subspace
    and are folded into RZ, which is a virtual gate on IBM hardware);
  * mixer: line (or ring) XY mixer, bonds applied even-then-odd, using
    XXPlusYYGate(2*beta) = exp(-i beta (XX+YY)/2).

Penalty QAOA
  * |+>^n, full penalised Ising cost, RX(2*beta) transverse mixer.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate

from qopt.agility.model import AgilityProblem
from qopt.quantum.fastsim import line_layers, ring_layers


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0):
    """
    y^T Q y + offset (Q upper triangular incl. diagonal, y in {0,1})
      = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + C,   with y = (1 - z) / 2.
    """
    Qu = np.triu(Q, 1)
    d = np.diag(Q)
    h = -d / 2.0 - (Qu.sum(axis=1) + Qu.sum(axis=0)) / 4.0
    J = Qu / 4.0
    C = offset + d.sum() / 2.0 + Qu.sum() / 4.0
    return h, J, C


def ising_energy_bits(h, J, C, y: np.ndarray) -> float:
    z = 1.0 - 2.0 * np.asarray(y, dtype=float)
    return float(h @ z + z @ J @ z + C)


def _w_state_line(qc: QuantumCircuit, qubits: list):
    k = len(qubits)
    qc.x(qubits[0])
    for j in range(k - 1):
        theta = 2.0 * np.arccos(np.sqrt(1.0 / (k - j)))
        qc.append(XXPlusYYGate(theta), [qubits[j], qubits[j + 1]])
    for j in range(1, k):                       # remove the (-i)^j Givens phases (virtual Z)
        qc.p(j * np.pi / 2.0, qubits[j])


def _cost_layer(qc, h, J, gamma, scale, tol=1e-12):
    n = len(h)
    for i in range(n):
        if abs(h[i]) > tol:
            qc.rz(2.0 * gamma * h[i] / scale, i)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > tol:
                qc.rzz(2.0 * gamma * J[i, j] / scale, i, j)


def xy_qaoa_circuit(prob: AgilityProblem, p: int, scale: float, mixer: str = "line",
                    measure: bool = True, compress: bool = True) -> QuantumCircuit:
    """compress=False keeps intra-block ZZ terms (only used to quantify the saving)."""
    if compress:
        Q, off = prob.qubo_feasible_compressed()
    else:
        Q, off = prob.qubo_penalized(0.0)
    h, J, _ = qubo_to_ising(Q, off)
    gam = ParameterVector("gamma", p)
    bet = ParameterVector("beta", p)
    qc = QuantumCircuit(prob.n, name=f"xyqaoa_p{p}")
    for blk in prob.blocks:
        _w_state_line(qc, list(blk))
    layer_fn = line_layers if mixer == "line" else ring_layers
    for layer in range(p):
        _cost_layer(qc, h, J, gam[layer], scale)
        for blk in prob.blocks:
            for bonds in layer_fn(len(blk)):
                for a, b in bonds:
                    qc.append(XXPlusYYGate(2.0 * bet[layer]), [blk[a], blk[b]])
    if measure:
        qc.measure_all()
    return qc


def penalty_qaoa_circuit(prob: AgilityProblem, p: int, scale: float, penalty: float,
                         measure: bool = True) -> QuantumCircuit:
    Q, off = prob.qubo_penalized(penalty)
    h, J, _ = qubo_to_ising(Q, off)
    gam = ParameterVector("gamma", p)
    bet = ParameterVector("beta", p)
    qc = QuantumCircuit(prob.n, name=f"penqaoa_p{p}")
    qc.h(range(prob.n))
    for layer in range(p):
        _cost_layer(qc, h, J, gam[layer], scale)
        for q in range(prob.n):
            qc.rx(2.0 * bet[layer], q)
    if measure:
        qc.measure_all()
    return qc


def w_state_circuit(prob: AgilityProblem) -> QuantumCircuit:
    """p = 0 control: block W states only (uniform over the feasible set)."""
    qc = QuantumCircuit(prob.n, name="wstate_p0")
    for blk in prob.blocks:
        _w_state_line(qc, list(blk))
    qc.measure_all()
    return qc


def bind(qc: QuantumCircuit, params) -> QuantumCircuit:
    """Bind [gamma_1..gamma_p, beta_1..beta_p] in that order."""
    p = len(params) // 2
    mapping = {}
    for prm in qc.parameters:
        idx = prm.index
        mapping[prm] = float(params[idx] if prm.vector.name == "gamma" else params[p + idx])
    return qc.assign_parameters(mapping)
