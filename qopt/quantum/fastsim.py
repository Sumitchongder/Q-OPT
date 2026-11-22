"""
Exact, fast state-vector simulators for the two QAOA variants studied.

XY-QAOA (constraint preserving)
    Lives entirely in the block-one-hot subspace of dimension prod_r k_r.
    In the Hamming-weight-1 sector of one block, (XX+YY)/2 on a bond (a, b)
    acts as an excitation hop |a><b| + |b><a|, so a layer of bond gates is
    exp(-i beta A_layer) with A_layer the adjacency matrix of that layer.
    We reproduce the *Trotterised* circuit exactly: for the line mixer the
    block unitary is U_odd(beta) U_even(beta), identical to the gate order
    emitted by ``qopt.quantum.circuits.xy_qaoa_circuit``. Agreement with
    Qiskit's Statevector is enforced by tests/test_quantum.py.

Penalty QAOA (textbook baseline)
    Full 2^n space, transverse-field mixer, |+>^n initial state, one-hot
    constraints enforced only through the penalty term.

Both simulators use the same energy scale ``scale`` as the circuits, so
trained angles transfer to hardware unchanged.
"""
from __future__ import annotations

import numpy as np

from qopt.agility.model import AgilityProblem
from qopt.agility.solvers import feasible_spectrum


def line_layers(k: int):
    even = [(a, a + 1) for a in range(0, k - 1, 2)]
    odd = [(a, a + 1) for a in range(1, k - 1, 2)]
    return [layer for layer in (even, odd) if layer]


def ring_layers(k: int):
    if k <= 2:
        return line_layers(k)
    layers = line_layers(k)
    layers.append([(k - 1, 0)])
    return layers


def _layer_unitary(k: int, bonds, beta: float) -> np.ndarray:
    U = np.eye(k, dtype=complex)
    for a, b in bonds:   # bonds in one layer are disjoint, so they commute
        G = np.eye(k, dtype=complex)
        G[a, a] = G[b, b] = np.cos(beta)
        G[a, b] = G[b, a] = -1j * np.sin(beta)
        U = G @ U
    return U


def block_mixer_unitary(k: int, beta: float, mixer: str = "line") -> np.ndarray:
    layers = line_layers(k) if mixer == "line" else ring_layers(k)
    U = np.eye(k, dtype=complex)
    for layer in layers:
        U = _layer_unitary(k, layer, beta) @ U
    return U


class XYQAOASim:
    def __init__(self, prob: AgilityProblem, scale: float | None = None, mixer: str = "line"):
        self.prob = prob
        self.sizes = prob.block_sizes
        self.R = len(self.sizes)
        self.E = feasible_spectrum(prob)               # mixed-radix order, block 0 fastest
        self.scale = scale if scale is not None else float(np.std(self.E)) or 1.0
        self.Es = self.E / self.scale
        self.mixer = mixer
        self.shape = tuple(self.sizes[::-1])           # C-order reshape: last axis = block 0

    def state(self, params) -> np.ndarray:
        p = len(params) // 2
        gam, bet = params[:p], params[p:]
        psi = np.full(self.E.size, 1.0 / np.sqrt(self.E.size), dtype=complex)
        for layer in range(p):
            psi = psi * np.exp(-1j * gam[layer] * self.Es)
            t = psi.reshape(self.shape)
            for r, k in enumerate(self.sizes):
                U = block_mixer_unitary(k, bet[layer], self.mixer)
                ax = self.R - 1 - r
                t = np.moveaxis(np.tensordot(U, t, axes=([1], [ax])), 0, ax)
            psi = t.reshape(-1)
        return psi

    def probs(self, params) -> np.ndarray:
        return np.abs(self.state(params)) ** 2

    def expectation(self, params) -> float:
        return float(self.probs(params) @ self.Es)


class PenaltyQAOASim:
    def __init__(self, prob: AgilityProblem, penalty: float, scale: float):
        self.prob = prob
        n = prob.n
        if n > 20:
            raise ValueError("full-space simulation limited to n <= 20")
        self.n = n
        Q, off = prob.qubo_penalized(penalty)
        idx = np.arange(2 ** n, dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(n)) & 1).astype(float)   # bits[:, q] = y_q
        self.E = np.einsum("bi,ij,bj->b", bits, Q, bits) + off
        self.scale = scale
        self.Es = self.E / scale
        feas = np.ones(2 ** n, dtype=bool)
        for b in prob.blocks:
            feas &= bits[:, b].sum(axis=1) == 1
        self.feasible = feas
        # map feasible full-space index -> mixed-radix feasible index
        sizes = np.array(prob.block_sizes)
        radix = np.concatenate([[1], np.cumprod(sizes[:-1])])
        fidx = np.zeros(2 ** n, dtype=np.int64)
        for r, b in enumerate(prob.blocks):
            loc = np.argmax(bits[:, b], axis=1)
            fidx += loc * radix[r]
        self.feasible_index = np.where(feas, fidx, -1)

    def state(self, params) -> np.ndarray:
        p = len(params) // 2
        gam, bet = params[:p], params[p:]
        n = self.n
        psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
        for layer in range(p):
            psi = psi * np.exp(-1j * gam[layer] * self.Es)
            c, s = np.cos(bet[layer]), -1j * np.sin(bet[layer])
            t = psi.reshape([2] * n)       # axis n-1-q is qubit q
            for q in range(n):
                ax = n - 1 - q
                t0 = np.take(t, 0, axis=ax)
                t1 = np.take(t, 1, axis=ax)
                t = np.stack([c * t0 + s * t1, s * t0 + c * t1], axis=ax)
            psi = t.reshape(-1)
        return psi

    def probs(self, params) -> np.ndarray:
        return np.abs(self.state(params)) ** 2

    def expectation(self, params) -> float:
        return float(self.probs(params) @ self.Es)

    def feasible_distribution(self, params):
        """(p_feasible, distribution over the mixed-radix feasible set)."""
        pr = self.probs(params)
        D = self.prob.n_feasible
        out = np.zeros(D)
        m = self.feasible
        np.add.at(out, self.feasible_index[m], pr[m])
        return float(pr[m].sum()), out
