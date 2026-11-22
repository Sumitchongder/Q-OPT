"""
QKD channel physics model.

We use a transparent, widely-used decoy-state BB84 operational model
(loss -> transmittance -> gain/QBER -> GLLP-type asymptotic secret key
rate). This is a standard textbook/paper-level abstraction (see e.g.
Lo, Ma & Chen 2005; Ma et al. 2005 decoy-state analysis) used here as
an *operational resource model*, not a claim of a new physical result.

All functions are deterministic given parameters + an explicit RNG,
so experiments are reproducible from a seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def h2(x: np.ndarray | float) -> np.ndarray | float:
    """Binary entropy function, safe at 0 and 1."""
    x = np.clip(np.asarray(x, dtype=float), 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


@dataclass
class ChannelParams:
    """Physical parameters of one QKD link (fixed per-link, per-scenario)."""
    fiber_loss_db_per_km: float = 0.2      # standard telecom fiber
    detector_efficiency: float = 0.65       # eta_d
    dark_count_rate: float = 1e-6           # per pulse, p_dark
    intrinsic_qber: float = 0.01            # e_detector (misalignment etc.)
    mean_photon_number: float = 0.5         # mu (signal state)
    pulse_rate_hz: float = 1.0e9            # source rep rate (1 GHz typical)
    post_processing_efficiency: float = 1.0  # f_ec >= 1 in denominator convention below
    error_correction_inefficiency: float = 1.16  # f_EC (Cascade/LDPC typical 1.1-1.2)


def transmittance(length_km: float, fiber_loss_db_per_km: float) -> float:
    """Channel transmittance eta_channel = 10^(-alpha*L/10)."""
    return float(10 ** (-fiber_loss_db_per_km * length_km / 10.0))


def gain_and_qber(length_km: float, p: ChannelParams,
                   qber_shift: float = 0.0) -> tuple[float, float]:
    """
    Compute the overall detection gain Q_mu and QBER E_mu for signal state mu,
    following the standard decoy-state BB84 detector model:

        eta = eta_channel * eta_d
        Y_0 = 2 * p_dark            (background/dark-count yield, both detectors)
        Q_mu = Y_0 + 1 - exp(-mu*eta)          (approx. gain)
        E_mu = [ e_0 * Y_0 + e_detector * (1 - exp(-mu*eta)) ] / Q_mu

    ``qber_shift`` is an additive attack/degradation term (e.g. Eve's
    intervention or detector aging), applied post-hoc and clipped to [0, 0.5].
    """
    eta_ch = transmittance(length_km, p.fiber_loss_db_per_km)
    eta = eta_ch * p.detector_efficiency
    Y0 = 2 * p.dark_count_rate
    signal_term = 1 - np.exp(-p.mean_photon_number * eta)
    Q_mu = Y0 + signal_term
    e0 = 0.5  # dark counts are random -> 50% error
    E_mu = (e0 * Y0 + p.intrinsic_qber * signal_term) / max(Q_mu, 1e-15)
    E_mu = float(np.clip(E_mu + qber_shift, 1e-6, 0.5 - 1e-6))
    return float(Q_mu), E_mu


def secret_key_rate_bits_per_pulse(Q_mu: float, E_mu: float, p: ChannelParams,
                                    decoy_state_efficiency: float = 0.9) -> float:
    """
    GLLP-type asymptotic secret key rate per pulse (decoy-state BB84, simplified):

        R = q * decoy_efficiency * Q_mu * [1 - f_EC * h2(E_mu) - h2(E_mu)]

    where q = 1/2 accounts for the sifting factor (basis reconciliation),
    and ``decoy_state_efficiency`` in (0,1] approximates the fraction of
    the ideal single-photon yield recovered by finite decoy-state
    statistics. This is a standard simplified/pedagogical form of the
    decoy-state key rate bound (cf. Lo-Ma-Chen 2005), sufficient as a
    resource model for network-level optimization (it is NOT used as a
    novel security proof in this project).
    """
    q = 0.5
    R = q * decoy_state_efficiency * Q_mu * (1 - p.error_correction_inefficiency * h2(E_mu) - h2(E_mu))
    return float(max(R, 0.0))


def secret_key_rate_lmc05(length_km: float, p: ChannelParams,
                          qber_shift: float = 0.0) -> tuple[float, float, float]:
    """
    Asymptotic decoy-state BB84 secret-key rate per pulse in the
    infinite-decoy limit (Lo, Ma & Chen, PRL 94, 230504 (2005); GLLP,
    QIC 4, 325 (2004)):

        R = q { -Q_mu f_EC h2(E_mu) + Q_1 [1 - h2(e_1)] }

        Y_1 = Y_0 + eta - Y_0 eta            single-photon yield
        Q_1 = Y_1 mu exp(-mu)                single-photon gain
        e_1 = [e_0 Y_0 + e_d (Y_1 - Y_0)] / Y_1

    ``qber_shift`` models an adversarial / degradation disturbance; it is
    applied to *both* E_mu and e_1 (an intercept-resend adversary acting on
    single-photon pulses raises e_1 one-for-one, which is the conservative
    choice for the defender).

    Returns (R_per_pulse, Q_mu, E_mu).
    """
    eta = transmittance(length_km, p.fiber_loss_db_per_km) * p.detector_efficiency
    Y0 = 2.0 * p.dark_count_rate
    mu = p.mean_photon_number
    e0, ed = 0.5, p.intrinsic_qber
    Q_mu = Y0 + 1.0 - np.exp(-mu * eta)
    E_mu = (e0 * Y0 + ed * (1.0 - np.exp(-mu * eta))) / max(Q_mu, 1e-15)
    Y1 = Y0 + eta - Y0 * eta
    Q1 = Y1 * mu * np.exp(-mu)
    e1 = (e0 * Y0 + ed * (Y1 - Y0)) / max(Y1, 1e-15)
    E_mu = float(np.clip(E_mu + qber_shift, 1e-9, 0.5))
    e1 = float(np.clip(e1 + qber_shift, 1e-9, 0.5))
    q = 0.5  # BB84 sifting (standard, non-efficient basis choice)
    R = q * (-Q_mu * p.error_correction_inefficiency * h2(E_mu) + Q1 * (1.0 - h2(e1)))
    return float(max(R, 0.0)), float(Q_mu), E_mu


def key_generation_rate_bps(length_km: float, p: ChannelParams,
                             qber_shift: float = 0.0,
                             decoy_state_efficiency: float = 0.9,
                             model: str = "lmc05") -> tuple[float, float, float]:
    """
    Returns (rate_bps, Q_mu, E_mu) for a link of given length.

    model="lmc05"      Lo-Ma-Chen infinite-decoy GLLP bound (default, v2.0+).
    model="simplified" the v0.1 pedagogical formula, kept only so that
                       legacy results under ``legacy/v1`` stay reproducible.
    """
    if model == "lmc05":
        R_per_pulse, Q_mu, E_mu = secret_key_rate_lmc05(length_km, p, qber_shift)
        return R_per_pulse * p.pulse_rate_hz, Q_mu, E_mu
    Q_mu, E_mu = gain_and_qber(length_km, p, qber_shift=qber_shift)
    R_per_pulse = secret_key_rate_bits_per_pulse(Q_mu, E_mu, p, decoy_state_efficiency)
    return R_per_pulse * p.pulse_rate_hz, Q_mu, E_mu


if __name__ == "__main__":
    p = ChannelParams()
    for L in [5, 20, 40, 60, 80, 100]:
        rate, Q, E = key_generation_rate_bps(L, p)
        print(f"L={L:5.1f} km  Q_mu={Q:.3e}  QBER={E:.4f}  rate={rate/1e3:9.3f} kbps")
