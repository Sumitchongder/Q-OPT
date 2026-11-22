# Methods

Notation: requests r = 1..R, options o = (path, mode) with mode in {QKD, PQC, HYB};
k_r options per request; binary y_{r,o}; block-one-hot feasibility
sum_o y_{r,o} = 1. Implementation: `qopt/agility/model.py`.

## 1. Physical layer

**Key rate.** Decoy-state BB84, infinite-decoy limit (Lo, Ma, Chen, PRL 94, 230504, 2005;
GLLP, QIC 4, 325, 2004):

    R = q [ -Q_mu f_EC h2(E_mu) + Q_1 (1 - h2(e_1)) ],
    Y_1 = Y_0 + eta - Y_0 eta,  Q_1 = Y_1 mu e^{-mu},  e_1 = (e_0 Y_0 + e_d (Y_1 - Y_0)) / Y_1,

eta = 10^{-alpha L/10} eta_d. An adversarial disturbance delta is added to E_mu and e_1;
with mu = 0.5 the rate reaches zero near 6 % QBER, so sustained attacks starve the link.
The operational rate is R x pulse rate x `key_rate_scale` (system efficiency, duty cycle,
finite-size overhead). Link budget per epoch: B_e = pool_e + rate_e x tau.

**Link exposure.** rho_e = 1 - (1 - a_e (1 - 0.8 d_e)) (1 - 0.02 (1 - trust_e)),
d_e = min(QBER_e / 0.11, 1): residual exposure is the attack probability times the
undetected fraction, plus a control-plane trust term. This is a proxy, stated as such.

## 2. Option attributes

| | QKD over path P | PQC (ML-KEM-768) | HYB |
|---|---|---|---|
| risk | rho_Q = 1 - prod_e (1-rho_e) prod_{n in int P} (1-rho_TN) | rho_PQC | rho_Q rho_PQC |
| key bits on each e in P | d_r | 0 | d_r |
| CPU at s, d | 0 | h_r | h_r |
| QRNG bits at s, d | 0 | 512 h_r | 512 h_r |
| latency | prop + relay x hops | prop + handshake | prop + max(relay x hops, handshake) + combine |

The HYB rule is the combiner property of dual-PRF / concatenation KDFs
(NIST SP 800-56C r2, ETSI TS 103 744): the session key is exposed only if both inputs are,
assuming independent failures.

## 3. Objective

    E(y) = sum_i c_i y_i + y^T W y,

    c_i = w_R rho_i / tau_r + w_L [ l_i/L_r + 4 max(l_i/L_r - 1, 0) ] + w_S max(rho_i/tau_r - 1, 0),

    W = w_K sum_e a_e a_e^T + w_C sum_n b_n b_n^T + w_Q sum_n g_n g_n^T,
    (a_e)_i = key_i,e / B_e,  (b_n)_i = cpu_i,n / Cap_n,  (g_n)_i = rng_i,n / Ent_n.

tau_r is the class tolerance (critical 0.004, high 0.015, normal 0.05, best-effort 0.25),
L_r the latency SLA. Weights (`DEFAULT_WEIGHTS`): w_R=1, w_L=0.25, w_S=4, w_K=1.5, w_C=0.5,
w_Q=0.5, fixed a priori and not tuned on evaluation seeds. Off-diagonal entries of W
between *different* requests are what makes the problem non-separable; v0.1 had none.

## 4. Classical solvers (`qopt/agility/solvers.py`)

Exact mixed-radix enumeration (vectorised), separable (v0.1 rule), tiered rule-based
controller (QBER <= 3 % and budget headroom -> QKD/HYB by class, <= 5.5 % -> HYB,
else PQC), sequential coupled greedy (critical first), block-move simulated annealing
(geometric schedule, restarts, final descent) and tabu search (best improvement,
aspiration), both with O(k) incremental local-field updates.

## 5. Digital twin (`qopt/agility/twin.py`)

Per epoch: Markov attack process per link (on 0.08, off 0.25), link denial (0.01);
LMC05 rates recomputed; R requests drawn from streams keyed by (seed, epoch) only
(common random numbers); demand normalised to nominal supply so that it does not
depend on the policy's pool state. Execution in priority order with **hard** budgets:
a QKD/HYB request without sufficient key on any hop is downgraded to PQC; CPU overload
adds queueing delay; QRNG shortfall forces a DRBG fallback (risk + 0.5 rho_PQC).
Compliance = realised risk <= class tolerance AND latency <= SLA.

## 6. Constraint-preserving QAOA (`qopt/quantum/`)

    |psi(gamma, beta)> = prod_l U_M(beta_l) e^{-i gamma_l H_C / s} (⊗_r |W_{k_r}>)

* **Initial state.** Per block, X on the first qubit, k-1 Givens rotations
  XXPlusYY(theta_j) with cos^2(theta_j/2) = 1/(k-j) along a line, and virtual
  phase gates P(j pi/2) that remove the Givens phases: an exact uniform W state.
* **Mixer.** U_M = prod_r U_odd,r(beta) U_even,r(beta), each bond XXPlusYY(2 beta) =
  exp(-i beta (XX+YY)/2). It acts as excitation hopping and conserves Hamming weight per
  block, so the state never leaves the feasible subspace (verified to < 1e-12).
* **Feasibility-compressed cost.** On the one-hot subspace y_i y_j = 0 within a block,
  so intra-block ZZ terms are dropped and their contribution folded into the linear
  (RZ, virtual) terms. H_C equals E(y) up to a constant on every feasible state.
* **Scale.** s = standard deviation of the feasible spectrum, so angles transfer
  across instances (E3).
* **Exact simulation.** XY-QAOA is simulated in the prod_r k_r-dimensional subspace by
  applying k x k block unitaries along tensor axes; penalty QAOA in the full 2^n space.
  Both agree with Qiskit `Statevector` to machine precision (`tests/test_v2.py`).
* **Training.** p=1 from a 24 x 24 grid + L-BFGS-B; p+1 from INTERP (Zhou et al.,
  PRX 10, 021067, 2020) + perturbed restarts; depth p+1 is never worse than p.

**Penalty baseline.** |+>^n, full penalised Ising cost with P = c x (E_max - E_min),
RX(2 beta) mixer. Reported both at c = 0.5 (hardware) and at the per-instance oracle
best c in {0.25, 0.5, 1, 2}.

## 7. Metrics

For feasible energies in [E_min, E_max]:
r_feas = (E_max - <E>_feas) / (E_max - E_min) (1 = optimal; the uniform-feasible value
r_rand is the floor); p_feas; r_eff = p_feas r_feas (infeasible shots scored as the worst
feasible assignment); p_opt; mean percentile. Hardware uncertainty: multinomial bootstrap
over shots (2 000 resamples), one-sided bootstrap p-values reported as (k+1)/(B+1).
Symmetry verification = discarding shots that violate block one-hot; greedy repair is
reported separately and never merged with raw numbers.

## 8. Statistics

Seed-level 95 % percentile-bootstrap CIs; paired two-sided Wilcoxon signed-rank tests
between policies on the same seeds; Holm-Bonferroni correction across baselines within
each sweep point.
