"""Parameters for the numerical experiments.

Every experiment parameter lives here as a module-level UPPER_CASE variable,
each documented with its meaning, unit, and the section/equation of
"On the Continuity of Martingale Optimal Transport in Higher Dimensions"
(Mattesini & Wiesel) from which it is taken. Where a value is not fixed by
the paper, the comment says so explicitly.
"""

import numpy as np

SEED = 0  # global RNG seed for reproducibility (not specified in the paper)

# ---------------------------------------------------------------------------
# Experiment 1: continuity and quantitative stability (Theorem 1.4, d = 2)
# ---------------------------------------------------------------------------
EXP1_D = 2  # ambient dimension, d >= 2 (Theorem 1.4)
EXP1_NU_MEAN = np.zeros(2)  # mean m of the target nu (Example 1.2)
EXP1_NU_HALF_WIDTH = 1.0  # nu = uniform on [-1,1]^2 (Example 1.2 / exp_1 data_requirements)
EXP1_NU_COV = (1.0 / 3.0) * np.eye(2)  # covariance Sigma of uniform([-1,1]^2): Var(U[-1,1]) = 1/3
EXP1_NU_RADIUS_R = np.sqrt(2.0)  # R such that supp(nu) subset B(0,R): corner distance of [-1,1]^2
EXP1_MU_HALF_WIDTH_R = 0.1  # r: mu = uniform on [-r,r]^2 (exp_1 data_requirements)
EXP1_THETA = 6.0 * EXP1_MU_HALF_WIDTH_R  # theta = 6r = R|Sigma^-1(x-m)| bound (exp_1 methodology)
EXP1_KAPPA = 1.0 - EXP1_THETA  # kappa = 1 - theta from Example 1.2, kappa = 0.4
EXP1_COST_L = 1.0  # Lipschitz constant L of c(x,y)=||x-y||_2 for |x-x'|+|y-y'| (exp_1 data_requirements)

# Grid sizes N_n = 2,...,40 used to discretize (mu_n, nu_n) (exp_1 data_requirements).
EXP1_N_VALUES_PAPER = list(range(2, 41))
# Reduced sweep actually executed; see "Deviations from the paper" in results/RESULTS.md.
EXP1_N_VALUES_RUN = list(range(2, 23))

EXP1_N_REF_PAPER = 80  # reference grid size N_ref for the "exact" V(mu,nu) (exp_1 data_requirements)
EXP1_N_REF_RUN = 44  # reference grid size actually executed (reduced to fit the available time)

# ---------------------------------------------------------------------------
# Experiment 2: tilted martingale coupling (Example 1.2, Section 2.4, Section 4.1)
# ---------------------------------------------------------------------------
EXP2_D = 2  # ambient dimension (Example 1.2)
EXP2_BOUNDED_N_GRID = 50  # N x N grid size for configuration 1 (exp_2 methodology, "Use N=50")
EXP2_BOUNDED_NU_HALF_WIDTH = 1.0  # nu = uniform on [-1,1]^2 (exp_2 data_requirements)
EXP2_BOUNDED_MU_HALF_WIDTH = 0.1  # mu = uniform on [-0.1,0.1]^2 (exp_2 data_requirements)
EXP2_BOUNDED_NU_COV = (1.0 / 3.0) * np.eye(2)  # Sigma = Cov(uniform[-1,1]^2) = (1/3) I
EXP2_BOUNDED_NU_RADIUS_R = np.sqrt(2.0)  # R, corner distance of [-1,1]^2 (exp_2 data_requirements)
EXP2_BOUNDED_THETA = 6.0 * EXP2_BOUNDED_MU_HALF_WIDTH  # theta = 6r = 0.6 (exp_2 data_requirements)
EXP2_BOUNDED_KAPPA = 1.0 - EXP2_BOUNDED_THETA  # kappa = 0.4 (exp_2 expected_outcomes)

EXP2_UNBOUNDED_N_SAMPLES = 20000  # Monte Carlo sample size for Gaussian nu (not specified; chosen
# large enough that the empirical Sigma_T and density bounds are stable to the third decimal)
EXP2_UNBOUNDED_T = 3.0  # truncation radius T for the Gaussian case (exp_2 methodology, "T=3")
EXP2_UNBOUNDED_CHI_INNER = 1.0  # chi(t) = 1 for t in [0,1] (Section 2.4, eq. (2.5))
EXP2_UNBOUNDED_CHI_OUTER = 2.0  # chi(t) = 0 for t in [2,infinity) (Section 2.4, eq. (2.5))
EXP2_UNBOUNDED_R_SAFETY = 0.5  # safety factor (<1) applied to the theoretical bound on r so that
# the truncated tilt density stays bounded away from zero (not specified in the paper)

# ---------------------------------------------------------------------------
# Experiment 3: Example 6.1 counterexample (Section 6)
# ---------------------------------------------------------------------------
EXP3_KAPPA = 1.0 / 16.0  # kappa for which every pair (mu_n, nu_n) satisfies (A_kappa) (Example 6.1)
EXP3_U_MU_HALF_WIDTH = 0.25  # X_1, W ~ U_{1/4} = Uniform[-1/4, 1/4] (Example 6.1)
EXP3_U_NU_HALF_WIDTH = 1.0  # Y_1, Z ~ U_1 = Uniform[-1, 1] (Example 6.1)
EXP3_N_VALUES = [2, 4, 8, 16, 32, 64]  # sequence index n tested (exp_3 data_requirements)
EXP3_N_SAMPLES_PAPER = 200  # Monte Carlo sample size per marginal (exp_3 implementation_notes)
EXP3_N_SAMPLES_RUN = 200  # sample size actually executed (kept at the paper's suggested value)
EXP3_M_1D_PAPER = 100  # quadrature points for the 1D limit problem (exp_3 implementation_notes)
EXP3_M_1D_RUN = 100  # grid size actually executed for the 1D limit problem
EXP3_V_Q = 5.0 / 16.0  # analytic limit value V_Q = integral (1+x)(1/3-x^2) U_{1/4}(dx) (Section 6)
EXP3_GAP = 1.0 / 540.0  # the proven discontinuity gap lim_n V(mu_n,nu_n) - V(mu,nu) >= 1/540 (Sec. 6)
