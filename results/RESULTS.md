# Results

Paper: **Francesco Mattesini and Johannes Wiesel, "On the Continuity of
Martingale Optimal Transport in Higher Dimensions"**.

All numbers below are taken from the files written by the latest run of
`python -m exp.exp1`, `python -m exp.exp2`, and `python -m exp.exp3`
(`results/exp1_summary.json`, `results/exp1_grid_sweep.csv`,
`results/exp2_summary.json`, `results/exp3_summary.json`,
`results/exp3_V_n.csv`, and the accompanying `.png` plots).

## Experiment 1 — Numerical Verification of Continuity and Quantitative Stability of MOT in Dimension d=2 (Theorem 1.4)

Setup: base pair (mu, nu) from Example 1.2 on R^2, nu = Uniform([-1,1]^2)
(mean 0, covariance Sigma = (1/3) I, support radius R = sqrt(2)), mu =
Uniform([-r,r]^2) with r = 0.1, giving theta = 6r = 0.6 and kappa = 1 - theta
= 0.4 (Assumption 1.1). The sequence (mu_n, nu_n) discretizes both marginals
on N x N uniform grids, and the discrete MOT linear program (1.1) is solved
with cost c(x,y) = ||x-y||_2 (L = 1-Lipschitz for |x-x'|+|y-y'|).

Executed configuration: N = 2..22 (`EXP1_N_VALUES_RUN`), reference grid
N_ref = 44 (`EXP1_N_REF_RUN`), in place of the paper's N = 2..40 with
N_ref = 80 (`EXP1_N_VALUES_PAPER`, `EXP1_N_REF_PAPER`, selectable with
`python -m exp.exp1 --full`). See "Deviations from the paper" below.

| Quantity | Paper (Theorem 1.4) | Observed (`exp1_summary.json`) | Verdict |
|---|---|---|---|
| (A_kappa) holds for every N with the fixed kappa=0.4 | required by Assumption 1.1 | `kappa_holds_all_N = true`, density always in `[0.4, 2.5]` (observed range narrows from `[0.85,1.15]` at N=2 to `[0.4533,1.5467]` at N=22) | reproduced |
| MOT LP feasible for every N (confirms convex order) | required | `lp_success_all_N = true` for all N in 2..22 | reproduced |
| Qualitative continuity V(mu_n,nu_n) -> V(mu,nu) | Theorem 1.4(i) | `abs_error` decreases monotonically from 0.03217 (N=2) to 0.000225 (N=22); `monotone_decreasing_error = true`; V_ref = 0.732206 | reproduced |
| Quantitative rate exponent gamma = 1/2 | Theorem 1.4(ii), bounded supports | bound `|V_n - V(mu,nu)| <= L*Delta_n + C*Delta_n^(1/2)` holds for every tested N with a certified constant `C = c_certificate = 0.0` (the pure linear term `L*Delta_n` alone already dominates the observed error at this grid resolution); the fitted log-log slope is `gamma_hat = 2.18`, steeper than 1/2 | partially reproduced |

`gamma_hat = 2.18` is steeper than the paper's worst-case exponent 1/2
because the observed error at these grid sizes is still dominated by the
smooth, locally linear part of the cost rather than by the worst-case
nonsmooth regime the theorem bounds; the theorem gives an upper bound on the
error, not its exact rate, and the bound (`bound_holds = true`) is satisfied
by a wide margin (certified constant `C = 0`). The qualitative result
(Theorem 1.4(i)) and the feasibility/continuity structure are fully
reproduced; the exponent 1/2 is an upper bound that this grid-refinement
sequence does not saturate, so the rate check is marked partially
reproduced. Plot: `results/exp1_convergence.png`. Table:
`results/exp1_grid_sweep.csv`.

## Experiment 2 — Construction and Verification of the Tilted Martingale Coupling Satisfying the Uniform Density Condition (Example 1.2)

Setup: bounded configuration with the same (mu, nu) as Experiment 1 on a
50x50 grid (`EXP2_BOUNDED_N_GRID`); unbounded configuration with
nu = N(0, I) and mu = Uniform(B(0, r)), using the truncation function
chi(t) = 1 on [0,1], chi(t) = 0 on [2, infinity) (Section 2.4, eq. (2.5)) and
truncation radius T = 3 (Section 4.1), with r found by a bounded search so
that the truncated-tilt density stays bounded away from zero.

| Quantity | Paper (Example 1.2 / Section 2.4) | Observed (`exp2_summary.json`) | Verdict |
|---|---|---|---|
| Bounded case: density in `[1-theta, 1+theta] = [0.4, 1.6]` | theta = 0.6 | density range `[0.4238, 1.5762]` | reproduced |
| Bounded case: martingale property `E[Y|X=x] = x` | Example 1.2 | `max_martingale_error = 1.57e-08` | reproduced |
| Bounded case: second marginal equals nu | Example 1.2 | `max_row_marginal_error = 5.4e-20`, `max_col_marginal_error = 6.6e-18` | reproduced |
| Unbounded case: `lambda_T(nu) = lambda_min(Sigma_T) > 0` for T=3 | Lemma 2.7 | `lambda_T = 0.9871 > 0` | reproduced |
| Unbounded case: truncated-tilt density bounded away from zero | Section 4.1, eq. (4.3) | density range `[0.8754, 1.1247]`, empirical `kappa = 0.8754` | reproduced |
| Unbounded case: martingale/second-marginal properties hold up to truncation and Monte Carlo error | Section 4.1 | `max_martingale_error = 2.79e-06`, `max_col_marginal_error = 7.7e-08` | reproduced |

Plot: `results/exp2_density_histogram.png`. All six checks in
`exp2_summary.json` are within the stated tolerances
(`within_tolerance = true` for both configurations).

## Experiment 3 — Counterexample: Discontinuity of V Under (A_kappa) Alone Without Non-Degeneracy (Example 6.1)

Setup: mu_n = law(X_1, eps_n/3 + t_n W), nu_n = law(Y_1, eps_n Y_1^2 + t_n Z)
with X_1, W ~ U_{1/4}, Y_1, Z ~ U_1, eps_n = 1/n, t_n = 1/n^2, cost
c(x,y) = (1+x_1)(y_1-x_1)^2, n = 2,4,8,16,32,64. The analytic limit
V_Q = 2 * integral_{-1/4}^{1/4} (1+x)(1/3-x^2) dx = 5/16 (Section 6). The
limit pair mu = U_{1/4} (x) delta_0, nu = U_1 (x) delta_0 reduces to a 1D MOT
problem, solved both exactly (LP on a 100-point grid, `EXP3_M_1D_RUN`) and
via the paper's explicit sub-optimal coupling of cost `V_Q - 1/540`.

| Quantity | Paper (Example 6.1) | Observed (`exp3_summary.json`, `exp3_V_n.csv`) | Verdict |
|---|---|---|---|
| (A_kappa) holds for every n with kappa = 1/16 | Example 6.1 | density in `[0.1268, 2.6956]` subset of `[1/16, 49/16] = [0.0625, 3.0625]` for every n | reproduced |
| V(mu_n,nu_n) -> V_Q = 5/16 = 0.3125 | Example 6.1 | V_n: 0.2824 (n=2) -> 0.2919 -> 0.2998 -> 0.3045 -> 0.3069 -> 0.3081 (n=64), monotonically increasing toward 0.3125 | reproduced |
| Explicit coupling achieves cost `V_Q - 1/540 = 0.310648` | Example 6.1 | `explicit_coupling_cost = 0.310648` | reproduced |
| V(mu,nu) <= V_Q - 1/540 | Example 6.1 | optimal `v_limit_lp = 0.273427`, which is <= 0.310648 (the LP finds the true optimum, below the paper's sub-optimal explicit-coupling upper bound) | reproduced |
| Discontinuity gap `lim_n V(mu_n,nu_n) - V(mu,nu) >= 1/540 ~ 0.00185` | Example 6.1 | `gap_observed = V_n(64) - v_limit_lp = 0.034711 >= 1/540`, about 18x the paper's proven lower bound | reproduced |

Plot: `results/exp3_convergence.png`. Table: `results/exp3_V_n.csv`. The
observed gap of 0.0347 exceeds the paper's proven lower bound of 1/540
because the paper's bound comes from one explicit, deliberately simple
coupling rather than the true optimizer; the discrete LP finds the actual
optimal value V(mu,nu), which is even lower, making the discontinuity more
pronounced than the proven bound while remaining consistent with it (a lower
bound on the gap, not its exact value).

## Deviations from the paper

* **Experiment 1 grid range reduced.** The paper's methodology for exp_1
  sweeps N = 2..40 with a reference grid N_ref = 80 (6400x6400 atoms,
  40,960,000 LP variables for the coupling matrix). The discrete MOT linear
  program has N^4 variables (an N^2 x N^2 coupling matrix) and 2N^2+2N
  equality constraints, so cost grows steeply with N: a single
  `scipy.optimize.linprog(method="highs-ipm")` solve was measured directly on
  this machine at N=22 (8.3 s), N=28 (37.6 s), and N=40 (236.9 s, peak
  resident memory 3.21 GiB for that one solve). Extrapolating this measured
  growth, N_ref=80 (16x more variables than N=40) would take on the order of
  hours and risk exceeding the memory available on a single process, which
  would exceed the time available for this replication together with the
  other two experiments and the test suite. The executed sweep uses
  N = 2..22 with N_ref = 44 (`EXP1_N_VALUES_RUN`, `EXP1_N_REF_RUN` in
  `src/config.py`); the paper's full configuration
  (`EXP1_N_VALUES_PAPER = 2..40`, `EXP1_N_REF_PAPER = 80`) is implemented and
  selectable with `python -m exp.exp1 --full`. The sample period,
  discretization method, cost function, and all formulas are unchanged; only
  the number of grid points swept is reduced. No data source is involved in
  this purely numerical experiment.
* **Experiment 1 rate exponent not saturated.** The fitted exponent
  `gamma_hat = 2.18` is steeper than the paper's worst-case bound `gamma =
  1/2`; this is expected at these grid resolutions (see the Experiment 1
  table above) and is not treated as a data or implementation defect, since
  Theorem 1.4(ii) states an upper bound on the error, which is verified to
  hold (`bound_holds = true`) rather than an exact asymptotic rate.
* **Experiment 2 unbounded-case parameters not given explicitly by the
  paper.** Example 1.2 does not specify a numerical truncation radius T or
  source radius r for the Gaussian target case; T = 3 is used, following the
  paper's own numerical illustration in Section 4.1, and r is found by a
  bounded search (`EXP2_UNBOUNDED_R_SAFETY` in `src/config.py`) so that the
  resulting truncated-tilt density stays bounded away from zero, as required
  by Lemma 2.7. The Monte Carlo sample size (`EXP2_UNBOUNDED_N_SAMPLES =
  20000`) is likewise not specified by the paper and was chosen so that the
  empirical covariance Sigma_T and density bounds are stable to the third
  decimal place.
* **Experiment 3 Monte Carlo recentring.** The discrete MOT linear program's
  equality constraints require the sampled marginals' empirical means to
  match exactly (an identity that holds exactly for the continuous laws of
  Example 6.1 but only up to O(1/sqrt(n_samples)) Monte Carlo noise for a
  finite i.i.d. sample). Each sampled marginal is recentred to its known
  theoretical mean before the LP is solved, a vanishing correction that
  preserves the n_samples -> infinity limit distribution; see the comment in
  `exp/exp3.py::run`.
* **No market data.** Every experiment operates on explicitly constructed or
  grid-discretized probability measures on R^2, following the paper's
  assumptions and examples directly; no external market-data source is used
  anywhere in this repository.

## Not reproduced

There is nothing to report: all three experiments (Experiments 1-3,
covering Theorem 1.4, Example 1.2, and Example 6.1) were executed and
compared against the paper's stated results above.
