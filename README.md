# On the Continuity of Martingale Optimal Transport in Higher Dimensions

Numerical replication of selected results from:

**Francesco Mattesini and Johannes Wiesel**, *On the Continuity of Martingale
Optimal Transport in Higher Dimensions*.

The paper studies qualitative and quantitative continuity of the martingale
optimal transport (MOT) value

```
V(mu, nu) = inf_{pi in Pi_M(mu, nu)} integral c dpi
```

under joint perturbations of both marginals in dimension `d >= 2`, assuming a
uniform density bound on a reference martingale coupling (Assumption 1.1,
condition `(A_kappa)`) and, for continuity, a non-degenerate limiting target
(Assumption 1.3). This repository implements three numerical experiments that
exercise these results directly on finitely supported measures, solving the
discrete MOT linear program with `scipy.optimize.linprog` (HiGHS).

## Experiments

* **`exp.exp1`** -- Theorem 1.4: qualitative continuity `V(mu_n,nu_n) ->
  V(mu,nu)` and the quantitative stability bound
  `|V(mu_n,nu_n) - V(mu,nu)| <= L*Delta_n + C*Delta_n^(1/2)` with exponent
  `gamma = 1/2` for uniformly bounded supports, on a grid-refinement sequence
  built from the Example 1.2 tilted-coupling pair in `R^2`.
* **`exp.exp2`** -- Example 1.2: construction and verification of the tilted
  martingale coupling `pi_bar_x(dy) = (1 + <Sigma^{-1}(x-m), y-m>) nu(dy)`
  satisfying `(A_kappa)`, in a bounded-support configuration and in an
  unbounded (Gaussian target) configuration using the truncated tilt of
  Section 2.4 / Section 4.1.
* **`exp.exp3`** -- Example 6.1 (Section 6): a counterexample showing that
  `(A_kappa)` alone, without a non-degenerate limiting target, does not imply
  continuity of `V`: `lim_n V(mu_n,nu_n) = V_Q = 5/16` while the limit value
  satisfies `V(mu,nu) <= V_Q - 1/540`, an explicit discontinuity gap.

No market data is used: every experiment operates on explicitly constructed
or finitely-discretized probability measures on `R^2`, as specified by the
paper's assumptions and examples.

## Repository layout

```
src/     grids, Wasserstein-1 distance, tilted couplings, the counterexample
         primitives, the discrete MOT linear program, and config.py
exp/     exp1.py, exp2.py, exp3.py -- one script per experiment
tests/   pytest unit tests for every src/ component
data/    empty package (no experiment loads external or synthetic data)
results/ CSV/JSON/PNG outputs of the experiments and RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1          # reduced grid sweep (see "Deviations from the paper")
python -m exp.exp1 --full   # paper configuration: N = 2..40, N_ref = 80
python -m exp.exp2
python -m exp.exp3
```

Each script writes its tables, summaries and plots directly to `results/`,
overwriting the previous run; see `results/RESULTS.md` for the comparison
against the paper.

## Reproduce everything

```bash
bash reproduce.sh
```
