# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A numerical replication of three results from Francesco Mattesini and
Johannes Wiesel, *On the Continuity of Martingale Optimal Transport in
Higher Dimensions*:

1. **Experiment 1** (`exp/exp1.py`) — qualitative continuity and the
   quantitative stability bound of Theorem 1.4 for the martingale optimal
   transport (MOT) value `V(mu, nu)` under joint W1-convergence of both
   marginals in `R^2`, using the Example 1.2 tilted-coupling base pair and a
   grid-refinement sequence.
2. **Experiment 2** (`exp/exp2.py`) — construction and verification of the
   tilted martingale coupling of Example 1.2, satisfying the uniform density
   condition `(A_kappa)`, in a bounded-support configuration and an unbounded
   (Gaussian target, truncated tilt) configuration.
3. **Experiment 3** (`exp/exp3.py`) — the Example 6.1 counterexample (Section
   6) showing that `(A_kappa)` alone, without a non-degenerate limiting
   target, does not imply continuity of `V`.

No market data is used anywhere: every measure is either an explicit
analytic distribution (uniform, Gaussian) or a finite grid/Monte Carlo
discretization of one, constructed directly from the paper's assumptions and
examples.

## Layout

```
src/     grids.py, wasserstein.py, tilted_coupling.py, mot_lp.py,
         counterexample.py — the shared library; config.py holds every
         experiment parameter as a module-level UPPER_CASE variable with a
         comment citing its paper section/equation
exp/     exp1.py, exp2.py, exp3.py — one script per experiment, calling only
         the src/ functions above
tests/   pytest unit tests for every src/ component
data/    empty package (no experiment loads external data; kept for layout
         consistency only)
results/ CSV/JSON/PNG outputs written by the experiment scripts, plus
         RESULTS.md with the paper comparison
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests (fast, no network, no `market_data` marker is used since no market
data is involved):

```bash
python -m pytest
```

Experiments (run from the repository root as `python -m exp.<name>`; each
script overwrites its files in `results/`):

```bash
python -m exp.exp1          # reduced grid sweep (N = 2..22, N_ref = 44)
python -m exp.exp1 --full   # paper configuration: N = 2..40, N_ref = 80
python -m exp.exp2
python -m exp.exp3
```

`bash reproduce.sh` runs the test suite followed by all three experiments in
their executed (non `--full`) configuration.

## Conventions

* All experiment parameters live only in `src/config.py` as module-level
  `UPPER_CASE` variables, each with a comment giving its meaning, unit, and
  the paper section/equation it is taken from (or stating that the paper
  does not specify it). No parameter is hardcoded inside `src/` or `exp/`.
* `src/` contains flat modules only (no subpackages); `exp/` scripts call
  `src/` functions with their documented signatures and do not invent extra
  keyword arguments.
* Every deviation from the paper (reduced grid ranges, parameters the paper
  leaves open, Monte Carlo recentring, etc.) is recorded only in
  `results/RESULTS.md`, never silently in code.
* No API keys are needed by this repository; if any were ever introduced
  they must be read from an uncommitted `.env`, never hardcoded.

## After a rerun

Running the experiment commands above overwrites every file in `results/`.
The numbers and verdicts in `results/RESULTS.md` are not regenerated
automatically: after a rerun, check them against the newly written
`results/*.json` and `results/*.csv` files and update `results/RESULTS.md`
by hand if they changed.
