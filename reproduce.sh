#!/usr/bin/env bash
# Runs the test suite and every experiment of this repository, writing all
# outputs to results/. Experiments use the time-budget-reduced configuration
# documented in results/RESULTS.md; pass --full to exp1 separately for the
# paper's full grid (N up to 40, N_ref = 80).
set -u

cd "$(dirname "$0")"

python -m pytest -q

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
