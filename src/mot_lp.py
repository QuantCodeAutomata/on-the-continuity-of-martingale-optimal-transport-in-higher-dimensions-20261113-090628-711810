"""Discrete martingale optimal transport (MOT) linear program.

Implements the primal LP formulation of (1.1):

    V(mu, nu) = inf_{pi in Pi_M(mu, nu)} integral c d pi

for finitely supported mu, nu on R^d, following the discretization described
in the exp_1/exp_3 methodology: variables pi_{ij} >= 0, row sums equal to the
mu weights, column sums equal to the nu weights, and d coordinate-wise
martingale constraints sum_j pi_{ij} y_j[k] = mu_weights[i] * x_i[k].

The constraint matrix is built with scipy.sparse.kron so that construction
cost stays linear in the number of nonzeros rather than in a Python loop over
every atom pair; this is what makes grids with hundreds of thousands of
variables tractable with scipy.optimize.linprog(method="highs").

# scipy.optimize.linprog (HiGHS sparse interior-point/simplex solver) is used
# directly for this LP rather than a generic modeling layer (e.g. cvxpy),
# because cvxpy's Python-level expression graph does not scale to the
# O(10^5)-O(10^6) variable problems used in Experiment 1's grid sweep.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def mot_cost_matrix(mu_atoms: np.ndarray, nu_atoms: np.ndarray, cost_fn) -> np.ndarray:
    """Pairwise cost matrix cost_fn(x_i, y_j) for all atoms of mu and nu.

    Parameters
    ----------
    mu_atoms : (m, d) array of source atoms.
    nu_atoms : (n, d) array of target atoms.
    cost_fn : callable(X, Y) -> (m, n) array, vectorized over broadcasting of
        X of shape (m, 1, d) and Y of shape (1, n, d).
    """
    x = mu_atoms[:, None, :]
    y = nu_atoms[None, :, :]
    return cost_fn(x, y)


def build_mot_constraints(
    mu_atoms: np.ndarray, mu_weights: np.ndarray, nu_atoms: np.ndarray, nu_weights: np.ndarray
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Sparse equality-constraint matrix A_eq and right-hand side b_eq.

    Row block 1 (m rows): first-marginal constraint sum_j pi[i,j] = mu_weights[i].
    Row block 2 (n rows): second-marginal constraint sum_i pi[i,j] = nu_weights[j].
    Row block 3 (d*m rows): martingale constraint
        sum_j pi[i,j] * nu_atoms[j,k] = mu_weights[i] * mu_atoms[i,k], for k = 0..d-1.
    """
    m, d = mu_atoms.shape
    n = nu_atoms.shape[0]

    ones_n = sp.csr_matrix(np.ones((1, n)))
    ones_m = sp.csr_matrix(np.ones((1, m)))
    eye_m = sp.identity(m, format="csr")

    a_row = sp.kron(eye_m, ones_n, format="csr")  # (m, m*n): first-marginal block
    a_col = sp.kron(ones_m, sp.identity(n, format="csr"), format="csr")  # (n, m*n)

    mart_blocks = []
    for k in range(d):
        row_k = sp.csr_matrix(nu_atoms[:, k].reshape(1, n))
        mart_blocks.append(sp.kron(eye_m, row_k, format="csr"))  # (m, m*n)
    a_mart = sp.vstack(mart_blocks, format="csr") if mart_blocks else sp.csr_matrix((0, m * n))

    a_eq = sp.vstack([a_row, a_col, a_mart], format="csr")
    b_eq = np.concatenate(
        [mu_weights, nu_weights, (mu_weights[:, None] * mu_atoms).T.reshape(-1)]
    )
    return a_eq, b_eq


def solve_mot(
    mu_atoms: np.ndarray,
    mu_weights: np.ndarray,
    nu_atoms: np.ndarray,
    nu_weights: np.ndarray,
    cost_matrix: np.ndarray,
    return_plan: bool = False,
):
    """Solve the discrete MOT LP (1.1) and return its optimal value.

    Parameters
    ----------
    mu_atoms, nu_atoms : (m, d), (n, d) atom coordinates.
    mu_weights, nu_weights : (m,), (n,) nonnegative weights summing to 1.
    cost_matrix : (m, n) array, cost_matrix[i, j] = c(x_i, y_j).
    return_plan : if True, also return the optimal coupling pi as an (m, n) array.

    Returns
    -------
    dict with keys "value" (float), "success" (bool), "status" (str), and,
    if requested, "pi" (m, n) array.
    """
    mu_atoms = np.asarray(mu_atoms, dtype=float)
    nu_atoms = np.asarray(nu_atoms, dtype=float)
    mu_weights = np.asarray(mu_weights, dtype=float)
    nu_weights = np.asarray(nu_weights, dtype=float)
    assert mu_atoms.ndim == 2 and nu_atoms.ndim == 2, "atoms must be 2D arrays (n_atoms, dim)"
    assert mu_atoms.shape[1] == nu_atoms.shape[1], "mu and nu must live in the same dimension"
    assert np.isclose(mu_weights.sum(), 1.0, atol=1e-8), "mu_weights must sum to 1"
    assert np.isclose(nu_weights.sum(), 1.0, atol=1e-8), "nu_weights must sum to 1"

    m, n = mu_atoms.shape[0], nu_atoms.shape[0]
    assert cost_matrix.shape == (m, n), f"cost_matrix shape {cost_matrix.shape} != ({m},{n})"

    a_eq, b_eq = build_mot_constraints(mu_atoms, mu_weights, nu_atoms, nu_weights)
    c = cost_matrix.reshape(-1)

    # The discrete transportation polytope is highly degenerate (many tied
    # vertices), which makes HiGHS's default simplex path ("highs") stall on
    # grids beyond N~18. The interior-point method ("highs-ipm") does not
    # suffer from this degeneracy and solves the same problems in seconds.
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs-ipm")

    out = {"value": float(res.fun) if res.success else float("nan"),
           "success": bool(res.success), "status": res.message}
    if return_plan:
        out["pi"] = res.x.reshape(m, n) if res.success else None
    return out
