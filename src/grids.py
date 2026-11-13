"""Grid discretizations of uniform distributions on squares in R^2.

Used to build the sequences (mu_n, nu_n) of Experiment 1 and the bounded
configuration of Experiment 2 (exp_1 methodology, Step 2; exp_2 methodology,
Step 1): an N x N grid on [-h, h]^2 with equal weights 1/N^2, grid points
{(-h + 2h*i/N, -h + 2h*j/N) : i, j = 0, ..., N-1}.
"""

from __future__ import annotations

import numpy as np


def uniform_grid_square(half_width: float, n: int, center: np.ndarray | None = None):
    """N x N grid on [center - h, center + h]^2 with uniform weights 1/N^2.

    Grid points are placed at cell midpoints, coord_i = -h + h*(2i+1)/n,
    i = 0, ..., n-1, rather than at the cell's left corner. This is the
    symmetric realization of the equally spaced N x N grid with weights
    1/N^2 described in the exp_1/exp_2 methodology: it has the same
    structure (N points per axis, uniform weights, domain [-h,h]^2) but its
    empirical mean is exactly the center for every N, so that the discrete
    pair (mu_n, nu_n) is exactly in convex order via the Example 1.2
    construction, rather than only up to an O(1/N) bias in the mean. Both
    placements have the same O(1/N^2) W1-approximation rate to the
    continuous uniform law.

    Parameters
    ----------
    half_width : h, half the side length of the square.
    n : number of grid points per axis; total atoms = n * n.
    center : (2,) array, defaults to the origin.

    Returns
    -------
    atoms : (n*n, 2) array of grid points.
    weights : (n*n,) array, each equal to 1 / n**2.
    """
    assert n >= 1, "grid size must be positive"
    center = np.zeros(2) if center is None else np.asarray(center, dtype=float)
    idx = np.arange(n)
    coord = -half_width + half_width * (2 * idx + 1) / n
    xx, yy = np.meshgrid(coord, coord, indexing="ij")
    atoms = np.stack([xx.ravel(), yy.ravel()], axis=1) + center[None, :]
    weights = np.full(n * n, 1.0 / (n * n))
    return atoms, weights


def discrete_mean_cov(atoms: np.ndarray, weights: np.ndarray):
    """Empirical mean and covariance of a finitely supported measure."""
    mean = weights @ atoms
    centered = atoms - mean[None, :]
    cov = (centered * weights[:, None]).T @ centered
    return mean, cov


def continuous_uniform_square_moments(half_width: float, center: np.ndarray | None = None):
    """Mean and covariance of the continuous uniform law on [-h, h]^2 + center.

    Var(Uniform[-h, h]) = h^2 / 3, used to build Sigma in Example 1.2.
    """
    center = np.zeros(2) if center is None else np.asarray(center, dtype=float)
    mean = center.copy()
    cov = (half_width**2 / 3.0) * np.eye(2)
    return mean, cov
