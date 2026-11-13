"""Tests for the tilted martingale coupling of Example 1.2 (Experiment 2).

Covers: the bounded-case density bound [1-theta, 1+theta], the second-marginal
and martingale identities (up to O(1/N^2) discretization error), the chi
truncation function, the unbounded-case Sigma_T/lambda_T estimator, and that
the truncated tilted coupling runs without error on a tiny synthetic example.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.grids import uniform_grid_square
from src.tilted_coupling import (
    bounded_tilted_coupling,
    chi_truncation,
    lambda_T,
    sigma_T,
    unbounded_tilted_coupling,
    verify_martingale_coupling,
)


def _build_bounded_pair(n: int):
    mean = np.zeros(2)
    mu_atoms, mu_weights = uniform_grid_square(config.EXP2_BOUNDED_MU_HALF_WIDTH, n, mean)
    nu_atoms, nu_weights = uniform_grid_square(config.EXP2_BOUNDED_NU_HALF_WIDTH, n, mean)
    sigma_inv = np.linalg.inv(config.EXP2_BOUNDED_NU_COV)
    pi, density = bounded_tilted_coupling(mu_atoms, mu_weights, nu_atoms, nu_weights, sigma_inv, mean)
    return mu_atoms, mu_weights, nu_atoms, nu_weights, pi, density


def test_bounded_density_within_theta_bound():
    """Density of the tilted kernel must lie in [1-theta, 1+theta] (Example 1.2)."""
    _, _, _, _, _, density = _build_bounded_pair(n=10)
    theta = config.EXP2_BOUNDED_THETA
    assert density.min() >= 1.0 - theta - 1e-9
    assert density.max() <= 1.0 + theta + 1e-9


def test_bounded_second_marginal_matches_nu():
    """The second marginal of the tilted coupling must equal nu_weights."""
    mu_atoms, mu_weights, nu_atoms, nu_weights, pi, _ = _build_bounded_pair(n=10)
    mart = verify_martingale_coupling(pi, mu_atoms, mu_weights, nu_atoms, nu_weights, atol=1e-6)
    assert mart["max_col_marginal_error"] < 1e-6


@pytest.mark.parametrize("n", [10, 20])
def test_bounded_martingale_property_holds_at_o_1_over_n2(n):
    """Martingale identity E[Y|X=x_i] = x_i holds up to O(1/N^2) discretization error."""
    mu_atoms, mu_weights, nu_atoms, nu_weights, pi, _ = _build_bounded_pair(n=n)
    mart = verify_martingale_coupling(pi, mu_atoms, mu_weights, nu_atoms, nu_weights, atol=20.0 / n**2)
    assert mart["max_martingale_error"] < 20.0 / n**2


def test_bounded_martingale_error_shrinks_with_grid_size():
    """The martingale discretization error should decrease as N increases."""
    errors = []
    for n in (10, 20):
        mu_atoms, mu_weights, nu_atoms, nu_weights, pi, _ = _build_bounded_pair(n=n)
        mart = verify_martingale_coupling(pi, mu_atoms, mu_weights, nu_atoms, nu_weights, atol=1.0)
        errors.append(mart["max_martingale_error"])
    assert errors[1] < errors[0]


def test_chi_truncation_boundary_values():
    """chi(t)=1 for t<=inner, chi(t)=0 for t>=outer, strictly between on (inner, outer)."""
    inner = config.EXP2_UNBOUNDED_CHI_INNER
    outer = config.EXP2_UNBOUNDED_CHI_OUTER
    assert chi_truncation(np.array([0.5]), inner, outer)[0] == 1.0
    mid = chi_truncation(np.array([1.5]), inner, outer)[0]
    assert 0.0 < mid < 1.0
    assert chi_truncation(np.array([3.0]), inner, outer)[0] == 0.0


def test_sigma_t_and_lambda_t_on_gaussian_samples():
    """Sigma_T on a Gaussian Monte Carlo sample should be finite, positive-definite."""
    rng = np.random.default_rng(config.SEED)
    n_samples = 2000
    mean = np.zeros(2)
    nu_atoms = rng.standard_normal((n_samples, 2))
    nu_weights = np.full(n_samples, 1.0 / n_samples)

    sigma_t = sigma_T(nu_atoms, mean, config.EXP2_UNBOUNDED_T,
                       config.EXP2_UNBOUNDED_CHI_INNER, config.EXP2_UNBOUNDED_CHI_OUTER, nu_weights)
    assert np.all(np.isfinite(sigma_t))
    assert sigma_t.shape == (2, 2)

    lam_t = lambda_T(sigma_t)
    assert np.isfinite(lam_t)
    assert lam_t > 0.0

    # Not an exact value (not specified by the paper), just a sanity check
    # that the Monte Carlo estimate is roughly isotropic (off-diagonals small
    # relative to the diagonal) for standard-normal samples.
    assert abs(sigma_t[0, 1]) < 0.2 * max(sigma_t[0, 0], sigma_t[1, 1])


def test_unbounded_tilted_coupling_runs_and_returns_finite_density():
    """unbounded_tilted_coupling must not raise and must return finite densities
    for a tiny synthetic example with mu concentrated near the mean."""
    rng = np.random.default_rng(config.SEED)
    mean = np.zeros(2)
    n_nu = 500
    nu_atoms = rng.standard_normal((n_nu, 2))
    nu_weights = np.full(n_nu, 1.0 / n_nu)

    # mu: a few points very close to the mean.
    mu_atoms = mean[None, :] + 1e-3 * rng.standard_normal((5, 2))
    mu_weights = np.full(5, 1.0 / 5)

    pi, density, sigma_t, v = unbounded_tilted_coupling(
        mu_atoms, mu_weights, nu_atoms, nu_weights, mean,
        config.EXP2_UNBOUNDED_T, config.EXP2_UNBOUNDED_CHI_INNER, config.EXP2_UNBOUNDED_CHI_OUTER,
    )

    assert pi.shape == (5, n_nu)
    assert density.shape == (5, n_nu)
    assert np.all(np.isfinite(pi))
    assert np.all(np.isfinite(density))
    # mu is essentially at the mean, so v(x) ~= 0 and density should be close to 1.
    assert np.all(np.abs(density - 1.0) < 0.5)
