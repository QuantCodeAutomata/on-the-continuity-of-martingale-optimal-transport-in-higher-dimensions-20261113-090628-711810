"""Tests for src/grids.py: grid discretization of uniform squares."""

from __future__ import annotations

import numpy as np
import pytest

from src.grids import continuous_uniform_square_moments, discrete_mean_cov, uniform_grid_square


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_uniform_grid_square_shapes_and_weights(n: int) -> None:
    atoms, weights = uniform_grid_square(1.0, n)
    assert atoms.shape == (n * n, 2)
    assert weights.shape == (n * n,)
    assert np.allclose(weights, 1.0 / (n * n))
    assert np.isclose(weights.sum(), 1.0)


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_uniform_grid_square_support_inside_box(n: int) -> None:
    half_width = 0.7
    atoms, _ = uniform_grid_square(half_width, n)
    assert np.all(atoms >= -half_width)
    assert np.all(atoms <= half_width)


def test_uniform_grid_square_centered_mean_is_exact() -> None:
    """The symmetric cell-midpoint placement gives empirical mean == center exactly."""
    for n in (1, 2, 3, 7):
        atoms, weights = uniform_grid_square(1.0, n, center=np.array([0.0, 0.0]))
        mean = weights @ atoms
        assert np.allclose(mean, 0.0, atol=1e-12)


def test_uniform_grid_square_respects_custom_center() -> None:
    atoms, weights = uniform_grid_square(0.5, 4, center=np.array([2.0, -3.0]))
    mean = weights @ atoms
    assert np.allclose(mean, [2.0, -3.0], atol=1e-12)


def test_uniform_grid_square_converges_to_continuous_covariance() -> None:
    """Empirical covariance of the N x N grid -> h^2/3 * I as N grows (Var(Uniform[-h,h])=h^2/3)."""
    half_width = 1.0
    errors = []
    for n in (5, 50):
        atoms, weights = uniform_grid_square(half_width, n)
        _, cov = discrete_mean_cov(atoms, weights)
        errors.append(np.max(np.abs(cov - (half_width**2 / 3.0) * np.eye(2))))
    assert errors[1] < errors[0]
    assert errors[1] < 1e-3


def test_discrete_mean_cov_matches_numpy_on_random_points() -> None:
    rng = np.random.default_rng(0)
    atoms = rng.standard_normal((500, 2))
    weights = np.full(500, 1.0 / 500)
    mean, cov = discrete_mean_cov(atoms, weights)
    assert np.allclose(mean, atoms.mean(axis=0), atol=1e-10)
    assert np.allclose(cov, np.cov(atoms.T, bias=True), atol=1e-8)


def test_continuous_uniform_square_moments_variance_formula() -> None:
    mean, cov = continuous_uniform_square_moments(2.0)
    assert np.allclose(mean, [0.0, 0.0])
    assert np.allclose(cov, (4.0 / 3.0) * np.eye(2))


def test_continuous_uniform_square_moments_custom_center() -> None:
    mean, cov = continuous_uniform_square_moments(1.0, center=np.array([1.0, 1.0]))
    assert np.allclose(mean, [1.0, 1.0])
    assert np.allclose(cov, (1.0 / 3.0) * np.eye(2))
