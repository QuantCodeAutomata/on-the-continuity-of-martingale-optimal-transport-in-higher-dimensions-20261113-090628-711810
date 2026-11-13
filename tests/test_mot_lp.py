"""Tests for src/mot_lp.py: the discrete martingale optimal transport LP."""

from __future__ import annotations

import numpy as np
import pytest

from src.grids import uniform_grid_square
from src.mot_lp import build_mot_constraints, mot_cost_matrix, solve_mot


def _euclidean_cost(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - y, axis=-1)


def test_mot_cost_matrix_shape_and_values() -> None:
    mu_atoms = np.array([[0.0, 0.0], [1.0, 1.0]])
    nu_atoms = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    cost = mot_cost_matrix(mu_atoms, nu_atoms, _euclidean_cost)
    assert cost.shape == (2, 3)
    assert cost[0, 0] == pytest.approx(0.0)
    assert cost[0, 1] == pytest.approx(2.0)
    assert cost[1, 0] == pytest.approx(np.sqrt(2.0))


def test_build_mot_constraints_shape() -> None:
    mu_atoms = np.array([[0.0, 0.0], [1.0, 0.0]])
    mu_weights = np.array([0.5, 0.5])
    nu_atoms = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    nu_weights = np.array([1 / 3, 1 / 3, 1 / 3])
    a_eq, b_eq = build_mot_constraints(mu_atoms, mu_weights, nu_atoms, nu_weights)
    m, n, d = 2, 3, 2
    assert a_eq.shape == (m + n + d * m, m * n)
    assert b_eq.shape == (m + n + d * m,)


def test_solve_mot_trivial_dirac_to_dirac() -> None:
    """mu = delta_x, nu = delta_x: the only martingale coupling is the identity, cost 0."""
    mu_atoms = np.array([[0.3, -0.2]])
    mu_weights = np.array([1.0])
    nu_atoms = np.array([[0.3, -0.2]])
    nu_weights = np.array([1.0])
    cost = mot_cost_matrix(mu_atoms, nu_atoms, _euclidean_cost)
    sol = solve_mot(mu_atoms, mu_weights, nu_atoms, nu_weights, cost)
    assert sol["success"]
    assert sol["value"] == pytest.approx(0.0, abs=1e-8)


def test_solve_mot_two_point_martingale_exact_value() -> None:
    """mu = delta_0, nu = (delta_{-1}+delta_{1})/2 is the unique martingale coupling.

    Cost c(x,y) = |x-y|: value = 0.5*1 + 0.5*1 = 1.
    """
    mu_atoms = np.array([[0.0, 0.0]])
    mu_weights = np.array([1.0])
    nu_atoms = np.array([[-1.0, 0.0], [1.0, 0.0]])
    nu_weights = np.array([0.5, 0.5])
    cost = mot_cost_matrix(mu_atoms, nu_atoms, _euclidean_cost)
    sol = solve_mot(mu_atoms, mu_weights, nu_atoms, nu_weights, cost, return_plan=True)
    assert sol["success"]
    assert sol["value"] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(sol["pi"], [[0.5, 0.5]], atol=1e-6)


def test_solve_mot_identity_coupling_when_mu_equals_nu() -> None:
    """When mu == nu on a grid, the identity coupling (cost 0) is feasible and optimal."""
    atoms, weights = uniform_grid_square(1.0, 4)
    cost = mot_cost_matrix(atoms, atoms, _euclidean_cost)
    sol = solve_mot(atoms, weights, atoms, weights, cost)
    assert sol["success"]
    assert sol["value"] == pytest.approx(0.0, abs=1e-6)


def test_solve_mot_value_is_nonnegative_for_nonnegative_cost() -> None:
    rng = np.random.default_rng(3)
    mu_atoms, mu_weights = uniform_grid_square(0.1, 3)
    nu_atoms, nu_weights = uniform_grid_square(1.0, 3)
    cost = mot_cost_matrix(mu_atoms, nu_atoms, _euclidean_cost)
    sol = solve_mot(mu_atoms, mu_weights, nu_atoms, nu_weights, cost)
    assert sol["success"]
    assert sol["value"] >= -1e-9


def test_solve_mot_rejects_mismatched_weight_sum() -> None:
    mu_atoms = np.array([[0.0, 0.0]])
    mu_weights = np.array([0.5])  # does not sum to 1
    nu_atoms = np.array([[0.0, 0.0]])
    nu_weights = np.array([1.0])
    cost = mot_cost_matrix(mu_atoms, nu_atoms, _euclidean_cost)
    with pytest.raises(AssertionError):
        solve_mot(mu_atoms, mu_weights, nu_atoms, nu_weights, cost)
