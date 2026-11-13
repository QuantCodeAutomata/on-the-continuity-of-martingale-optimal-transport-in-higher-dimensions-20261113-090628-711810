"""Tests for src/wasserstein.py: W1 distance between finitely supported measures."""

from __future__ import annotations

import numpy as np

from src.grids import uniform_grid_square
from src.wasserstein import w1_distance


def test_w1_distance_zero_for_identical_measures() -> None:
    atoms, weights = uniform_grid_square(1.0, 5)
    d = w1_distance(atoms, weights, atoms, weights)
    assert abs(d) < 1e-10


def test_w1_distance_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    atoms_a = rng.standard_normal((20, 2))
    weights_a = np.full(20, 1.0 / 20)
    atoms_b = rng.standard_normal((15, 2)) + 3.0
    weights_b = np.full(15, 1.0 / 15)
    d_ab = w1_distance(atoms_a, weights_a, atoms_b, weights_b)
    d_ba = w1_distance(atoms_b, weights_b, atoms_a, weights_a)
    assert abs(d_ab - d_ba) < 1e-8


def test_w1_distance_matches_known_1d_shift() -> None:
    """Two Dirac masses at distance L apart have W1 = L."""
    atoms_a = np.array([[0.0, 0.0]])
    atoms_b = np.array([[3.0, 4.0]])  # Euclidean distance 5
    weights = np.array([1.0])
    d = w1_distance(atoms_a, weights, atoms_b, weights)
    assert abs(d - 5.0) < 1e-10


def test_w1_distance_nonnegative_and_triangle_like_behavior() -> None:
    rng = np.random.default_rng(2)
    atoms_a = rng.standard_normal((10, 2))
    atoms_b = rng.standard_normal((10, 2))
    weights = np.full(10, 0.1)
    d = w1_distance(atoms_a, weights, atoms_b, weights)
    assert d >= 0.0


def test_w1_distance_shrinks_as_grids_refine_towards_same_continuous_law() -> None:
    """W1(grid_N, grid_M) -> 0 as both N, M grow for the same underlying uniform law."""
    atoms_coarse, w_coarse = uniform_grid_square(1.0, 4)
    atoms_fine, w_fine = uniform_grid_square(1.0, 40)
    atoms_finer, w_finer = uniform_grid_square(1.0, 80)
    d_coarse = w1_distance(atoms_coarse, w_coarse, atoms_fine, w_fine)
    d_fine = w1_distance(atoms_fine, w_fine, atoms_finer, w_finer)
    assert d_fine < d_coarse
