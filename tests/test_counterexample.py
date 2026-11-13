"""Tests for the Example 6.1 counterexample primitives (src/counterexample.py).

Checks the building blocks used by exp/exp3.py: Monte Carlo sampling of
(mu_n, nu_n), the (A_kappa) tilted density, the cost function, the analytic
limit value V_Q, the deterministic 1D limit grid, the explicit near-optimal
1D coupling cost, and a small end-to-end MOT LP regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.counterexample import (
    cost_example_6_1,
    explicit_limit_coupling_cost,
    limit_1d_grid,
    sample_marginals,
    tilted_density_example_6_1,
    v_q_analytic,
)
from src.mot_lp import mot_cost_matrix, solve_mot


@pytest.mark.parametrize("n,n_samples", [(2, 50), (8, 120)])
def test_sample_marginals_shapes_and_weights(n: int, n_samples: int) -> None:
    rng = np.random.default_rng(config.SEED + n)
    mu_atoms, mu_weights, nu_atoms, nu_weights, aux = sample_marginals(
        n, n_samples, rng, config.EXP3_U_MU_HALF_WIDTH, config.EXP3_U_NU_HALF_WIDTH
    )

    assert mu_atoms.shape == (n_samples, 2)
    assert nu_atoms.shape == (n_samples, 2)
    assert mu_weights.shape == (n_samples,)
    assert nu_weights.shape == (n_samples,)

    assert np.isclose(mu_weights.sum(), 1.0)
    assert np.isclose(nu_weights.sum(), 1.0)
    assert np.allclose(mu_weights, 1.0 / n_samples)
    assert np.allclose(nu_weights, 1.0 / n_samples)

    for key in ("x1", "w", "y1", "z"):
        assert aux[key].shape == (n_samples,)
    assert aux["eps_n"] == pytest.approx(1.0 / n)
    assert aux["t_n"] == pytest.approx(1.0 / n**2)


@pytest.mark.parametrize("n", [2, 4, 16, 64])
def test_tilted_density_within_theoretical_bounds(n: int) -> None:
    rng = np.random.default_rng(config.SEED + n)
    _, _, _, _, aux = sample_marginals(
        n, 200, rng, config.EXP3_U_MU_HALF_WIDTH, config.EXP3_U_NU_HALF_WIDTH
    )
    density = tilted_density_example_6_1(aux)

    assert density.shape == (200, 200)
    # (A_kappa) with kappa = 1/16 (Example 6.1): density >= kappa = 1/16.
    assert density.min() >= config.EXP3_KAPPA - 1e-9
    # Tighter theoretical bound stated in the paper: density <= 49/16 <= 1/kappa = 16.
    assert density.max() <= 49.0 / 16.0 + 1e-9
    assert density.max() <= 1.0 / config.EXP3_KAPPA + 1e-9


@pytest.mark.parametrize(
    "x1,x2,y1,y2",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.2, -0.1, 0.5, 0.3),
        (-0.25, 0.1, 1.0, -1.0),
    ],
)
def test_cost_example_6_1_matches_formula(x1: float, x2: float, y1: float, y2: float) -> None:
    x = np.array([[[x1, x2]]])  # shape (1, 1, 2)
    y = np.array([[[y1, y2]]])  # shape (1, 1, 2)
    expected = (1.0 + x1) * (y1 - x1) ** 2
    result = cost_example_6_1(x, y)
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(expected)


def test_v_q_analytic_equals_five_sixteenths() -> None:
    assert abs(v_q_analytic() - 5.0 / 16.0) < 1e-6
    assert abs(v_q_analytic() - config.EXP3_V_Q) < 1e-6


@pytest.mark.parametrize("m_points", [10, 100])
def test_limit_1d_grid_shapes_and_support(m_points: int) -> None:
    mu_atoms_1d, w_mu, nu_atoms_1d, w_nu = limit_1d_grid(
        m_points, config.EXP3_U_MU_HALF_WIDTH, config.EXP3_U_NU_HALF_WIDTH
    )

    assert mu_atoms_1d.ndim == 1 and mu_atoms_1d.shape == (m_points,)
    assert nu_atoms_1d.ndim == 1 and nu_atoms_1d.shape == (m_points,)
    assert w_mu.shape == (m_points,)
    assert w_nu.shape == (m_points,)

    assert np.isclose(w_mu.sum(), 1.0)
    assert np.isclose(w_nu.sum(), 1.0)

    assert np.all(mu_atoms_1d >= -config.EXP3_U_MU_HALF_WIDTH)
    assert np.all(mu_atoms_1d <= config.EXP3_U_MU_HALF_WIDTH)
    assert np.all(nu_atoms_1d >= -config.EXP3_U_NU_HALF_WIDTH)
    assert np.all(nu_atoms_1d <= config.EXP3_U_NU_HALF_WIDTH)


def test_explicit_limit_coupling_cost_close_to_v_q_minus_gap() -> None:
    cost = explicit_limit_coupling_cost(
        config.EXP3_M_1D_RUN, config.EXP3_U_MU_HALF_WIDTH, config.EXP3_U_NU_HALF_WIDTH
    )
    expected = config.EXP3_V_Q - config.EXP3_GAP
    assert abs(cost - expected) < 1e-3


def test_mot_lp_regression_small_instance_is_finite_and_nonnegative() -> None:
    n, n_samples = 4, 30
    rng = np.random.default_rng(config.SEED + n)
    mu_atoms, mu_weights, nu_atoms, nu_weights, aux = sample_marginals(
        n, n_samples, rng, config.EXP3_U_MU_HALF_WIDTH, config.EXP3_U_NU_HALF_WIDTH
    )

    # Recenter coordinate-wise to the known theoretical means (0 and eps_n/3) so that
    # the exact-equality MOT LP constraints are feasible for this finite Monte Carlo
    # sample (see exp/exp3.py for the full explanation of this necessary correction).
    eps_n = aux["eps_n"]
    mu_atoms = mu_atoms.copy()
    nu_atoms = nu_atoms.copy()
    mu_atoms[:, 0] -= mu_atoms[:, 0].mean()
    nu_atoms[:, 0] -= nu_atoms[:, 0].mean()
    mu_atoms[:, 1] += eps_n / 3.0 - mu_atoms[:, 1].mean()
    nu_atoms[:, 1] += eps_n / 3.0 - nu_atoms[:, 1].mean()

    cost_matrix = mot_cost_matrix(mu_atoms, nu_atoms, cost_example_6_1)
    assert cost_matrix.shape == (n_samples, n_samples)
    # c(x,y) = (1+x1)(y1-x1)^2 >= 0.75 * 0 = 0 since x1 in [-1/4, 1/4] => 1+x1 >= 0.75 > 0.
    assert np.all(cost_matrix >= -1e-12)

    sol = solve_mot(mu_atoms, mu_weights, nu_atoms, nu_weights, cost_matrix)
    assert sol["success"]
    assert np.isfinite(sol["value"])
    assert sol["value"] >= -1e-9
