"""Example 6.1: (A_kappa) alone does not imply continuity of V (Section 6).

Marginals (page 25):
    eps_n = 1/n, t_n = 1/n^2,
    X_1, W ~ U_{1/4} = Uniform[-1/4, 1/4],  Y_1, Z ~ U_1 = Uniform[-1, 1],  all independent,
    mu_n := law(X_1, eps_n/3 + t_n W),   nu_n := law(Y_1, eps_n Y_1^2 + t_n Z),
    c(x, y) := (1 + x_1)(y_1 - x_1)^2.

mu_n -> mu := U_{1/4} (x) delta_0 and nu_n -> nu := U_1 (x) delta_0 in W1, and
every pair (mu_n, nu_n) satisfies (A_kappa) with kappa = 1/16, yet
lim_n V(mu_n, nu_n) >= V(mu, nu) + 1/540.
"""

from __future__ import annotations

import numpy as np


def _factor_pair(n: int) -> tuple[int, int]:
    """The factor pair (k1, k2) of n with k1*k2 = n closest to a square (k1 <= k2)."""
    for k1 in range(int(np.floor(np.sqrt(n))), 0, -1):
        if n % k1 == 0:
            return k1, n // k1
    return 1, n


def _equispaced_quadrature(half_width: float, k: int) -> np.ndarray:
    """k equal-weight midpoints of Uniform[-half_width, half_width] (deterministic quadrature)."""
    return np.linspace(-half_width, half_width, k, endpoint=False) + half_width / k


def sample_marginals(n: int, n_samples: int, rng: np.random.Generator, u_mu_half_width: float, u_nu_half_width: float):
    """Deterministic-quadrature atoms of mu_n and nu_n for a given index n (Example 6.1).

    Each underlying uniform pair (X_1, W) and (Y_1, Z) is discretized on the
    Cartesian product of two 1D equispaced quadratures of U_{1/4} resp. U_1
    (exp_3 implementation_notes: "N_samples Monte Carlo samples ... or
    deterministic quadrature on the underlying 1D uniforms"). This avoids the
    O(1/sqrt(n_samples)) noise of i.i.d. Monte Carlo sampling, which otherwise
    swamps the O(1/n) discontinuity gap of Example 6.1, and gives atoms whose
    empirical first moments equal the population means (0, eps_n/3) exactly
    by symmetry, matching exp_1's grid-quadrature treatment of continuous
    marginals. ``rng`` is accepted for interface stability but unused: the
    quadrature is deterministic.
    """
    eps_n = 1.0 / n
    t_n = 1.0 / n**2

    k1, k2 = _factor_pair(n_samples)
    x1_grid = _equispaced_quadrature(u_mu_half_width, k1)
    w_grid = _equispaced_quadrature(u_mu_half_width, k2)
    y1_grid = _equispaced_quadrature(u_nu_half_width, k1)
    z_grid = _equispaced_quadrature(u_nu_half_width, k2)

    x1 = np.repeat(x1_grid, k2)
    w = np.tile(w_grid, k1)
    y1 = np.repeat(y1_grid, k2)
    z = np.tile(z_grid, k1)

    mu_atoms = np.stack([x1, eps_n / 3.0 + t_n * w], axis=1)
    nu_atoms = np.stack([y1, eps_n * y1**2 + t_n * z], axis=1)
    weights = np.full(n_samples, 1.0 / n_samples)

    aux = {"x1": x1, "w": w, "y1": y1, "z": z, "eps_n": eps_n, "t_n": t_n}
    return mu_atoms, weights, nu_atoms, weights, aux


def tilted_density_example_6_1(aux: dict) -> np.ndarray:
    """Density of the reference coupling pi_bar_n w.r.t. mu_n (x) nu_n.

    Pi_bar_n is the pushforward, under Phi(x1,w)=(x1, eps_n/3+t_n w) and
    Psi(y1,z)=(y1, eps_n y1^2+t_n z), of the product tilt
    (1+3 x1 y1)(1+3 w z) U_{1/4}(dx1) U_{1/4}(dw) U_1(dy1) U_1(dz)
    (exp_3 methodology, Step 2). Its density w.r.t. mu_n (x) nu_n, evaluated
    on the sampled atoms, is exactly this product of Example-1.2 tilts in
    each coordinate, since X_1 ~ U_{1/4} has Sigma^{-1} = 3 and W ~ U_{1/4}
    likewise.
    """
    x1, w, y1, z = aux["x1"], aux["w"], aux["y1"], aux["z"]
    density = (1.0 + 3.0 * np.outer(x1, y1)) * (1.0 + 3.0 * np.outer(w, z))
    return density


def cost_example_6_1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """c(x, y) = (1 + x_1)(y_1 - x_1)^2, broadcasting over (m,1,d) and (1,n,d)."""
    x1 = x[..., 0]
    y1 = y[..., 0]
    return (1.0 + x1) * (y1 - x1) ** 2


def v_q_analytic() -> float:
    """V_Q = integral (1+x)(1/3 - x^2) U_{1/4}(dx) = 5/16, computed in closed form (Section 6)."""
    # U_{1/4} has density 2 on [-1/4, 1/4]; the odd terms x/3 - x^3 vanish by symmetry.
    from scipy.integrate import quad

    density = 2.0
    val, _ = quad(lambda x: (1 + x) * (1 / 3 - x**2) * density, -0.25, 0.25)
    return float(val)


def limit_1d_grid(m_points: int, u_mu_half_width: float, u_nu_half_width: float):
    """Uniform grids discretizing U_{1/4} and U_1 for the 1D limit MOT problem."""
    mu_atoms_1d = np.linspace(-u_mu_half_width, u_mu_half_width, m_points, endpoint=False) + u_mu_half_width / m_points
    nu_atoms_1d = np.linspace(-u_nu_half_width, u_nu_half_width, m_points, endpoint=False) + u_nu_half_width / m_points
    weights = np.full(m_points, 1.0 / m_points)
    return mu_atoms_1d, weights, nu_atoms_1d, weights


def explicit_limit_coupling_cost(m_points: int, u_mu_half_width: float, u_nu_half_width: float) -> float:
    """Cost of the explicit 1D coupling with density (1+3xy-x(y^2-1/3)) w.r.t. U_1(dy) (Section 6).

    Its conditional second moment is 1/3 - (4/45) x, and its cost equals
    V_Q - (4/45) integral x^2 U_{1/4}(dx) = V_Q - 1/540 (Section 6, page 27-28).
    """
    from scipy.integrate import dblquad

    def density(x, y):
        return 1.0 + 3.0 * x * y - x * (y**2 - 1.0 / 3.0)

    def integrand(y, x):
        # joint density w.r.t. dx dy: U_{1/4}(dx) has density 2, U_1(dy) has density 1/2.
        fx = 2.0
        fy = 0.5
        cost = (1 + x) * (y - x) ** 2
        return cost * density(x, y) * fx * fy

    val, _ = dblquad(
        integrand, -u_mu_half_width, u_mu_half_width, lambda x: -u_nu_half_width, lambda x: u_nu_half_width
    )
    return float(val)
