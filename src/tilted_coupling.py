"""Tilted martingale coupling of Example 1.2 and its truncated extension.

Bounded case (Example 1.2, page 3): given nu with mean m, invertible
covariance Sigma and support in B(m, R), and mu with mean m such that
R |Sigma^{-1}(x - m)| <= theta < 1 on supp(mu), the kernel

    pi_bar_x(dy) := (1 + <Sigma^{-1}(x - m), y - m>) nu(dy)

is a martingale coupling of (mu, nu) satisfying (A_kappa) with kappa = 1 - theta.

Unbounded case (Section 2.4, eq. (2.5)-(2.7), and Section 4.1): replacing the
linear map y - m by its truncation

    psi_T(y) := (y - m) chi(|y - m| / T),   phi_T := psi_T - integral psi_T dnu,

with chi continuous, nonincreasing, chi = 1 on [0,1], chi = 0 on [2, infinity),
gives a tilt that remains dominated by a multiple of nu even when supp(nu) is
unbounded. Writing Sigma_T(nu) := integral (y-m)(y-m)^T chi(|y-m|/T) nu(dy),
lambda_T(nu) := lambda_min(Sigma_T(nu)), and v(x) := Sigma_T(nu)^{-1} (x - m),
the kernel

    pi_bar_x(dy) := (1 + <v(x), phi_T(y)>) nu(dy)

has the same martingale and marginal properties (Lemma 2.7(i)), with its
density controlled by |tau_x| <= 4T |v(x)| nu (eq. (4.3)).
"""

from __future__ import annotations

import numpy as np


def chi_truncation(t: np.ndarray, inner: float = 1.0, outer: float = 2.0) -> np.ndarray:
    """Continuous nonincreasing chi: [0, inf) -> [0, 1], chi=1 on [0,inner], chi=0 on [outer, inf).

    Piecewise linear interpolation in between (Section 2.4): a concrete, admissible
    instance of the class of cutoffs chi required by the paper; the paper does not
    fix a specific chi beyond these boundary conditions.
    """
    t = np.asarray(t, dtype=float)
    out = np.clip((outer - t) / (outer - inner), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Bounded case (Example 1.2)
# ---------------------------------------------------------------------------

def bounded_tilt_density(mu_atoms: np.ndarray, nu_atoms: np.ndarray, sigma_inv: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Density matrix d[i,j] = 1 + <Sigma^{-1}(x_i - m), y_j - m> of Example 1.2."""
    dx = mu_atoms - mean[None, :]
    dy = nu_atoms - mean[None, :]
    inner = dx @ sigma_inv @ dy.T  # (m, n): <Sigma^{-1}(x_i-m), y_j-m>
    return 1.0 + inner


def bounded_tilted_coupling(mu_atoms, mu_weights, nu_atoms, nu_weights, sigma_inv, mean):
    """Discrete tilted coupling pi_bar_{ij} = alpha_i * beta_j * density[i,j] (Example 1.2)."""
    density = bounded_tilt_density(mu_atoms, nu_atoms, sigma_inv, mean)
    pi = mu_weights[:, None] * nu_weights[None, :] * density
    return pi, density


def verify_martingale_coupling(pi: np.ndarray, mu_atoms, mu_weights, nu_atoms, nu_weights, atol: float):
    """Check that pi has the correct marginals and the martingale property.

    Returns a dict of diagnostic maxima so the caller can report the actual
    discretization error rather than only a pass/fail boolean.
    """
    row_sums = pi.sum(axis=1)
    col_sums = pi.sum(axis=0)
    cond_mean = pi @ nu_atoms  # should equal mu_weights[:, None] * mu_atoms
    target_mean = mu_weights[:, None] * mu_atoms

    max_row_err = float(np.max(np.abs(row_sums - mu_weights)))
    max_col_err = float(np.max(np.abs(col_sums - nu_weights)))
    max_mart_err = float(np.max(np.abs(cond_mean - target_mean)))
    return {
        "max_row_marginal_error": max_row_err,
        "max_col_marginal_error": max_col_err,
        "max_martingale_error": max_mart_err,
        "within_tolerance": max(max_row_err, max_col_err, max_mart_err) <= atol,
    }


# ---------------------------------------------------------------------------
# Unbounded case (Section 2.4, Section 4.1)
# ---------------------------------------------------------------------------

def psi_T(y: np.ndarray, mean: np.ndarray, T: float, inner: float, outer: float) -> np.ndarray:
    """psi_T(y) = (y - m) chi(|y-m|/T), eq. (2.5)."""
    centered = y - mean[None, :]
    r = np.linalg.norm(centered, axis=1) / T
    weight = chi_truncation(r, inner, outer)
    return centered * weight[:, None]


def phi_T(y: np.ndarray, mean: np.ndarray, T: float, inner: float, outer: float, nu_weights: np.ndarray) -> np.ndarray:
    """phi_T(y) = psi_T(y) - integral psi_T dnu, eq. (2.5)."""
    psi = psi_T(y, mean, T, inner, outer)
    psi_mean = nu_weights @ psi
    return psi - psi_mean[None, :]


def sigma_T(y: np.ndarray, mean: np.ndarray, T: float, inner: float, outer: float, nu_weights: np.ndarray) -> np.ndarray:
    """Sigma_T(nu) = integral (y-m)(y-m)^T chi(|y-m|/T) nu(dy), eq. (2.6)."""
    centered = y - mean[None, :]
    r = np.linalg.norm(centered, axis=1) / T
    weight = chi_truncation(r, inner, outer)
    weighted = centered * (nu_weights * weight)[:, None]
    return weighted.T @ centered


def lambda_T(sigma_t_matrix: np.ndarray) -> float:
    """lambda_T(nu) = lambda_min(Sigma_T(nu)), eq. (2.6)."""
    eigvals = np.linalg.eigvalsh(sigma_t_matrix)
    return float(np.min(eigvals))


def unbounded_tilted_coupling(mu_atoms, mu_weights, nu_atoms, nu_weights, mean, T, inner, outer):
    """Discrete truncated tilted coupling pi_bar_{ij} = alpha_i * beta_j * density[i,j].

    density[i,j] = 1 + <v(x_i), phi_T(y_j)>, v(x) = Sigma_T(nu)^{-1} (x - m)
    (Section 4.1, with the linear tilt of Example 1.2 replaced by its
    truncated version so that nu may have unbounded support).
    """
    sigma_t = sigma_T(nu_atoms, mean, T, inner, outer, nu_weights)
    sigma_t_inv = np.linalg.inv(sigma_t)
    phi = phi_T(nu_atoms, mean, T, inner, outer, nu_weights)  # (n, d)

    v = (mu_atoms - mean[None, :]) @ sigma_t_inv  # (m, d), v(x_i) = Sigma_T^{-1}(x_i - m)
    inner_prod = v @ phi.T  # (m, n)
    density = 1.0 + inner_prod
    pi = mu_weights[:, None] * nu_weights[None, :] * density
    return pi, density, sigma_t, v
