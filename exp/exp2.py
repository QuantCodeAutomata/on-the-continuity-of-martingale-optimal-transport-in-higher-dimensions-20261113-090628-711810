"""Experiment 2: construction and verification of the tilted martingale coupling
satisfying the uniform density condition (Example 1.2).

Configuration 1 (bounded support): mu = Uniform([-r,r]^2), nu = Uniform([-1,1]^2),
both centered at m = (0,0), discretized on N x N grids (exp_2 methodology,
Step 1). The tilted kernel

    pi_bar_x(dy) = (1 + <Sigma^{-1}(x - m), y - m>) nu(dy)

is built via src.tilted_coupling.bounded_tilted_coupling and checked against
the paper's theoretical density bound [1 - theta, 1 + theta] (Example 1.2) and
against the martingale/marginal identities up to O(1/N^2) discretization error.

Configuration 2 (unbounded support): nu = N(0, I) in R^2, approximated by
Monte Carlo samples; mu = Uniform on a disk B(m, r_mu) with r_mu chosen (with
a safety factor < 1) below the theoretical radius r_max = lambda_T(nu) / (4T)
at which the truncated tilt density is guaranteed to stay non-negative
(Section 2.4, Section 4.1, eq. (4.3)). The truncated kernel is built via
src.tilted_coupling.unbounded_tilted_coupling and the resulting density range
and martingale/marginal errors (now also subject to Monte Carlo noise) are
reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config
from src.grids import uniform_grid_square
from src.tilted_coupling import (
    bounded_tilted_coupling,
    lambda_T,
    sigma_T,
    unbounded_tilted_coupling,
    verify_martingale_coupling,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_bounded_configuration() -> dict:
    """Configuration 1: tilted coupling for the bounded Example 1.2 pair."""
    mean = np.zeros(2)
    mu_atoms, mu_weights = uniform_grid_square(
        config.EXP2_BOUNDED_MU_HALF_WIDTH, config.EXP2_BOUNDED_N_GRID, mean
    )
    nu_atoms, nu_weights = uniform_grid_square(
        config.EXP2_BOUNDED_NU_HALF_WIDTH, config.EXP2_BOUNDED_N_GRID, mean
    )
    sigma_inv = np.linalg.inv(config.EXP2_BOUNDED_NU_COV)

    pi, density = bounded_tilted_coupling(mu_atoms, mu_weights, nu_atoms, nu_weights, sigma_inv, mean)

    theta = config.EXP2_BOUNDED_THETA
    density_min = float(density.min())
    density_max = float(density.max())
    assert 1.0 - theta - 1e-9 <= density_min, (
        f"density_min={density_min} violates the theoretical lower bound 1-theta={1 - theta}"
    )
    assert density_max <= 1.0 + theta + 1e-9, (
        f"density_max={density_max} violates the theoretical upper bound 1+theta={1 + theta}"
    )

    n = config.EXP2_BOUNDED_N_GRID
    mart = verify_martingale_coupling(pi, mu_atoms, mu_weights, nu_atoms, nu_weights, atol=10.0 / n**2)
    if not mart["within_tolerance"]:
        print(
            "WARNING: bounded configuration martingale/marginal check exceeded the "
            f"O(1/N^2) tolerance (N={n}); the paper does not fix this constant. "
            f"Diagnostics: {mart}"
        )

    result = {
        "density_min": density_min,
        "density_max": density_max,
        "theta": theta,
        "kappa": 1.0 - theta,
        "max_row_marginal_error": mart["max_row_marginal_error"],
        "max_col_marginal_error": mart["max_col_marginal_error"],
        "max_martingale_error": mart["max_martingale_error"],
        "within_tolerance": mart["within_tolerance"],
    }
    return result


def run_unbounded_configuration(return_density: bool = False):
    """Configuration 2: truncated tilted coupling for unbounded Gaussian nu.

    Returns the result dict; if `return_density` is True, also returns the
    (m, n) density matrix of the last attempt (used only to plot the optional
    histogram in `main`).
    """
    mean = np.zeros(2)
    rng = np.random.default_rng(config.SEED)

    n_samples = config.EXP2_UNBOUNDED_N_SAMPLES
    nu_atoms = rng.standard_normal((n_samples, 2))
    nu_weights = np.full(n_samples, 1.0 / n_samples)

    # unbounded_tilted_coupling / verify_martingale_coupling build dense (m, n)
    # matrices (pi, density, ...). With m = n = N_SAMPLES = 20000 those matrices
    # would be 20000 x 20000 x 8 bytes ~= 3.2 GB *each*, far above the
    # per-process memory limit used here. nu keeps the full N_SAMPLES (needed for a stable Monte
    # Carlo estimate of Sigma_T / lambda_T, an O(N) computation); mu is capped
    # at a smaller count so the (m, n) matrices stay small (O(10-100 MB)) while
    # still being large enough for stable marginal/martingale diagnostics.
    mu_n_samples = min(n_samples, 2000)

    t = config.EXP2_UNBOUNDED_T
    inner = config.EXP2_UNBOUNDED_CHI_INNER
    outer = config.EXP2_UNBOUNDED_CHI_OUTER

    sigma_t = sigma_T(nu_atoms, mean, t, inner, outer, nu_weights)
    lam_t = lambda_T(sigma_t)
    assert lam_t > 0, f"lambda_T must be positive, got {lam_t}"

    r_max = lam_t / (4.0 * t)

    attempts = []
    r_mu = config.EXP2_UNBOUNDED_R_SAFETY * r_max
    for attempt in range(2):
        angle = rng.uniform(0.0, 2.0 * np.pi, mu_n_samples)
        radius = r_mu * np.sqrt(rng.uniform(0.0, 1.0, mu_n_samples))
        mu_atoms = mean[None, :] + radius[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        mu_weights = np.full(mu_n_samples, 1.0 / mu_n_samples)

        pi, density, sigma_t2, v = unbounded_tilted_coupling(
            mu_atoms, mu_weights, nu_atoms, nu_weights, mean, t, inner, outer
        )
        density_min = float(density.min())
        density_max = float(density.max())
        attempts.append({"r_mu": float(r_mu), "density_min": density_min, "density_max": density_max})

        if density_min > 0:
            break
        print(
            f"WARNING: attempt {attempt} with r_mu={r_mu:.6f} gave non-positive "
            f"density_min={density_min:.6f}; halving r_mu and retrying."
        )
        r_mu = r_mu / 2.0
    else:
        print("WARNING: density remained non-positive after retry; reporting both attempts.")

    mart = verify_martingale_coupling(pi, mu_atoms, mu_weights, nu_atoms, nu_weights, atol=0.05)
    if not mart["within_tolerance"]:
        print(
            "WARNING: unbounded configuration martingale/marginal check exceeded the "
            f"Monte Carlo + truncation tolerance. Diagnostics: {mart}"
        )

    kappa_empirical = density_min if density_min > 0 else None

    result = {
        "lambda_T": float(lam_t),
        "r_max": float(r_max),
        "r_mu": float(r_mu),
        "density_min": density_min,
        "density_max": density_max,
        "kappa_empirical": kappa_empirical,
        "attempts": attempts,
        "max_row_marginal_error": mart["max_row_marginal_error"],
        "max_col_marginal_error": mart["max_col_marginal_error"],
        "max_martingale_error": mart["max_martingale_error"],
        "within_tolerance": mart["within_tolerance"],
    }
    if return_density:
        return result, density
    return result


def _save_histogram(bounded_density: np.ndarray, unbounded_density: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(bounded_density.ravel(), bins=50)
    axes[0].set_title("Bounded tilt density")
    axes[0].set_xlabel("density")
    axes[1].hist(unbounded_density.ravel(), bins=50)
    axes[1].set_title("Unbounded (truncated) tilt density")
    axes[1].set_xlabel("density")
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "exp2_density_histogram.png", dpi=150)
    plt.close(fig)


def main() -> None:
    bounded_result = run_bounded_configuration()

    mean = np.zeros(2)
    mu_atoms, mu_weights = uniform_grid_square(
        config.EXP2_BOUNDED_MU_HALF_WIDTH, config.EXP2_BOUNDED_N_GRID, mean
    )
    nu_atoms, nu_weights = uniform_grid_square(
        config.EXP2_BOUNDED_NU_HALF_WIDTH, config.EXP2_BOUNDED_N_GRID, mean
    )
    sigma_inv = np.linalg.inv(config.EXP2_BOUNDED_NU_COV)
    _, bounded_density = bounded_tilted_coupling(mu_atoms, mu_weights, nu_atoms, nu_weights, sigma_inv, mean)

    unbounded_result, unbounded_density = run_unbounded_configuration(return_density=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {"bounded": bounded_result, "unbounded": unbounded_result}
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _save_histogram(bounded_density, unbounded_density)

    print("=== Experiment 2: tilted martingale coupling (Example 1.2) ===")
    print(
        f"[Bounded]   theta={bounded_result['theta']:.3f}  kappa={bounded_result['kappa']:.3f}  "
        f"density in [{bounded_result['density_min']:.4f}, {bounded_result['density_max']:.4f}]  "
        f"max_martingale_error={bounded_result['max_martingale_error']:.3e}  "
        f"max_col_marginal_error={bounded_result['max_col_marginal_error']:.3e}"
    )
    print(
        f"[Unbounded] lambda_T={unbounded_result['lambda_T']:.4f}  r_max={unbounded_result['r_max']:.4f}  "
        f"r_mu={unbounded_result['r_mu']:.4f}  "
        f"density in [{unbounded_result['density_min']:.4f}, {unbounded_result['density_max']:.4f}]  "
        f"max_martingale_error={unbounded_result['max_martingale_error']:.3e}  "
        f"max_col_marginal_error={unbounded_result['max_col_marginal_error']:.3e}"
    )
    print(f"Summary written to {RESULTS_DIR / 'exp2_summary.json'}")


if __name__ == "__main__":
    main()
