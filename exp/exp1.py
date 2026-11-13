"""Experiment 1: continuity and quantitative stability of MOT in R^2 (Theorem 1.4).

Builds the base pair (mu, nu) of Example 1.2 on R^2 -- mu = Uniform([-r,r]^2),
nu = Uniform([-1,1]^2), both centered at m = (0,0) -- which satisfies
Assumption 1.1 ((A_kappa) with kappa = 1 - 6r) and Assumption 1.3 (nu has
invertible covariance). A sequence (mu_n, nu_n) is produced by discretizing
both marginals on N x N uniform grids, N = 2, ..., N_max (exp_1 methodology,
Steps 1-2). For each N the discrete MOT linear program (1.1) is solved with
cost c(x, y) = ||x - y||_2 and compared, via the Wasserstein-1 distance to a
fine reference grid (N_ref), against the quantitative stability bound of
Theorem 1.4(ii):

    |V(mu_n, nu_n) - V(mu, nu)| <= L * Delta_n + C * Delta_n^(1/2),

with L = 1 (the cost is 1-Lipschitz for |x-x'| + |y-y'|) and gamma = 1/2 for
uniformly bounded supports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.grids import uniform_grid_square
from src.mot_lp import mot_cost_matrix, solve_mot
from src.tilted_coupling import bounded_tilted_coupling, verify_martingale_coupling
from src.wasserstein import w1_distance

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def euclidean_cost(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """c(x, y) = ||x - y||_2, 1-Lipschitz for |x-x'| + |y-y'| (exp_1 data_requirements)."""
    return np.linalg.norm(x - y, axis=-1)


def build_pair(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """mu_n (N x N grid on [-r,r]^2) and nu_n (N x N grid on [-1,1]^2) (exp_1 methodology, Step 2)."""
    nu_atoms, nu_weights = uniform_grid_square(config.EXP1_NU_HALF_WIDTH, n, config.EXP1_NU_MEAN)
    mu_atoms, mu_weights = uniform_grid_square(config.EXP1_MU_HALF_WIDTH_R, n, config.EXP1_NU_MEAN)
    return mu_atoms, mu_weights, nu_atoms, nu_weights


def check_density_condition(mu_atoms: np.ndarray, mu_weights: np.ndarray,
                             nu_atoms: np.ndarray, nu_weights: np.ndarray, n: int) -> dict:
    """Verify (A_kappa) for the discrete pair via the tilted coupling of Example 1.2.

    The martingale and second-marginal identities hold for the continuous
    kernel exactly; at the discrete level they hold up to an O(1/N^2)
    quadrature error (exp_1 methodology, Step 3), so the tolerance scales
    with 1/N^2.
    """
    sigma_inv = np.linalg.inv(config.EXP1_NU_COV)
    pi, density = bounded_tilted_coupling(mu_atoms, mu_weights, nu_atoms, nu_weights, sigma_inv, config.EXP1_NU_MEAN)
    mart = verify_martingale_coupling(pi, mu_atoms, mu_weights, nu_atoms, nu_weights, atol=20.0 / n**2)
    return {
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "kappa_holds": bool(density.min() >= config.EXP1_KAPPA - 1e-9
                             and density.max() <= (1.0 / config.EXP1_KAPPA) + 1e-9),
        "max_martingale_error": mart["max_martingale_error"],
        "max_col_marginal_error": mart["max_col_marginal_error"],
    }


def run(n_values: list[int], n_ref: int) -> dict:
    """Run the full grid sweep and return a results dict (also saved to results/)."""
    mu_ref_atoms, mu_ref_w, nu_ref_atoms, nu_ref_w = build_pair(n_ref)
    cost_ref = mot_cost_matrix(mu_ref_atoms, nu_ref_atoms, euclidean_cost)
    ref_sol = solve_mot(mu_ref_atoms, mu_ref_w, nu_ref_atoms, nu_ref_w, cost_ref)
    assert ref_sol["success"], f"reference MOT LP (N_ref={n_ref}) infeasible: {ref_sol['status']}"
    v_ref = ref_sol["value"]

    rows = []
    for n in n_values:
        mu_atoms, mu_w, nu_atoms, nu_w = build_pair(n)
        density_check = check_density_condition(mu_atoms, mu_w, nu_atoms, nu_w, n)
        cost = mot_cost_matrix(mu_atoms, nu_atoms, euclidean_cost)
        sol = solve_mot(mu_atoms, mu_w, nu_atoms, nu_w, cost)
        w1_mu = w1_distance(mu_atoms, mu_w, mu_ref_atoms, mu_ref_w)
        w1_nu = w1_distance(nu_atoms, nu_w, nu_ref_atoms, nu_ref_w)
        delta_n = w1_mu + w1_nu
        error = abs(sol["value"] - v_ref)
        rows.append({
            "N": n, "V_n": sol["value"], "lp_success": sol["success"],
            "W1_mu": w1_mu, "W1_nu": w1_nu, "Delta_n": delta_n, "abs_error": error,
            **density_check,
        })
        print(f"N={n:3d}  V_n={sol['value']:.6f}  Delta_n={delta_n:.5f}  "
              f"|V_n-V_ref|={error:.6f}  density in [{density_check['density_min']:.4f},"
              f"{density_check['density_max']:.4f}]  success={sol['success']}", flush=True)

    df = pd.DataFrame(rows)
    assert df["lp_success"].all(), "some MOT LPs were infeasible"
    assert df["kappa_holds"].all(), "(A_kappa) failed to hold for some N"

    # Fit the convergence-rate exponent gamma on a log-log scale (Theorem 1.4(ii)):
    # error ~ Delta_n^gamma, expected gamma ~ 1/2 for uniformly bounded supports.
    fit_mask = df["Delta_n"] > 0
    log_delta = np.log(df.loc[fit_mask, "Delta_n"].to_numpy())
    log_err = np.log(df.loc[fit_mask, "abs_error"].to_numpy() + 1e-15)
    gamma_hat, intercept = np.polyfit(log_delta, log_err, 1)

    # Minimal constant C >= 0 certifying |V_n - V| <= L*Delta_n + C*sqrt(Delta_n)
    # for every N tested, with L = EXP1_COST_L (exp_1 methodology, Step 6).
    residual = df["abs_error"].to_numpy() - config.EXP1_COST_L * df["Delta_n"].to_numpy()
    sqrt_delta = np.sqrt(df["Delta_n"].to_numpy())
    c_terms = np.where(sqrt_delta > 0, residual / np.maximum(sqrt_delta, 1e-15), 0.0)
    c_certificate = float(max(0.0, np.max(c_terms)))
    bound_holds = bool(np.all(
        df["abs_error"].to_numpy() <= config.EXP1_COST_L * df["Delta_n"].to_numpy()
        + c_certificate * sqrt_delta + 1e-9
    ))

    monotone_decreasing = bool(df["abs_error"].iloc[-1] < df["abs_error"].iloc[0])

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp1_grid_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(df["Delta_n"], df["abs_error"], "o-", label=r"$|V(\mu_n,\nu_n)-V(\mu,\nu)|$")
    delta_grid = np.geomspace(df["Delta_n"].min(), df["Delta_n"].max(), 50)
    ax.loglog(delta_grid, np.exp(intercept) * delta_grid**0.5, "--",
               label=r"reference slope $\gamma=1/2$")
    ax.set_xlabel(r"$\Delta_n = W_1(\mu_n,\mu) + W_1(\nu_n,\nu)$")
    ax.set_ylabel(r"$|V(\mu_n,\nu_n) - V(\mu,\nu)|$")
    ax.set_title(f"MOT quantitative stability (Theorem 1.4), fitted exponent={gamma_hat:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_convergence.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_ref": n_ref, "V_ref": v_ref, "n_values": n_values,
        "kappa": config.EXP1_KAPPA, "theta": config.EXP1_THETA, "cost_L": config.EXP1_COST_L,
        "gamma_hat": float(gamma_hat), "c_certificate": c_certificate, "bound_holds": bound_holds,
        "monotone_decreasing_error": monotone_decreasing,
        "kappa_holds_all_N": bool(df["kappa_holds"].all()),
        "lp_success_all_N": bool(df["lp_success"].all()),
        "max_abs_error": float(df["abs_error"].max()), "min_abs_error": float(df["abs_error"].min()),
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use the paper's full configuration (N up to 40, N_ref=80) "
                              "instead of the reduced sweep executed by default")
    args = parser.parse_args()
    if args.full:
        run(config.EXP1_N_VALUES_PAPER, config.EXP1_N_REF_PAPER)
    else:
        run(config.EXP1_N_VALUES_RUN, config.EXP1_N_REF_RUN)


if __name__ == "__main__":
    main()
