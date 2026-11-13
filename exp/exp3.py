"""Experiment 3: Example 6.1 counterexample (Section 6).

Shows that (A_kappa) alone, without a non-degenerate limit target, does not
imply continuity of the MOT value V. The sequence (mu_n, nu_n) collapses in
W1 to (mu, nu), both supported on the hyperplane {x_2 = 0}, while every
(mu_n, nu_n) satisfies (A_kappa) with kappa = 1/16, and

    lim_n V(mu_n, nu_n) = V_Q = 5/16,

whereas V(mu, nu) <= V_Q - 1/540, an explicit discontinuity gap (exp_3
data_requirements / methodology).
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
from src.counterexample import (
    cost_example_6_1,
    explicit_limit_coupling_cost,
    limit_1d_grid,
    sample_marginals,
    tilted_density_example_6_1,
    v_q_analytic,
)
from src.mot_lp import mot_cost_matrix, solve_mot
from src.tilted_coupling import verify_martingale_coupling

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(n_values: list[int], n_samples: int, m_1d: int) -> dict:
    """Run the Example 6.1 sweep over n_values and return the summary dict."""
    rows = []
    for n in n_values:
        rng = np.random.default_rng(config.SEED + n)
        mu_atoms, mu_weights, nu_atoms, nu_weights, aux = sample_marginals(
            n, n_samples, rng, config.EXP3_U_MU_HALF_WIDTH, config.EXP3_U_NU_HALF_WIDTH
        )

        density = tilted_density_example_6_1(aux)
        assert density.min() >= config.EXP3_KAPPA - 1e-6, (
            f"n={n}: density.min()={density.min()} below kappa={config.EXP3_KAPPA}"
        )
        assert density.max() <= 1.0 / config.EXP3_KAPPA + 1e-6, (
            f"n={n}: density.max()={density.max()} above 1/kappa={1.0 / config.EXP3_KAPPA}"
        )
        assert density.max() <= 49.0 / 16.0 + 1e-6, (
            f"n={n}: density.max()={density.max()} above the tighter bound 49/16"
        )

        pi_ref = mu_weights[:, None] * nu_weights[None, :] * density
        mart = verify_martingale_coupling(pi_ref, mu_atoms, mu_weights, nu_atoms, nu_weights, atol=0.1)
        if not mart["within_tolerance"]:
            print(
                f"WARNING: n={n} reference coupling outside Monte Carlo tolerance "
                f"(row_err={mart['max_row_marginal_error']:.4f}, "
                f"col_err={mart['max_col_marginal_error']:.4f}, "
                f"mart_err={mart['max_martingale_error']:.4f})",
                flush=True,
            )

        # The discrete MOT LP's equality constraints (mu_lp.build_mot_constraints)
        # imply, after summing the per-atom martingale rows over i and using the
        # second-marginal block, the *exact* necessary condition
        #   sum_i mu_weights[i] * mu_atoms[i,k] == sum_j nu_weights[j] * nu_atoms[j,k]
        # for every coordinate k. In the continuum this holds exactly (E[X_1]=E[Y_1]=0,
        # E[mu_2]=E[nu_2]=eps_n/3), but i.i.d. Monte Carlo sampling only reproduces it up
        # to O(1/sqrt(n_samples)) noise, which makes the exact-equality LP infeasible as
        # sampled. We recenter each coordinate to its known theoretical mean -- a
        # vanishing (O(1/sqrt(n_samples))) shift that preserves the n_samples -> infinity
        # limit distribution -- which restores feasibility without altering src/.
        mu_atoms_lp = mu_atoms.copy()
        nu_atoms_lp = nu_atoms.copy()
        eps_n = aux["eps_n"]
        mu_atoms_lp[:, 0] -= mu_atoms_lp[:, 0].mean()
        nu_atoms_lp[:, 0] -= nu_atoms_lp[:, 0].mean()
        mu_atoms_lp[:, 1] += eps_n / 3.0 - mu_atoms_lp[:, 1].mean()
        nu_atoms_lp[:, 1] += eps_n / 3.0 - nu_atoms_lp[:, 1].mean()

        cost_matrix = mot_cost_matrix(mu_atoms_lp, nu_atoms_lp, cost_example_6_1)
        sol = solve_mot(mu_atoms_lp, mu_weights, nu_atoms_lp, nu_weights, cost_matrix)
        assert sol["success"], f"n={n}: MOT LP failed: {sol['status']}"
        v_n = sol["value"]

        rows.append({
            "n": n, "V_n": v_n,
            "density_min": float(density.min()), "density_max": float(density.max()),
            "max_martingale_error": mart["max_martingale_error"],
        })
        print(
            f"n={n:3d}  V_n={v_n:.6f}  density in [{density.min():.4f},{density.max():.4f}]  "
            f"max_mart_err={mart['max_martingale_error']:.4f}  success={sol['success']}",
            flush=True,
        )

    df = pd.DataFrame(rows)

    # Limit quantities.
    v_q = v_q_analytic()
    assert abs(v_q - config.EXP3_V_Q) < 1e-6, f"v_q_analytic()={v_q} != EXP3_V_Q={config.EXP3_V_Q}"

    mu1d, w_mu, nu1d, w_nu = limit_1d_grid(m_1d, config.EXP3_U_MU_HALF_WIDTH, config.EXP3_U_NU_HALF_WIDTH)
    mu_atoms_2d = np.stack([mu1d, np.zeros_like(mu1d)], axis=1)
    nu_atoms_2d = np.stack([nu1d, np.zeros_like(nu1d)], axis=1)
    limit_cost_matrix = mot_cost_matrix(mu_atoms_2d, nu_atoms_2d, cost_example_6_1)
    limit_sol = solve_mot(mu_atoms_2d, w_mu, nu_atoms_2d, w_nu, limit_cost_matrix)
    assert limit_sol["success"], f"limit MOT LP failed: {limit_sol['status']}"
    v_limit_lp = limit_sol["value"]

    explicit_cost = explicit_limit_coupling_cost(m_1d, config.EXP3_U_MU_HALF_WIDTH, config.EXP3_U_NU_HALF_WIDTH)

    gap_observed = float(df["V_n"].iloc[-1] - v_limit_lp)
    gap_passes = bool(gap_observed >= config.EXP3_GAP * 0.5)
    print(
        f"v_q_analytic={v_q:.6f}  v_limit_lp={v_limit_lp:.6f}  explicit_cost={explicit_cost:.6f}  "
        f"gap_observed={gap_observed:.6f}  gap_theory={config.EXP3_GAP:.6f}  "
        f"gap_observed>=0.5*gap_theory: {gap_passes}",
        flush=True,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "exp3_V_n.csv", index=False)

    summary = {
        "v_q_analytic": v_q,
        "v_limit_lp": v_limit_lp,
        "explicit_coupling_cost": explicit_cost,
        "gap_observed": gap_observed,
        "gap_theory": config.EXP3_GAP,
        "kappa": config.EXP3_KAPPA,
    }
    with open(RESULTS_DIR / "exp3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df["n"], df["V_n"], "o-", label=r"$V(\mu_n,\nu_n)$")
    ax.axhline(v_q, color="tab:red", linestyle="--", label=r"$V_Q=5/16$ (analytic limit)")
    ax.axhline(v_limit_lp, color="tab:green", linestyle=":", label=r"$V(\mu,\nu)$ (limit LP)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$V(\mu_n,\nu_n)$")
    ax.set_title("Example 6.1: discontinuity of V under $(A_\\kappa)$ alone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_convergence.png", dpi=150)
    plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run(config.EXP3_N_VALUES, config.EXP3_N_SAMPLES_RUN, config.EXP3_M_1D_RUN)


if __name__ == "__main__":
    main()
