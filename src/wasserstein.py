"""Wasserstein-1 distance between finitely supported measures.

# POT (Python Optimal Transport) ot.emd2 is used for the W1 computation
# (exp_1 implementation_notes): W1(mu, nu) = min_pi integral |x-y| dpi over
# couplings of mu and nu (no martingale constraint), which is exactly the
# classical (non-martingale) optimal transport problem that ot.emd2 solves.
"""

from __future__ import annotations

import numpy as np
import ot


def w1_distance(atoms_a: np.ndarray, weights_a: np.ndarray, atoms_b: np.ndarray, weights_b: np.ndarray) -> float:
    """Wasserstein-1 distance between two finitely supported measures in R^d."""
    cost = ot.dist(atoms_a, atoms_b, metric="euclidean")
    weights_a = np.asarray(weights_a, dtype=float)
    weights_b = np.asarray(weights_b, dtype=float)
    return float(ot.emd2(weights_a, weights_b, cost))
