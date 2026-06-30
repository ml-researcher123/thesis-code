"""Composition of two stages under controllable hardness dependence (for E3b).

E3 (v1) composed the retrieval and compression stages assuming *independent* per-item
errors, giving pipeline recall = p_R * p_C. Reality need not be independent: whether the
same items are hard for both stages (positive correlation) or hard for different stages
(negative correlation) changes the composition.

We model this with a Gaussian copula on the per-item success events while holding the
marginals (p_R, p_C, measured from the real E1/E2 walls) fixed:

  - draw (g_R, g_C) ~ N(0, [[1, rho],[rho, 1]])
  - item succeeds at a stage iff its gaussian < Phi^{-1}(marginal)  (so P(success)=marginal)
  - pipeline succeeds iff BOTH succeed

Limits (exact):
  rho = +1 -> pipeline = min(p_R, p_C)              (aligned hardness: redundant failures)
  rho =  0 -> pipeline = p_R * p_C                  (independent: the E3 v1 baseline)
  rho = -1 -> pipeline = max(0, p_R + p_C - 1)      (misaligned: Frechet lower bound)

So negative rho (misalignment) is the only regime that compounds *super-multiplicatively*
(pipeline < p_R * p_C). The sign of rho in real corpora is an empirical question this frames.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def pipeline_recall(p_R: float, p_C: float, rho: float, n: int = 200_000, seed: int = 0) -> float:
    """Monte-Carlo pipeline recall under a Gaussian copula with success correlation rho."""
    if p_R <= 0 or p_C <= 0:
        return 0.0
    if p_R >= 1 and p_C >= 1:
        return 1.0
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    zR = norm.ppf(min(max(p_R, 1e-6), 1 - 1e-9))
    zC = norm.ppf(min(max(p_C, 1e-6), 1 - 1e-9))
    s_R = z[:, 0] < zR
    s_C = z[:, 1] < zC
    return float(np.mean(s_R & s_C))


def frechet_bounds(p_R: float, p_C: float) -> tuple[float, float]:
    """(lower, upper) = (max(0, p_R+p_C-1), min(p_R, p_C))."""
    return max(0.0, p_R + p_C - 1.0), min(p_R, p_C)
