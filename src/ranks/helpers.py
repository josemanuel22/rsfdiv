import numpy as np
from numpy.polynomial.hermite import hermgauss

# -----------------------------------------------------------------------------
# Analytical helpers for JS with Gaussians (for validation)
# -----------------------------------------------------------------------------


def js_normal_1d(mu0: float, sigma0: float, mu1: float, sigma1: float, n_nodes: int = 80) -> float:
    """
    Jensen–Shannon divergence JS(P||Q) for 1D normals
    P=N(mu0,sigma0^2), Q=N(mu1,sigma1^2).
    Uses Gauss–Hermite quadrature:
        E_{X~N(μ,σ^2)}[g(X)] = 1/sqrt(pi) * sum w_i g(μ+√2 σ x_i).
    Returns nats.
    """
    x, w = hermgauss(n_nodes)

    def logp(x_):
        return -0.5 * np.log(2 * np.pi * sigma0**2) - 0.5 * ((x_ - mu0) / sigma0) ** 2

    def logq(x_):
        return -0.5 * np.log(2 * np.pi * sigma1**2) - 0.5 * ((x_ - mu1) / sigma1) ** 2

    # E_P[log p - log m]
    xp = mu0 + np.sqrt(2.0) * sigma0 * x
    lp = logp(xp)
    lq_at_xp = logq(xp)
    lm_p = np.logaddexp(lp, lq_at_xp) - np.log(2.0)  # log m(xp)
    Ep = (w @ (lp - lm_p)) / np.sqrt(np.pi)

    # E_Q[log q - log m]
    xq = mu1 + np.sqrt(2.0) * sigma1 * x
    lq = logq(xq)
    lp_at_xq = logp(xq)
    lm_q = np.logaddexp(lp_at_xq, lq) - np.log(2.0)  # log m(xq)
    Eq = (w @ (lq - lm_q)) / np.sqrt(np.pi)

    return 0.5 * (Ep + Eq)


def _log_phi(x):
    return -0.5 * np.log(2 * np.pi) - 0.5 * x**2


def _log_mix_pm1(x, Delta):
    a = _log_phi(x - Delta)
    b = _log_phi(x + Delta)
    m = np.maximum(a, b)
    return np.log(0.5 * np.exp(a - m) + 0.5 * np.exp(b - m)) + m


def js_true_mixture_vs_gaussian(Delta, n_mc: int = 200_000, rng_seed: int = 2) -> float:
    """
    Approximate JS between a symmetric two-Gaussian mixture (means ±Delta, var=1)
    and a standard normal, via simple Monte Carlo.
    """
    rng_local = np.random.default_rng(rng_seed)
    x_half = n_mc // 2
    xP = np.concatenate(
        [
            rng_local.normal(-Delta, 1.0, size=x_half),
            rng_local.normal(Delta, 1.0, size=n_mc - x_half),
        ]
    )
    xQ = rng_local.normal(0.0, 1.0, size=n_mc)
    lpP = _log_mix_pm1(xP, Delta)
    lqP = _log_phi(xP)
    lpQ = _log_mix_pm1(xQ, Delta)
    lqQ = _log_phi(xQ)
    mP = np.maximum(lpP, lqP)
    lmP = np.log(0.5 * np.exp(lpP - mP) + 0.5 * np.exp(lqP - mP)) + mP
    mQ = np.maximum(lpQ, lqQ)
    lmQ = np.log(0.5 * np.exp(lpQ - mQ) + 0.5 * np.exp(lqQ - mQ)) + mQ
    return 0.5 * (np.mean(lpP - lmP) + np.mean(lqQ - lmQ))
