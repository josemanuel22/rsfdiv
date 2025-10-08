# rsfdiv.py
# -----------------------------------------------------------------------------
# Rank-Statistic f-Divergence (and Sliced version) from finite samples
#
# This module implements:
#   1) The 1D rank pmf Q^{(K)}_{mu|nu} via the mixture-of-Binomial identity
#      Q(n) = E_{Y~mu}[ BinomPMF(n; K, F_nu(Y)) ].
#   2) The rank-statistic f-divergence:
#         D^{(K)}_f(mu||nu) = (1/(K+1)) * sum_{n=0}^K f( (K+1) Q(n) ).
#   3) The sliced multivariate version by averaging D^{(K)}_f over random
#      directions on the unit sphere (pushforward by x -> s^T x).
# -----------------------------------------------------------------------------


from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Tuple, Optional, Dict
from dataclasses import dataclass
from math import isfinite
from scipy.special import loggamma  # robust binomial coefficients

import numpy as np
from numpy.polynomial.hermite import hermgauss


def js_normal_1d(mu0: float, sigma0: float, mu1: float, sigma1: float, n_nodes: int = 80) -> float:
    """
    Jensen–Shannon divergence JS(P||Q) for 1D normals P=N(mu0,sigma0^2), Q=N(mu1,sigma1^2).
    Uses Gauss–Hermite quadrature: E_{X~N(μ,σ^2)}[g(X)] = 1/sqrt(pi) * sum w_i g(μ+√2 σ x_i).
    Returns nats.
    """
    x, w = hermgauss(n_nodes)

    def logp(x):
        return -0.5*np.log(2*np.pi*sigma0**2) - 0.5*((x-mu0)/sigma0)**2

    def logq(x):
        return -0.5*np.log(2*np.pi*sigma1**2) - 0.5*((x-mu1)/sigma1)**2

    # E_P[log p - log m]
    xp = mu0 + np.sqrt(2.0)*sigma0*x
    lp, lq_at_xp = logp(xp), logq(xp)
    lm_p = np.logaddexp(lp, lq_at_xp) - np.log(2.0)     # log m(xp)
    Ep = (w @ (lp - lm_p)) / np.sqrt(np.pi)

    # E_Q[log q - log m]
    xq = mu1 + np.sqrt(2.0)*sigma1*x
    lq, lp_at_xq = logq(xq), logp(xq)
    lm_q = np.logaddexp(lp_at_xq, lq) - np.log(2.0)     # log m(xq)
    Eq = (w @ (lq - lm_q)) / np.sqrt(np.pi)

    return 0.5*(Ep + Eq)

def log_phi(x):
    return -0.5*np.log(2*np.pi) - 0.5*x**2

def log_mix_pm1(x, Delta):
    a = log_phi(x - Delta)
    b = log_phi(x + Delta)
    m = np.maximum(a, b)
    return np.log(0.5*np.exp(a-m) + 0.5*np.exp(b-m)) + m

def js_true_mixture_vs_gaussian(Delta, n_mc=200000, rng_seed=2):
    rng_local = np.random.default_rng(rng_seed)
    x_half = n_mc//2
    xP = np.concatenate([rng_local.normal(-Delta, 1.0, size=x_half),
                         rng_local.normal( Delta, 1.0, size=n_mc - x_half)])
    xQ = rng_local.normal(0.0, 1.0, size=n_mc)
    lpP = log_mix_pm1(xP, Delta); lqP = log_phi(xP)
    lpQ = log_mix_pm1(xQ, Delta); lqQ = log_phi(xQ)
    mP = np.maximum(lpP, lqP); lmP = np.log(0.5*np.exp(lpP-mP) + 0.5*np.exp(lqP-mP)) + mP
    mQ = np.maximum(lpQ, lqQ); lmQ = np.log(0.5*np.exp(lpQ-mQ) + 0.5*np.exp(lqQ-mQ)) + mQ
    return 0.5*(np.mean(lpP - lmP) + np.mean(lqQ - lmQ))


# ---------------------------
# Entropy functions f(t)
# ---------------------------


def f_kl(t: np.ndarray) -> np.ndarray:
    """
    Kullback–Leibler generator: f(t) = t log t - (t - 1), with f(0) := 1 (limit).

    Parameters
    ----------
    t : ndarray
        Nonnegative array (likelihood ratios).

    Returns
    -------
    ndarray
        Elementwise f(t).
    """
    t = np.asarray(t)
    out = np.empty_like(t, dtype=np.float64)
    mask_pos = t > 0
    out[mask_pos] = t[mask_pos] * np.log(t[mask_pos]) - (t[mask_pos] - 1.0)
    out[~mask_pos] = 1.0  # limit t->0+
    return out


def f_revkl(t: np.ndarray) -> np.ndarray:
    """
    Reverse KL generator: f(t) = -log t + t - 1  (domain t>0, clipped at eps).

    Parameters
    ----------
    t : ndarray

    Returns
    -------
    ndarray
    """
    t = np.asarray(t)
    eps = 1e-300
    return -(np.log(np.clip(t, eps, None))) + t - 1.0


def f_tv(t: np.ndarray) -> np.ndarray:
    """
    Total variation generator used by ISL: f(t) = |t - 1|.

    Parameters
    ----------
    t : ndarray

    Returns
    -------
    ndarray
    """
    return np.abs(np.asarray(t) - 1.0)


def f_hellinger2(t: np.ndarray) -> np.ndarray:
    """
    Squared Hellinger generator: f(t) = (sqrt(t) - 1)^2, f(t<0) = +inf.

    Parameters
    ----------
    t : ndarray

    Returns
    -------
    ndarray
    """
    t = np.asarray(t)
    out = np.empty_like(t, dtype=np.float64)
    mask = t >= 0
    out[mask] = (np.sqrt(t[mask]) - 1.0) ** 2
    out[~mask] = np.inf
    return out


def f_pearson_chi2(t: np.ndarray) -> np.ndarray:
    """
    Pearson chi^2 generator: f(t) = (t - 1)^2.

    Parameters
    ----------
    t : ndarray

    Returns
    -------
    ndarray
    """
    t = np.asarray(t)
    return (t - 1.0) ** 2


def f_neyman_chi2(t: np.ndarray) -> np.ndarray:
    """
    Neyman chi^2 generator: f(t) = (1 - t)^2 / t, with f(0) = +inf.
    """
    t = np.asarray(t)
    eps = 1e-300
    return ((1.0 - t) ** 2) / np.clip(t, eps, None)


def f_js(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    eps = 1e-300
    u = np.clip(t, eps, None)
    return 0.5 * (u * (np.log(2.0 * u) - np.log1p(u)) + (np.log(2.0) - np.log1p(u)))


F_LIBRARY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "kl": f_kl,
    "reverse_kl": f_revkl,
    "tv": f_tv,
    "hellinger2": f_hellinger2,
    "pearson_chi2": f_pearson_chi2,
    "neyman_chi2": f_neyman_chi2,
    "js": f_js,
}


def get_f(name_or_callable: "str|Callable[[np.ndarray], np.ndarray]") -> Callable[[np.ndarray], np.ndarray]:
    """
    Return an entropy function f(t) either by name or as-is if a callable is provided.

    Parameters
    ----------
    name_or_callable : str or Callable
        Either one of: 'kl', 'reverse_kl', 'tv', 'hellinger2', 'pearson_chi2',
        'neyman_chi2', 'js'; or a custom callable f(t)->array.

    Returns
    -------
    Callable
        The function f(t).

    Raises
    ------
    KeyError
        If a string key is unknown.
    """
    if callable(name_or_callable):
        return name_or_callable
    key = str(name_or_callable).lower()
    if key not in F_LIBRARY:
        raise KeyError(f"Unknown f-divergence '{name_or_callable}'. "
                       f"Available: {', '.join(sorted(F_LIBRARY.keys()))} or pass a callable.")
    return F_LIBRARY[key]

# ---------------------------
# Utilities
# ---------------------------


def empirical_cdf_at(x_ref_sorted: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """
    Right-continuous empirical CDF of the reference sample at query points.

    Computes:
        F_hat(x) = (1/N_ref) * #{ ref_j <= x }.

    Parameters
    ----------
    x_ref_sorted : ndarray, shape (N_ref,)
        Sorted reference sample (nu).
    x_query : ndarray, shape (N_mu,)
        Query points (mu samples).

    Returns
    -------
    ndarray, shape (N_mu,)
        F_hat(x_query[i]) for each query.
    """
    idx = np.searchsorted(x_ref_sorted, x_query, side="right")
    return idx / x_ref_sorted.size


def log_binom_pmf(n: int, K: int, t: np.ndarray) -> np.ndarray:
    """
    Log pmf of Binomial(K, t) at n with careful boundary handling.

    Parameters
    ----------
    n : int
        Target count (0..K).
    K : int
        Number of trials.
    t : ndarray
        Success probabilities in [0,1].

    Returns
    -------
    ndarray
        log P(N=n | K, t).
    """
    t = np.asarray(t, dtype=np.float64)
    out = np.full_like(t, -np.inf)

    # exact mass at boundaries
    mask0 = t <= 0.0
    if n == 0:
        out[mask0] = 0.0  # log(1) = 0
    else:
        out[mask0] = -np.inf

    mask1 = t >= 1.0
    if n == K:
        out[mask1] = 0.0
    else:
        out[mask1] = -np.inf

    mask = (~mask0) & (~mask1)
    if np.any(mask):
        tt = t[mask]
        C = loggamma(K + 1.0) - loggamma(n + 1.0) - loggamma(K - n + 1.0)
        out[mask] = C + n * np.log(tt) + (K - n) * np.log1p(-tt)
    return out


def binom_pmf(n: int, K: int, t: np.ndarray) -> np.ndarray:
    """
    Binomial pmf at n given K and vector of probabilities t.

    Returns
    -------
    ndarray
        P(N=n | K, t) elementwise.
    """
    return np.exp(log_binom_pmf(n, K, t))

# ---------------------------
# Core: Q^{(K)} and D^{(K)}_f
# ---------------------------


@dataclass
class RSFDivergenceResult:
    """
    Container for 1D rank-statistic f-divergence results.

    Attributes
    ----------
    D : float
        Divergence value D^{(K)}_f(mu||nu).
    Q : ndarray, shape (K+1,)
        Rank pmf Q^{(K)}_{mu|nu}(n), n=0..K.
    t_values : ndarray, shape (N_mu,)
        Diagnostics: t_i = F_hat_nu(y_i) for y_i ~ mu.
    """
    D: float
    Q: np.ndarray               # shape (K+1,)
    # F_nu(y_i) for each y_i (diagnostic), shape (N_mu,)
    t_values: np.ndarray


def rank_statistic_pmf_1d(
    mu_samples: ArrayLike,
    nu_samples: ArrayLike,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Q^{(K)}_{mu|nu}(n) for n=0..K via the mixture-of-Binomial formula.

    Q(n) = E_{Y~mu}[ BinomPMF(n; K, F_hat_nu(Y)) ].

    Parameters
    ----------
    mu_samples : array-like, shape (N_mu,)
        Samples from mu (target).
    nu_samples : array-like, shape (N_nu,)
        Samples from nu (reference).
    K : int
        Rank degree (number of reference points used in the rank).

    Returns
    -------
    Q : ndarray, shape (K+1,)
        Discrete pmf over n=0..K.
    t_values : ndarray, shape (N_mu,)
        Empirical CDF values F_hat_nu(y_i).

    Raises
    ------
    ValueError
        If inputs are empty or K < 0.
    """
    y = np.asarray(mu_samples, dtype=np.float64).ravel()
    y_tilde = np.asarray(nu_samples, dtype=np.float64).ravel()
    if y.size == 0 or y_tilde.size == 0:
        raise ValueError("mu_samples and nu_samples must be non-empty.")
    if K < 0:
        raise ValueError("K must be a non-negative integer.")
    y_tilde_sorted = np.sort(y_tilde)
    t = empirical_cdf_at(y_tilde_sorted, y)  # shape (N_mu,)

    Q = np.empty(K + 1, dtype=np.float64)
    for n in range(K + 1):
        Q[n] = binom_pmf(n, K, t).mean()
    return Q, t


def rs_f_divergence_1d(
    mu_samples: ArrayLike,
    nu_samples: ArrayLike,
    K: int,
    f: "str|Callable[[np.ndarray], np.ndarray]" = "kl",
) -> RSFDivergenceResult:
    """
    Rank-statistic f-divergence in 1D:
        D^{(K)}_f = (1/(K+1)) * sum_{n=0}^K f( (K+1) * Q(n) ).

    Parameters
    ----------
    mu_samples : array-like, shape (N_mu,)
    nu_samples : array-like, shape (N_nu,)
    K : int
        Rank degree.
    f : str or Callable, default="kl"
        Entropy generator. Use a registered name or a custom callable f(t).

    Returns
    -------
    RSFDivergenceResult
        (D, Q, t_values) as defined above.
    """
    f_fun = get_f(f)
    Q, t = rank_statistic_pmf_1d(mu_samples, nu_samples, K)
    D = (f_fun((K + 1.0) * Q)).mean()
    return RSFDivergenceResult(D=float(D), Q=Q, t_values=t)

# ---------------------------
# Sliced multivariate version
# ---------------------------


def _sample_unit_sphere(d: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample L random directions uniformly on the unit sphere S^{d-1} by
    normalizing i.i.d. Gaussian vectors.
    """
    M = rng.normal(size=(L, d))
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


@dataclass
class SlicedRSFDivergenceResult:
    """
    Container for sliced/multivariate results.

    Attributes
    ----------
    D : float
        Sliced divergence: mean over directions.
    per_direction : ndarray, shape (L,)
        Divergence values per direction.
    directions : ndarray, shape (L, d)
        The sampled unit directions used.
    """
    D: float
    per_direction: np.ndarray     # shape (L,)
    directions: np.ndarray        # shape (L,d)


def rs_f_divergence_sliced(
    mu_samples: ArrayLike,
    nu_samples: ArrayLike,
    K: int,
    f: "str|Callable[[np.ndarray], np.ndarray]" = "kl",
    L: int = 64,
    random_state: Optional[int] = 0,
) -> SlicedRSFDivergenceResult:
    """
    Sliced rank-statistic f-divergence in R^d:
        SR^{(K)}_f = E_{s ~ Unif(S^{d-1})}[ D^{(K)}_f( s#mu || s#nu ) ].

    Parameters
    ----------
    mu_samples : array-like, shape (N_mu, d) or (N_mu,)
        Samples from mu.
    nu_samples : array-like, shape (N_nu, d) or (N_nu,)
        Samples from nu.
    K : int
        Rank degree.
    f : str or Callable, default="kl"
        Entropy generator.
    L : int, default=64
        Number of random directions to average over.
    random_state : int or None, default=0
        Seed for reproducibility.

    Returns
    -------
    SlicedRSFDivergenceResult
        Mean value, per-direction values, and the sampled directions.

    Notes
    -----
    - If inputs are 1D, we fall back to the 1D routine directly.
    - For high d and/or large L, this is embarrassingly parallel over directions.
    """
    X = np.asarray(mu_samples, dtype=np.float64)
    Z = np.asarray(nu_samples, dtype=np.float64)

    if X.ndim == 1:
        # fall back to 1D directly
        res = rs_f_divergence_1d(X, Z, K, f)
        return SlicedRSFDivergenceResult(D=res.D, per_direction=np.array([res.D]), directions=np.array([[1.0]]))

    if X.ndim != 2 or Z.ndim != 2:
        raise ValueError(
            "mu_samples and nu_samples must be 1D or 2D arrays of shape (N, d).")
    if X.shape[1] != Z.shape[1]:
        raise ValueError(
            "mu_samples and nu_samples must have the same dimensionality.")
    if X.shape[0] == 0 or Z.shape[0] == 0:
        raise ValueError("mu_samples and nu_samples must be non-empty.")

    d = X.shape[1]
    rng = np.random.default_rng(random_state)
    S = _sample_unit_sphere(d, L, rng)  # (L,d)
    f_fun = get_f(f)

    vals = np.empty(L, dtype=np.float64)
    for i, s in enumerate(S):
        y = X @ s
        y_tilde = Z @ s
        Q, _ = rank_statistic_pmf_1d(y, y_tilde, K)
        vals[i] = (f_fun((K + 1.0) * Q)).mean()

    return SlicedRSFDivergenceResult(D=float(vals.mean()), per_direction=vals, directions=S)


# ---------------------------
# Quick examples / smoke tests
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(123)
    
    K = 100
    nP = 10000
    nQ = 10000

    # 1) 1D: N(0,1) vs N(1,1)
    
    mu = rng.normal(loc=0.0, scale=1.0, size=10000)
    nu = rng.normal(loc=1.0, scale=1.0, size=10000)
    res = rs_f_divergence_1d(mu, nu, K=K, f="kl")
    print(f"1D KL (K={K}): {res.D:.6f}")
    print(f"True KL: {0.5*((0.0-1.0)**2)/1.0:.6f}")
    
    #1D: Hellinger^2 for N(mu0, s0^2) vs N(mu1, s1^2) (unequal variances)
    mu0, s0 = 0.0, 1.0
    mu1, s1 = 0.5, 1.3
    mu = rng.normal(loc=mu0, scale=s0, size=10000)
    nu = rng.normal(loc=mu1, scale=s1, size=10000)

    res = rs_f_divergence_1d(mu, nu, K=K, f="hellinger2")

    den = s0**2 + s1**2
    H2_true_full = 2.0 * (1.0 - np.sqrt(2.0*s0*s1/den) * np.exp(-(mu0-mu1)**2/(4.0*den)))
    print(f"1D Hellinger^2 (K={K}): {res.D:.6f}")
    print("True Hellinger^2 (full):", H2_true_full)
    print(f"Ratio D^(K)/true:      {res.D / H2_true_full:.3f}")
    
    # 1D: Jensen–Shannon for Gaussians

    mu0, s0 = 0.0, 1.0
    mu1, s1 = 0.6, 1.2
    mu = rng.normal(loc=mu0, scale=s0, size=10000)
    nu = rng.normal(loc=mu1, scale=s1, size=10000)

    res = rs_f_divergence_1d(mu, nu, K=K, f="js")
    JS_true = js_normal_1d(mu0, s0, mu1, s1)

    print(f"1D JS (K={K}): {res.D:.6f}")
    print(f"True JS:       {JS_true:.6f}")
    print(f"Ratio:         {res.D / JS_true:.3f}")
    
    # Extreme 1D JS example: approach log(2)
    mu = rng.normal(0.0, 1.0, 10000)
    nu = rng.normal(10.0, 1.0, 10000)

    res = rs_f_divergence_1d(mu, nu, K=K, f="js")
    JS_true = js_normal_1d(0.0, 1.0, 10.0, 1.0)

    print(f"1D JS (K={K}):   {res.D:.6f}")
    print(f"True JS:          {JS_true:.6f}  (log 2 ≈ {np.log(2):.6f})")
    
    # Small sweep in Δ for JS (visual) + variance
    deltas = np.linspace(0.0, 2.0, 9)
    R = 10  # repeats per Δ

    vals_hat_mean, vals_hat_var, vals_hat_std = [], [], []
    vals_true_mean, vals_true_var, vals_true_std = [], [], []

    for d in deltas:
        hat_r, true_r = [], []
        for _ in range(R):
            xP = np.concatenate([rng.normal(-d, 1.0, size=nP//2),
                                rng.normal( d, 1.0, size=nP - nP//2)])
            xQ = rng.normal(0.0, 1.0, size=nQ)
            hat_r.append(float(rs_f_divergence_1d(xP, xQ, K=K, f="js").D))
            true_r.append(float(js_true_mixture_vs_gaussian(d, n_mc=nP)))  # increase n_mc for lower MC variance

        hat_r = np.array(hat_r, dtype=float)
        true_r = np.array(true_r, dtype=float)

        vals_hat_mean.append(hat_r.mean())
        vals_hat_var.append(hat_r.var(ddof=1))
        vals_hat_std.append(hat_r.std(ddof=1))

        vals_true_mean.append(true_r.mean())
        vals_true_var.append(true_r.var(ddof=1))
        vals_true_std.append(true_r.std(ddof=1))

    # (optional) turn into arrays if you’ll plot with error bars
    vals_hat_mean = np.array(vals_hat_mean); vals_hat_std = np.array(vals_hat_std)
    vals_true_mean = np.array(vals_true_mean); vals_true_std = np.array(vals_true_std)

    print(f"Mean std ranks-div: {np.mean(vals_hat_std):.12f}, max std ranks-div:{np.max(vals_hat_std):.12f}")
    print(f"Mean std mc: {np.mean(vals_true_std):.12f},  max std mc:{np.max(vals_true_std):.12f}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(deltas, vals_true_mean, marker='x', linestyle=':', label="True JS (MC)")
    plt.plot(deltas, vals_hat_mean, marker='o', linestyle='-', label=f"Rank (K={K}) JS")
    plt.fill_between(deltas, vals_hat_mean - vals_hat_std, vals_hat_mean + vals_hat_std, alpha=0.2)
    plt.xlabel("Δ (mixture separation)")
    plt.ylabel("Jensen–Shannon divergence (nats)")
    plt.title("Mixture (0.5 N(-Δ,1) + 0.5 N(Δ,1))  vs  Q=N(0,1)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    # 2) Sliced 2D: mean shift along x-axis (KL)
    L = 128
    X = rng.normal(size=(nP, 2))
    Z = rng.normal(loc=(0.5, 0.0), scale=1.0, size=(nQ, 2))

    sres = rs_f_divergence_sliced(X, Z, K=K, f="kl", L=L, random_state=1)
    print(f"Sliced KL (K={K}, L={L}): {sres.D:.6f}")

    # True KL for equal covariances (I2): 0.5 * ||Δ||^2, here Δ = (0.5, 0)
    KL_true = 0.5 * (0.5**2)
    print(f"True KL (closed form):    {KL_true:.6f}")

    # Check the factor in d=2
    print(f"d * Sliced KL:             {2*sres.D:.6f}  (should match true)")

    # 2) Sliced 2D: mean shift along x-axis
    L = 128
    X = rng.normal(size=(nP, 2))
    Z = rng.normal(loc=(0.5, 0.0), scale=1.0, size=(nQ, 2))
    sres = rs_f_divergence_sliced(
        X, Z, K=K, f="hellinger2", L=L, random_state=1)
    print(f"Sliced Hellinger^2 (K={K}, L={L}): {sres.D:.6f}")

    def hellinger2_gaussians(mu1, Sigma1, mu2, Sigma2):
        mu1 = np.asarray(mu1); mu2 = np.asarray(mu2)
        Sigma1 = np.asarray(Sigma1); Sigma2 = np.asarray(Sigma2)
        dmu = mu1 - mu2
        Sigma_avg = 0.5 * (Sigma1 + Sigma2)

        # log-determinants for numerical stability
        s1, logdet1 = np.linalg.slogdet(Sigma1)
        s2, logdet2 = np.linalg.slogdet(Sigma2)
        sa, logdetA = np.linalg.slogdet(Sigma_avg)
        if s1 <= 0 or s2 <= 0 or sa <= 0:
            raise ValueError("Covariances must be SPD.")

        # log Bhattacharyya coefficient
        quad = dmu @ np.linalg.solve(Sigma_avg, dmu)
        log_BC = 0.25 * (logdet1 + logdet2) - 0.5 * logdetA - 0.125 * quad
        BC = np.exp(log_BC)
        return 2.0 * (1.0 - BC)

    # For your case: N([0,0], I2) vs N([0.5,0], I2)
    H2_true = hellinger2_gaussians(mu1=[0.0, 0.0], Sigma1=np.eye(2),
                                mu2=[0.5, 0.0], Sigma2=np.eye(2))
    print(f"True Hellinger^2 (closed form): {H2_true:.6f}")
    print(f"Gap (true - sliced):            {H2_true - sres.D:.6f}")
