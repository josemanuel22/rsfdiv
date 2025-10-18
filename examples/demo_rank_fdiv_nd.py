#!/usr/bin/env python3
# ------------------------------------------------------------------
# nD sliced hard-rank f-divergence demo vs true values (NumPy backend)
# ------------------------------------------------------------------

import argparse
import numpy as np
from ranks.ranks_hard import rs_f_divergence_sliced
from ranks.helpers import js_normal_1d
from scipy.special import gammaln

# ---------- analytic formulas ----------

def kl_gaussians(mu0, Sigma0, mu1, Sigma1):
    """KL(N0||N1) in R^d"""
    d = len(mu0)
    inv_S1 = np.linalg.inv(Sigma1)
    diff = mu1 - mu0
    term_trace = np.trace(inv_S1 @ Sigma0)
    term_quad = diff.T @ inv_S1 @ diff
    logdet = np.log(np.linalg.det(Sigma1) / np.linalg.det(Sigma0))
    return 0.5 * (term_trace + term_quad - d + logdet)


def hellinger2_gaussians(mu0, Sigma0, mu1, Sigma1):
    """Squared Hellinger between multivariate Gaussians"""
    Sigma_avg = 0.5 * (Sigma0 + Sigma1)
    diff = mu0 - mu1
    log_det_term = (
        0.25 * np.log(np.linalg.det(Sigma0) * np.linalg.det(Sigma1))
        - 0.5 * np.log(np.linalg.det(Sigma_avg))
    )
    quad = -0.125 * diff.T @ np.linalg.inv(Sigma_avg) @ diff
    BC = np.exp(log_det_term + quad)
    return 2.0 * (1.0 - BC)


def js_gaussians(mu0, Sigma0, mu1, Sigma1):
    """
    Approximate multivariate JS divergence between two Gaussians
    using the Gaussian-mixture moment matching approximation.

    JS ≈ ½ KL(N0 || M) + ½ KL(N1 || M),
    where M = N(μ_m, Σ_m) with mixture mean/covariance.

    Parameters
    ----------
    mu0, mu1 : ndarray, shape (d,)
        Means of the Gaussians.
    Sigma0, Sigma1 : ndarray, shape (d,d)
        Covariances.
    Returns
    -------
    float
        Approximate JS divergence (nats).
    """
    mu_m = 0.5 * (mu0 + mu1)
    # mixture covariance (moment matching)
    diff0 = (mu0 - mu_m).reshape(-1, 1)
    diff1 = (mu1 - mu_m).reshape(-1, 1)
    Sigma_m = 0.5 * (Sigma0 + Sigma1) + 0.25 * (diff0 @ diff0.T + diff1 @ diff1.T)

    kl0 = kl_gaussians(mu0, Sigma0, mu_m, Sigma_m)
    kl1 = kl_gaussians(mu1, Sigma1, mu_m, Sigma_m)
    return 0.5 * (kl0 + kl1)


# ---------- helpers ----------

def pretty_print(name, d_shift, est, scaled, true, scale_label):
    ratio = scaled / true if true > 0 else np.nan
    print(f"Δ={d_shift:>3.1f} → sliced={est:.6f}, {scale_label}={scaled:.6f}, "
          f"true={true:.6f}, ratio={ratio:.3f}")


# ---------- experiments ----------

def mean_shift_kl(rng, K, L, n_mu, n_nu, dim):
    print(f"\n=== {dim}D KL: N(0,I) vs N([Δ,0,...,0],I) ===")
    for d_shift in [0.0, 0.5, 1.0]:
        mu0 = np.zeros(dim)
        mu1 = np.zeros(dim)
        mu1[0] = d_shift
        Sigma = np.eye(dim)
        X = rng.multivariate_normal(mu0, Sigma, size=n_mu)
        Z = rng.multivariate_normal(mu1, Sigma, size=n_nu)

        res = rs_f_divergence_sliced(X, Z, K=K, f="kl", L=L, random_state=1)
        est, true = res.D, kl_gaussians(mu0, Sigma, mu1, Sigma)
        scaled = dim * est  # linear scaling
        pretty_print("KL", d_shift, est, scaled, true, f"d×sliced")


def mean_shift_hellinger(rng, K, L, n_mu, n_nu, dim):
    print(f"\n=== {dim}D Hellinger²: N(0,I) vs N([Δ,0,...,0],I) ===")
    for d_shift in [0.0, 0.5, 1.0]:
        mu0 = np.zeros(dim)
        mu1 = np.zeros(dim)
        mu1[0] = d_shift
        Sigma = np.eye(dim)
        X = rng.multivariate_normal(mu0, Sigma, size=n_mu)
        Z = rng.multivariate_normal(mu1, Sigma, size=n_nu)

        res = rs_f_divergence_sliced(X, Z, K=K, f="hellinger2", L=L, random_state=2)
        est = res.D
        true = hellinger2_gaussians(mu0, Sigma, mu1, Sigma)
        scaled = dim * est  # also roughly linear for small Δ
        pretty_print("H2", d_shift, est, scaled, true, f"d×sliced")


def mean_shift_js(rng, K, L, n_mu, n_nu, dim):
    print(f"\n=== {dim}D Jensen–Shannon: N(0,I) vs N([Δ,0,...,0],I) ===")
    for d_shift in [0.0, 0.5, 1.0]:
        mu0 = np.zeros(dim)
        mu1 = np.zeros(dim)
        mu1[0] = d_shift
        Sigma = np.eye(dim)

        X = rng.multivariate_normal(mu0, Sigma, size=n_mu)
        Z = rng.multivariate_normal(mu1, Sigma, size=n_nu)

        # Rank-based sliced estimate
        res = rs_f_divergence_sliced(X, Z, K=K, f="js", L=L, random_state=3)
        est = res.D

        # True full-dimensional JS divergence (moment-matched Gaussian approximation)
        true = js_gaussians(mu0, Sigma, mu1, Sigma)

        # For isotropic Gaussian shifts, scaling by d matches the true divergence
        scaled = dim * est
        pretty_print("JS", d_shift, est, scaled, true, f"d×sliced")
        
def scale_change_js(rng, K, L, n_mu, n_nu, dim):
    print(f"\n=== {dim}D JS: N(0,I) vs N(0,σ²I) ===")
    for sigma in [1.0, 1.2, 1.5, 2.0]:
        mu0 = np.zeros(dim)
        mu1 = np.zeros(dim)
        Sigma0 = np.eye(dim)
        Sigma1 = (sigma ** 2) * np.eye(dim)

        X = rng.multivariate_normal(mu0, Sigma0, size=n_mu)
        Z = rng.multivariate_normal(mu1, Sigma1, size=n_nu)

        res = rs_f_divergence_sliced(X, Z, K=K, f="js", L=L, random_state=4)
        est = res.D
        true = js_gaussians(mu0, Sigma0, mu1, Sigma1)
        scaled = dim * est
        pretty_print("JS-scale", sigma, est, scaled, true, f"d×sliced")
        
def anisotropic_js(rng, K, L, n_mu, n_nu, dim):
    print(f"\n=== {dim}D JS: anisotropic covariances ===")
    Sigma0 = np.eye(dim)
    Sigma1 = np.diag(np.linspace(1.0, 2.0, dim))  # stretch along axes
    mu0 = np.zeros(dim)
    mu1 = np.zeros(dim)

    X = rng.multivariate_normal(mu0, Sigma0, size=n_mu)
    Z = rng.multivariate_normal(mu1, Sigma1, size=n_nu)

    res = rs_f_divergence_sliced(X, Z, K=K, f="js", L=L, random_state=5)
    est = res.D
    true = js_gaussians(mu0, Sigma0, mu1, Sigma1)
    scaled = dim * est
    pretty_print("JS-aniso", 0.0, est, scaled, true, f"d×sliced")
    
    
def laplace_vs_gaussian_js(rng, K, L, n_mu, n_nu, dim):
    print(f"\n=== {dim}D JS: Laplace(0,I) vs N(0,I) ===")
    scale = 1.0
    X = rng.laplace(0.0, scale, size=(n_mu, dim))
    Z = rng.normal(0.0, 1.0, size=(n_nu, dim))

    res = rs_f_divergence_sliced(X, Z, K=K, f="js", L=L, random_state=10)
    est = res.D

    # --- numerically stable true JS via log-sum-exp tricks ---
    def logp(x): return -np.sum(np.abs(x), axis=1) - dim * np.log(2)
    def logq(x): return -0.5 * np.sum(x**2, axis=1) - 0.5 * dim * np.log(2 * np.pi)

    allx = np.concatenate([X, Z], axis=0)
    lp, lq = logp(allx), logq(allx)

    # stable log(0.5(p+q))
    m = np.maximum(lp, lq)
    logm = m + np.log(0.5 * (np.exp(lp - m) + np.exp(lq - m)))

    Ep = np.mean(lp[:n_mu] - logm[:n_mu])
    Eq = np.mean(lq[n_mu:] - logm[n_mu:])
    js_true = 0.5 * (Ep + Eq)
    js_true = max(js_true, 0.0)  # numerical guard

    scaled = dim * est
    ratio = scaled / js_true if js_true > 0 else np.nan
    print(f"JS(rank)={scaled:.6f}, JS(true)={js_true:.6f}, ratio={ratio:.3f}")
    

def student_vs_gaussian_kl(rng, K, L, n_mu, n_nu, dim):
    print(f"\n=== {dim}D KL: Student-t(ν=3) vs N(0,I) ===")
    df = 3
    # heavy-tailed Student-t vs standard normal
    X = rng.standard_t(df=df, size=(n_mu, dim))
    Z = rng.normal(0.0, 1.0, size=(n_nu, dim))

    res = rs_f_divergence_sliced(X, Z, K=K, f="kl", L=L, random_state=11)
    est = res.D

    # ---------- True KL by Monte-Carlo ----------
    # log-pdfs
    def logp(x):
        c = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi)
        return np.sum(c - 0.5 * (df + 1) * np.log1p((x**2) / df), axis=1)

    def logq(x):
        return -0.5 * np.sum(x**2, axis=1) - 0.5 * dim * np.log(2 * np.pi)

    # expectation under t-distribution
    kl_true_total = np.mean(logp(X) - logq(X))
    kl_true_total = max(kl_true_total, 0.0)

    # per-dimension true KL for fair comparison with sliced estimator
    kl_true_per_dim = kl_true_total / dim

    # scaling of sliced KL (roughly linear in d)
    scaled = dim * est

    print(f"sliced={est:.6f}, d×sliced={scaled:.6f}, "
          f"true_total={kl_true_total:.6f}, true/d={kl_true_per_dim:.6f}, "
          f"ratio(d×sliced/true_total)={scaled/kl_true_total if kl_true_total>0 else np.nan:.3f}, "
          f"ratio(sliced/true/d)={est/kl_true_per_dim if kl_true_per_dim>0 else np.nan:.3f}")
    
def mixture_vs_gaussian_js_nd(rng, K, L, n_mu, n_nu, dim):
    print(f"\n=== {dim}D JS: ½N(-Δe₁,I)+½N(+Δe₁,I) vs N(0,I) ===")
    Δ = 1.0
    half = n_mu // 2
    X = np.vstack([
        rng.normal(-Δ, 1.0, size=(half, dim)),
        rng.normal(+Δ, 1.0, size=(n_mu - half, dim))
    ])
    Z = rng.normal(0.0, 1.0, size=(n_nu, dim))

    res = rs_f_divergence_sliced(X, Z, K=K, f="js", L=L, random_state=12)
    est = res.D

    # ---------- stable true JS ----------
    def log_mix(x):
        # log(0.5 N(x|-Δ,I) + 0.5 N(x|+Δ,I))
        lp1 = -0.5 * np.sum((x + Δ)**2, axis=1) - 0.5 * dim * np.log(2*np.pi)
        lp2 = -0.5 * np.sum((x - Δ)**2, axis=1) - 0.5 * dim * np.log(2*np.pi)
        m = np.maximum(lp1, lp2)
        return m + np.log(0.5 * (np.exp(lp1 - m) + np.exp(lp2 - m)))

    def logq(x):
        return -0.5 * np.sum(x**2, axis=1) - 0.5 * dim * np.log(2*np.pi)

    # use stable log-sum-exp for mixture vs gaussian JS
    allx = np.concatenate([X, Z], axis=0)
    lp, lq = log_mix(allx), logq(allx)
    m = np.maximum(lp, lq)
    lm = m + np.log(0.5 * (np.exp(lp - m) + np.exp(lq - m)))

    Ep = np.mean(lp[:n_mu] - lm[:n_mu])
    Eq = np.mean(lq[n_mu:] - lm[n_mu:])
    js_true = 0.5 * (Ep + Eq)
    js_true = max(js_true, 0.0)

    scaled = dim * est
    pretty_print("JS-mix", Δ, est, scaled, js_true, f"d×sliced")




# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=64, help="Rank degree K")
    p.add_argument("--L", type=int, default=128, help="Number of random directions")
    p.add_argument("--dim", type=int, default=2, help="Dimension d")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--n-mu", type=int, default=10000, help="Samples from μ")
    p.add_argument("--n-nu", type=int, default=10000, help="Samples from ν")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    dim = args.dim

    mean_shift_kl(rng, args.K, args.L, args.n_mu, args.n_nu, dim)
    mean_shift_hellinger(rng, args.K, args.L, args.n_mu, args.n_nu, dim)
    mean_shift_js(rng, args.K, args.L, args.n_mu, args.n_nu, dim)
    scale_change_js(rng, args.K, args.L, args.n_mu, args.n_nu, dim)
    anisotropic_js(rng, args.K, args.L, args.n_mu, args.n_nu, dim)
    laplace_vs_gaussian_js(rng, args.K, args.L, args.n_mu, args.n_nu, dim)
    student_vs_gaussian_kl(rng, args.K, args.L, args.n_mu, args.n_nu, dim)
    mixture_vs_gaussian_js_nd(rng, args.K, args.L, args.n_mu, args.n_nu, dim)


if __name__ == "__main__":
    main()
