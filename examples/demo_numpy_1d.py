#!/usr/bin/env python3
# ------------------------------------------------------------------
# 1D hard-rank f-divergence demo vs true values (NumPy backend)
# ------------------------------------------------------------------

import argparse
import numpy as np
from ranks.helpers import js_normal_1d, js_true_mixture_vs_gaussian
from ranks.ranks_hard import rs_f_divergence_1d


# ---------- analytic formulas ----------

def kl_gaussians(mu0, s0, mu1, s1):
    """KL(N0||N1)"""
    return np.log(s1/s0) + (s0**2 + (mu0 - mu1)**2)/(2*s1**2) - 0.5

def hellinger2_gaussians(mu0, s0, mu1, s1):
    """Squared Hellinger between 1D normals"""
    den = s0**2 + s1**2
    return 2.0 * (1.0 - np.sqrt(2.0*s0*s1/den) * np.exp(-(mu0-mu1)**2/(4.0*den)))

def reverse_kl_gaussians(mu0, s0, mu1, s1):
    """KL(N1||N0)"""
    return kl_gaussians(mu1, s1, mu0, s0)


# ---------- examples ----------

def mean_shift_js(rng, K, n_mu, n_nu):
    print("\n=== Jensen–Shannon: N(0,1) vs N(Δ,1) ===")
    for d in [0.0, 0.5, 1.0, 2.0]:
        mu = rng.normal(0.0, 1.0, n_mu)
        nu = rng.normal(d, 1.0, n_nu)
        est = rs_f_divergence_1d(mu, nu, K=K, f="js").D
        true = js_normal_1d(0.0, 1.0, d, 1.0)
        print(f"Δ={d:>4.1f} → rank={est:.6f}, true={true:.6f}, ratio={est/true if true>0 else np.nan:.3f}")

def mean_shift_kl(rng, K, n_mu, n_nu):
    print("\n=== KL: N(0,1) vs N(Δ,1) ===")
    for d in [0.0, 0.5, 1.0, 2.0]:
        mu = rng.normal(0.0, 1.0, n_mu)
        nu = rng.normal(d, 1.0, n_nu)
        est = rs_f_divergence_1d(mu, nu, K=K, f="kl").D
        true = kl_gaussians(0.0, 1.0, d, 1.0)
        print(f"Δ={d:>4.1f} → rank={est:.6f}, true={true:.6f}, ratio={est/true if true>0 else np.nan:.3f}")

def scale_change_kl(rng, K, n_mu, n_nu):
    print("\n=== KL: N(0,1) vs N(0,σ) ===")
    for s in [1.0, 1.2, 1.5, 2.0]:
        mu = rng.normal(0.0, 1.0, n_mu)
        nu = rng.normal(0.0, s, n_nu)
        est = rs_f_divergence_1d(mu, nu, K=K, f="kl").D
        true = kl_gaussians(0.0, 1.0, 0.0, s)
        print(f"σ={s:>4.1f} → rank={est:.6f}, true={true:.6f}, ratio={est/true if true>0 else np.nan:.3f}")

def scale_change_hellinger(rng, K, n_mu, n_nu):
    print("\n=== Hellinger²: N(0,1) vs N(0,σ) ===")
    for s in [1.0, 1.2, 1.5, 2.0]:
        mu = rng.normal(0.0, 1.0, n_mu)
        nu = rng.normal(0.0, s, n_nu)
        est = rs_f_divergence_1d(mu, nu, K=K, f="hellinger2").D
        true = hellinger2_gaussians(0.0, 1.0, 0.0, s)
        print(f"σ={s:>4.1f} → rank={est:.6f}, true={true:.6f}, ratio={est/true if true>0 else np.nan:.3f}")

def mixture_vs_gaussian_js(rng, K, n_mu, n_nu):
    print("\n=== JS: mixture(±Δ,1) vs N(0,1) ===")
    for d in [0.0, 0.5, 1.0, 2.0]:
        xP = np.concatenate([
            rng.normal(-d, 1.0, n_mu // 2),
            rng.normal(+d, 1.0, n_mu - n_mu // 2),
        ])
        xQ = rng.normal(0.0, 1.0, n_nu)
        est = rs_f_divergence_1d(xP, xQ, K=K, f="js").D
        true = js_true_mixture_vs_gaussian(d)
        print(f"Δ={d:>4.1f} → rank={est:.6f}, true≈{true:.6f}, ratio={est/true if true>0 else np.nan:.3f}")

def heavy_tail_js(rng, K, n_mu, n_nu):
    print("\n=== JS: Laplace vs Gaussian (approx true by MC) ===")
    mu = rng.laplace(0.0, 1.0, n_mu)
    nu = rng.normal(0.0, 1.0, n_nu)
    est = rs_f_divergence_1d(mu, nu, K=K, f="js").D

    # Correct Monte Carlo approximation
    def logp(x): return -np.abs(x) - np.log(2.0)
    def logq(x): return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)

    lp, lq = logp(mu), logq(mu)
    Ep = np.mean(np.log(2.0) + lp - np.logaddexp(lp, lq))
    lp, lq = logp(nu), logq(nu)
    Eq = np.mean(np.log(2.0) + lq - np.logaddexp(lp, lq))
    js_true = 0.5 * (Ep + Eq)

    print(f"Laplace(0,1) vs N(0,1): rank={est:.6f}, true≈{js_true:.6f}, ratio={est/js_true:.3f}")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=64, help="Rank degree K")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--n-mu", type=int, default=8000, help="Number of μ samples")
    p.add_argument("--n-nu", type=int, default=8000, help="Number of ν samples")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    mean_shift_js(rng, args.K, args.n_mu, args.n_nu)
    mean_shift_kl(rng, args.K, args.n_mu, args.n_nu)
    scale_change_kl(rng, args.K, args.n_mu, args.n_nu)
    scale_change_hellinger(rng, args.K, args.n_mu, args.n_nu)
    mixture_vs_gaussian_js(rng, args.K, args.n_mu, args.n_nu)
    heavy_tail_js(rng, args.K, args.n_mu, args.n_nu)


if __name__ == "__main__":
    main()
