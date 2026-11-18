#!/usr/bin/env python3
"""Reproducible sample-only f-divergence tutorial (NumPy + SciPy).

From the repository root:
    PYTHONPATH=./src python examples/tutorial_fdivergence_estimators.py
"""

import argparse

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ranks.ranks_hard import rs_f_divergence_1d, rs_f_divergence_sliced


def js_gaussian_shift(delta: float, nodes: int = 100) -> float:
    """Numerical JS(N(0,1), N(delta,1)) via Gauss-Hermite quadrature."""
    z, w = hermgauss(nodes)
    x_p = np.sqrt(2.0) * z
    x_q = delta + np.sqrt(2.0) * z

    def log_p(x):
        return -0.5 * (x**2 + np.log(2.0 * np.pi))

    def log_q(x):
        return -0.5 * ((x - delta) ** 2 + np.log(2.0 * np.pi))

    # JS = 0.5 E_P[log(p/m)] + 0.5 E_Q[log(q/m)], m = (p+q)/2.
    ep = np.dot(w, log_p(x_p) - np.logaddexp(log_p(x_p), log_q(x_p)) + np.log(2.0))
    eq = np.dot(w, log_q(x_q) - np.logaddexp(log_p(x_q), log_q(x_q)) + np.log(2.0))
    return float(0.5 * (ep + eq) / np.sqrt(np.pi))


def dv_linear_kl(p_train, q_train, p_test, q_test):
    """Held-out Donsker-Varadhan KL estimate with T_a(x)=a*x.

    This is a deliberately small variational family, not a neural baseline.
    """

    def objective(a):
        log_q_exp_t = logsumexp(a * q_train) - np.log(q_train.size)
        return -(a * p_train.mean() - log_q_exp_t)

    opt = minimize_scalar(objective, method="bounded", bounds=(-5.0, 5.0))
    if not opt.success:
        raise RuntimeError("Variational optimization did not converge")
    a = float(opt.x)
    held_out = a * p_test.mean() - (logsumexp(a * q_test) - np.log(q_test.size))
    return float(held_out), a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4000, help="Samples per distribution in each split")
    parser.add_argument("--K", type=int, default=64, help="Rank degree")
    parser.add_argument("--delta", type=float, default=1.0, help="Normal mean shift")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--L", type=int, default=32, help="Directions in optional sliced example")
    args = parser.parse_args()
    if args.n < 2 or args.K < 1 or args.L < 1:
        parser.error("Require n >= 2, K >= 1, L >= 1")

    rng = np.random.default_rng(args.seed)
    # Independent training/evaluation splits; rank estimates use evaluation split only.
    p_train = rng.normal(0.0, 1.0, args.n)
    q_train = rng.normal(args.delta, 1.0, args.n)
    p_test = rng.normal(0.0, 1.0, args.n)
    q_test = rng.normal(args.delta, 1.0, args.n)

    true = {
        "KL": 0.5 * args.delta**2,
        "JS": js_gaussian_shift(args.delta),
        "Hellinger2": 2.0 * (1.0 - np.exp(-args.delta**2 / 8.0)),
    }
    print(f"P=N(0,1), Q=N({args.delta:g},1); n_test={args.n}, K={args.K}")
    print("{:<16} {:>13} {:>13} {:>13}".format("Divergence", "Rank estimate", "Population", "Abs. error"))
    for label, generator in [("KL", "kl"), ("JS", "js"), ("Hellinger2", "hellinger2")]:
        estimate = rs_f_divergence_1d(p_test, q_test, K=args.K, f=generator).D
        print(f"{label:<16} {estimate:13.6f} {true[label]:13.6f} {abs(estimate - true[label]):13.6f}")

    dv, a = dv_linear_kl(p_train, q_train, p_test, q_test)
    print(f"\nHeld-out variational KL (linear T_a(x)=a*x): {dv:.6f} (fitted a={a:.3f})")
    print("The linear variational family is tailored to this equal-variance Gaussian example.")
    print("It uses a separate training split, unlike the rank estimator; not a benchmark.")

    # In higher dimensions, a sliced divergence is generally not the full divergence.
    p2 = rng.normal(0.0, 1.0, size=(args.n, 2))
    q2 = rng.normal(0.0, 1.0, size=(args.n, 2))
    q2[:, 0] += args.delta
    sliced = rs_f_divergence_sliced(p2, q2, K=args.K, f="kl", L=args.L, random_state=args.seed)
    print(f"\n2D sliced rank-KL (L={args.L}): {sliced.D:.6f}")
    print(f"2D full KL (Gaussian ground truth): {true['KL']:.6f}")
    print("Do not identify the sliced quantity with the full-dimensional KL.")


if __name__ == "__main__":
    main()
