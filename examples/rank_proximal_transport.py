#!/usr/bin/env python3
# ---------------------------------------------------------------
# Rank–Proximal Transport using *Bernstein* rank pmf (via ranks_soft)
# ---------------------------------------------------------------

import os, math, argparse
import torch
import torch.nn.functional as F

# ---- import the shared rank-stat toolkit ----
from ranks.ranks_torch import (
    soft_ecdf_Q_of_x,         # smooth ECDF σ((x - y)/τ)
    bernstein_basis,          # B_{n,K}(u)
    discrete_f_div_from_pmf,  # ∑_n q f(p_n/q), with q = 1/(K+1)
    get_f,                    # f-generator fetcher: 'js' | 'kl' | 'hellinger2'
)

def _get_f_any(name: str, alpha: float):
    key = (name or "").lower()
    if key in ("js", "jensen-shannon", "kl", "hellinger2", "hell2"):
        return get_f(key)
    if key in ("rkl", "reverse_kl", "revkl"):
        return _f_reverse_kl
    if key in ("alpha", "a-div", "adiv"):
        return _f_alpha(alpha)
    raise KeyError(f"Unknown f-divergence '{name}'. "
                   "Available: js, kl, hellinger2, reverse_kl, alpha")

# ---------------------------- slices ----------------------------

def make_slices(L: int, d: int, device=None, dtype=None, anchored=True):
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    S_list = []
    if anchored:
        I = torch.eye(d, device=device, dtype=dtype)
        S_list += [I, -I]
        if d >= 2:
            diag = F.normalize(torch.ones(d, device=device, dtype=dtype), dim=0)
            S_list += [diag[None, :], (-diag)[None, :]]
    S = torch.cat(S_list, dim=0) if S_list else torch.zeros(0, d, device=device, dtype=dtype)
    rem = max(0, L - S.size(0))
    if rem > 0:
        R = torch.randn(rem, d, device=device, dtype=dtype)
        R = F.normalize(R, dim=1)
        S = torch.cat([S, R], dim=0)
    if S.size(0) > L:
        S = S[:L]
    return F.normalize(S, dim=1)

# ---------------------- soft ECDF (library wrapper) -----------------------

def _soft_ecdf_standardized(x_proj: torch.Tensor, y_proj: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Match the original behavior:
      xs=(x-μ)/σ, ys=(y-μ)/σ, diff=(xs-ys)/tau
    which is equivalent to using τ_eff = tau * σ and centering by μ.
    """
    mu = y_proj.mean()
    sd = y_proj.std().clamp_min(1e-6)
    return soft_ecdf_Q_of_x(x_proj - mu, y_proj - mu, tau=tau * sd).clamp(1e-6, 1 - 1e-6)

# --------------------- Bernstein rank histogram -----------------

def _bernstein_histogram(U: torch.Tensor, K: int, alpha: float = 1e-4):
    """
    Smooth pmf p over {0,...,K} by averaging Bernstein rows + tiny Dirichlet prior.
    Uses bernstein_basis from the shared library.
    """
    B = bernstein_basis(U, K)       # (N, K+1), rows sum to 1
    p = B.mean(dim=0)               # (K+1,)
    if alpha is not None and alpha > 0.0:
        p = (p + alpha) / (p.sum() + (K + 1) * alpha)
    else:
        p = p / p.sum()
    return p

# ---------------------- proximal update in ranks -----------------

def prox_rank_step(U0: torch.Tensor,
                   K: int,
                   eta: float = 0.5,         # strong prox
                   steps: int = 5,           # a few inner steps
                   f_name: str = "js",       # now defaults to library names
                   alpha: float = 1.6,
                   alpha_hist: float = 1e-4,
                   debug: bool = False) -> torch.Tensor:
    """
    Prox on ranks using a tiny inner optimizer over U:
      minimize  D_f(p_Bernstein(U) || uniform) + (1/(2η))||U-U0||^2
    D_f is computed via ranks_soft.discrete_f_div_from_pmf with f-generator.
    """
    f_fn = _get_f_any(f_name, alpha)

    with torch.enable_grad():
        U = torch.nn.Parameter(U0.detach().clamp(1e-6, 1 - 1e-6))
        optU = torch.optim.SGD([U], lr=float(eta), momentum=0.0)

        for k in range(max(1, steps)):
            optU.zero_grad(set_to_none=True)
            p = _bernstein_histogram(U, K=K, alpha=alpha_hist)
            D = discrete_f_div_from_pmf(p, f_fn)
            quad = 0.5 * (1.0 / max(1e-8, eta)) * ((U - U0) ** 2).mean()
            loss = D + quad
            if debug and k == 0:
                assert U.requires_grad and p.requires_grad and D.requires_grad and loss.requires_grad
            loss.backward()
            optU.step()
            with torch.no_grad():
                U.data.clamp_(1e-6, 1 - 1e-6)
        return U.detach()

# ------------------- inverse CDF via interpolation ----------------

def quantile_from_samples(y_proj: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Approximate F^{-1}(u) by sorting y and linear-interpolating.
    y_proj: (M,), u: (N,) in [0,1]  -> returns (N,)
    """
    y_sorted, _ = torch.sort(y_proj)
    M = y_sorted.numel()
    pos = (u.clamp(0.0, 1.0) * (M - 1)).to(dtype=torch.float32)
    lo = torch.clamp(pos.floor().long(), 0, M - 1)
    hi = torch.clamp(lo + 1, 0, M - 1)
    w = (pos - lo.to(pos.dtype)).clamp(0, 1)
    z = (1.0 - w) * y_sorted[lo] + w * y_sorted[hi]
    return z

# ---------------- Rank-Proximal Transport (one step) ---------------

@torch.no_grad()
def rank_proximal_transport(
    X: torch.Tensor,          # (N, d)
    Y: torch.Tensor,          # (M, d)
    *,
    L: int = 128,
    K: int = 96,
    tau: float = 0.25,        # interpreted as τ_eff = tau * std(y·s)
    f_name: str = "js",
    alpha_div: float = 1.6,   # only used if f_name in {'alpha','a-div','adiv'}
    eta: float = 0.5,         # strong prox
    inner_steps: int = 5,
    eps_move: float = 0.20,
    anchored: bool = True,
    antithetic: bool = True,
    per_slice_cap: float = 0.40,
    debug_every: int = 0
) -> torch.Tensor:
    """
    One RPT update with Bernstein rank pmf (library-backed):
      - prox in rank space (Bernstein histogram)
      - inverse-CDF pullback
      - per-slice trust region + d-scaled averaged update
    """
    dev, dt = X.device, X.dtype
    N, d = X.shape
    S = make_slices(L, d, device=dev, dtype=dt, anchored=anchored)  # (L,d)
    if antithetic:
        S = torch.cat([S, -S], dim=0)
    nS = S.size(0)

    dX = torch.zeros_like(X)

    if debug_every:
        r0 = X.norm(dim=1).mean().item()

    for i, s in enumerate(S):
        xs = X @ s
        ys = Y @ s

        # Soft ranks via library (standardization-compatible)
        U0 = _soft_ecdf_standardized(xs, ys, tau=tau)

        # Prox step in rank-space with library f-div
        U1 = prox_rank_step(
            U0, K=K, eta=eta, steps=inner_steps,
            f_name=f_name, alpha=alpha_div
        )

        # Pull back via approximate quantile on y·s
        z_star   = quantile_from_samples(ys, U1)
        delta_pr = z_star - xs

        # Per-slice trust region
        scale = (per_slice_cap / (delta_pr.abs() + 1e-8)).clamp(max=1.0)
        dX    += (eps_move * scale * delta_pr)[:, None] * s[None, :]

        if debug_every and (i % debug_every == 0):
            mean_du = (U1 - U0).abs().mean().item()
            mean_dp = delta_pr.abs().mean().item()
            print(f"[slice {i:03d}] ⟨|ΔU|⟩={mean_du:.3e}  ⟨|Δproj|⟩={mean_dp:.3e}")

    # ---- KEY: scale the average by d to counter E[ss^T]=(1/d)I ----
    X_plus = X + (d / nS) * dX

    if debug_every:
        r1 = X_plus.norm(dim=1).mean().item()
        print(f"[rpt] ⟨r⟩: {r0:.4f} → {r1:.4f}  (Δ {r1 - r0:+.4f})")

    return X_plus

# --------------------------- toy data ------------------------------

def make_ring(n=6000, R=2.0, noise=0.05, device="cpu"):
    th = torch.rand(n, device=device) * (2*math.pi)
    base = torch.stack([torch.cos(th), torch.sin(th)], dim=1) * R
    return base + noise * torch.randn(n, 2, device=device)

def make_two_blobs(n=6000, Delta=2.0, std=1.0, p=0.5, device="cpu"):
    n1 = int(round(n * p)); n2 = n - n1
    a = torch.randn(n1, 2, device=device) * std + torch.tensor([-Delta, 0.0], device=device)
    b = torch.randn(n2, 2, device=device) * std + torch.tensor([+Delta, 0.0], device=device)
    return torch.cat([a, b], 0)

# ------------------------------ demo -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--snap-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=str, default="ring", choices=["ring","two_blobs"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--f", type=str, default="js", choices=["js","kl","hellinger2","reverse_kl","alpha"])
    ap.add_argument("--alpha", type=float, default=1.6, help="alpha for alpha-div (if --f alpha)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if (args.device=="auto" and torch.cuda.is_available()) else
              ("cuda" if args.device=="cuda" else "cpu"))

    # Data
    if args.data == "ring":
        Y = make_ring(n=8000, R=2.0, noise=0.05, device=device)
    else:
        Y = make_two_blobs(n=8000, Delta=2.5, std=0.9, device=device)

    # Particles
    X = 0.7 * torch.randn(6000, 2, device=device)

    # Headless-safe plotting
    import matplotlib
    if os.environ.get("DISPLAY","")=="":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = args.steps
    for t in range(1, steps + 1):
        a = t / steps
        # Anneal: τ down, K up, step slightly down
        tau_t = 0.30 - 0.20 * a                 # 0.30 → 0.10 (as τ_scale wrt std)
        K_t   = int(round(80 + (128 - 80) * a)) # 80   → 128
        eps_t = 0.20 * (1.0 - 0.25 * a)         # 0.20 → 0.15

        rad_before = X.norm(dim=1).mean().item()
        X = rank_proximal_transport(
            X, Y,
            L=128, K=K_t, tau=tau_t,
            f_name=args.f, alpha_div=args.alpha,
            eta=0.5, inner_steps=5,
            eps_move=eps_t,
            anchored=True, antithetic=True,
            per_slice_cap=0.40,
        )
        rad_after  = X.norm(dim=1).mean().item()
        print(f"[outer {t:03d}] mean radius {rad_before:.3f} → {rad_after:.3f}, Δ={rad_after-rad_before:+.3f}")

        if args.snap_every and (t % args.snap_every == 0 or t in {1,5}):
            plt.figure(figsize=(5,5))
            plt.scatter(Y[:,0].detach().cpu(), Y[:,1].detach().cpu(), s=6, alpha=0.55, label="data")
            plt.scatter(X[:,0].detach().cpu(), X[:,1].detach().cpu(), s=6, alpha=0.55, label=f"RPT step {t}")
            plt.axis("equal"); plt.legend(); plt.tight_layout()
            fname = f"rpt_bern_step_{t:03d}.png"
            plt.savefig(fname, dpi=160); plt.close()
            print(f"[SNAP] saved {os.path.abspath(fname)}")

    plt.figure(figsize=(5,5))
    plt.scatter(Y[:,0].detach().cpu(), Y[:,1].detach().cpu(), s=6, alpha=0.60, label="data")
    plt.scatter(X[:,0].detach().cpu(), X[:,1].detach().cpu(), s=6, alpha=0.60, label="RPT final")
    plt.axis("equal"); plt.legend(); plt.tight_layout()
    plt.savefig("rpt_bern_final.png", dpi=180); plt.close()
    print("✅ Saved rpt_bern_final.png")

if __name__ == "__main__":
    main()
