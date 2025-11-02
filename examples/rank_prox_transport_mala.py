#!/usr/bin/env python3
# ---------------------------------------------------------------
# Rank–Proximal Transport using *Bernstein* rank pmf
#   - inner prox over ranks U with: SGD | ULA | MALA (logit-space)
#   - outer particle update with optional Langevin kick
#
# Usage:
#   PYTHONPATH=src python rank_prox_transport_mala.py --rank-sampler mala --steps 400
# ---------------------------------------------------------------

import os, math, argparse
from typing import Optional
import torch
import torch.nn.functional as F

# ---- import the shared rank-stat toolkit (your repo) ----
from ranks.ranks_torch import (
    soft_ecdf_Q_of_x,         # smooth ECDF σ((x - y)/τ)
    bernstein_basis,          # B_{n,K}(u)
    discrete_f_div_from_pmf,  # ∑_n q f(p_n/q), with q = 1/(K+1)
    get_f,                    # 'js' | 'kl' | 'hellinger2'
)

# ---------------------------- f-generators ----------------------------

def _f_reverse_kl():
    # f(r) for D_RKL(P||Q) = ∑ q f(p/q): f(r) = -log r + r - 1
    return lambda r: -torch.log(r + 1e-32) + r - 1.0

def _f_alpha(alpha: float):
    # α-divergence generator (α ≠ 0,1): f(r) = (r^α - α r + α - 1)/(α(α-1))
    a = float(alpha)
    if abs(a) < 1e-12 or abs(a - 1.0) < 1e-12:
        return get_f("kl") if abs(a - 1.0) < 1e-12 else _f_reverse_kl()
    def f(r):
        r = torch.clamp(r, 1e-32, 1e32)
        return (torch.pow(r, a) - a * r + a - 1.0) / (a * (a - 1.0))
    return f

def _get_f_any(name: str, alpha: float):
    key = (name or "").lower()
    if key in ("js", "jensen-shannon", "kl", "hellinger2", "hell2"):
        return get_f(key)
    if key in ("rkl", "reverse_kl", "revkl"):
        return _f_reverse_kl()
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
    Standardize by μ,σ of y·s; equivalently τ_eff = tau * std(y·s).
    """
    mu = y_proj.mean()
    sd = y_proj.std().clamp_min(1e-6)
    return soft_ecdf_Q_of_x(x_proj - mu, y_proj - mu, tau=tau * sd).clamp(1e-6, 1 - 1e-6)

# --------------------- Bernstein rank histogram -----------------

def _bernstein_histogram(U: torch.Tensor, K: int, alpha: float = 1e-4):
    """
    Smooth pmf p over {0,...,K} by averaging Bernstein rows + tiny Dirichlet prior.
    """
    B = bernstein_basis(U, K)       # (N, K+1), rows sum to 1
    p = B.mean(dim=0)               # (K+1,)
    if alpha is not None and alpha > 0.0:
        p = (p + alpha) / (p.sum() + (K + 1) * alpha)
    else:
        p = p / p.sum()
    return p

# ---------------------- inner prox: SGD / ULA / MALA ----------------------

def prox_rank_step_sgd(
    U0: torch.Tensor, K: int, *,
    eta: float = 0.5, steps: int = 5,
    f_name: str = "js", alpha: float = 1.6, alpha_hist: float = 1e-4,
) -> torch.Tensor:
    f_fn = _get_f_any(f_name, alpha)
    with torch.enable_grad():
        U = U0.detach().clamp(1e-6, 1 - 1e-6).requires_grad_(True)
        opt = torch.optim.SGD([U], lr=float(eta), momentum=0.0)
        for _ in range(max(1, steps)):
            opt.zero_grad(set_to_none=True)
            p = _bernstein_histogram(U, K=K, alpha=alpha_hist)
            D = discrete_f_div_from_pmf(p, f_fn)
            quad = 0.5 * (1.0 / max(1e-8, eta)) * ((U - U0) ** 2).mean()
            loss = D + quad
            loss.backward()
            opt.step()
            with torch.no_grad():
                U.clamp_(1e-6, 1 - 1e-6)
        return U.detach()

def prox_rank_step_ula(
    U0: torch.Tensor, K: int, *,
    eta: float = 0.5, steps: int = 5,
    f_name: str = "js", alpha: float = 1.6, alpha_hist: float = 1e-4,
    lr_U: Optional[float] = None, T: float = 0.05,
) -> torch.Tensor:
    f_fn = _get_f_any(f_name, alpha)
    lr = float(lr_U if lr_U is not None else eta)
    with torch.enable_grad():
        U = U0.detach().clamp(1e-6, 1 - 1e-6).requires_grad_(True)
        for _ in range(max(1, steps)):
            p = _bernstein_histogram(U, K=K, alpha=alpha_hist)
            D = discrete_f_div_from_pmf(p, f_fn)
            quad = 0.5 * (1.0 / max(1e-8, eta)) * ((U - U0) ** 2).mean()
            loss = D + quad
            (gU,) = torch.autograd.grad(loss, U, create_graph=False)
            with torch.no_grad():
                U.add_(-lr * gU)
                if T > 0.0:
                    U.add_(math.sqrt(2.0 * lr * T) * torch.randn_like(U))
                U.clamp_(1e-6, 1 - 1e-6)
            U.requires_grad_(True)
        return U.detach()

def prox_rank_step_mala(
    U0: torch.Tensor, K: int, *,
    eta: float = 0.5, steps: int = 1,
    f_name: str = "js", alpha: float = 1.6, alpha_hist: float = 1e-4,
    lr: Optional[float] = None, clip_v: float = 8.0, debug: bool = False,
    return_accept: bool = False,
):
    """
    Metropolis-Adjusted Langevin (logit space):
      target π(U) ∝ exp(-Φ(U)), Φ(U)=D_f(p_Bern(U)||unif) + (1/(2η))||U-U0||^2
      Φ̃(V)=Φ(σ(V)) - Σ log σ'(V), proposal:
      V' ~ N(V - lr ∇Φ̃(V), 2 lr I)
    """
    f_fn = _get_f_any(f_name, alpha)
    lr = float(lr if lr is not None else eta)
    assert lr > 0.0
    eps = 1e-6
    dev, dt = U0.device, U0.dtype
    U0c = U0.detach().clamp(eps, 1 - eps)

    def phi_U(U: torch.Tensor) -> torch.Tensor:
        p = _bernstein_histogram(U, K=K, alpha=alpha_hist)
        D = discrete_f_div_from_pmf(p, f_fn)
        quad = 0.5 * (1.0 / max(1e-8, eta)) * ((U - U0) ** 2).mean()
        return D + quad

    with torch.enable_grad():
        def phi_tilde_and_grad(V: torch.Tensor):
            U = torch.sigmoid(V)
            Phi = phi_U(U)
            jac = -torch.log(U * (1.0 - U) + 1e-32).sum()  # -Σ log σ'(V)
            Phi_tilde = Phi + jac
            (gV,) = torch.autograd.grad(Phi_tilde, V, create_graph=False)
            return Phi_tilde, gV, U

        V = torch.logit(U0c).detach().to(dtype=dt).requires_grad_(True)
        PhiV, gV, _ = phi_tilde_and_grad(V)
        acc_sum = 0.0

        for _ in range(max(1, steps)):
            with torch.no_grad():
                mean = V - lr * gV
                V_prop = mean + math.sqrt(2.0 * lr) * torch.randn_like(V)
                V_prop.clamp_(-clip_v, clip_v)

            V_prop = V_prop.detach().requires_grad_(True)
            PhiV_prop, gV_prop, _ = phi_tilde_and_grad(V_prop)

            with torch.no_grad():
                def log_q(y, x, grad_x):
                    m = x - lr * grad_x
                    diff = y - m
                    return -(diff.pow(2).sum()) / (4.0 * lr)
                log_acc = (-PhiV_prop + PhiV) + (log_q(V, V_prop, gV_prop) - log_q(V_prop, V, gV))
                accept = (torch.log(torch.rand((), device=dev, dtype=dt)) < log_acc)
                acc_sum += float(accept)
                if debug:
                    print(f"[MALA] Φ̃_cur={float(PhiV):.4f} Φ̃_prop={float(PhiV_prop):.4f} "
                          f"Δ={float(PhiV_prop - PhiV):+.4f}  accept={float(accept)}")
                if accept:
                    V = V_prop.detach().requires_grad_(True)
                    PhiV, gV, _ = phi_tilde_and_grad(V)
                else:
                    V = V.detach().requires_grad_(True)

        U_final = torch.sigmoid(V).clamp(eps, 1 - eps).detach()

    if return_accept:
        return U_final, acc_sum / float(max(1, steps))
    return U_final

# ------------------- inverse CDF via interpolation ----------------

def quantile_from_samples(y_proj: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Approximate F^{-1}(u) by sorting y and linear-interpolating.
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
    tau: float = 0.25,        # τ_eff = tau * std(y·s)
    f_name: str = "js",
    alpha_div: float = 1.6,   # for α-div only
    eta: float = 0.5,         # prox strength / MALA step
    inner_steps: int = 5,
    eps_move: float = 0.20,
    anchored: bool = True,
    antithetic: bool = True,
    per_slice_cap: float = 0.40,
    rank_sampler: str = "mala",     # 'sgd' | 'ula' | 'mala'
    ula_T: float = 0.05,            # ULA temperature
    x_langevin_std: float = 0.0,    # isotropic noise on X
    debug_every: int = 0
) -> torch.Tensor:
    """
    One RPT update with selectable inner sampler and optional outer noise.
    """
    dev, dt = X.device, X.dtype
    N, d = X.shape
    S = make_slices(L, d, device=dev, dtype=dt, anchored=anchored)  # (L,d)
    if antithetic:
        S = torch.cat([S, -S], dim=0)
    nS = S.size(0)

    dX = torch.zeros_like(X)
    mala_acc = 0.0
    mala_cnt = 0

    if debug_every:
        r0 = X.norm(dim=1).mean().item()

    for i, s in enumerate(S):
        xs = X @ s
        ys = Y @ s

        # Soft ranks via library (standardization-compatible)
        U0 = _soft_ecdf_standardized(xs, ys, tau=tau)

        # Prox step in rank-space
        if rank_sampler == "sgd":
            U1 = prox_rank_step_sgd(U0, K=K, eta=eta, steps=inner_steps,
                                    f_name=f_name, alpha=alpha_div)
        elif rank_sampler == "ula":
            U1 = prox_rank_step_ula(U0, K=K, eta=eta, steps=inner_steps,
                                    f_name=f_name, alpha=alpha_div, lr_U=None, T=ula_T)
        elif rank_sampler == "mala":
            U1, acc = prox_rank_step_mala(U0, K=K, eta=eta, steps=max(1, inner_steps//2),
                                          f_name=f_name, alpha=alpha_div,
                                          lr=None, clip_v=8.0, debug=False,
                                          return_accept=True)
            mala_acc += acc
            mala_cnt += 1
        else:
            raise ValueError("rank_sampler must be one of {'sgd','ula','mala'}")

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

    # scale by d to counter E[ss^T]=(1/d)I
    X_plus = X + (d / nS) * dX

    # optional particle Langevin kick
    if x_langevin_std > 0.0:
        X_plus = X_plus + x_langevin_std * torch.randn_like(X_plus)

    if debug_every:
        r1 = X_plus.norm(dim=1).mean().item()
        extra = ""
        if rank_sampler == "mala" and mala_cnt > 0:
            extra = f" | MALA acc ≈ {mala_acc / mala_cnt:.2f}"
        print(f"[rpt] ⟨r⟩: {r0:.4f} → {r1:.4f}  (Δ {r1 - r0:+.4f}){extra}")

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
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--snap-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=str, default="ring", choices=["ring","two_blobs"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--f", type=str, default="js", choices=["js","kl","hellinger2","reverse_kl","alpha"])
    ap.add_argument("--alpha", type=float, default=1.6, help="alpha for alpha-div (if --f alpha)")
    ap.add_argument("--rank-sampler", type=str, default="mala", choices=["sgd","ula","mala"])
    ap.add_argument("--eta", type=float, default=0.5, help="prox strength / inner step size")
    ap.add_argument("--inner-steps", type=int, default=5)
    ap.add_argument("--x-noise", type=float, default=0.0, help="outer particle Langevin std (overridden by anneal)")
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
        # U-space & X-space noise schedules
        T_ula   = 0.08 * (1.0 - a)              # ULA temperature
        sigma_X = (args.x_noise if args.x_noise > 0 else 0.04 * (1.0 - 0.5 * a))

        rad_before = X.norm(dim=1).mean().item()
        X = rank_proximal_transport(
            X, Y,
            L=128, K=K_t, tau=tau_t,
            f_name=args.f, alpha_div=args.alpha,
            eta=args.eta, inner_steps=args.inner_steps,
            eps_move=eps_t,
            anchored=True, antithetic=True,
            per_slice_cap=0.40,
            rank_sampler=args.rank_sampler,
            ula_T=T_ula,
            x_langevin_std=sigma_X if t < int(0.7 * steps) else 0.0,  # kill outer noise late
            debug_every=0 if (t % 50) else 1
        )
        rad_after  = X.norm(dim=1).mean().item()
        print(f"[outer {t:03d}] mean radius {rad_before:.3f} → {rad_after:.3f}, Δ={rad_after-rad_before:+.3f}")

        if args.snap_every and (t % args.snap_every == 0 or t in {1,5}):
            plt.figure(figsize=(5,5))
            plt.scatter(Y[:,0].detach().cpu(), Y[:,1].detach().cpu(), s=6, alpha=0.55, label="data")
            plt.scatter(X[:,0].detach().cpu(), X[:,1].detach().cpu(), s=6, alpha=0.55, label=f"{args.rank_sampler.upper()} step {t}")
            plt.axis("equal"); plt.legend(); plt.tight_layout()
            fname = f"rpt_{args.rank_sampler}_step_{t:03d}.png"
            plt.savefig(fname, dpi=160); plt.close()
            print(f"[SNAP] saved {os.path.abspath(fname)}")

    plt.figure(figsize=(5,5))
    plt.scatter(Y[:,0].detach().cpu(), Y[:,1].detach().cpu(), s=6, alpha=0.60, label="data")
    plt.scatter(X[:,0].detach().cpu(), X[:,1].detach().cpu(), s=6, alpha=0.60, label=f"{args.rank_sampler.upper()} final")
    plt.axis("equal"); plt.legend(); plt.tight_layout()
    plt.savefig(f"rpt_{args.rank_sampler}_final.png", dpi=180); plt.close()
    print(f"✅ Saved rpt_{args.rank_sampler}_final.png")

if __name__ == "__main__":
    main()
