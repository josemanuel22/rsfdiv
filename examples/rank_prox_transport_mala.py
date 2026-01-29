#!/usr/bin/env python3
# ---------------------------------------------------------------
# Rank–Proximal Transport using *Bernstein* rank pmf
#   - inner prox over ranks U with: SGD | ULA | MALA (logit-space)
#   - FIX 1: per-slice monotone coupling (prevents mass collapse)
#   - FIX 2: SWF-style entropy regularization -> diffusion noise
#   - MORE 2D toy datasets
#   - CONSISTENT PLOTTING: fixed canvas size + fixed limits (no tight bbox),
#     no axes/labels/legend; step_000 is particles-only in blue.
#
# Usage:
#   PYTHONPATH=src python rank_prox_transport_mala.py --data two_blobs --rank-sampler mala --steps 600
#   PYTHONPATH=src python rank_prox_transport_mala.py --data spirals   --rank-sampler mala --steps 800 --entropy-lambda 0.03
# ---------------------------------------------------------------

import os, math, argparse
from typing import Optional, Tuple
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
    # f(r) for D_RKL(P||Q): f(r) = -log r + r - 1
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
    raise KeyError(
        f"Unknown f-divergence '{name}'. Available: js, kl, hellinger2, reverse_kl, alpha"
    )

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
  # ensure normalized
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

# ---------------------- FIX 1: monotone coupling ----------------------

def monotone_rearrange(U: torch.Tensor, xs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Reassign values of U in increasing order to particles ordered by xs.
    Preserves the multiset/histogram of U, but enforces a monotone map xs -> U.
    """
    U = U.clamp(eps, 1.0 - eps)
    # tiny jitter to break ties (optional but helps with identical xs/U)
    Uj = (U + eps * torch.randn_like(U)).clamp(eps, 1.0 - eps)

    idx = torch.argsort(xs)
    U_sorted = torch.sort(Uj).values
    U_mono = torch.empty_like(Uj)
    U_mono[idx] = U_sorted
    return U_mono

def uniform_grid_in_xs_order(xs: torch.Tensor) -> torch.Tensor:
    """
    Deterministic monotone ranks (0.5/N, ..., (N-0.5)/N) assigned in xs-order.
    """
    N = xs.numel()
    dev, dt = xs.device, xs.dtype
    u_grid = (torch.arange(N, device=dev, dtype=dt) + 0.5) / float(N)
    idx = torch.argsort(xs)
    out = torch.empty_like(u_grid)
    out[idx] = u_grid
    return out

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
    Metropolis-Adjusted Langevin in logit space.

    Target in U-space:
        π(U) ∝ exp(-Φ(U)),
        Φ(U) = D_f(p_Bern(U) || unif) + (1/(2η)) ||U - U0||^2.

    Work in V = logit(U), Φ̃(V) = Φ(σ(V)) - Σ log σ'(V).
    """
    f_fn = _get_f_any(f_name, alpha)
    lr = float(lr if lr is not None else eta)
    assert lr > 0.0
    eps = 1e-6
    dev, dt = U0.device, U0.dtype

    U0c = U0.clamp(eps, 1.0 - eps)

    def phi_tilde_and_grad(V_in: torch.Tensor):
        with torch.enable_grad():
            V = V_in.detach().to(dtype=dt).requires_grad_(True)
            U = torch.sigmoid(V)

            p = _bernstein_histogram(U, K=K, alpha=alpha_hist)
            D = discrete_f_div_from_pmf(p, f_fn)
            quad = 0.5 * (1.0 / max(1e-8, eta)) * ((U - U0c) ** 2).mean()
            Phi = D + quad

            jac = -torch.log(U * (1.0 - U) + 1e-32).sum()
            Phi_tilde = Phi + jac

            (gV,) = torch.autograd.grad(Phi_tilde, V, create_graph=False)
        return Phi_tilde.detach(), gV.detach()

    V = torch.logit(U0c).to(dtype=dt)
    PhiV, gV = phi_tilde_and_grad(V)
    acc_sum = 0.0

    for _ in range(max(1, steps)):
        with torch.no_grad():
            mean = V - lr * gV
            V_prop = mean + math.sqrt(2.0 * lr) * torch.randn_like(V)
            V_prop.clamp_(-clip_v, clip_v)

        PhiV_prop, gV_prop = phi_tilde_and_grad(V_prop)

        with torch.no_grad():
            def log_q(y, x, grad_x):
                m = x - lr * grad_x
                diff = y - m
                return -(diff.pow(2).sum()) / (4.0 * lr)

            log_acc = (-PhiV_prop + PhiV) + (log_q(V, V_prop, gV_prop) - log_q(V_prop, V, gV))
            u = torch.rand((), device=dev, dtype=dt)
            accept = (torch.log(u) < log_acc)
            acc_sum += float(accept)

            if debug:
                print(
                    f"[MALA] Φ̃_cur={float(PhiV):.4f} Φ̃_prop={float(PhiV_prop):.4f} "
                    f"Δ={float(PhiV_prop - PhiV):+.4f} accept={float(accept)}"
                )

            if accept:
                V = V_prop.detach()
                PhiV, gV = PhiV_prop, gV_prop
            else:
                V = V.detach()

    U_final = torch.sigmoid(V).clamp(eps, 1.0 - eps)

    if return_accept:
        return U_final.detach(), acc_sum / float(max(1, steps))
    return U_final.detach()

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
    tau: float = 0.25,
    f_name: str = "js",
    alpha_div: float = 1.6,   # for α-div only
    eta: float = 0.5,
    inner_steps: int = 5,
    eps_move: float = 0.02,   # also used as "h" for diffusion scaling
    anchored: bool = True,
    antithetic: bool = True,
    per_slice_cap: float = 0.40,
    rank_sampler: str = "mala",
    ula_T: float = 0.05,

    # FIX 1
    monotone_coupling: bool = True,
    u_mix_beta: float = 0.0,

    # FIX 2 (entropy -> diffusion)
    entropy_lambda: float = 0.0,

    # optional extra isotropic noise std (variance-additive)
    x_langevin_std: float = 0.0,

    debug_every: int = 0
) -> torch.Tensor:
    dev, dt = X.device, X.dtype
    N, d = X.shape

    S = make_slices(L, d, device=dev, dtype=dt, anchored=anchored)
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

        U0 = _soft_ecdf_standardized(xs, ys, tau=tau)

        if rank_sampler == "sgd":
            U1 = prox_rank_step_sgd(U0, K=K, eta=eta, steps=inner_steps,
                                    f_name=f_name, alpha=alpha_div)
        elif rank_sampler == "ula":
            U1 = prox_rank_step_ula(U0, K=K, eta=eta, steps=inner_steps,
                                    f_name=f_name, alpha=alpha_div, lr_U=None, T=ula_T)
        elif rank_sampler == "mala":
            U1, acc = prox_rank_step_mala(U0, K=K, eta=eta, steps=max(1, inner_steps // 2),
                                          f_name=f_name, alpha=alpha_div,
                                          lr=None, clip_v=8.0, debug=False,
                                          return_accept=True)
            mala_acc += acc
            mala_cnt += 1
        else:
            raise ValueError("rank_sampler must be one of {'sgd','ula','mala'}")

        # ---- FIX 1: enforce monotone coupling ----
        if monotone_coupling:
            U1 = monotone_rearrange(U1, xs)

        # optional: mix toward deterministic uniform grid ranks (still monotone)
        if u_mix_beta and u_mix_beta > 0.0:
            Ugrid = uniform_grid_in_xs_order(xs)
            U1 = ((1.0 - float(u_mix_beta)) * U1 + float(u_mix_beta) * Ugrid).clamp(1e-6, 1.0 - 1e-6)

        # pull back via projected quantile
        z_star   = quantile_from_samples(ys, U1)
        delta_pr = z_star - xs

        # per-slice trust region
        scale = (per_slice_cap / (delta_pr.abs() + 1e-8)).clamp(max=1.0)
        dX    += (eps_move * scale * delta_pr)[:, None] * s[None, :]

        if debug_every and (i % debug_every == 0):
            mean_du = (U1 - U0).abs().mean().item()
            mean_dp = delta_pr.abs().mean().item()
            print(f"[slice {i:03d}] ⟨|ΔU|⟩={mean_du:.3e}  ⟨|Δproj|⟩={mean_dp:.3e}")

    # scale by d to counter E[ss^T]=(1/d)I
    X_plus = X + (d / nS) * dX

    # ---- FIX 2: diffusion (entropy regularization) + optional extra noise ----
    var = 0.0
    if entropy_lambda and entropy_lambda > 0.0:
        h = float(eps_move)
        var += 2.0 * float(entropy_lambda) * max(0.0, h)
    if x_langevin_std and x_langevin_std > 0.0:
        var += float(x_langevin_std) ** 2
    if var > 0.0:
        X_plus = X_plus + math.sqrt(var) * torch.randn_like(X_plus)

    if debug_every:
        r1 = X_plus.norm(dim=1).mean().item()
        extra = ""
        if rank_sampler == "mala" and mala_cnt > 0:
            extra += f" | MALA acc ≈ {mala_acc / mala_cnt:.2f}"
        if entropy_lambda and entropy_lambda > 0.0:
            extra += f" | diff std ≈ {math.sqrt(2.0*float(entropy_lambda)*float(eps_move)):.4f}"
        print(f"[rpt] ⟨r⟩: {r0:.4f} → {r1:.4f} (Δ {r1 - r0:+.4f}){extra}")

    return X_plus

# --------------------------- toy data (MORE examples) ------------------------------

def make_ring(n=6000, R=2.0, noise=0.05, device="cpu"):
    th = torch.rand(n, device=device) * (2*math.pi)
    base = torch.stack([torch.cos(th), torch.sin(th)], dim=1) * R
    return base + noise * torch.randn(n, 2, device=device)

def make_rings2(n=8000, R1=1.2, R2=2.4, noise=0.05, p=0.5, device="cpu"):
    n1 = int(round(n * p)); n2 = n - n1
    a = make_ring(n1, R=R1, noise=noise, device=device)
    b = make_ring(n2, R=R2, noise=noise, device=device)
    return torch.cat([a, b], 0)

def make_two_blobs(n=6000, Delta=2.0, std=1.0, p=0.5, device="cpu"):
    n1 = int(round(n * p)); n2 = n - n1
    a = torch.randn(n1, 2, device=device) * std + torch.tensor([-Delta, 0.0], device=device)
    b = torch.randn(n2, 2, device=device) * std + torch.tensor([+Delta, 0.0], device=device)
    return torch.cat([a, b], 0)

def make_two_moons(n=8000, noise=0.08, device="cpu"):
    n1 = n // 2
    n2 = n - n1
    t1 = torch.rand(n1, device=device) * math.pi
    t2 = torch.rand(n2, device=device) * math.pi
    moon1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
    moon2 = torch.stack([1.0 - torch.cos(t2), -torch.sin(t2) - 0.5], dim=1)
    X = torch.cat([moon1, moon2], 0)
    X = X + noise * torch.randn_like(X)
    X = X - X.mean(dim=0, keepdim=True)
    return X

def make_spirals(n=8000, noise=0.06, turns=2.0, device="cpu"):
    n1 = n // 2
    n2 = n - n1
    t1 = torch.rand(n1, device=device)
    t2 = torch.rand(n2, device=device)
    r1 = t1
    r2 = t2
    a1 = turns * 2 * math.pi * t1
    a2 = turns * 2 * math.pi * t2 + math.pi
    s1 = torch.stack([r1 * torch.cos(a1), r1 * torch.sin(a1)], dim=1)
    s2 = torch.stack([r2 * torch.cos(a2), r2 * torch.sin(a2)], dim=1)
    X = torch.cat([s1, s2], 0)
    X = 3.0 * X
    X = X + noise * torch.randn_like(X)
    X = X - X.mean(dim=0, keepdim=True)
    return X

def make_pinwheel(n=8000, noise=0.15, K=5, radial_std=0.3, tangential_std=0.1, rate=0.25, device="cpu"):
    N = n
    dev = device
    r = torch.randn(N, device=dev) * radial_std + 1.0
    t = torch.randn(N, device=dev) * tangential_std
    k = torch.randint(0, K, (N,), device=dev)
    base_angle = (2 * math.pi / K) * k.to(torch.float32)
    angle = base_angle + rate * r + t
    X = torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=1)
    X = X + noise * torch.randn_like(X) * 0.05
    X = X - X.mean(dim=0, keepdim=True)
    return X

def make_eight_gaussians(n=8000, radius=2.5, std=0.15, device="cpu"):
    centers = []
    for j in range(8):
        a = 2 * math.pi * j / 8.0
        centers.append([radius * math.cos(a), radius * math.sin(a)])
    C = torch.tensor(centers, device=device, dtype=torch.float32)  # (8,2)
    idx = torch.randint(0, 8, (n,), device=device)
    X = C[idx] + std * torch.randn(n, 2, device=device)
    X = X - X.mean(dim=0, keepdim=True)
    return X

def make_checkerboard(n=8000, device="cpu"):
    # 2D checkerboard on [-2,2]^2 with alternating squares
    x = (torch.rand(n, device=device) * 4.0) - 2.0
    y = (torch.rand(n, device=device) * 4.0) - 2.0
    xi = torch.floor(x + 2.0).long()
    yi = torch.floor(y + 2.0).long()
    mask = ((xi + yi) % 2 == 0)
    X = torch.stack([x[mask], y[mask]], dim=1)
    while X.size(0) < n:
        x = (torch.rand(n, device=device) * 4.0) - 2.0
        y = (torch.rand(n, device=device) * 4.0) - 2.0
        xi = torch.floor(x + 2.0).long()
        yi = torch.floor(y + 2.0).long()
        mask = ((xi + yi) % 2 == 0)
        X = torch.cat([X, torch.stack([x[mask], y[mask]], dim=1)], dim=0)
    X = X[:n]
    X = X - X.mean(dim=0, keepdim=True)
    return X

def make_banana(n=8000, noise=0.25, device="cpu"):
    x = torch.randn(n, device=device)
    y = 0.25 * (x ** 2 - 1.0) + noise * torch.randn(n, device=device)
    X = torch.stack([x, y], dim=1)
    X = 2.0 * X
    X = X - X.mean(dim=0, keepdim=True)
    return X

def make_dataset(name: str, n: int, device: str, noise: float):
    key = name.lower()
    if key == "ring":
        return make_ring(n=n, R=2.0, noise=max(1e-6, noise), device=device)
    if key == "rings2":
        return make_rings2(n=n, R1=1.2, R2=2.4, noise=max(1e-6, noise), p=0.5, device=device)
    if key == "two_blobs":
        return make_two_blobs(n=n, Delta=2.5, std=0.9, p=0.5, device=device)
    if key == "two_moons":
        return make_two_moons(n=n, noise=max(1e-6, noise), device=device)
    if key == "spirals":
        return make_spirals(n=n, noise=max(1e-6, noise), turns=2.0, device=device)
    if key == "pinwheel":
        return make_pinwheel(n=n, noise=0.15, K=5, radial_std=0.3, tangential_std=0.1, rate=0.25, device=device)
    if key == "eight_gaussians":
        return make_eight_gaussians(n=n, radius=2.5, std=0.15, device=device)
    if key == "checkerboard":
        return make_checkerboard(n=n, device=device)
    if key == "banana":
        return make_banana(n=n, noise=max(1e-6, noise), device=device)
    raise ValueError(f"Unknown dataset '{name}'")

# ---------------------- CONSISTENT PLOTTING HELPERS ----------------------

def compute_fixed_limits(Y: torch.Tensor, X: Optional[torch.Tensor] = None, pad_frac: float = 0.05) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """
    Fixed x/y limits shared by *all* frames, to keep the same viewport.
    Uses Y plus optional X (e.g. initial particles) to avoid cropping step_000.
    """
    XY = Y if X is None else torch.cat([Y, X], dim=0)
    mn = XY.min(dim=0).values
    mx = XY.max(dim=0).values
    span = (mx - mn).max().item()
    pad = pad_frac * span
    xlim = (float(mn[0] - pad), float(mx[0] + pad))
    ylim = (float(mn[1] - pad), float(mx[1] + pad))
    return xlim, ylim

def save_frame(fname: str, X: torch.Tensor, Y: Optional[torch.Tensor],
               *, xlim, ylim, dpi: int, figsize: float, s: float, alpha: float,
               x_color: Optional[str] = None):
    """
    Save a frame with:
      - fixed canvas size (figsize,dpi)
      - fixed limits (xlim,ylim)
      - no axes/legend/labels
      - NO bbox_inches='tight' (keeps identical pixel sizes)
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)

    if Y is not None:
        ax.scatter(Y[:, 0].detach().cpu(), Y[:, 1].detach().cpu(), s=s, alpha=alpha)

    ax.scatter(X[:, 0].detach().cpu(), X[:, 1].detach().cpu(), s=s, alpha=alpha, color=x_color)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    fig.savefig(fname)   # IMPORTANT: no tight bbox
    plt.close(fig)

# ------------------------------ demo -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--snap-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--data", type=str, default="ring",
                    choices=["ring","rings2","two_blobs","two_moons","spirals","pinwheel",
                             "eight_gaussians","checkerboard","banana"])
    ap.add_argument("--data-n", type=int, default=8000)
    ap.add_argument("--data-noise", type=float, default=0.08, help="used by some datasets (moons/spirals/banana/rings)")

    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--f", type=str, default="js", choices=["js","kl","hellinger2","reverse_kl","alpha"])
    ap.add_argument("--alpha", type=float, default=1.6, help="alpha for alpha-div (if --f alpha)")
    ap.add_argument("--rank-sampler", type=str, default="mala", choices=["sgd","ula","mala"])
    ap.add_argument("--eta", type=float, default=0.5, help="prox strength / inner step size")
    ap.add_argument("--inner-steps", type=int, default=5)
    ap.add_argument("--L", type=int, default=64, help="number of projection directions (slices)")

    # FIX 1 flags
    ap.add_argument("--no-monotone", action="store_true",
                    help="disable monotone coupling (NOT recommended for mixtures)")
    ap.add_argument("--u-mix-beta", type=float, default=0.5,
                    help="optional small mix toward uniform monotone ranks (e.g. 0.05-0.15)")

    # FIX 2 flag (entropy regularization -> diffusion)
    ap.add_argument("--entropy-lambda", type=float, default=0.1,
                    help="adds diffusion with std sqrt(2*lambda*eps_move)")

    # Optional extra noise (variance-additive; usually leave 0 if using entropy-lambda)
    ap.add_argument("--x-noise", type=float, default=0.0, help="extra isotropic noise std (optional)")

    # Plot controls (fixed size across frames)
    ap.add_argument("--figsize", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--point-size", type=float, default=6.0)
    ap.add_argument("--point-alpha", type=float, default=0.55)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
              ("cuda" if args.device == "cuda" else "cpu"))

    # Data
    Y = make_dataset(args.data, n=args.data_n, device=device, noise=args.data_noise)

    # Headless-safe plotting (MUST come before pyplot import)
    import matplotlib
    if os.environ.get("DISPLAY","") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # expose to helpers
    globals()["plt"] = plt

    # Particles
    X = 0.7 * torch.randn(6000, 2, device=device)

    # Fixed limits for ALL frames (ensures LaTeX images have identical aspect/canvas)
    xlim, ylim = compute_fixed_limits(Y, X, pad_frac=0.05)

    # Step 000: particles only in blue (no data, no orange), same limits/canvas as all frames
    fname0 = f"rpt_{args.rank_sampler}_{args.data}_step_000.pdf"
    save_frame(
        fname0, Y, Y=None,
        xlim=xlim, ylim=ylim,
        dpi=args.dpi, figsize=args.figsize, s=args.point_size, alpha=args.point_alpha,
        x_color="tab:blue",
    )
    print(f"[SNAP] saved {os.path.abspath(fname0)}")

    steps = args.steps
    for t in range(1, steps + 1):
        a = t / steps

        # Anneal: τ down, K up, step slightly down
        tau_t = 0.30 - 0.20 * a                 # 0.30 → 0.10
        K_t   = int(round(80 + (128 - 80) * a)) # 80   → 128
        eps_t = 0.20 * (1.0 - 0.25 * a)         # 0.20 → 0.15

        # ULA temp (if used)
        T_ula = 0.08 * (1.0 - a)

        rad_before = X.norm(dim=1).mean().item()
        X = rank_proximal_transport(
            X, Y,
            L=args.L, K=K_t, tau=tau_t,
            f_name=args.f, alpha_div=args.alpha,
            eta=args.eta, inner_steps=args.inner_steps,
            eps_move=eps_t,
            anchored=True, antithetic=True,
            per_slice_cap=0.40,
            rank_sampler=args.rank_sampler,
            ula_T=T_ula,

            monotone_coupling=(not args.no_monotone),
            u_mix_beta=float(args.u_mix_beta),

            entropy_lambda=float(args.entropy_lambda),
            x_langevin_std=float(args.x_noise),

            debug_every=0 if (t % 50) else 1
        )
        rad_after = X.norm(dim=1).mean().item()
        print(f"[outer {t:03d}] mean radius {rad_before:.3f} → {rad_after:.3f}, Δ={rad_after-rad_before:+.3f}")

        if args.snap_every and (t % args.snap_every == 0 or t in {1, 5}):
            fname = f"rpt_{args.rank_sampler}_{args.data}_step_{t:03d}.pdf"
            save_frame(
                fname, X, Y,
                xlim=xlim, ylim=ylim,
                dpi=args.dpi, figsize=args.figsize, s=args.point_size, alpha=args.point_alpha,
                x_color=None,  # default (orange) for particles; data stays default (blue)
            )
            print(f"[SNAP] saved {os.path.abspath(fname)}")

    # Final frame
    fnameF = f"rpt_{args.rank_sampler}_{args.data}_final.pdf"
    save_frame(
        fnameF, X, Y,
        xlim=xlim, ylim=ylim,
        dpi=args.dpi, figsize=args.figsize, s=args.point_size, alpha=args.point_alpha,
        x_color=None,
    )
    print(f"✅ Saved {fnameF}")

if __name__ == "__main__":
    main()
