# ranks/ranks_soft.py
# ---------------------------------------------------------------------
# Differentiable (soft) rank-statistic f-divergence and sliced version.
# ---------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F

# -----------------------------------------------------------
# f-divergence generators
# -----------------------------------------------------------

def f_js(t: torch.Tensor) -> torch.Tensor:
    """Jensen–Shannon generator f(t)"""
    u = torch.clamp(t, 1e-30, None)
    return 0.5 * (u * (torch.log(2*u) - torch.log1p(u)) + (math.log(2.) - torch.log1p(u)))

def f_kl(t: torch.Tensor) -> torch.Tensor:
    """KL generator f(t) = t log t"""
    u = torch.clamp(t, 1e-30, None)
    return u * torch.log(u)

def f_hellinger2(t: torch.Tensor) -> torch.Tensor:
    """Squared Hellinger generator f(t) = (sqrt(t)-1)^2"""
    u = torch.clamp(t, 0.0, None)
    return (torch.sqrt(u) - 1.0)**2

F_LIBRARY = {
    "js": f_js,
    "jensen-shannon": f_js,
    "kl": f_kl,
    "hellinger2": f_hellinger2,
    "hell2": f_hellinger2,
}

def get_f(name: str):
    key = name.lower()
    if key not in F_LIBRARY:
        raise KeyError(f"Unknown f-divergence '{name}'. Available: {', '.join(F_LIBRARY)}")
    return F_LIBRARY[key]

# -----------------------------------------------------------
# Soft ECDF (CDF smoothing via logistic sigmoid)
# -----------------------------------------------------------

def soft_ecdf_Q_of_x(x_proj, y_proj, tau):
    """
    Differentiable empirical CDF of y_proj evaluated at x_proj.
    Uses a smooth comparator σ((x - y)/τ).
    """
    diff = (x_proj[:, None] - y_proj[None, :]) / tau
    U = torch.sigmoid(diff).mean(dim=1)
    return U.clamp(1e-6, 1 - 1e-6)

# -----------------------------------------------------------
# Bernstein basis & discrete f-divergence
# -----------------------------------------------------------

def bernstein_basis(U: torch.Tensor, K: int):
    """
    Compute Bernstein basis functions B_{n,K}(u) for u in (0,1).
    Returns matrix of shape (B, K+1).
    """
    n = torch.arange(K + 1, device=U.device, dtype=U.dtype)
    logC = torch.lgamma(torch.tensor(K + 1., device=U.device)) \
          - torch.lgamma(n + 1.) - torch.lgamma(torch.tensor(K, device=U.device) - n + 1.)
    logB = logC + n[None, :] * torch.log(U[:, None]) + (K - n)[None, :] * torch.log(1 - U[:, None])
    return torch.exp(logB)

def discrete_f_div_from_pmf(p_hat: torch.Tensor, f_fn):
    """
    Compute discrete f-divergence between estimated histogram p_hat and uniform q.
    """
    K = p_hat.numel() - 1
    q = 1.0 / (K + 1)
    ratio = torch.clamp(p_hat / q, 1e-30, None)
    return (q * f_fn(ratio)).sum()

# -----------------------------------------------------------
# Sliced soft-rank f-divergence
# -----------------------------------------------------------

def sliced_rank_fdiv(X, Y, L=64, K=64, f_name="js", tau_scale=0.3, rng=None):
    """
    Differentiable sliced rank-statistic f-divergence between samples X~μ, Y~ν.
    Parameters
    ----------
    X, Y : torch.Tensor
        Input samples, shape (B,d) and (M,d) or flattened images.
    L : int
        Number of random directions.
    K : int
        Bernstein histogram degree.
    f_name : str
        Name of the f-divergence generator ('js', 'kl', 'hellinger2').
    tau_scale : float
        Temperature = tau_scale * std(y_proj) + 1e-4.
    rng : torch.Generator or None
        Optional RNG for reproducibility.
    Returns
    -------
    D : torch.Tensor
        Estimated sliced f-divergence (scalar, differentiable).
    """
    f_fn = get_f(f_name)
    device = X.device
    B, M, d = X.size(0), Y.size(0), X.size(1)

    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(1234)

    D_total = 0.0
    for _ in range(L):
        # random direction on sphere
        s = torch.randn(d, device=device, generator=rng)
        s = s / (s.norm() + 1e-12)

        x_proj = X @ s
        y_proj = Y @ s
        tau = tau_scale * (y_proj.std() + 1e-6) + 1e-4

        U = soft_ecdf_Q_of_x(x_proj, y_proj, tau)
        Bmat = bernstein_basis(U, K)
        p_hat = Bmat.mean(dim=0)
        D_total = D_total + discrete_f_div_from_pmf(p_hat, f_fn)

    return D_total / L

# -----------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(512, 2)
    Y = torch.randn(512, 2) + 0.5
    D = sliced_rank_fdiv(X, Y, L=32, K=32, f_name="js")
    print(f"Sliced soft-rank JS(μ||ν) ≈ {D.item():.6f}")
