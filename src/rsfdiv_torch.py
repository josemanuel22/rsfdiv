# ==== Differentiable RS f-divergence in PyTorch (1D + sliced) =================
import math
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Normal, MixtureSameFamily


@torch.no_grad()
def _unit_sphere(L, d, device):
    v = torch.randn(L, d, device=device)
    return v / (v.norm(dim=1, keepdim=True) + 1e-12)


def _normal_cdf(x):
    # Φ(x) = 0.5 * (1 + erf(x/√2))
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _soft_ecdf(y, y_ref, sigma):
    """
    Differentiable ECDF: F̂_σ(y) = (1/N) Σ_j Φ((y - y_ref_j)/σ)
    y, y_ref: (B,) tensors; sigma>0 (float or tensor)
    """
    diff = (y[:, None] - y_ref[None, :]) / sigma
    return _normal_cdf(diff).mean(dim=1)


def _f_apply(name, x, eps=1e-12):
    x = torch.clamp(x, min=eps)
    if name == "kl":
        return x * torch.log(x) - (x - 1.0)
    if name == "reverse_kl":
        return -torch.log(x) + x - 1.0
    if name == "hellinger2":
        return (torch.sqrt(x) - 1.0) ** 2
    if name == "tv":
        return (x - 1.0).abs()
    if name == "pearson_chi2":
        return (x - 1.0) ** 2
    if name == "neyman_chi2":
        return (1.0 - x) ** 2 / x
    if name == "js":
        return x * (torch.log(2*x) - torch.log1p(x)) + (math.log(2.0) - torch.log1p(x))
    raise ValueError(f"Unknown f: {name}")


def rsf_loss_1d_torch(y_gen, y_ref, K=64, f="kl", sigma=None):
    """
    y_gen, y_ref: (B,) float tensors (requires_grad on y_gen for training).
    sigma: smoothing for ECDF; default = 0.2 * std(y_ref) (no grad).
    """
    device, dtype = y_gen.device, y_gen.dtype
    if sigma is None:
        with torch.no_grad():
            s = y_ref.std(unbiased=False) + 1e-6
        sigma = 0.2 * s
    t = _soft_ecdf(y_gen, y_ref, sigma)                   # (B,)
    t = torch.clamp(t, 1e-7, 1.0 - 1e-7)

    # Vectorized Binomial pmfs over n=0..K for all samples
    n = torch.arange(K+1, device=device, dtype=dtype)[:, None]  # (K+1,1)
    logC = (torch.lgamma(torch.tensor(K+1., device=device, dtype=dtype))
            - torch.lgamma(n+1.) - torch.lgamma(torch.tensor(K-n+1., device=device, dtype=dtype)))
    logpmf = logC + n*torch.log(t)[None, :] + \
        (K-n)*torch.log1p(-t)[None, :]   # (K+1,B)
    Q = logpmf.exp().mean(dim=1)                                            # (K+1,)

    # scalar
    return _f_apply(f, (K+1.0) * Q).mean()


def rsf_loss_sliced_torch(X_gen, X_ref, K=64, f="kl", L=64, sigma=None):
    """
    X_gen, X_ref: (B,d). Draw L random directions and average 1D RS losses.
    """
    device, dtype = X_gen.device, X_gen.dtype
    d = X_gen.shape[1]
    S = _unit_sphere(L, d, device)                        # (L,d)
    losses = []
    for s in S:                                           # small L -> loop is fine; vectorize if needed
        y_g = (X_gen @ s)                                 # (B,)
        y_r = (X_ref @ s)
        losses.append(rsf_loss_1d_torch(y_g, y_r, K=K, f=f, sigma=sigma))
    return torch.stack(losses).mean()

# ==== Minimal training example =================================================
# Learn a 1D generator G(z) to match target N(1,1) by minimizing RS KL


class Gen1D(nn.Module):
    def __init__(self, zdim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z): return self.net(z).squeeze(-1)


B, zdim = 10000, 4
G = Gen1D(zdim).cuda() if torch.cuda.is_available() else Gen1D(zdim)
opt = torch.optim.Adam(G.parameters(), lr=1e-2)

device = next(G.parameters()).device  # or torch.device("cpu"/"cuda")

target_dist = torch.distributions.Normal(
    loc=torch.tensor(10.0), scale=torch.tensor(2.0))

weights = torch.tensor([0.7, 0.3], device=device)
means = torch.tensor([0.0, 5.0], device=device)
scales = torch.tensor([1.0, 0.8], device=device)

mix = Categorical(probs=weights)          # ()
comp = Normal(loc=means, scale=scales)     # batch_shape=(2,), event_shape=()
target_dist = MixtureSameFamily(mix, comp)  # mixture in R

# training loop usage:
# x_real = target_dist.sample((B,))          # shape (B,)

for step in range(200):
    # real batch from target ν
    x_real = target_dist.sample((B,)).to(device)

    # fake batch μθ via reparam z->Gθ(z)
    z = torch.randn(B, zdim, device=device)
    x_fake = G(z)

    # anneal smoothing: start smooth, decrease over time (helps stability)
    sigma = max(0.05, 0.5 * math.exp(-step / 1000.0)) * \
        (x_real.std(unbiased=False).item() + 1e-6)

    loss = rsf_loss_1d_torch(x_fake, x_real, K=10000, f="kl", sigma=sigma)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if step % 100 == 0:
        with torch.no_grad():
            m, s = x_fake.mean().item(), x_fake.std(unbiased=False).item()
        print(
            f"step {step:4d} | loss {loss.item():.4f} | gen mean {m:.3f} std {s:.3f}")

with torch.no_grad():
    n_eval = 20_000
    z = torch.randn(n_eval, zdim, device=device)
    x_fake_eval_t = G(z).detach().cpu()                        # (n_eval,)
    x_real_eval_t = target_dist.sample((n_eval,)).to(device).cpu()

    # convert tensors -> Python lists (no NumPy needed)
    x_fake_eval = x_fake_eval_t.tolist()
    x_real_eval = x_real_eval_t.tolist()

    # common plotting range
    lo = float(min(min(x_fake_eval), min(x_real_eval)))
    hi = float(max(max(x_fake_eval), max(x_real_eval)))

    # true N(1,1) pdf with torch, then to lists
    xs_t = torch.linspace(lo, hi, 400)
    pdf_true_t = (1.0 / math.sqrt(2.0 * math.pi)) * \
        torch.exp(-0.5 * (xs_t - 1.0) ** 2)
    xs = xs_t.tolist()
    pdf_true = pdf_true_t.tolist()

    plt.figure(figsize=(10, 4))

    # --- PDF / histogram overlay
    plt.subplot(1, 2, 1)
    plt.hist(x_real_eval, bins=80, range=(lo, hi),
             density=True, alpha=0.6, label="target")
    plt.hist(x_fake_eval, bins=80, range=(lo, hi),
             density=True, alpha=0.6, label="generator")
    plt.plot(xs, pdf_true, linewidth=2.0, label="true pdf N(1,1)")
    plt.title("PDF / Histogram")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend(loc="best")

    # --- Empirical CDF overlay (pure Python lists)

    def ecdf_list(a):
        a_sorted = sorted(a)
        n = len(a_sorted)
        y = [(i+1)/n for i in range(n)]
        return a_sorted, y

    xr, yr = ecdf_list(x_real_eval)
    xg, yg = ecdf_list(x_fake_eval)

    plt.subplot(1, 2, 2)
    plt.plot(xr, yr, label="target ECDF")
    plt.plot(xg, yg, label="generator ECDF")
    plt.title("Empirical CDF")
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig("rsfdiv_final_result.png", dpi=150)
    plt.show()
    print("Saved plot to rsfdiv_final_result.png")
