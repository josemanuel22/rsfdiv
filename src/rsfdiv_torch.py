# train_mnist_sliced_rank_fdiv.py
# Minimal MNIST generator trained with sliced rank f-divergence (JS by default).

import math, argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

# -----------------------------
# Generator: DCGAN-ish for 28x28
# -----------------------------
class Gen28(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256*7*7),
            nn.BatchNorm1d(256*7*7),
            nn.ReLU(True),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 7 -> 14
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 14 -> 28
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh(),  # outputs in [-1,1]
        )

    def forward(self, z):
        x = self.net(z)
        x = x.view(z.size(0), 256, 7, 7)
        x = self.up(x)
        return x

# -----------------------------
# f-generators (natural logs)
# -----------------------------
def f_js(t):
    u = torch.clamp(t, 1e-30, None)
    return 0.5 * (u * (torch.log(2*u) - torch.log1p(u)) + (math.log(2.) - torch.log1p(u)))

def f_kl(t):
    u = torch.clamp(t, 1e-30, None)
    return u * torch.log(u)

def f_hellinger2(t):
    # (sqrt(t)-1)^2
    u = torch.clamp(t, 0.0, None)
    return (torch.sqrt(u) - 1.0)**2

def pick_f(name):
    name = name.lower()
    if name in ("js","jensen-shannon"): return f_js
    if name in ("kl",): return f_kl
    if name in ("hellinger2","hell2"): return f_hellinger2
    raise ValueError("f must be one of: js, kl, hellinger2")

# -----------------------------
# Soft CDF & Bernstein rank pmf
# -----------------------------
def soft_ecdf_Q_of_x(x_proj, y_proj, tau):
    """
    x_proj: (B,) fake projections
    y_proj: (M,) real projections (reference)
    tau: temperature > 0 (scale ~ std(y_proj)*c)
    Returns U in [0,1]^B approximating F_Q(x).
    """
    # pairwise (x - y)/tau -> sigmoid
    diff = (x_proj[:, None] - y_proj[None, :]) / tau
    U = torch.sigmoid(diff).mean(dim=1)
    # clamp away from 0/1 for numerical stability
    return U.clamp(1e-6, 1-1e-6)

def bernstein_basis(U, K):
    """
    U: (B,) in (0,1)
    Returns Bmat: shape (B, K+1) with B_{K,n}(U) = C(K,n) U^n (1-U)^{K-n}
    """
    B = U.shape[0]
    n = torch.arange(K+1, device=U.device, dtype=U.dtype)  # (K+1,)
    # log comb using lgamma
    logC = torch.lgamma(torch.tensor(K+1., device=U.device, dtype=U.dtype)) \
         - torch.lgamma(n+1.) - torch.lgamma(torch.tensor(K, device=U.device, dtype=U.dtype)-n+1.)
    # (B,1) and (1,K+1)
    Ue = U[:, None]
    logB = logC + n[None, :]*torch.log(Ue) + (K - n)[None, :]*torch.log(1 - Ue)
    return torch.exp(logB)  # (B, K+1)

def discrete_f_div_from_pmf(p_hat, f_fn):
    """
    p_hat: (K+1,) probabilities summing to 1
    q_hat: uniform on {0,...,K}
    """
    K = p_hat.numel() - 1
    q = 1.0 / (K+1)
    ratio = torch.clamp(p_hat / q, 1e-30, None)
    return (q * f_fn(ratio)).sum()

def sliced_rank_fdiv(X, Y, L=64, K=64, f_name="js", tau_scale=0.3, rng=None):
    """
    X: fake images in [-1,1], shape (B,1,28,28)
    Y: real images in [-1,1], shape (M,1,28,28)
    L: number of random directions
    K: Bernstein degree (rank resolution)
    f_name: 'js', 'kl', or 'hellinger2'
    tau_scale: temperature = tau_scale * std(y_proj) + 1e-4
    """
    f_fn = pick_f(f_name)
    device = X.device
    B = X.size(0); M = Y.size(0)
    XF = X.view(B, -1)  # (B, 784)
    YF = Y.view(M, -1)  # (M, 784)

    if rng is None:
        rng = torch.Generator(device=device)
        rng.manual_seed(1234)

    D_total = 0.0
    for _ in range(L):
        # random direction on sphere
        s = torch.randn(XF.size(1), device=device, generator=rng)
        s = s / (s.norm() + 1e-12)
        x_proj = XF @ s  # (B,)
        y_proj = YF @ s  # (M,)

        tau = tau_scale * (y_proj.std() + 1e-6) + 1e-4
        U = soft_ecdf_Q_of_x(x_proj, y_proj, tau)  # (B,)
        Bmat = bernstein_basis(U, K)               # (B, K+1)
        p_hat = Bmat.mean(dim=0)                   # (K+1,)
        D_total = D_total + discrete_f_div_from_pmf(p_hat, f_fn)

    return D_total / L

# -----------------------------
# Training loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./runs/mnist_srfdiv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--f", type=str, default="js", choices=["js","kl","hellinger2"])
    parser.add_argument("--tau_scale", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Data: map to [-1,1]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Model & opt
    G = Gen28(z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_z = torch.randn(64, args.z_dim, device=device)

    step = 0
    for epoch in range(1, args.epochs+1):
        for (real, _) in loader:
            real = real.to(device)  # (B,1,28,28) in [-1,1]
            z = torch.randn(real.size(0), args.z_dim, device=device)
            fake = G(z)  # (B,1,28,28)

            # Sliced rank f-divergence (minimize D_f(model || data))
            D = sliced_rank_fdiv(fake, real, L=args.L, K=args.K, f_name=args.f, tau_scale=args.tau_scale)

            opt.zero_grad(set_to_none=True)
            D.backward()
            opt.step()

            if step % 200 == 0:
                with torch.no_grad():
                    grid = vutils.make_grid(G(fixed_z), nrow=8, normalize=True, value_range=(-1,1))
                    vutils.save_image(grid, os.path.join(args.out_dir, f"samples_e{epoch:03d}_s{step:06d}.png"))
                print(f"epoch {epoch:03d} | step {step:06d} | D_{args.f} {D.item():.4f}")
            step += 1

    # final samples
    with torch.no_grad():
        grid = vutils.make_grid(G(fixed_z), nrow=8, normalize=True, value_range=(-1,1))
        vutils.save_image(grid, os.path.join(args.out_dir, f"samples_final.png"))

if __name__ == "__main__":
    main()
