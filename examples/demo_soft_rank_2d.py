#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Soft-rank f-divergence demos in 2D
----------------------------------
Train simple 2D generators on toy datasets using the differentiable
sliced-rank f-divergence from `ranks.ranks_soft`.

Datasets:
  - two_moons
  - blobs_imbalanced
  - ring

Generators:
  - mlp       (nonlinear)
  - gaussian  (linear A z + b)

Example:
    python examples/demo_soft_rank_2d.py --data ring --f js --steps 1200 --snap-every 20 --gif
"""

import os, math, argparse, torch, torch.nn as nn
import matplotlib.pyplot as plt
from ranks.ranks_torch import sliced_rank_fdiv

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def plot_snapshot(G, Xd, z_fixed, step, f_name, snap_dir):
    G.eval()
    Xg = G(z_fixed).detach().cpu()
    Xd = Xd.detach().cpu()
    plt.figure(figsize=(5, 5))
    plt.scatter(Xd[:, 0], Xd[:, 1], s=6, alpha=0.55, label="data")
    plt.scatter(Xg[:, 0], Xg[:, 1], s=6, alpha=0.55, label="generator")
    plt.axis("equal"); plt.legend()
    plt.title(f"Step {step} | f={f_name}")
    plt.tight_layout()
    path = os.path.join(snap_dir, f"step_{step:05d}.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"[snap] {path}")

def assemble_gif(snap_dir, gif_name="evolution.gif", fps=6):
    try:
        import imageio.v2 as imageio
        frames = [imageio.imread(os.path.join(snap_dir, f))
                  for f in sorted(os.listdir(snap_dir)) if f.endswith(".png")]
        out = os.path.join(snap_dir, gif_name)
        imageio.mimsave(out, frames, duration=1/fps)
        print(f"[gif] saved {out}")
    except Exception as e:
        print(f"[gif] skipped ({e})")

# ---------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------
def make_two_moons(n=4000, noise=0.06, seed=0, device="cpu"):
    torch.manual_seed(seed)
    n1 = n // 2
    n2 = n - n1
    t1 = torch.rand(n1, device=device) * math.pi
    t2 = torch.rand(n2, device=device) * math.pi
    x1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
    x2 = torch.stack([1.0 - torch.cos(t2), -torch.sin(t2) + 0.5], dim=1)
    X = torch.cat([x1, x2], dim=0)
    return X + noise * torch.randn_like(X)

def make_blobs_imbalanced(n=4000, w_small=0.15, dist=3.0, std_big=0.35, std_small=0.20,
                          seed=0, device="cpu"):
    torch.manual_seed(seed)
    n_small = int(n * w_small)
    n_big = n - n_small
    mu_big = torch.tensor([-dist/2, 0.0], device=device)
    mu_small = torch.tensor([+dist/2, 0.0], device=device)
    X_big = mu_big + std_big * torch.randn(n_big, 2, device=device)
    X_small = mu_small + std_small * torch.randn(n_small, 2, device=device)
    return torch.cat([X_big, X_small], dim=0)

def make_ring(n=4000, r=2.0, ring_std=0.05, seed=0, device="cpu"):
    torch.manual_seed(seed)
    theta = torch.rand(n, device=device) * (2 * math.pi)
    base = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * r
    return base + ring_std * torch.randn(n, 2, device=device)

def get_dataset(name, device, seed=0, n=4000):
    name = name.lower()
    if name == "two_moons": return make_two_moons(n, seed=seed, device=device)
    if name == "blobs_imbalanced": return make_blobs_imbalanced(n, seed=seed, device=device)
    if name == "ring": return make_ring(n, seed=seed, device=device)
    raise ValueError(f"Unknown dataset {name}")

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class GenMLP(nn.Module):
    def __init__(self, zdim=2, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zdim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 2),
        )
    def forward(self, z):
        return 2.0 * torch.tanh(self.net(z))

class GenGaussian(nn.Module):
    def __init__(self, zdim=2):
        super().__init__()
        self.A = nn.Parameter(torch.randn(2, zdim) * 0.5)
        self.b = nn.Parameter(torch.zeros(2))
    def forward(self, z):
        return z @ self.A.t() + self.b

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train_soft_rank(
    data="two_moons", gen_type="mlp",
    f="js", K=32, L=32, tau_scale=0.3,
    steps=1500, lr=3e-4, bs=2048, seed=0,
    zdim=2, h=64, snap_every=0, snap_n=2000,
    snap_dir="snapshots", gif=False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    Xd = get_dataset(data, device, seed=seed, n=6000)
    G = GenMLP(zdim, h).to(device) if gen_type == "mlp" else GenGaussian(zdim).to(device)
    opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.99))

    z_fixed = torch.randn(snap_n, zdim, device=device)
    if snap_every:
        ensure_dir(snap_dir)
        plot_snapshot(G, Xd, z_fixed, 0, f, snap_dir)

    for step in range(1, steps + 1):
        z = torch.randn(bs, zdim, device=device)
        Xg = G(z)
        D = sliced_rank_fdiv(Xg, Xd, L=L, K=K, f_name=f, tau_scale=tau_scale)
        opt.zero_grad(set_to_none=True)
        D.backward()
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"[{step:04d}] {f.upper()}={D.item():.5f}")

        if snap_every and step % snap_every == 0:
            plot_snapshot(G, Xd, z_fixed, step, f, snap_dir)

    if gif and snap_every:
        assemble_gif(snap_dir)
    return G, Xd

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="two_moons",
                   choices=["two_moons", "blobs_imbalanced", "ring"])
    p.add_argument("--gen", type=str, default="mlp", choices=["mlp", "gaussian"])
    p.add_argument("--f", type=str, default="js", choices=["js", "kl", "hellinger2"])
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--tau-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snap-every", type=int, default=20)
    p.add_argument("--snap-n", type=int, default=2000)
    p.add_argument("--snap-dir", type=str, default="snapshots_soft")
    p.add_argument("--gif", action="store_true")
    args = p.parse_args()

    G, Xd = train_soft_rank(
        data=args.data, gen_type=args.gen, f=args.f,
        K=args.K, L=args.L, tau_scale=args.tau_scale,
        steps=args.steps, lr=args.lr, bs=args.bs, seed=args.seed,
        snap_every=args.snap_every, snap_n=args.snap_n,
        snap_dir=args.snap_dir, gif=args.gif
    )

if __name__ == "__main__":
    main()
