# train_mnist_sliced_rank_fdiv_hq.py
# MNIST generator with high-quality sliced rank f-divergence (patch-sliced, multiscale, EMA).

import math, argparse, os, collections
import torch
import torch.nn as nn
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
_EPS = 1e-12

def f_js(u):
    v = torch.clamp(u, min=_EPS)
    # Binary-entropy form (very stable near 0/∞)
    w = 1.0 / (1.0 + v)
    return 0.5 * (1.0 + v) * (math.log(2.0) + torch.log(torch.clamp(w, min=_EPS)) + torch.log1p(-w))

def f_kl(u):
    v = torch.clamp(u, min=_EPS)
    return v * torch.log(v)  # f(1)=0

def f_hellinger2(u):
    v = torch.clamp(u, min=0.0)
    return (torch.sqrt(v) - 1.0)**2

def pick_f(name):
    name = name.lower()
    if name in ("js","jensen-shannon"): return f_js
    if name in ("kl",): return f_kl
    if name in ("hellinger2","hell2"): return f_hellinger2
    raise ValueError("f must be one of: js, kl, hellinger2")

# -----------------------------
# Helpers (binomial, ECDF)
# -----------------------------
def precompute_logC(K, device, dtype):
    n = torch.arange(K+1, device=device, dtype=dtype)
    return (torch.lgamma(torch.tensor(float(K+1), dtype=dtype, device=device))
            - torch.lgamma(n + 1.) - torch.lgamma(torch.tensor(float(K), dtype=dtype, device=device) - n + 1.))

@torch.no_grad()
def smooth_ecdf_chunked(x_proj, y_proj, tau, ref_chunk=16384):
    """Low-memory smooth ECDF: mean_j sigmoid((x - y_j)/tau)."""
    N, M = x_proj.numel(), y_proj.numel()
    out = torch.zeros(N, device=x_proj.device, dtype=x_proj.dtype)
    for a in range(0, M, ref_chunk):
        b = min(a+ref_chunk, M)
        z = (x_proj[:, None] - y_proj[None, a:b]) / tau
        out += torch.sigmoid(z).mean(dim=1) * (b - a)
    return out / M

def bernstein_mean(U, K, logC, chunk=65536):
    """Compute mean over i of B_{K,n}(U_i) without building (N, K+1) all at once."""
    U = torch.clamp(U, 1e-6, 1-1e-6)
    n = torch.arange(K+1, device=U.device, dtype=U.dtype)
    total = torch.zeros(K+1, device=U.device, dtype=U.dtype)
    N = U.numel()
    for a in range(0, N, chunk):
        b = min(a+chunk, N)
        u = U[a:b]
        logB = (logC[None, :] + u[:, None].log()*n[None, :] + (1.0 - u)[:, None].log()*(K - n)[None, :])
        total += torch.exp(logB).sum(dim=0)
    return total / max(1, N)  # (K+1,)

def discrete_f_div_from_pmf(p_hat, f_fn):
    K = p_hat.numel() - 1
    q = 1.0 / (K+1)
    ratio = torch.clamp(p_hat / q, min=_EPS)
    return (q * f_fn(ratio)).sum()

# -----------------------------
# Patch-sliced SRF divergence
# -----------------------------
class RSFPatchLoss2D(nn.Module):
    """
    Random conv filters as projection directions over patches.
    For each filter ℓ:
       y = conv(real), x = conv(fake)  -> vectors
       U_i = F̂(y)(x_i) via smooth-ECDF
       Q(n) = mean_i Binom(K, U_i)[n] = mean_i Bernstein_{K,n}(U_i)
       D_f = mean_ℓ f((K+1)Q)^mean
    """
    def __init__(self, C=1, K=96, L=128, k_list=(3,5,7), tau_scales=(0.30,0.18,0.10),
                 stride=2, f_name="js", device="cuda", seed=1234, fix_filters=True):
        super().__init__()
        self.C, self.K, self.L = C, K, L
        self.k_list = tuple(k_list)
        self.tau_scales = tuple(tau_scales)
        self.stride = stride
        self.f_fn = pick_f(f_name)
        self.device = device
        self.register_buffer("logC", precompute_logC(K, device, torch.float32))
        self.fix_filters = fix_filters
        self.seed = seed
        if fix_filters:
            self.filters = self._make_filters()

    def _make_filters(self):
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        banks = []
        for k in self.k_list:
            W = torch.randn(self.L, self.C, k, k, device=self.device, generator=g)
            W = W / (W.flatten(1).norm(dim=1, keepdim=True).view(self.L,1,1,1) + 1e-8)
            banks.append(W)
        return banks  # list of (L,C,k,k)

    @torch.no_grad()
    def _conv_flat(self, X, W):
        A = torch.conv2d(X, W, bias=None, stride=self.stride, padding=W.shape[-1]//2)  # (B,L,H',W')
        return A.permute(1,0,2,3).contiguous().view(W.shape[0], -1)  # (L, N)

    def forward(self, Xg, Xd):
        loss = 0.0
        denom = 0
        # sample filters per step if not fixed
        banks = self.filters if self.fix_filters else self._make_filters()
        for bank, k, tau_scale in zip(banks, self.k_list, self.tau_scales):
            yg = self._conv_flat(Xg, bank)  # (L, Ng)
            yd = self._conv_flat(Xd, bank)  # (L, Nd)
            for l in range(self.L):
                y = yd[l]; x = yg[l]
                # per-direction temperature using real std
                tau = tau_scale * (y.std() + 1e-6) + 1e-4
                U = smooth_ecdf_chunked(x, y, tau)            # (Ng,)
                p_hat = bernstein_mean(U, self.K, self.logC)  # (K+1,)
                loss = loss + discrete_f_div_from_pmf(p_hat, self.f_fn)
            denom += self.L
        return loss / max(1, denom)

# -----------------------------
# Real replay buffer (bigger M)
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity, C=1, H=28, W=28, device="cuda"):
        self.capacity = capacity
        self.C, self.H, self.W = C, H, W
        self.device = device
        self.buf = torch.empty(capacity, C, H, W, device=device)
        self.size = 0
        self.ptr = 0

    @torch.no_grad()
    def add(self, x):
        B = x.size(0)
        if B >= self.capacity:
            self.buf.copy_(x[-self.capacity:])
            self.size = self.capacity
            self.ptr = 0
            return
        end = min(self.ptr + B, self.capacity)
        if end <= self.capacity:
            self.buf[self.ptr:end] = x[:(end - self.ptr)]
        if end < self.capacity and end - self.ptr < B:
            rem = B - (end - self.ptr)
            self.buf[0:rem] = x[-rem:]
        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.capacity, self.size + B)

    @torch.no_grad()
    def get(self):
        return self.buf[:self.size] if self.size > 0 else None

# -----------------------------
# EMA wrapper
# -----------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.m = Gen28(model.z_dim).to(next(model.parameters()).device)
        self.m.load_state_dict(model.state_dict())
        for p in self.m.parameters(): p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for p_ema, p in zip(self.m.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p.data, alpha=1-self.decay)

# -----------------------------
# Training loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./runs/mnist_srfdiv_hq")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--K", type=int, default=96)
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--f", type=str, default="js", choices=["js","kl","hellinger2"])
    parser.add_argument("--tau_scales", type=str, default="0.30,0.18,0.10")
    parser.add_argument("--k_list", type=str, default="3,5,7")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--buffer", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--save_every", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Data: light augmentation + normalize to [-1,1]
    tfm = transforms.Compose([
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # tiny shifts help
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    # Model, EMA, opt
    G = Gen28(z_dim=args.z_dim).to(device)
    ema = EMA(G, decay=args.ema_decay)
    opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Patch-sliced SRF loss
    k_list = tuple(int(k) for k in args.k_list.split(",") if k.strip())
    tau_scales = tuple(float(t) for t in args.tau_scales.split(",") if t.strip())
    assert len(k_list) == len(tau_scales), "k_list and tau_scales must have same length"
    loss_fn = RSFPatchLoss2D(C=1, K=args.K, L=args.L, k_list=k_list, tau_scales=tau_scales,
                             stride=args.stride, f_name=args.f, device=device, seed=args.seed, fix_filters=True)

    # Replay buffer for real images
    buffer = ReplayBuffer(capacity=args.buffer, C=1, H=28, W=28, device=device)

    fixed_z = torch.randn(64, args.z_dim, device=device)
    step = 0
    for epoch in range(1, args.epochs+1):
        for (real, _) in loader:
            real = real.to(device, non_blocking=True)  # (B,1,28,28)
            buffer.add(real)

            # If buffer not yet filled, fall back to current batch
            real_ref = buffer.get()
            if real_ref is None or real_ref.shape[0] < args.batch_size:
                real_ref = real

            z = torch.randn(real.size(0), args.z_dim, device=device)
            fake = G(z)

            # Sliced rank f-divergence on patch projections
            D = loss_fn(fake, real_ref)

            opt.zero_grad(set_to_none=True)
            D.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt.step(); ema.update(G)

            if step % args.save_every == 0:
                with torch.no_grad():
                    grid = vutils.make_grid(ema.m(fixed_z), nrow=8, normalize=True, value_range=(-1,1))
                    vutils.save_image(grid, os.path.join(args.out_dir, f"samples_e{epoch:03d}_s{step:06d}.png"))
                print(f"epoch {epoch:03d} | step {step:06d} | D_{args.f} {D.item():.4f}")
            step += 1

    # final samples (EMA)
    with torch.no_grad():
        grid = vutils.make_grid(ema.m(fixed_z), nrow=8, normalize=True, value_range=(-1,1))
        vutils.save_image(grid, os.path.join(args.out_dir, f"samples_final.png"))

if __name__ == "__main__":
    main()
