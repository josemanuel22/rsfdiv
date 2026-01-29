# rsfdiv — Rank-Statistic *f*-Divergences (RS-*f*-div)

This repository contains code to **estimate and use rank-statistic approximations of *f*-divergences**. The core idea is to avoid explicit density-ratio estimation by working with **ranks / rank histograms**, and (optionally) extend the construction to high-dimensional data via **random projections (“slicing”)**.

> Typical uses:
> - divergence estimation between two sample sets (1D and sliced/high-D),
> - two-sample testing / discrepancy measurement,
> - training implicit generative models with RS-*f*-divergence–based losses (e.g., as pretraining or auxiliary objectives).

---

## Contents (high level)

You will typically find (names may differ slightly):
- **`src/` or package module**: estimator + utilities (rank histograms, discrete *f*-divs, slicing).
- **`scripts/`**: runnable experiments (benchmarks, training loops, sweeps).
- **`configs/`** (optional): Hydra / YAML configs for experiments.
- **`notebooks/`** (optional): exploratory plots and sanity checks.
- **`results/` or `outputs/`** (optional): logs, figures, checkpoints.

---

## Setup

### 1) Create an environment

```bash
python3 -m venv .venv

# 2) Activate it
source .venv/bin/activate

# 3) Upgrade pip tooling (recommended)
python -m pip install --upgrade pip setuptools wheel

# 4) Install the repo requirements
pip install -r requirements.txt

# (Optional) if the repo is a package and you want editable install:
# pip install -e .

# 5) Quick sanity check
python -c "import sys; print('OK, python =', sys.executable)"
``` 

### 2)Run a small sanity check

Run the 1D numerical example:

```bash
PYTHONPATH=./src python ./examples/demo_numpy_1d.py --K 32 --n-mu 10000 --n-nu 10000
```

### 3)Run proximal rank f-div on 2d examples

Here is a small example of how to run the proximal-rank algorithm on 2D toy datasets:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./src python ./examples/rank_prox_transport_mala.py --f kl --data checkerboard --L 10 --steps 400 --u-mix-beta 0.4
```

### 4)Run proximal rank f-div algorithm in celebA

Here is a small example of how to run the proximal-rank algorithm on CelebA:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./src python ./examples/celeba_co_rpt_fid.py --feature-space pixel --images-dir .../data/celeba --n-target 2000 --eval-fid-pr --steps 20000 --snap-every 2000
```
