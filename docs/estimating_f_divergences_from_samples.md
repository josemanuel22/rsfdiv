# Estimating KL, Jensen–Shannon and Hellinger divergences directly from samples (without neural networks)

**A practical guide to estimating *f*-divergences without explicit densities or density ratios.** This tutorial uses **rank statistics** to approximate KL, Jensen–Shannon (JS), and squared Hellinger divergences directly from independent samples. It includes working NumPy/SciPy code, a small variational KL comparison, and a multidimensional sliced example.

**Paper:** Viktor Stein and José Manuel de Frutos, [*Approximating f-Divergences with Rank Statistics* (ICML 2026)](https://proceedings.mlr.press/v306/stein26b.html) ([arXiv](https://arxiv.org/abs/2601.22784)).  
**Implementation:** [`src/ranks/ranks_hard.py`](../src/ranks/ranks_hard.py)  
**Reproducible experiment:** [`examples/tutorial_fdivergence_estimators.py`](../examples/tutorial_fdivergence_estimators.py)

> **Scope.** In one dimension, the population rank construction approximates the corresponding *f*-divergence as the rank degree grows under the paper's assumptions. In multiple dimensions, the **sliced** construction targets an average of one-dimensional projected divergences, generally **not the full-dimensional divergence**. Population lower bounds are not automatically finite-sample lower bounds.

## 1. The problem: two sets of samples, no densities

For probability distributions $P$ and $Q$ with $P\ll Q$, the *f*-divergence is

$$
D_f(P\|Q)=\mathbb E_{X\sim Q}\left[f\left(\frac{dP}{dQ}(X)\right)\right],
$$

where $f$ is convex and $f(1)=0$. Standard estimators may rely on density estimation, nearest neighbors, or a variational optimization problem. But what if we only have samples $X_1,\ldots,X_m\sim P$ and $Z_1,\ldots,Z_n\sim Q$?

The rank-statistic construction replaces density-ratio estimation with **comparisons between samples**.

## 2. From ranks to a discrete *f*-divergence

Let $X\sim P$ and let $Z_1,\ldots,Z_K\sim Q$ be independent. Define the comparison rank

$$
A_K=\sum_{j=1}^{K}\mathbf{1}\{Z_j\le X\}\in\{0,\ldots,K\},
$$

and let $R_K(n)=\Pr(A_K=n)$ denote its probability mass function. When $P=Q$ and $Q$ is continuous, this rank is uniform on $\{0,\ldots,K\}$.

Compare its law with $U_K$, the uniform distribution on $K+1$ points:

$$
\boxed{D_f^{(K)}(P\|Q)
=D_f(R_K\|U_K)
=\frac{1}{K+1}\sum_{n=0}^{K}f((K+1)R_K(n)).}
$$

The associated binomial-mixture identity is

$$
R_K(n)=\mathbb E_{X\sim P}\left[
\binom{K}{n}F_Q(X)^n(1-F_Q(X))^{K-n}
\right].
$$

The provided implementation uses the empirical CDF of the $Q$ sample and a sample average over observations from $P$. There is **no density estimator, trained discriminator, or neural network** in this hard-rank calculation.

The paper develops population approximation, monotonicity, and lower-bound results under stated conditions. **Finite-sample plug-in estimates may be above or below the population target and need not be monotonic in $K$.**

### Which divergences?

| Name | Generator $f(t)$ | Convention |
| --- | --- | --- |
| Kullback–Leibler (KL) | $t\log t-(t-1)$ | KL in nats |
| Jensen–Shannon (JS) | $\frac12\left[t\log\frac{2t}{1+t}+\log\frac{2}{1+t}\right]$ | JS in nats, bounded by $\log 2$ |
| Squared Hellinger | $(\sqrt t-1)^2$ | $\int(\sqrt p-\sqrt q)^2\,dx$: **twice** the commonly used $H^2=\frac12\int(\sqrt p-\sqrt q)^2\,dx$ |

## 3. Try it in two commands

From the [repository root](https://github.com/josemanuel22/rsfdiv):

```bash
python -m pip install numpy scipy
PYTHONPATH=./src python examples/tutorial_fdivergence_estimators.py --n 4000 --K 64 --seed 2026
```

The script samples $P=\mathcal N(0,1)$ and $Q=\mathcal N(\delta,1)$, with default $\delta=1$, and reports rank estimates alongside known population values. For this example,

$$
D_{\rm KL}(P\|Q)=\delta^2/2,\qquad
H_{\rm repo}^2(P,Q)=2(1-e^{-\delta^2/8}).
$$

JS is computed by deterministic Gauss–Hermite quadrature. With $\delta=1$, the population KL is $0.5$ and the repository's squared Hellinger is approximately $0.2350$. The script **does not hard-code expected rank estimates**: these change with $K$, sample size, and random seed.

### Minimal Python API

```python
import numpy as np
from ranks.ranks_hard import rs_f_divergence_1d

rng = np.random.default_rng(2026)
p_samples = rng.normal(0.0, 1.0, size=5000)
q_samples = rng.normal(1.0, 1.0, size=5000)

for divergence in ("kl", "js", "hellinger2"):
    res = rs_f_divergence_1d(p_samples, q_samples, K=64, f=divergence)
    print(divergence, res.D)
```

Use `PYTHONPATH=./src` when running your own script from the repository root. The input order matters for asymmetric divergences.

### Choose the rank degree $K$

$K$ controls the number of comparisons in a rank and the number $K+1$ of possible rank values. Larger $K$ improves the **population** approximation under the assumptions of the paper, but it may increase runtime or sampling noise.

```bash
for K in 16 32 64 128; do
  PYTHONPATH=./src python examples/tutorial_fdivergence_estimators.py --K "$K" --n 4000 --seed 2026
done
```

For an empirical accuracy study, use multiple independent seeds, confidence intervals, and a known ground truth.

## 4. Comparison with variational estimation

For KL, the Donsker–Varadhan representation is

$$
D_{\rm KL}(P\|Q)
=\sup_T\left\{\mathbb E_P[T(X)]-\log\mathbb E_Q[e^{T(Z)}]\right\}.
$$

Variational estimators fit a function $T$ (often a neural network) by optimizing a sample-based objective. The companion script includes a **deliberately simple comparison**: optimize $T_a(x)=ax$ using SciPy, then evaluate the objective on **independent held-out samples**.

This witness is **particularly well specified** for the equal-variance Gaussian shift example, because its true log-density ratio is affine in $x$. It is **not** a neural baseline or a comprehensive benchmark; it is intended to show what optimization-based estimation looks like alongside rank estimation.

| | Hard-rank estimation | Variational KL estimation |
| --- | --- | --- |
| Requires density values | No | No |
| Optimization of a discriminator/test function | No | Yes |
| Key approximation control | Rank degree $K$ | Function class and optimization |
| KL, JS and Hellinger via one interface | Yes | Requires divergence-specific formulations |
| Common practical caveat | Finite-$K$ and finite-sample effects | Function-class bias, optimization, overfitting |

Do not interpret an empirical variational objective as a guaranteed lower bound on population KL. Its finite-sample value may be optimistic, especially when evaluated on its training sample.

## 5. Higher dimensions: sliced rank divergences

In $\mathbb R^d$, project samples onto random unit directions $\theta_\ell$ and average the resulting one-dimensional rank divergences:

$$
\widehat{SD}^{(K,L)}_f(P\|Q)
=\frac1L\sum_{\ell=1}^L
D_f^{(K)}(\theta_\ell\#P\|\theta_\ell\#Q).
$$

```python
import numpy as np
from ranks.ranks_hard import rs_f_divergence_sliced

rng = np.random.default_rng(2026)
p = rng.normal(size=(5000, 5))
q = rng.normal(size=(5000, 5)) + np.array([1, 0, 0, 0, 0])

result = rs_f_divergence_sliced(p, q, K=64, f="js", L=64, random_state=2026)
print("Sliced rank-JS:", result.D)
```

**This estimates a sliced divergence, not the full 5D JS divergence.** By data processing, the population divergence of any projection is at most the full-dimensional divergence; the average is likewise at most the full divergence. A finite-sample plug-in estimate need not obey that inequality exactly.

The runnable script also compares a two-dimensional sliced rank-KL to the analytically known full KL to make this distinction visible.

## 6. When is the method useful?

**Good candidates:** one-dimensional sample-only divergence estimation; optimization-free KL/JS/Hellinger baselines; discrepancies between samples from implicit generative models; and comparisons where the **sliced** target is scientifically meaningful.

**Important limitations:** extreme tails can be difficult at finite $K$, especially for unbounded divergences; large $K$ relative to sample size can increase noise; ties or discrete distributions require care with the uniform-rank null; and hard ranks are not automatically differentiable for training. For model optimization, consult the repository's soft-rank implementations.

When choosing between methods, account for the target estimand, statistical error, computational cost, and function-class assumptions. There is no universal winner.

## 7. Frequently asked questions

**Can I estimate KL divergence from samples without a neural network?**  
Yes. The hard-rank approximation above needs only independent samples and a rank degree $K$. Other non-neural estimators also exist.

**Is this the same as estimating a density ratio?**  
No. The rank histogram is computed from comparisons or the empirical CDF; it does not explicitly reconstruct $p/q$.

**Does increasing $K$ always increase my estimate?**  
Not necessarily with finite data. The monotonic approximation statement concerns the population construction, not arbitrary plug-in estimates.

**Does sliced rank-KL equal the true KL in high dimensions?**  
Generally no. It approaches a sliced target, not automatically the full-dimensional KL.

## Paper and further reading

- Stein, V. and de Frutos, J. M. (2026), [*Approximating f-Divergences with Rank Statistics*](https://proceedings.mlr.press/v306/stein26b.html), *Proceedings of Machine Learning Research*, vol. 306. [arXiv:2601.22784](https://arxiv.org/abs/2601.22784).
- Nowozin et al., [*f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization*](https://arxiv.org/abs/1606.00709).

If this implementation is useful in your research, please cite the ICML 2026 paper.
