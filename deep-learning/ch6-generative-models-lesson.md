# Chapter 6 — Generative Models: Autoregression, VAEs, GANs & Diffusion

Generative modelling estimates or samples from a data distribution. Different families trade likelihood, sample quality, mode coverage, latent structure, training stability, and sampling speed.

No single score captures all these properties.

## 6.1 Autoregressive factorization

Any joint distribution can be factorized:

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^Tp(x_t\mid x_{<t}).
$$

Autoregressive models train by maximum likelihood with teacher forcing. They provide tractable likelihood and strong mode coverage when expressive, but sampling is sequential.

Ordering is an architectural choice. Text has a natural order; images require raster, multiscale, or tokenized order that affects dependencies and generation speed.

## 6.2 Latent-variable models

Introduce latent $z$:

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)dz.
$$

The latent can represent compressed factors of variation. The integral is usually intractable, motivating variational inference.

## 6.3 The variational lower bound

Choose approximate posterior $q_\phi(z\mid x)$. Then

$$
\log p_\theta(x)
\ge
E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{KL}(q_\phi(z\mid x)\|p(z)).
$$

This ELBO contains:

- reconstruction/log-likelihood term encouraging $z$ to explain $x$;
- KL term making encoded latents compatible with the prior.

The gap is $D_{KL}(q_\phi(z\mid x)\|p_\theta(z\mid x))$.

## 6.4 Reparameterization

For Gaussian encoder

$$
q_\phi(z\mid x)=N(\mu_\phi(x),\operatorname{diag}\sigma_\phi(x)^2),
$$

sample as

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,
\qquad\epsilon\sim N(0,I).
$$

Randomness is moved to parameter-free $\epsilon$, allowing low-variance pathwise gradients through $\mu$ and $\sigma$.

VAEs often produce smooth latent spaces and broad mode coverage, while simple likelihood choices can yield visually blurry samples.

## 6.5 Posterior collapse

With a powerful decoder, the model may ignore $z$, giving $q(z\mid x)\approx p(z)$ and near-zero KL. The decoder models data autoregressively without using latent information.

KL warmup, free bits, weaker decoder conditioning, richer priors/posteriors, or modified objectives can help. Monitor mutual-information proxies and reconstructions, not only total ELBO.

## 6.6 Generative adversarial networks

A GAN trains generator $G(z)$ against discriminator $D(x)$:

$$
\min_G\max_D
E_{x\sim p_{data}}[\log D(x)]
+E_{z\sim p(z)}[\log(1-D(G(z)))].
$$

The discriminator learns to distinguish real from generated; the generator learns to fool it. GANs can produce sharp samples without explicit likelihood, but the game can be unstable.

Mode collapse occurs when many latent values map to a small set of outputs. A strong discriminator can also yield poor generator gradients.

## 6.7 Wasserstein perspective

Wasserstein GAN replaces classification divergence with an approximation to earth-mover distance and constrains the critic to be Lipschitz, often using a gradient penalty. This can improve gradient behaviour and provide a more meaningful training signal.

It does not eliminate architecture, optimization, or mode-coverage problems.

## 6.8 Diffusion forward process

Diffusion gradually adds Gaussian noise:

$$
q(x_t\mid x_{t-1})
=N(\sqrt{1-\beta_t}x_{t-1},\beta_tI).
$$

Let $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$. Then sample any noise level directly:

$$
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad\epsilon\sim N(0,I).
$$

At large $t$, $x_t$ approaches Gaussian noise.

## 6.9 Learning the reverse process

A network predicts noise, clean data, velocity, or score from $(x_t,t,\text{condition})$. A common simplified objective is

$$
E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right].
$$

Reverse sampling starts from noise and repeatedly denoises. U-Nets provide multiscale image structure; attention injects global and conditional information.

Diffusion training is stable and covers modes well, but iterative sampling is slower than one-pass generators.

## 6.10 Score interpretation

The score is

$$
\nabla_x\log p_t(x).
$$

It points toward higher-density regions of the noisy distribution. Denoising score matching learns this vector field across noise levels. Reverse-time SDE/ODE formulations connect diffusion sampling to continuous dynamics.

## 6.11 Guidance and sampling

Classifier guidance uses gradients from a separate classifier. Classifier-free guidance combines conditional and unconditional predictions:

$$
\epsilon_{guided}
=\epsilon_{uncond}+s(\epsilon_{cond}-\epsilon_{uncond}).
$$

Larger guidance scale strengthens condition adherence but can reduce diversity and create artefacts. DDIM and numerical ODE/SDE solvers reduce steps or alter stochasticity, trading speed and sample properties.

## 6.12 Comparing families

| Family | Likelihood | Sampling | Typical strength | Typical weakness |
|---|---|---|---|---|
| Autoregressive | Tractable | Sequential | Density modelling, mode coverage | Slow generation |
| VAE | Lower bound | One/few passes | Latent structure, stable training | Likelihood gap, blur under simple decoders |
| GAN | Usually implicit | One pass | Sharp fast samples | Instability, mode collapse |
| Diffusion | Variational/score view | Iterative | Quality, coverage, stable training | Slow and compute-heavy sampling |

Hybrid systems combine discrete autoencoders, autoregressive priors, diffusion decoders, or adversarial/perceptual losses.

## 6.13 Evaluation

Evaluate likelihood where meaningful, sample quality, diversity/mode coverage, precision–recall in feature space, downstream utility, memorization/privacy, conditional adherence, and subgroup behaviour.

FID compares Gaussian approximations to feature distributions and depends on feature extractor and sample count. Good FID does not guarantee factual, semantic, or safety quality.

## 6.14 Failure modes

- Comparing likelihood across incompatible data preprocessing or discretization.
- Calling a low VAE reconstruction loss a good generative density.
- Ignoring posterior or mode collapse.
- Treating discriminator loss as a direct sample-quality metric.
- Increasing guidance without checking diversity and artefacts.
- Reporting one feature-space score as comprehensive evaluation.
- Failing to test training-data memorization.

## 6.15 Knowledge checks

1. Derive the ELBO by introducing $q(z\mid x)$ and Jensen's inequality.
2. Why does reparameterization reduce gradient difficulty?
3. What is mode collapse in a GAN?
4. How can $x_t$ be sampled directly from $x_0$ in diffusion?
5. What trade-off does classifier-free guidance control?

### Solution outlines

1. Rewrite marginal likelihood as an expectation of $p(x,z)/q(z\mid x)$ and apply Jensen.
2. It expresses samples as a differentiable parameterized transformation of fixed noise.
3. Many latent inputs generate few modes, leaving parts of the data distribution uncovered.
4. Use the closed-form Gaussian with $\bar\alpha_t$ and one noise sample.
5. Conditional adherence versus diversity/naturalness.

## 6.16 What to retain

- Generative families optimize different approximations and make different sampling trade-offs.
- VAEs balance reconstruction with a prior-compatible posterior.
- GANs learn through a game and can sacrifice mode coverage.
- Diffusion learns denoising/score fields over noise scales.
- Evaluation must separate likelihood, quality, diversity, utility, and memorization.

Next: [Chapter 7 — Transfer, Scaling & Failure Diagnosis](ch7-transfer-scaling-diagnosis-viewer.html).
