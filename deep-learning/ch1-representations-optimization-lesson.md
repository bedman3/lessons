# Chapter 1 — Representation Learning, Gradients & Optimization

Classical feature engineering chooses a representation before fitting a model. Deep learning fits layers of representation and prediction jointly. Its power comes from composition; its difficulty comes from optimizing and validating a large, adaptive system.

## 1.1 Representations as coordinate systems

A representation $h=f_\theta(x)$ maps raw input into coordinates useful for downstream tasks. A good representation makes relevant distinctions simple and suppresses nuisance variation.

Layers compose transformations:

$$
h^{(\ell)}=\phi(W^{(\ell)}h^{(\ell-1)}+b^{(\ell)}).
$$

Early layers often capture local/simple patterns; later layers combine them into task-specific abstractions. This hierarchy is an empirical tendency, not a guarantee of human-interpretable concepts.

## 1.2 Tensor shapes are part of the model

For a batch of $B$ examples with $d_{in}$ features,

$$
X\in\mathbb R^{B\times d_{in}},\quad
W\in\mathbb R^{d_{in}\times d_{out}},\quad
H=XW+b\in\mathbb R^{B\times d_{out}}.
$$

Broadcasting adds $b\in\mathbb R^{d_{out}}$ to every row. Tracking shapes exposes many bugs before calculus begins.

A layer has $d_{in}d_{out}+d_{out}$ trainable parameters. A network $100\to256\to64\to10$ has

$$
100(256)+256+256(64)+64+64(10)+10=42{,}954
$$

parameters.

## 1.3 Nonlinearity creates expressive composition

Without nonlinear activations, multiple affine layers collapse to one affine map. Common choices:

- ReLU: $\max(0,z)$, simple and sparse but can create dead units;
- GELU/SiLU: smooth gating-like activations common in Transformers;
- tanh: bounded and zero-centred but saturates;
- sigmoid: useful for gates/probabilities, saturates in deep hidden stacks.

Activation choice interacts with initialization and normalization through signal variance and gradient flow.

## 1.4 Computational graphs and backpropagation

A computational graph records intermediate operations. Reverse-mode differentiation applies the chain rule from scalar loss backward.

For $z=Wx+b$, $h=\phi(z)$, and upstream gradient $g_h=\partial L/\partial h$,

$$
g_z=g_h\odot\phi'(z),
$$

$$
\frac{\partial L}{\partial W}=g_zx^T,
\qquad
\frac{\partial L}{\partial x}=W^Tg_z.
$$

Gradients accumulate when a value influences loss through multiple paths. Backprop is dynamic programming over the graph, reusing local derivatives rather than expanding every chain.

## 1.5 Vanishing and exploding gradients

Across many layers, gradients multiply Jacobians:

$$
\frac{\partial h^{(L)}}{\partial h^{(0)}}
=J_LJ_{L-1}\cdots J_1.
$$

Repeated singular values below one shrink gradients; above one amplify them. Saturating activations, poorly scaled weights, and long recurrences worsen the problem.

Monitor gradient norms by layer, activation distributions, fraction of dead/saturated units, and update-to-weight ratios. Gradient clipping limits explosions but does not repair the cause of persistent vanishing.

## 1.6 Initialization

If weights are too small, signals vanish; too large, activations or gradients explode. Variance-preserving schemes approximate stable propagation:

- Xavier/Glorot for symmetric activations:

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{d_{in}+d_{out}};
$$

- He initialization for ReLU:

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{d_{in}}.
$$

These rely on independence and distribution approximations that training soon violates, but they provide a workable starting scale.

## 1.7 Normalization

Batch normalization standardizes activations using batch statistics and learns scale/shift. It can smooth optimization and regularize, but behaviour depends on batch size and differs between training and inference.

Layer normalization standardizes across features within one example:

$$
\operatorname{LN}(h)=\gamma\odot\frac{h-\mu_h}{\sqrt{\sigma_h^2+\epsilon}}+\beta.
$$

It is independent of other batch examples and suits variable-length sequence models. RMSNorm omits mean subtraction and normalizes root-mean-square scale.

Normalization does not make input preprocessing, learning-rate control, or monitoring unnecessary.

## 1.8 Residual connections

A residual block computes

$$
h_{\ell+1}=h_\ell+F_\ell(h_\ell).
$$

The identity path gives gradient Jacobian $I+J_F$, allowing information and gradients to bypass poorly conditioned transformations. The network can learn small refinements rather than reconstructing every representation.

Pre-normalization in Transformers places normalization before the residual branch and often improves very-deep training stability.

## 1.9 Optimization

Mini-batch SGD estimates the full gradient:

$$
\theta_{t+1}=\theta_t-\eta_t\widehat\nabla L(\theta_t).
$$

Momentum averages gradient direction. Adam combines momentum with coordinate-wise second-moment scaling. AdamW decouples weight decay from the adaptive gradient update.

Warmup prevents unstable early steps; decay schedules reduce step size later. Larger batches reduce gradient noise but can require learning-rate changes and may alter generalization.

Training loss should be interpreted with learning rate, batch size, gradient scale, and data order—not optimizer name alone.

## 1.10 Regularization

- Weight decay discourages large weights.
- Dropout multiplies activations by random masks during training.
- Data augmentation encodes invariances through label-preserving transformations.
- Early stopping limits adaptation to training noise.
- Label smoothing prevents extreme classification confidence.
- Mixup interpolates examples and targets, encouraging local linearity.

Regularization can improve validation loss while worsening probability calibration or rare-case recall, so evaluate the intended decision.

## 1.11 A debugging ladder

1. Overfit one tiny batch; failure suggests code, target, or optimization errors.
2. Verify loss against a hand calculation and random/uniform baseline.
3. Check gradients numerically on a small network.
4. Inspect activations, gradients, and parameter updates by layer.
5. Scale data/model gradually and compare learning curves.
6. Separate train/eval modes for dropout and normalization.
7. Audit data leakage and labels before tuning architecture.

## 1.12 Failure modes

- Counting examples while ignoring tensor or mask shapes.
- Adding layers without nonlinearities and expecting more expressiveness.
- Using training-mode batch normalization at inference.
- Treating gradient clipping as a cure for unstable architecture.
- Confusing $L_2$ penalty with decoupled AdamW weight decay.
- Tuning on a test set because training is expensive.
- Scaling model size without measuring data quality or baseline error.

## 1.13 Knowledge checks

1. Why does a stack of affine layers collapse to one affine map?
2. Derive gradients for one affine-plus-activation layer.
3. How do residual paths help gradient flow?
4. Contrast batch normalization and layer normalization.
5. What does the tiny-batch overfit test diagnose?

### Solution outlines

1. Composition $(W_2(W_1x+b_1)+b_2)$ is another affine transformation.
2. Multiply upstream gradient by activation derivative; outer-product with input for weight gradient; multiply by $W^T$ for input gradient.
3. The identity term lets gradients propagate without relying entirely on every residual Jacobian.
4. Batch norm uses cross-example statistics per feature; layer norm uses within-example feature statistics.
5. Whether the implementation and optimizer can fit a trivial subset before generalization becomes relevant.

## 1.14 What to retain

- Deep learning jointly learns representations and predictors through composition.
- Tensor shapes and computational graphs are first-class reasoning tools.
- Initialization, normalization, residuals, and optimization jointly control gradient flow.
- Regularization encodes stability preferences, not universal improvement.
- Diagnose data, loss, gradients, and modes before scaling architecture.

Next: [Chapter 2 — Convolutional Networks](ch2-convolutional-networks-viewer.html).
