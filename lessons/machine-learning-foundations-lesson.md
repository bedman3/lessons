# Machine Learning & Deep Learning Foundations

> From decision trees to Transformers — the mathematics, the architectures, the training tricks, and the interview questions that separate practitioners from theorists.

## 1. The learning problem

### 1.1 What machine learning is

Machine learning is the art of finding functions that generalise. You have training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ drawn from some unknown distribution $P_{\text{data}}$. You pick a hypothesis space $\mathcal{H}$ (the set of functions your model can represent) and use an algorithm to find $h \in \mathcal{H}$ that minimises the expected loss on *unseen* data:

$$
h^* = \arg\min_{h \in \mathcal{H}} \;\mathbb{E}_{(x,y) \sim P_{\text{data}}}\!\big[\ell(h(x), y)\big].
$$

You cannot compute this expectation directly. Instead, you minimise the **empirical risk** (the average loss on training data) while controlling the gap between empirical and true risk — the **generalisation gap**.

### 1.2 The three paradigms

| Paradigm | What you have | What you learn |
|---|---|---|
| **Supervised** | $(x_i, y_i)$ pairs | $f: x \mapsto y$ — a mapping from input to output. |
| **Unsupervised** | Only $x_i$ | Structure in $P(x)$ — clusters, densities, latent representations. |
| **Reinforcement** | $(s, a, r, s')$ transitions | A policy $\pi(a \mid s)$ that maximises cumulative reward. |

### 1.3 Bias–variance decomposition

For squared-error loss, the expected prediction error at a point $x$ decomposes cleanly:

$$
\mathbb{E}_{\mathcal{D}}\!\big[(y - \hat{f}(x))^2\big] =
\underbrace{\big(\mathbb{E}[\hat{f}(x)] - f_{\text{true}}(x)\big)^2}_{\text{Bias}^2}
+ \underbrace{\mathbb{E}\!\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]}_{\text{Variance}}
+ \underbrace{\sigma^2_{\epsilon}}_{\text{Irreducible noise}}.
$$

- **High bias (underfitting):** the model is too simple to capture the true pattern. Fix: more capacity, more features, less regularisation.
- **High variance (overfitting):** the model fits the noise in this particular training set. Fix: more data, regularisation, simpler model, ensembling.

The **double descent** phenomenon challenges the classical U-shaped curve: in overparametrised models (where parameters > data points), test error can *decrease again* after the interpolation threshold. Modern deep learning operates in this regime.

---

## 2. Decision trees and ensemble methods

### 2.1 What a tree does

A decision tree recursively partitions the feature space with axis-aligned splits, assigning a constant prediction to each leaf region $R_m$:

$$
\hat{y}_m = \begin{cases}
\arg\max_k \sum_{x_i \in R_m} \mathbf{1}[y_i = k] & \text{(classification)} \\[4pt]
\frac{1}{N_m}\sum_{x_i \in R_m} y_i & \text{(regression)}
\end{cases}
$$

### 2.2 Splitting criteria

**Classification — Gini impurity:**

$$
G = \sum_{k=1}^{K} \hat{p}_k (1 - \hat{p}_k) = 1 - \sum_{k=1}^{K} \hat{p}_k^2.
$$

**Classification — Entropy (information gain):**

$$
H = -\sum_{k=1}^{K} \hat{p}_k \log_2 \hat{p}_k,
\qquad
\text{Gain} = H(\text{parent}) - \sum_{\text{child}} \frac{N_{\text{child}}}{N_{\text{parent}}} H(\text{child}).
$$

**Regression — MSE reduction:**

$$
\Delta\text{MSE} = \text{MSE}(\text{parent}) - \sum_{\text{child}} \frac{N_{\text{child}}}{N_{\text{parent}}} \text{MSE}(\text{child}).
$$

### 2.3 Trees are greedy and high-variance

Trees are built greedily — optimize one split at a time, no lookahead. This is $O(N \log N)$ per level but does not guarantee global optimality. Critically, trees are **high-variance**: a small change in the training data can produce a structurally different tree. This is why ensembles dominate.

### 2.4 Regularisation and pruning

- **Pre-pruning:** max depth, min samples per leaf, min impurity decrease.
- **Cost-complexity pruning:** $R_\alpha(T) = R(T) + \alpha |T|$. Larger $\alpha$ → fewer leaves. Choose $\alpha$ by cross-validation.

### 2.5 Random Forest

Train $M$ trees on bootstrap samples. At each split, consider only $\sqrt{d}$ (classification) or $d/3$ (regression) randomly chosen features. Final prediction: majority vote or average. This **decorrelates** the trees — without the feature subsampling, all trees would be dominated by the same few strong features, and averaging would not reduce variance.

### 2.6 Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)

Train trees sequentially. Each new tree fits the **negative gradient** (pseudo-residual) of the current ensemble's loss:

$$
r_i^{(t)} = -\left[\frac{\partial \ell(y_i, \hat{y}_i)}{\partial \hat{y}_i}\right]_{\hat{y}=\hat{y}^{(t-1)}}.
$$

The new tree $f_t$ is fitted to these residuals, and added with a shrinkage factor $\eta$ (learning rate, typically 0.01–0.1):

$$
\hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta \cdot f_t(x).
$$

**Why boosting wins on tabular data:** it sequentially focuses on hard examples; it handles heterogeneous features without extensive preprocessing; it learns complex interactions greedily. The cost: more prone to overfitting than random forests; needs careful tuning of $\eta$, tree depth, and number of trees.

### 2.7 Tree traps

| Trap | Response |
|---|---|
| Extrapolation failure | Trees predict constants within training-data ranges. They cannot extrapolate trends. |
| Gini vs entropy | They rarely disagree. The choice matters far less than the regularisation. |
| Overfitting | An unconstrained tree memorises noise. Use depth limits, min-samples, or pruning. |
| Feature importance bias | Split-based importance favours high-cardinality features. Use permutation importance instead. |

---

## 3. Objective functions

### 3.1 What a loss function does

Training means solving $\theta^* = \arg\min_\theta \mathcal{L}(\theta)$. The choice of $\mathcal{L}$ encodes what "wrong" costs.

| Problem | Loss | Formula |
|---|---|---|
| Regression (MSE) | Mean squared error | $\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2$ |
| Regression (MAE) | Mean absolute error | $\frac{1}{N}\sum_i |y_i - \hat{y}_i|$ |
| Binary classification | Binary cross-entropy | $-\frac{1}{N}\sum_i [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ |
| Multi-class | Categorical cross-entropy | $-\frac{1}{N}\sum_i \sum_k y_{ik} \log \hat{y}_{ik}$ |

### 3.2 Why cross-entropy for classification

Cross-entropy + softmax/sigmoid is the standard pairing because:

1. **Gradient cancellation.** The sigmoid derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ appears in the chain rule. Cross-entropy's derivative $-\frac{y}{\sigma(z)} + \frac{1-y}{1-\sigma(z)}$ cancels it, producing the clean gradient:

$$
\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y.
$$

With MSE + sigmoid, the gradient contains a $\hat{y}(1-\hat{y})$ factor that vanishes when the prediction is confident-but-wrong — learning stalls.

2. **Proper scoring rule.** The minimiser of cross-entropy is the true conditional probability $\Pr(Y \mid X)$. Cross-entropy is the only proper scoring rule (up to affine transformations) that decomposes additively across examples.

3. **Statistical interpretation.** Minimising cross-entropy = maximising likelihood under a Bernoulli/categorical model. The Hessian of cross-entropy is positive semi-definite for logistic regression (the problem is convex).

### 3.3 The gradient vector

$$
\nabla_\theta \mathcal{L} = \begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial \theta_1} &
\frac{\partial \mathcal{L}}{\partial \theta_2} &
\cdots &
\frac{\partial \mathcal{L}}{\partial \theta_d}
\end{bmatrix}^T.
$$

It points in the direction of steepest **ascent**. The negative gradient $-\nabla_\theta \mathcal{L}$ is the direction of steepest descent. The gradient is the compass; the learning rate is the step size.

### 3.4 Gradients for common models

**Linear regression (MSE):** $f_\theta(x) = \theta^T x$

$$
\nabla_\theta \mathcal{L} = \frac{2}{N}\sum_{i=1}^{N} (\theta^T x_i - y_i)\,x_i.
$$

**Logistic regression (binary cross-entropy):** $\hat{y}_i = \sigma(\theta^T x_i)$

$$
\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)\,x_i.
$$

Identical form — this is by design (canonical link in GLMs).

### 3.5 Loss function traps

| Trap | Symptom | Fix |
|---|---|---|
| Wrong loss for the task | MSE on classification: poor calibration, slow learning. | Match loss to output distribution. |
| Non-convex landscape | Gradient descent trapped in local minima. | Multiple restarts, momentum, SGD noise. |
| Ill-conditioned Hessian | Different parameter directions have vastly different curvature. | Normalise features; use Adam. |
| Saddle points | Gradient is zero but it is not an extremum — ubiquitous in high dimensions. | Momentum, adaptive methods. |

---

## 4. Gradient descent and its variants

### 4.1 Batch gradient descent

Gradient on the **entire training set** each step:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \ell(f_\theta(x_i), y_i).
$$

Exact gradient, $O(N)$ per step. Deterministic — converges to a sharp (potentially poor-generalising) minimum.

### 4.2 Stochastic gradient descent (SGD)

One random example per step:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \ell(f_\theta(x_{i_t}), y_{i_t}).
$$

$O(1)$ per step. Unbiased but high-variance gradient estimate. The noise helps escape saddle points and biases toward **flatter minima** that generalise better. Must decay $\eta$ for convergence: $\eta_t = \eta_0 / (1 + \lambda t)$ or similar.

### 4.3 Mini-batch SGD

The practical default. Batch of $B$ examples:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B}\sum_{i \in \mathcal{B}_t} \nabla_\theta \ell(f_\theta(x_i), y_i).
$$

Interpolates between $B=1$ (SGD) and $B=N$ (batch). Variance scales as $\sim 1/B$; GPU throughput improves with larger $B$. Typical sizes: 32, 64, 128, 256.

### 4.4 Momentum

Maintain a velocity vector that smooths gradients:

$$
\begin{aligned}
v_{t+1} &= \beta v_t + \eta \,\nabla_\theta \mathcal{L}(\theta_t), \\
\theta_{t+1} &= \theta_t - v_{t+1}.
\end{aligned}
$$

$\beta = 0.9$ is standard. Think of a heavy ball rolling downhill — consistent gradient directions accumulate speed; oscillating directions cancel out. Momentum accelerates convergence in narrow valleys and helps escape shallow local minima.

**Nesterov accelerated gradient (NAG):**

$$
v_{t+1} = \beta v_t + \eta \,\nabla_\theta \mathcal{L}(\theta_t - \beta v_t).
$$

Evaluate the gradient *after* the momentum step ("look-ahead"). Often converges faster theoretically and empirically.

### 4.5 Adam

Combines momentum (first moment) with per-parameter adaptive learning rates (second moment):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\,g_t \\
v_t &= \beta_2 v_{t-1}  + (1 - \beta_2)\,g_t^2 \\
\hat{m}_t &= m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t) \\
\theta_{t+1} &= \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
\end{aligned}
$$

Defaults: $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

The bias correction ($\hat{m}_t$, $\hat{v}_t$) accounts for the fact that $m_0 = v_0 = 0$ — without it, the first few steps would be severely underestimated.

**AdamW** decouples weight decay from the adaptive learning rate, which improves generalisation. Use AdamW over vanilla Adam for most deep learning.

**When SGD beats Adam.** Well-tuned SGD + momentum sometimes finds better-generalising solutions than Adam on vision tasks. The adaptivity that makes Adam fast can steer it toward sharp minima. For new projects, start with AdamW; if generalisation matters more than training speed, try SGD + momentum with a cosine schedule.

### 4.6 Learning rate schedules

Constant $\eta$ is rarely optimal:

| Schedule | Formula | When |
|---|---|---|
| Step decay | $\eta \leftarrow \eta \times 0.1$ every $T$ epochs | Simple baseline. |
| Cosine annealing | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t\pi}{T}))$ | Modern default; smooth decay. |
| Linear warmup | $\eta_t = \eta_{\max} \cdot t / T_{\text{warm}}$ for $t < T_{\text{warm}}$ | Transformers; prevents early instability. |
| ReduceLROnPlateau | Halve $\eta$ when val loss plateaus | When you do not know the schedule ahead of time. |

---

## 5. Epochs, batches, and the training loop

### 5.1 Vocabulary

| Term | Meaning |
|---|---|
| **Iteration** | One parameter update (forward + backward on one mini-batch). |
| **Batch / mini-batch** | The $B$ examples used in one iteration. |
| **Epoch** | One full pass through the training set. $\lceil N/B \rceil$ iterations. |
| **Step** | Synonym for iteration in most frameworks. |

### 5.2 Batch size trade-offs

| Small $B$ (32) | Large $B$ (8192) |
|---|---|
| More gradient noise → implicit regularisation | Less noise → stabler, more "exact" gradients |
| Poor GPU utilisation | Excellent GPU utilisation (high throughput) |
| Favours flatter minima | Risk of sharper minima, worse generalisation |
| Lower memory per step | Higher memory; may need gradient accumulation |

**Linear scaling rule (Goyal et al.):** When you multiply $B$ by $k$, multiply $\eta$ by $k$. This holds when gradients are approximately constant across the batch. Breaks down at very large $B$; use **LARS** (layer-wise adaptive rate scaling) as a fix.

**Gradient accumulation:** if $B$ does not fit in GPU memory, run multiple forward/backward passes with smaller micro-batches, accumulating gradients, and call `optimizer.step()` only after processing the equivalent of the target $B$.

---

## 6. Neural network foundations

### 6.1 The artificial neuron

$$
a = g(w^T x + b) = g\!\left(\sum_j w_j x_j + b\right).
$$

Without the nonlinearity $g$, stacking layers just produces an affine transformation: depth adds no expressivity.

### 6.2 Activation functions

| Name | $g(z)$ | $g'(z)$ | Notes |
|---|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $g(z)(1-g(z))$ | Saturates for $|z|>5$. Use in output layer for binary classification only. |
| Tanh | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | $1-g(z)^2$ | Zero-centred. Still saturates. |
| ReLU | $\max(0,z)$ | $\mathbf{1}[z>0]$ | Default for hidden layers. Non-saturating for $z>0$. |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha=0.01$ | $\mathbf{1}[z>0] + \alpha\mathbf{1}[z \le 0]$ | Prevents dying ReLU. |
| GELU | $z \cdot \Phi(z)$ | $\Phi(z) + z\varphi(z)$ | Smooth ReLU approximation. Default in Transformers. |
| Swish/SiLU | $z \cdot \sigma(z)$ | $\sigma(z) + z\sigma(z)(1-\sigma(z))$ | Used in EfficientNet and some LLMs. |
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | $\hat{y}_k(\delta_{kj} - \hat{y}_j)$ | Output layer for multi-class. |

### 6.3 The universal approximation theorem

A feedforward network with one hidden layer and any non-polynomial activation can approximate any continuous function on a compact set arbitrarily well — **given enough hidden units**. The theorem:

- Guarantees **existence** of weights, not that gradient descent can find them.
- Places no bound on the width needed (could be exponential in the input dimension).
- Is why **depth** matters: deep networks can represent certain functions with exponentially fewer units than shallow ones. Depth enables hierarchical feature learning — edges → textures → parts → objects.

### 6.4 Fully connected (linear) layer

For an input $x \in \mathbb{R}^{d_{\text{in}}}$ and output $y \in \mathbb{R}^{d_{\text{out}}}$:

$$
y = W x + b, \quad W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}},\; b \in \mathbb{R}^{d_{\text{out}}}.
$$

Parameter count: $d_{\text{in}} \cdot d_{\text{out}} + d_{\text{out}}$. This grows quadratically — why dense layers are expensive and why convolutions and attention are prefered for high-dimensional inputs.

---

## 7. Advanced neural network techniques

### 7.1 Weight initialisation

Starting weights matter enormously. Too large → exploding activations/gradients. Too small → vanishing.

**Xavier/Glorot (tanh, sigmoid):**

$$
w \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad\text{or}\quad \mathcal{U}\!\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right].
$$

**He/Kaiming (ReLU, Leaky ReLU):**

$$
w \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}}}\right).
$$

The factor of 2 compensates for ReLU zeroing half the activations. Using Xavier with ReLU causes activations to shrink layer by layer; using He with tanh/sigmoid has the opposite problem.

### 7.2 Normalisation layers

**Batch Normalisation.** For each feature $k$ across a mini-batch $\mathcal{B}$:

$$
\hat{x}^{(k)} = \frac{x^{(k)} - \mu_{\mathcal{B}}^{(k)}}{\sqrt{\sigma_{\mathcal{B}}^{2(k)} + \epsilon}},\qquad
y^{(k)} = \gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}.
$$

$\gamma$ and $\beta$ are learned. At test time, use running averages of $\mu$ and $\sigma^2$.

Batch norm enables higher learning rates, reduces sensitivity to initialisation, and acts as a mild regulariser (via mini-batch noise). Its original "internal covariate shift" explanation has been challenged; the modern view is that it smooths the optimisation landscape, making gradients more Lipschitz.

**Layer Normalisation.** Normalises across features for each example individually. Used in Transformers because: (1) batch size varies; (2) sequence length varies; (3) no dependence on batch statistics means clean train/test behaviour.

**RMSNorm.** A simplified LayerNorm (no mean subtraction, no bias). Popular in Llama and recent LLMs — faster, equally effective for Transformers.

### 7.3 Regularisation

| Method | Mechanism | When |
|---|---|---|
| **L2 (weight decay)** | $\mathcal{L}_{\text{total}} = \mathcal{L} + \frac{\lambda}{2}\|w\|^2_2$ | Always; integrated into optimiser (AdamW). |
| **L1** | $\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda\|w\|_1$ | When you want sparse weights. |
| **Dropout** | Randomly zero activations with probability $p$ during training. Test time: scale by $1-p$. | Dense layers: $p=0.5$. Input: $p=0.2$. Transformers: $p=0.1$. |
| **DropConnect** | Drop weights, not activations. | Alternative to dropout; rarer. |
| **Stochastic Depth** | Randomly drop entire layers during training. | Very deep networks (ResNets with hundreds of layers). |
| **Label smoothing** | Replace hard target $y=1$ with $y=1-\epsilon$, distribute $\epsilon/(K-1)$ to other classes. | Prevents overconfident predictions; improves calibration. |
| **Data augmentation** | Generate new training examples: flip, crop, colour jitter, Cutout, Mixup, CutMix. | Essential for vision; increasingly used in NLP (back-translation, EDA). |
| **Early stopping** | Stop when validation loss stops improving. | Always. The simplest and most reliable regulariser. |

### 7.4 Skip connections (ResNets)

Instead of learning a direct mapping $H(x)$, learn the residual:

$$
y = \mathcal{F}(x, \{W_i\}) + x.
$$

If the optimal function is close to the identity, the residual $\mathcal{F}(x)$ is small and easy to optimise. Skip connections:

- Allow gradients to flow directly backward through identity paths — no vanishing gradient.
- Make the optimisation landscape smoother (loss surface is more convex-like).
- Enable training of networks with 100–1000+ layers.
- The idea generalises: Transformers use residual connections around each attention and FFN sublayer.

### 7.5 Double descent

In classical statistics, test error follows a U-shape: it decreases, then increases as model complexity grows (bias–variance tradeoff). In modern deep learning, test error often follows a **double descent**:

1. Test error decreases with more parameters (classical regime).
2. Test error peaks at the interpolation threshold (where the model just fits the training data).
3. Test error decreases **again** as parameters increase further (overparametrised regime).

The implication: bigger models can be *both* better-fitting and better-generalising. This contradicts classical intuition and is an active area of theory. The practical lesson: when in doubt, a larger model with appropriate regularisation often outperforms a carefully right-sized one.

---

## 8. Backpropagation

### 8.1 The idea

Backpropagation is **reverse-mode automatic differentiation** applied to neural networks. It computes $\frac{\partial \mathcal{L}}{\partial \theta}$ for every parameter $\theta$ in a computation graph in $O(\text{#edges})$ time, regardless of how many parameters there are.

This is remarkable: a naive approach would require one forward pass per parameter (tens of millions). Backprop does it in one forward + one backward pass.

### 8.2 The forward and backward passes

**Forward pass** (layer $\ell$): $z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}$, $a^{(\ell)} = g^{(\ell)}(z^{(\ell)})$.

**Backward pass:** define the error signal $\delta^{(\ell)} = \frac{\partial \mathcal{L}}{\partial z^{(\ell)}}$. For the output layer, $\delta^{(L)}$ comes from the loss. For earlier layers:

$$
\delta^{(\ell)} = \big((W^{(\ell+1)})^T \delta^{(\ell+1)}\big) \odot g'^{(\ell)}(z^{(\ell)}).
$$

Then parameter gradients:

$$
\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \delta^{(\ell)} (a^{(\ell-1)})^T,\qquad
\frac{\partial \mathcal{L}}{\partial b^{(\ell)}} = \delta^{(\ell)}.
$$

### 8.3 Worked example: two-layer network

Forward: $z_1 = W_1 x + b_1$, $h = \sigma(z_1)$, $\hat{y} = W_2 h + b_2$, $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$.

Backward:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \hat{y}} &= \hat{y} - y,\quad &
\frac{\partial \mathcal{L}}{\partial W_2} &= (\hat{y} - y) \cdot h^T,\quad &
\frac{\partial \mathcal{L}}{\partial b_2} &= \hat{y} - y, \\[4pt]
\frac{\partial \mathcal{L}}{\partial h} &= W_2^T(\hat{y} - y),\quad &
\frac{\partial \mathcal{L}}{\partial z_1} &= \frac{\partial \mathcal{L}}{\partial h} \odot \sigma'(z_1), \\[4pt]
\frac{\partial \mathcal{L}}{\partial W_1} &= \frac{\partial \mathcal{L}}{\partial z_1} \cdot x^T,\quad &
\frac{\partial \mathcal{L}}{\partial b_1} &= \frac{\partial \mathcal{L}}{\partial z_1}.
\end{aligned}
$$

You must be able to do this by hand for a tiny network. This is a classic interview question.

### 8.4 Autograd in practice

Modern frameworks (PyTorch `autograd`, JAX `grad`, TensorFlow `GradientTape`) build a dynamic computation graph as you execute the forward pass. When `loss.backward()` is called, they traverse the graph in reverse topological order, applying the chain rule at each node.

Key debugging rules:
- Always `optimizer.zero_grad()` before `loss.backward()` — gradients accumulate by default.
- Never modify a tensor that is needed for backward pass in-place — it invalidates the saved values.
- For the fused log-softmax + NLL loss, use `CrossEntropyLoss` (PyTorch) — manual softmax + log is numerically unstable.

### 8.5 Gradient checking

The finite-difference check verifies your backprop:

$$
\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon},
$$

with $\epsilon \approx 10^{-7}$. The central difference is $O(\epsilon^2)$ accurate. Compare analytical and numerical gradients — relative error should be $< 10^{-7}$ for double precision.

Do this with a tiny network on a few data points. It catches: forgetting to zero gradients, incorrect reshaping, transposition errors, and sign errors in custom layers.

### 8.6 Gradient flow traps

| Trap | Symptom | Fix |
|---|---|---|
| Vanishing gradients | Early-layer gradients → 0. Network stops learning. | ReLU/GeLU, He init, batch/layer norm, skip connections. |
| Exploding gradients | Loss → NaN. Gradients → ±∞. | Gradient clipping, proper init, lower $\eta$, normalisation. |
| Dead ReLUs | Large fraction of neurons output only zero. | Lower $\eta$, Leaky ReLU/ELU, better init. |
| Gradient noise dominates | Loss oscillates wildly. | Larger batch, lower $\eta$, gradient clipping. |

---

## 9. Convolutional and recurrent architectures (the precursors)

### 9.1 CNNs in one paragraph

Convolutional layers exploit spatial locality and translation equivariance. A filter $K \in \mathbb{R}^{k \times k \times c_{\text{in}} \times c_{\text{out}}}$ slides over the input, computing:

$$
y_{p,q,o} = \sum_{i=0}^{k-1}\sum_{j=0}^{k-1}\sum_{c=0}^{c_{\text{in}}-1} K_{i,j,c,o} \cdot x_{p+i,\,q+j,\,c} + b_o.
$$

Key ideas: **weight sharing** (the same filter is applied everywhere — drastically fewer parameters than a dense layer), **pooling** (downsample spatially to increase receptive field), and **hierarchical features** (early layers detect edges; deeper layers detect objects).

CNNs dominated vision from AlexNet (2012) through EfficientNet (2019). They are now being displaced by Vision Transformers (ViT), which treat images as sequences of patches.

### 9.2 RNNs and LSTMs

An RNN processes a sequence $x_1, \ldots, x_T$ by maintaining a hidden state $h_t$:

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b).
$$

Problem: gradients vanish or explode when unrolling over long sequences — the repeated multiplication by $W_{hh}$ causes eigenvalues $<1$ to decay and $>1$ to explode.

**LSTM** introduces gates to control information flow:

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \quad &\text{(forget gate)} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \quad &\text{(input gate)} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \quad &\text{(output gate)} \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \quad &\text{(candidate cell)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad &\text{(cell state)} \\
h_t &= o_t \odot \tanh(c_t) \quad &\text{(hidden state)}.
\end{aligned}
$$

The cell state $c_t$ acts as a gradient highway — the forget gate can preserve information over long distances. This mitigates, but does not fully solve, vanishing gradients.

Transformers eventually overtook RNNs/LSTMs because: (1) attention provides direct connections between any two positions — no bottleneck through a fixed-size hidden state; (2) non-sequential computation enables massive parallelism; (3) they scale better to very long contexts.

---

## 10. Transformers

### 10.1 The big idea

Transformers (Vaswani et al., 2017, "Attention Is All You Need") replace recurrence with **self-attention** and **position-wise feedforward networks**. The key insight: instead of processing tokens sequentially like an RNN, let every position attend to every other position directly. This gives:

- $O(1)$ path length between any two tokens (vs $O(T)$ for RNNs).
- Fully parallelisable computation across sequence positions.
- Scalability to huge datasets and models.

### 10.2 Scaled dot-product attention

For a query $Q$, key $K$, and value $V$ (all matrices of shape $T \times d$):

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V.
$$

**Intuition.** Each query vector asks "which keys are relevant to me?" The dot product $QK^T$ measures relevance. The softmax normalises to a probability distribution over positions (the attention weights). These weights then average the value vectors. Output: for each query position, a context-aware representation as a weighted sum of values from all positions.

**Why $\sqrt{d_k}$?** Without it, for large $d_k$, the dot products become large and the softmax saturates (gradients → 0). The scaling keeps the variance of the dot products at 1, keeping the softmax in a reasonable regime.

### 10.3 Multi-head attention

Instead of one attention function with $d_{\text{model}}$-dimensional Q/K/V, project to $h$ lower-dimensional heads ($d_k = d_{\text{model}} / h$), run attention in parallel, and concatenate:

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O, \\
\text{head}_i &= \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V).
\end{aligned}
$$

Different heads learn different relationships: one might attend to syntactic structure, another to semantic similarity, another to positional proximity. The concatenated output is projected back to $d_{\text{model}}$ dimensions.

Typical choices: $d_{\text{model}} = 512$ (base) or $768$ (BERT-base) or $4096$ (GPT-3), with $h = 8$–$64$ heads.

### 10.4 Self-attention vs cross-attention vs masked attention

| Type | $Q$ from | $K,V$ from | Used in |
|---|---|---|---|
| **Self-attention** | This layer's input | Same as $Q$ | Encoder, decoder (each layer attends to itself). |
| **Cross-attention** | Decoder | Encoder output | Decoder attending to encoded source. |
| **Causal (masked) self-attention** | This layer's input | Same as $Q$, but mask future positions | Decoder self-attention; GPT / autoregressive LMs. |

The causal mask is an upper-triangular matrix of $-\infty$ added before softmax, so position $t$ cannot attend to positions $> t$. This preserves the autoregressive property: each token can only depend on previous tokens.

### 10.5 Positional encoding

Self-attention is permutation-equivariant — it has no notion of order. Positional encodings inject sequence order information.

**Sinusoidal (original Transformer):**

$$
\begin{aligned}
\text{PE}(pos, 2i) &= \sin(pos / 10000^{2i/d_{\text{model}}}), \\
\text{PE}(pos, 2i+1) &= \cos(pos / 10000^{2i/d_{\text{model}}}).
\end{aligned}
$$

These are added to the input embeddings. The sinusoid has the property that $\text{PE}(pos+k)$ can be expressed as a linear function of $\text{PE}(pos)$ — the model can learn to attend to relative positions.

**Learned positional embeddings (GPT, BERT):** just learn a lookup table $E_{\text{pos}} \in \mathbb{R}^{T_{\max} \times d}$. Simpler, but cannot extrapolate beyond $T_{\max}$.

**RoPE (Rotary Position Embedding, Llama, GPT-NeoX, most modern LLMs):** rotates the query and key vectors by an angle proportional to their position. After rotation, the dot product $q_m^T k_n$ depends only on the relative position $m - n$. RoPE combines the flexibility of learned embeddings with the relative-position property of sinusoids. It is the current standard.

**ALiBi:** adds a static, non-learned bias to attention scores that penalises distant positions. Simple, and enables extrapolation to longer sequences than seen in training.

### 10.6 The encoder-decoder architecture (original Transformer)

**Encoder:** $N$ identical layers, each = Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm.

**Decoder:** $N$ identical layers, each = Masked Multi-Head Self-Attention → Add & Norm → Cross-Attention (to encoder output) → Add & Norm → FFN → Add & Norm.

The FFN is a two-layer MLP with a ReLU/GeLU activation:

$$
\text{FFN}(x) = W_2 \cdot \text{GeLU}(W_1 x + b_1) + b_2.
$$

Typically $d_{\text{ff}} = 4 \times d_{\text{model}}$. The FFN provides depth and capacity after the attention mixing.

**Pre-LN vs Post-LN:** The original Transformer applied layer norm *after* the residual (Post-LN). Modern practice puts it *before* (Pre-LN), which stabilises training — gradients flow unimpeded through the residual path, and the normalisation is applied to the inputs of each sublayer, not the outputs.

### 10.7 Computational complexity

For a sequence of length $T$ and dimension $d$:

| Component | Complexity |
|---|---|
| Self-attention | $O(T^2 \cdot d)$ |
| FFN | $O(T \cdot d^2)$ |
| Overall (one layer) | $O(T^2 d + T d^2)$ |

The $T^2$ attention term dominates for long sequences. This is the core challenge: standard attention scales quadratically in sequence length. Solutions include sparse attention, linear attention, FlashAttention (IO-aware exact attention), and state-space models (Mamba).

---

## 11. Large Language Models (LLMs)

### 11.1 What is an LLM?

An LLM is an autoregressive language model trained to predict the next token:

$$
\Pr(w_1, \ldots, w_T) = \prod_{t=1}^{T} \Pr(w_t \mid w_1, \ldots, w_{t-1}).
$$

Each conditional $\Pr(w_t \mid w_{<t})$ is modelled by a Transformer decoder with causal masking. The model is trained on vast text corpora (trillions of tokens) using a cross-entropy (next-token prediction) loss:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log \Pr(w_t \mid w_{<t}; \theta).
$$

### 11.2 The GPT architecture

A GPT-style (decoder-only) Transformer stacks $L$ layers, each consisting of:

1. **Causal multi-head self-attention** (with mask to prevent attending to future tokens).
2. **Feedforward network** (typically with GeLU or SwiGLU activation).
3. **Pre-layer normalisation** (RMSNorm in Llama).
4. **Residual connections** around each sublayer.

GPT-3: 175B parameters, $L=96$, $d_{\text{model}}=12288$, $h=96$ heads, trained on ~300B tokens.
GPT-4: estimated ~1.8T parameters (MoE), details undisclosed.
Llama 3 (405B): $L=126$, $d_{\text{model}}=16384$, $h=128$, trained on 15T tokens.

### 11.3 Tokenisation

Text must be converted to discrete tokens. Modern LLMs use **subword tokenisation** (BPE — byte-pair encoding; or SentencePiece):

- Common words become single tokens ("the", "model").
- Rare words are split into subwords ("tokenisation" → "token" + "isation").
- The vocabulary size is typically 32K–256K tokens.

The tokeniser matters enormously: it determines the model's effective context length (more subwords per word = shorter effective context), multilingual support, and code/maths handling.

### 11.4 Scaling laws

Kaplan et al. (2020) and Hoffmann et al. (2022, Chinchilla) found that model performance follows power-law scaling:

$$
\mathcal{L}(N, D) = \left(\frac{A}{N}\right)^{\alpha} + \left(\frac{B}{D}\right)^{\beta} + L_{\infty},
$$

where $N$ = parameter count, $D$ = training tokens, and $L_\infty$ is the irreducible loss.

**Chinchilla optimal:** for a given compute budget, you should scale parameters and training data roughly equally: $N \propto C^{0.5}$, $D \propto C^{0.5}$, yielding ~20 tokens per parameter. Many models before Chinchilla (including GPT-3) were **undertrained** — they had too many parameters for their data budget.

**Emergent abilities:** certain capabilities (reasoning, code generation, instruction following) appear only above certain scale thresholds. They are not present in smaller models and cannot be predicted by extrapolating from small-scale performance.

### 11.5 Training pipeline

1. **Pretraining.** Train a base model on internet-scale text (trillions of tokens). This teaches the model grammar, facts, reasoning patterns, and world knowledge. Training takes thousands of GPUs for weeks to months.

2. **Instruction tuning / supervised fine-tuning (SFT).** Fine-tune on high-quality (prompt, response) pairs. Teaches the model the "chat" format and to follow instructions.

3. **RLHF (Reinforcement Learning from Human Feedback):**
   - Collect human preference comparisons between model outputs.
   - Train a reward model to predict human preferences.
   - Fine-tune the policy (LLM) with PPO to maximise the reward, with a KL penalty to prevent the model from drifting too far from the SFT model:

   $$
   \max_\theta\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y\mid x)}\!\big[r(x, y)\big] - \beta \cdot D_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x) \parallel \pi_{\text{SFT}}(\cdot\mid x)\big).
   $$

4. **DPO (Direct Preference Optimisation):** an alternative to RLHF that directly optimises the policy from preference pairs without training a separate reward model. Mathematically equivalent to RLHF under the Bradley-Terry preference model, but simpler to implement.

### 11.6 Inference and decoding

Generating text from an LLM is autoregressive sampling:

```python
tokens = tokenizer.encode(prompt)
for _ in range(max_new_tokens):
    logits = model(tokens)
    next_token = sample(logits[-1] / temperature)  # or greedy: argmax
    tokens.append(next_token)
    if next_token == eos: break
return tokenizer.decode(tokens)
```

**Decoding strategies:**

| Strategy | What it does | When |
|---|---|---|
| Greedy | Take argmax at each step. | Deterministic; tends to be repetitive. |
| Temperature sampling | Divide logits by $T$ before softmax. $T<1$: sharper; $T>1$: flatter. | $T=0.7$–$0.9$ is standard for creative generation. |
| Top-k sampling | Sample only from the $k$ most probable tokens. | $k=50$ is common. |
| Top-p (nucleus) sampling | Sample from the smallest set of tokens whose cumulative probability ≥ $p$. | $p=0.9$–$0.95$ dynamically adapts. |
| Beam search | Maintain $B$ candidate sequences, expand all, keep top $B$. | Machine translation; less useful for open-ended generation. |

### 11.7 KV caching

During autoregressive generation, each new token attends to *all* previous tokens. Without caching, you recompute the keys and values for all previous positions at every step — $O(T^2)$ per token generated.

**KV cache:** store the key and value tensors for each layer from previous steps. Each new token only computes attention against the cached K/V and appends its own K/V. This reduces the per-step cost from $O(T^2 d)$ to $O(T d)$.

The KV cache size is $2 \cdot L \cdot T \cdot d_{\text{model}}$ elements. For large models and long contexts, this dominates inference memory. Techniques like **Grouped Query Attention (GQA)** and **Multi-Query Attention (MQA)** reduce the KV cache by sharing K/V heads across multiple query heads.

### 11.8 Prompting and in-context learning

LLMs exhibit **in-context learning**: they can perform tasks from a few examples provided in the prompt, without any weight updates. This emerges at scale and is not present in small models.

Key prompting techniques:
- **Few-shot:** provide 2–10 examples of (input → output) in the prompt.
- **Chain-of-thought (CoT):** prepend "Let's think step by step." Forces the model to verbalise intermediate reasoning, dramatically improving multi-step reasoning accuracy.
- **Self-consistency:** sample multiple CoT paths and take the majority answer.
- **ReAct / tool use:** interleave reasoning traces with calls to external tools (search, calculator, code executor).

### 11.9 The Transformer / LLM traps

| Trap | What happens | Response |
|---|---|---|
| Quadratic attention cost | $O(T^2)$ memory/time — 8K context uses 64× more attention than 1K. | FlashAttention, GQA, sliding windows. |
| Position extrapolation | Performance degrades beyond training context length. | RoPE with scaling, ALiBi, position interpolation. |
| Hallucination | Model generates plausible but factually incorrect text. | Retrieval-augmented generation (RAG), factuality tuning. |
| Prompt injection | User input overrides system instructions. | Input/output separation, careful prompt engineering. |
| KV cache explosion | Memory from KV cache exceeds model weights for long sequences. | GQA/MQA, quantised KV cache, paged attention (vLLM). |
| Training–serving skew | Differences in tokenisation or prompting between training and deployment. | Version and test the full pipeline end-to-end. |
| Evaluation contamination | Benchmark data leaked into training corpus. | Use held-out benchmarks, n-gram overlap detection. |

---

## 12. The complete training loop

```
# Initialisation
model = Transformer(vocab_size, d_model, n_heads, n_layers)
model.apply(weight_init)       # He/Kaiming for non-attention, special init for embeddings
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

for epoch in range(num_epochs):
    # Training
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward
        logits = model(X_batch)                    # autoregressive or parallel
        loss   = F.cross_entropy(logits, y_batch)  # most common LLM loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (crucial for Transformers!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
    
    # Checkpointing
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), f'checkpoint_epoch{epoch}.pt')
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break  # early stopping
```

This loop, plus the architecture and the data, is the heart of modern deep learning. Everything else is a variation.

---

## 13. Interview practice

### 13.1 The response pattern

1. **Frame the problem** — supervised/unsupervised, classification/regression, data modality.
2. **Choose architecture and loss** — justify the pairing.
3. **Describe optimisation** — optimiser, batch size, LR schedule, epochs.
4. **Name failure modes** — overfitting, vanishing gradients, distribution shift, data leakage.
5. **Describe validation** — hold-out test set; cross-validation; calibration; ablation studies.

### 13.2 Gradient of cross-entropy with sigmoid

**Question.** Derive $\frac{\partial \mathcal{L}}{\partial w}$ for binary cross-entropy with a sigmoid output.

**Answer.** $\hat{y} = \sigma(w^T x + b)$, $\mathcal{L} = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$.

$$
\frac{\partial \mathcal{L}}{\partial w} = (\hat{y} - y)\, x.
$$

The sigmoid derivative cancels with the log derivative. This is why sigmoid + cross-entropy works and sigmoid + MSE does not.

### 13.3 Explain gradient descent non-technically

**Answer.** "You are on a mountain in fog, trying to reach the valley floor. You cannot see the whole landscape, but you can feel which way the ground slopes beneath your feet. You take small steps downhill, re-checking the slope each time. The step size matters: too large, you jump over the valley; too small, you inch along forever. Gradient descent is a systematic way to do this in high dimensions, with rules for choosing the right step size."

### 13.4 Backprop through a tiny network

**Question.** Given a 2-layer network (input → hidden with sigmoid → output with identity → MSE loss), compute all gradients by hand.

**Answer.** This is a test of whether you understand the chain rule in practice. Do the full forward pass, then the backward pass step by step. Show that $\frac{\partial \mathcal{L}}{\partial W_1}$ involves $\sigma'(z_1)$ and the weight matrix $W_2$. The interviewer is checking whether you actually understand backprop, not whether you can recite it.

### 13.5 Transformer self-attention

**Question.** Write the formula for scaled dot-product attention and explain each component.

**Answer.**

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V.
$$

- $Q$: queries — "what am I looking for?"
- $K$: keys — "what do I contain?"
- $V$: values — "what information should I pass on?"
- $QK^T$: dot product ~ relevance score between each query and each key.
- $\sqrt{d_k}$: scaling to prevent saturated softmax.
- softmax: normalises scores to a probability distribution.
- Multiply by $V$: weighted average of values; each query position gets a context-aware representation.

Explain multi-head attention, the causal mask, and positional encoding if the interviewer digs deeper.

### 13.6 Design: "Build a chatbot from scratch"

**Answer.** "Use a decoder-only Transformer (GPT architecture). Tokenise text with BPE. Pretrain on a large text corpus with next-token prediction loss (cross-entropy). Fine-tune on instruction-following data (SFT). Apply RLHF or DPO for alignment with human preferences. At inference, use KV caching for efficiency; apply top-p sampling with temperature ~0.8. For production, quantise to 4-bit or 8-bit; serve with vLLM or TensorRT-LLM for high throughput. Ground factual responses with retrieval-augmented generation (RAG)."

### 13.7 Decision tree from scratch

**Question.** Implement `best_split(X, y)` for a classification tree.

**Answer.** Iterate over features and thresholds; compute Gini gain; return the feature and threshold with maximum gain. $O(N \cdot d)$ for sorted features per split. Discuss: greedy = no global optimality guarantee; this is the CART algorithm.

### 13.8 "Training loss is zero — good model?"

**Answer.** No. You have memorised the training set. Validation loss is almost certainly high. You need: more data, regularisation, simpler model, data augmentation. A model with more parameters than training points can achieve exactly zero training loss on any dataset; this is *overfitting by construction*.

### 13.9 "BatchNorm vs LayerNorm — when to use which?"

**Answer.** BatchNorm normalises across the batch dimension — effective for CNNs with large, stable batch sizes. LayerNorm normalises across the feature dimension — independent of batch size, so it is used in Transformers and RNNs where batch sizes vary or are small. BatchNorm behaves differently at train vs test time (running statistics); LayerNorm is identical in both modes.

### 13.10 "Why do Transformers scale better than RNNs?"

**Answer.** (1) Attention provides $O(1)$ path length between any two tokens vs $O(T)$ for RNNs — gradients flow directly. (2) Non-sequential computation enables full parallelism during training. (3) The fixed computation pattern is highly optimisable on GPUs/TPUs. (4) The architecture has proven remarkably amenable to scaling — just add more layers, wider dimensions, and more data. The cost is $O(T^2)$ memory/FLOPs, which is addressed by FlashAttention, sparse attention, and efficient implementations.

---

## 14. Cheat sheet

| Concept | Remember |
|---|---|
| Bias–variance | $\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}$. More capacity = lower bias, higher variance. |
| Double descent | Test error can *decrease* after interpolation threshold — bigger ≠ worse. |
| Decision tree | Greedy axis-aligned splits; max impurity reduction. High variance. |
| Random forest | Bootstrap + random feature subsets → decorrelated trees, lower variance. |
| Boosting | Sequential trees fit to residual; XGBoost/LightGBM dominate tabular data. |
| Cross-entropy | $-\sum y \log \hat{y}$. Pair with softmax/sigmoid for gradient cancellation. |
| Gradient descent | $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$ |
| SGD | One/mini-batch example: unbiased, noisy, generalises better via flat minima. |
| Momentum | $v \leftarrow \beta v + \eta \nabla\mathcal{L}$; smooths, accelerates. $\beta=0.9$. |
| Adam / AdamW | Adaptive per-parameter LR from first + second moments. Default optimiser. |
| BatchNorm | $\hat{x} = (x - \mu)/(\sigma + \epsilon)$; $\gamma$, $\beta$ learned. Smooths landscape. |
| LayerNorm | Normalises per-example across features. Standard in Transformers. |
| ReLU | $\max(0, z)$. Default activation. Watch for dead neurons. |
| Backprop | Chain rule on computation graph, backward from loss. $O(\text{#edges})$. |
| Residual connection | $y = F(x) + x$. Gradient highway. Enables very deep networks. |
| Self-attention | $\text{softmax}(QK^T/\sqrt{d_k})V$. Direct pairwise interactions; $O(T^2 d)$. |
| Multi-head attention | $h$ parallel attention heads, different projections → diverse patterns. |
| Positional encoding | Adds order information. Sinusoidal, learned, or RoPE (current standard). |
| Causal mask | Prevents attending to future tokens in autoregressive models. |
| KV cache | Store past keys/values to avoid recomputation during autoregressive generation. |
| LLM | Autoregressive Transformer trained on next-token prediction at scale. |
| RLHF | Align LLM to human preferences via reward model + PPO (or DPO directly). |
| Scaling law | $\mathcal{L}(N, D) \propto N^{-\alpha} + D^{-\beta} + L_\infty$. Chinchilla: ~20 tokens/param. |
| In-context learning | LLM performs tasks from prompt examples; no weight updates. Emerges with scale. |
| Chain-of-thought | "Let's think step by step" — verbalised reasoning improves accuracy. |
| Hallucination | Plausible-sounding but factually wrong output. Mitigate with RAG. |
| FlashAttention | IO-aware exact attention; makes $O(T^2)$ practical via tiling + recomputation. |

---

## 15. Final checklist

Before training or deploying a model, verify:

- **Data.** Is the train/val/test split chronological (time series) or stratified? Any leakage? Are features available at prediction time?
- **Loss.** Does the loss match the task and output distribution? Are you using the fused `CrossEntropyLoss`, not manual softmax + log?
- **Architecture.** Appropriate for the data modality (CNN/ViT for images, Transformer for sequences, GBDT for tabular)? Skip connections present for depth > 10?
- **Initialisation.** Activation-appropriate init (He for ReLU, Xavier for tanh)? Biases initialised to zero (or small positive for forget gates)?
- **Optimisation.** `optimizer.zero_grad()` before `backward()`? Gradient clipping for Transformers? LR warmup if needed?
- **Validation.** Held-out test set used exactly once? Calibration checked? Compared against a simple baseline?
- **Deployment.** KV cache working for autoregressive generation? Model quantised? Monitoring for drift in production?

If you can explain these choices, derive cross-entropy and backprop gradients by hand, build a Transformer from the attention formula, and diagnose when training diverges or plateaus, you have the practical ML/DL fluency that the most demanding interviews and projects require.
