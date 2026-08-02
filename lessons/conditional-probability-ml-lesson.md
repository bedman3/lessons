# Conditional Probability & Machine Learning Foundations

> A practical tour through the mathematical backbone of ML — from Bayes to backprop, with the traps and interview questions that matter.

## 1. Why these topics together

Machine learning is, at its core, a set of techniques for learning conditional relationships from data. Whether you are classifying with a tree, optimising with gradient descent, or training a neural network, you are estimating something like:

$$
\Pr(Y \mid X) \quad\text{or}\quad \mathbb{E}[Y \mid X] \quad\text{or}\quad \arg\min_\theta\; \mathcal{L}(\theta \mid \mathcal{D}).
$$

Conditional probability gives you the language to describe what the model is trying to capture. The optimisation machinery — gradient descent, backpropagation, objective functions — gives you the tools to actually fit the model to data.

This lesson connects the two: we start with the probability foundations, build through classical ML models and their optimisation, and end with neural networks and backpropagation. Each section names the traps that practitioners hit and the questions that interviewers ask.

---

## 2. Conditional probability: the language of learning

### 2.1 Definitions

The conditional probability of event $A$ given $B$ is:

$$
\Pr(A \mid B) = \frac{\Pr(A \cap B)}{\Pr(B)}, \quad \Pr(B) > 0.
$$

Equivalently, the **chain rule** (product rule) factorises a joint distribution:

$$
\Pr(A, B) = \Pr(A \mid B)\,\Pr(B) = \Pr(B \mid A)\,\Pr(A).
$$

For continuous random variables, replace probability with density:

$$
f_{X\mid Y}(x \mid y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}.
$$

### 2.2 Bayes' theorem

This is the workhorse. It inverts a conditional:

$$
\Pr(A \mid B) = \frac{\Pr(B \mid A)\,\Pr(A)}{\Pr(B)}
               = \frac{\Pr(B \mid A)\,\Pr(A)}{\sum_i \Pr(B \mid A_i)\,\Pr(A_i)}.
$$

In ML terms, with parameters $\theta$ and data $\mathcal{D}$:

$$
\underbrace{\Pr(\theta \mid \mathcal{D})}_{\text{posterior}}
= \frac{\overbrace{\Pr(\mathcal{D} \mid \theta)}^{\text{likelihood}}\;
        \overbrace{\Pr(\theta)}^{\text{prior}}}
       {\underbrace{\Pr(\mathcal{D})}_{\text{evidence}}}.
$$

**Key insight.** Maximum likelihood estimation (MLE) drops the prior and evidence and maximises $\Pr(\mathcal{D} \mid \theta)$. Maximum a posteriori (MAP) maximises the numerator $\Pr(\mathcal{D} \mid \theta)\,\Pr(\theta)$. Full Bayesian inference keeps the denominator and works with the whole posterior.

### 2.3 Independence and conditional independence

- **Independence:** $\Pr(A, B) = \Pr(A)\,\Pr(B)$, equivalently $\Pr(A \mid B) = \Pr(A)$.
- **Conditional independence:** $\Pr(A, B \mid C) = \Pr(A \mid C)\,\Pr(B \mid C)$.

Conditional independence is the secret engine behind naïve Bayes, graphical models, and the factorisation assumptions that make high-dimensional learning tractable. It says: *once you know $C$, knowing $B$ tells you nothing extra about $A$.*

### 2.4 The ML connection

Most supervised learning can be framed as estimating a conditional:

| Task | What you are modelling |
|---|---|
| Binary classification | $\Pr(Y = 1 \mid X = x)$ |
| Multi-class classification | $\Pr(Y = k \mid X = x)$ for $k = 1,\ldots,K$ |
| Regression | $\mathbb{E}[Y \mid X = x]$ or the full conditional $f_{Y\mid X}(y \mid x)$ |
| Density estimation | $f_X(x)$ (unconditional) or $f_{X\mid Z}(x \mid z)$ |

### 2.5 Common traps

| Trap | What it looks like | How to catch it |
|---|---|---|
| Confusing $\Pr(A\mid B)$ with $\Pr(B\mid A)$ | "Most defaulted loans had low income, so low-income borrowers always default" — the prosecutor's fallacy. | Always check: is this the right direction of conditioning? Compute both if needed. |
| Ignoring the base rate | A test with 95% accuracy is applied to a disease with 0.1% prevalence — most positives are false. | Always apply Bayes with the prior. Draw the contingency table. |
| Assuming independence | "Features are uncorrelated so the naïve Bayes assumption holds" — uncorrelated $\neq$ independent. | Check conditional independence, not marginal correlation. |
| Forgetting the denominator | Comparing posteriors across models without the evidence term. | For model *selection*, use the ratio: $\frac{\Pr(\mathcal{D}\mid M_1)}{\Pr(\mathcal{D}\mid M_2)}$ times the prior ratio. |

---

## 3. Decision trees: the geometry of axis-aligned splits

### 3.1 What a tree does

A decision tree recursively partitions the feature space into axis-aligned hyper-rectangles, then assigns a constant prediction to each region. For classification, that constant is typically the majority class; for regression, the mean of the training points in that leaf.

Given a region $R_m$ with $N_m$ training points, the prediction is:

$$
\hat{y}_m = \begin{cases}
\arg\max_k \sum_{x_i \in R_m} \mathbf{1}[y_i = k] & \text{(classification)} \\[4pt]
\frac{1}{N_m}\sum_{x_i \in R_m} y_i & \text{(regression)}
\end{cases}
$$

### 3.2 Splitting criteria

At each node, the tree chooses a feature $j$ and threshold $s$ to split the data into two child nodes. Different criteria measure the "goodness" of a split.

**Classification — Gini impurity:**

$$
G = \sum_{k=1}^{K} \hat{p}_k (1 - \hat{p}_k) = 1 - \sum_{k=1}^{K} \hat{p}_k^2,
$$

where $\hat{p}_k$ is the proportion of class $k$ in the node. Gini ranges from 0 (pure) to $1 - 1/K$ (uniform). The split that maximises the *reduction* in impurity is chosen.

**Classification — Entropy (information gain):**

$$
H = -\sum_{k=1}^{K} \hat{p}_k \log_2 \hat{p}_k.
$$

The information gain from a split is $H(\text{parent}) - \sum_{\text{child}} \frac{N_{\text{child}}}{N_{\text{parent}}} H(\text{child})$. Entropy penalises impurity more heavily than Gini; in practice they rarely lead to very different trees.

**Regression — Mean squared error:**

$$
\text{MSE} = \frac{1}{N_m}\sum_{x_i \in R_m} (y_i - \hat{y}_m)^2.
$$

Again, pick the split that gives the largest reduction in MSE.

### 3.3 The greedy nature and its consequences

Trees are built greedily: at each node, pick the best split without looking ahead. This makes them fast to train ($O(N \log N)$ per node with sorting) but means:

- The globally optimal tree is not guaranteed.
- An early mediocre split can cascade into a suboptimal structure.
- Trees are high-variance: small data changes produce different trees.

### 3.4 Regularisation and pruning

An unconstrained tree will memorise the training data. Control complexity with:

- **Pre-pruning (early stopping):** maximum depth, minimum samples per leaf, minimum impurity decrease.
- **Post-pruning (cost-complexity pruning):** grow a large tree, then prune back. Add a penalty $\alpha \cdot |T|$ (number of leaves) to the loss:

$$
R_\alpha(T) = R(T) + \alpha |T|.
$$

Larger $\alpha$ → smaller tree. Choose $\alpha$ by cross-validation.

### 3.5 Ensemble methods in one paragraph

A single tree is weak. Ensembles dominate:

- **Bagging (Random Forest):** train many trees on bootstrap samples; at each split, consider only a random subset of features. Reduces variance without increasing bias. The forest prediction is the majority vote (classification) or average (regression).
- **Boosting (Gradient Boosted Trees / XGBoost / LightGBM):** train trees sequentially, each one fitting the residual (negative gradient) of the current ensemble. Reduces bias. More prone to overfitting; needs careful regularisation.

### 3.6 Tree interview traps

| Question | Good answer direction |
|---|---|
| "Why not just use linear regression?" | Trees capture non-linearities and interactions without manual feature engineering. But they extrapolate poorly beyond the training range. |
| "Gini vs entropy — which is better?" | They usually agree. Entropy is slightly more sensitive to class imbalance. In practice, the choice matters far less than the regularisation. |
| "Why does a tree overfit?" | It can keep splitting until every leaf has one point. Without a complexity penalty, it achieves zero training error and learns noise. |
| "How do trees handle missing values?" | By using surrogate splits (find a correlated feature to approximate the split) or by sending the observation down both branches with weights. Different implementations differ — know yours. |

---

## 4. Objective functions and their derivatives

### 4.1 What an objective function is

An objective (loss, cost) function $\mathcal{L}(\theta)$ measures how badly your model with parameters $\theta$ fits the training data. Training means finding $\theta^*$ that makes $\mathcal{L}$ as small as possible:

$$
\theta^* = \arg\min_\theta \mathcal{L}(\theta).
$$

Common losses:

| Problem | Loss | Formula |
|---|---|---|
| Regression (MSE) | Mean squared error | $\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} (y_i - f_\theta(x_i))^2$ |
| Regression (MAE) | Mean absolute error | $\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} |y_i - f_\theta(x_i)|$ |
| Binary classification | Binary cross-entropy (log loss) | $\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$ |
| Multi-class | Categorical cross-entropy | $\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik} \log \hat{y}_{ik}$ |

### 4.2 Why cross-entropy?

For classification, why not use MSE? Three reasons:

1. **Gradient saturation.** With a sigmoid output, the MSE gradient contains $\hat{y}(1-\hat{y})$ factors — when the prediction is confidently wrong, the gradient vanishes and learning stalls. Cross-entropy cancels the sigmoid derivative, giving clean gradients proportional to $(\hat{y} - y)$.
2. **Probabilistic interpretation.** Minimising cross-entropy is equivalent to maximising the likelihood under a Bernoulli / categorical model. The loss is *proper*: its minimiser is the true conditional probability.
3. **Calibration.** Cross-entropy encourages calibrated probabilities; MSE does not.

### 4.3 Derivatives and gradients

The **gradient** $\nabla_\theta \mathcal{L}$ is the vector of partial derivatives. It points in the direction of steepest *ascent*. To minimise, we move opposite the gradient:

$$
\theta \leftarrow \theta - \eta \,\nabla_\theta \mathcal{L}(\theta),
$$

where $\eta$ is the learning rate. The gradient is the compass; the learning rate is the step size.

**Example: MSE for linear regression.** With $f_\theta(x) = \theta^T x$ (no intercept for simplicity):

$$
\mathcal{L}(\theta) = \frac{1}{2N}\sum_{i=1}^{N} (\theta^T x_i - y_i)^2,
\qquad
\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} (\theta^T x_i - y_i)\,x_i.
$$

The factor $\frac{1}{2}$ is a convenience — it cancels the 2 from the derivative of the square.

**Example: Binary cross-entropy with logistic model.** For $\hat{y}_i = \sigma(\theta^T x_i)$ where $\sigma(z) = \frac{1}{1+e^{-z}}$:

$$
\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)\,x_i.
$$

The gradient has the same simple form as linear regression — this is not an accident; it is a property of matching the loss to the output nonlinearity (the "canonical link" in GLM terms).

### 4.4 The chain rule is everything

When your model is a composition of functions — and every neural network is — you need the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial f}\,
                                            \frac{\partial f}{\partial g}\,
                                            \frac{\partial g}{\partial \theta}.
$$

If this seems trivial, you are underestimating it. Backpropagation is *nothing but* the chain rule applied to a computation graph, with a clever bookkeeping trick to avoid redundant computation. We will return to this in Section 7.

### 4.5 Objective function traps

| Trap | Symptom | Fix |
|---|---|---|
| Wrong loss for the task | MSE on classification: slow learning, poor calibration. | Match the loss to the output distribution. |
| Non-convex landscape | Gradient descent finds a local minimum, not the global one. | Multiple random initialisations; momentum; stochasticity. For neural nets, local minima are often "good enough." |
| Ill-conditioned Hessian | Very different curvature along different directions. | Normalise features; use adaptive optimisers (Adam, RMSProp). |
| Saddle points | Gradient is zero but it is not a minimum — common in high dimensions. | Momentum and adaptive methods help escape. |
| Exploding/vanishing gradients | In deep nets, gradients grow or shrink exponentially with depth. | Proper weight initialisation; batch/layer normalisation; skip connections. |

---

## 5. Gradient descent and its variants

### 5.1 Batch gradient descent

Compute the gradient on the **entire training set** at every step:

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \ell(f_\theta(x_i), y_i).
$$

| Property | Implication |
|---|---|
| Exact gradient | Moves directly toward the local minimum of the full-batch loss surface. |
| $O(N)$ per step | Impractical for large datasets. |
| Deterministic | Same starting point → same path. Gets stuck in sharp local minima. |
| No gradient noise | Can converge to a sharper (worse-generalising) minimum. |

### 5.2 Stochastic gradient descent (SGD)

Use a **single randomly chosen example** per step:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \ell(f_\theta(x_i), y_i), \quad i \sim \text{Uniform}(1,\ldots,N).
$$

| Property | Implication |
|---|---|
| $O(1)$ per step | Extremely fast per iteration. |
| Noisy gradient estimate | $\mathbb{E}[\nabla \ell_i] = \nabla \mathcal{L}$ — it is unbiased, but high-variance. |
| Escapes local minima | The noise lets SGD bounce out of shallow minima and saddle points. |
| Converges slower near optimum | Need to decay $\eta$ for convergence; constant $\eta$ jitters around the minimum. |

**Why SGD generalises better.** The noise in the gradient estimate biases SGD toward *flatter* minima — small perturbations of the parameters do not change the loss much. Flat minima tend to generalise better than sharp ones, a phenomenon confirmed empirically and partially explained by Bayesian and PAC-Bayes analyses.

### 5.3 Mini-batch SGD

The practical default: use $B$ examples per step ($B$ is the batch size):

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_\theta \ell(f_\theta(x_i), y_i),
$$

where $\mathcal{B}$ is a random mini-batch of size $B$.

This interpolates between SGD ($B=1$) and batch GD ($B=N$). It gives:

- Lower gradient variance than SGD (variance scales as $\sim 1/B$).
- GPU-efficient matrix operations (vectorised over the batch).
- Still some noise for escaping bad minima.

### 5.4 Momentum

Momentum smooths the gradient by maintaining a velocity vector:

$$
v_{t+1} = \beta v_t + \eta \,\nabla_\theta \mathcal{L}(\theta_t),
\qquad
\theta_{t+1} = \theta_t - v_{t+1}.
$$

Common choice: $\beta = 0.9$. Think of it as a heavy ball rolling down the loss surface — it accumulates speed in consistent directions and dampens oscillations in high-curvature directions.

**Nesterov accelerated gradient (NAG)** evaluates the gradient *after* the momentum step, giving a "look-ahead" correction:

$$
v_{t+1} = \beta v_t + \eta \,\nabla_\theta \mathcal{L}(\theta_t - \beta v_t).
$$

It often converges faster in theory and practice.

### 5.5 Adaptive methods (Adam, RMSProp)

Adam combines momentum with per-parameter adaptive learning rates:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\,g_t  \quad &\text{(first moment — momentum)} \\
v_t &= \beta_2 v_{t-1}  + (1 - \beta_2)\,g_t^2 \quad &\text{(second moment — RMS)} \\
\hat{m}_t &= m_t / (1 - \beta_1^t) \quad &\text{(bias correction)} \\
\hat{v}_t &= v_t   / (1 - \beta_2^t)  \quad &\text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
\end{aligned}
$$

Default values: $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

Adam is the default optimiser for most deep learning. Watch out: Adam can sometimes find worse-generalising solutions than well-tuned SGD with momentum — the adaptivity that makes it fast can also lead it to sharp minima.

### 5.6 Epoch, batch, iteration — the vocabulary

| Term | Meaning |
|---|---|
| **Iteration** | One parameter update (one forward + backward pass on one mini-batch). |
| **Batch / mini-batch** | The $B$ examples used in one iteration. |
| **Epoch** | One full pass through the entire training set. For $N$ examples and batch size $B$, one epoch = $\lceil N/B \rceil$ iterations. |
| **Batch size** | $B$: the number of examples per mini-batch. Typical choices: 32, 64, 128, 256. Powers of 2 for GPU efficiency. |

**Why batch size matters:**

| Small $B$ (e.g. 32) | Large $B$ (e.g. 8192) |
|---|---|
| More noise → better regularisation | Less noise → stabler gradients |
| Slower wall-clock (poor GPU utilisation) | Faster wall-clock (good GPU utilisation) |
| Easier to escape sharp minima | Can converge to sharper minima |
| Lower memory | Higher memory |

The "linear scaling rule": when you multiply $B$ by $k$, multiply $\eta$ by $k$ as a starting heuristic (holds when gradients are approximately consistent across the batch).

### 5.7 Learning rate scheduling

Constant $\eta$ is rarely optimal. Common schedules:

- **Step decay:** $\eta \leftarrow \eta \times 0.1$ every $T$ epochs.
- **Cosine annealing:** $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))$. Smooth decay to near-zero; popular in modern training.
- **Warmup:** Start from a small $\eta$, linearly increase to the target $\eta$ over the first few epochs, then decay. Prevents early instability when weights are random.
- **Reduce-on-plateau:** Halve $\eta$ when validation loss stops improving.

---

## 6. Neural network concepts

### 6.1 The neuron as a building block

A single artificial neuron computes:

$$
a = g\!\left(\sum_{j=1}^{d} w_j x_j + b\right) = g(w^T x + b),
$$

where $w$ are weights, $b$ is the bias, and $g$ is a nonlinear **activation function**. Without $g$, stacking layers just produces an affine transformation — no more expressive than linear regression.

### 6.2 Activation functions

| Name | Formula | Derivative | Notes |
|---|---|---|---|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | Saturates for $|z|>5$; outputs in $(0,1)$. Used in output layer for binary classification. |
| Tanh | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | Zero-centered (unlike sigmoid); still saturates. |
| ReLU | $\max(0, z)$ | $\mathbf{1}[z>0]$ | Non-saturating for $z>0$; dead neurons for $z<0$. Default for hidden layers. |
| Leaky ReLU | $\max(0.01z, z)$ | $\mathbf{1}[z>0] + 0.01 \cdot \mathbf{1}[z \le 0]$ | Fixes the "dying ReLU" problem with a small negative slope. |
| GELU | $z \cdot \Phi(z)$ (approx) | — | Smooth approximation to ReLU; used in transformers. |
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | $\hat{y}_k(\delta_{kj} - \hat{y}_j)$ | Maps logits to a probability vector. Used in the output layer for multi-class classification. |

**The dying ReLU problem.** If a neuron always receives negative pre-activations, its gradient is zero forever — it "dies." Causes: too-high learning rate, poor initialisation, large negative bias. Leaky ReLU and proper initialisation mitigate this.

### 6.3 The universal approximation theorem

A feedforward network with one hidden layer and a non-polynomial activation can approximate any continuous function on a compact set to arbitrary accuracy, *given enough hidden units*. 

Caveats:
- It says nothing about how many units are needed (could be exponential).
- It says nothing about *learning* — only that a set of weights exists.
- Deep networks can represent certain functions with exponentially fewer units than shallow ones.

### 6.4 Architecture: width, depth, and skip connections

**Depth** (more layers) lets the network learn hierarchical features. In vision: edges → textures → parts → objects. In NLP: characters → words → phrases → semantics.

**Width** (more units per layer) increases capacity per representation level. Overly narrow layers create information bottlenecks.

**Skip connections (ResNets).** Instead of learning $H(x)$ directly, learn the residual $F(x) = H(x) - x$, so the output is $F(x) + x$. This:

- Allows gradients to flow directly through identity paths.
- Enables training of very deep networks (100+ layers).
- Makes the optimisation landscape smoother.

### 6.5 Initialisation

If weights start too large, activations and gradients explode. If too small, they vanish. Two principled approaches:

**Xavier/Glorot (for tanh/sigmoid):**

$$
w \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right).
$$

**He/Kaiming (for ReLU):**

$$
w \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}}}\right).
$$

The factor of 2 accounts for the fact that ReLU zeros out half the activations. Using the wrong initialisation for your activation function causes training to stall silently.

### 6.6 Regularisation tricks

| Technique | What it does | When to use |
|---|---|---|
| **L2 (weight decay)** | Adds $\lambda \|w\|^2_2$ to the loss. Shrinks all weights toward zero. | Always a sensible starting point. |
| **L1** | Adds $\lambda \|w\|_1$ to the loss. Encourages sparse weights. | When you need feature selection. |
| **Dropout** | Randomly zeroes out a fraction $p$ of activations during training. At test time, scale by $1-p$ (or use inverted dropout). | Dense layers; $p=0.5$ for hidden, $p=0.2$ for input. |
| **Batch normalisation** | Normalises each feature to zero mean / unit variance over the mini-batch, then applies learned scale $\gamma$ and shift $\beta$. | After linear layers, before activation. Reduces internal covariate shift; acts as a mild regulariser. |
| **Layer normalisation** | Like batch norm but normalises *per example* across features. | Transformers, RNNs; where batch norm breaks. |
| **Early stopping** | Stop training when validation loss stops improving. | Always. The simplest and most effective regulariser. |
| **Data augmentation** | Generate additional training examples through label-preserving transformations. | Images (flip, crop, colour jitter); audio (pitch shift, time stretch); text (back-translation, synonym replacement). |

### 6.7 Batch normalisation: the maths

For a mini-batch $\mathcal{B}$ of size $B$, for each feature dimension $k$:

$$
\begin{aligned}
\mu_k    &= \frac{1}{B}\sum_{i \in \mathcal{B}} x_i^{(k)}  \\
\sigma_k^2 &= \frac{1}{B}\sum_{i \in \mathcal{B}} (x_i^{(k)} - \mu_k)^2  \\
\hat{x}_i^{(k)} &= \frac{x_i^{(k)} - \mu_k}{\sqrt{\sigma_k^2 + \epsilon}}  \\
y_i^{(k)} &= \gamma_k \,\hat{x}_i^{(k)} + \beta_k.
\end{aligned}
$$

$\gamma$ and $\beta$ are learned parameters that restore representational power — the network can learn to undo the normalisation if that helps. At test time, use the running averages of $\mu$ and $\sigma^2$ from training.

---

## 7. Backpropagation

### 7.1 The idea

Backpropagation is an algorithm for computing the gradient of the loss with respect to every parameter in a computation graph. It is **reverse-mode automatic differentiation** applied to neural networks.

The key insight: computing all partial derivatives naively would be immensely redundant — the same sub-expressions appear in many derivatives. Backprop avoids this by computing derivatives from the output backward, reusing intermediate results.

### 7.2 Forward and backward passes

**Forward pass:** Compute all activations layer by layer:

$$
z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)},
\qquad
a^{(\ell)} = g^{(\ell)}(z^{(\ell)}),
$$

starting from $a^{(0)} = x$ and ending with the loss $\mathcal{L}$ computed from $a^{(L)}$.

**Backward pass:** Compute gradients recursively from the output layer backward. Define the **error signal** at layer $\ell$:

$$
\delta^{(\ell)} = \frac{\partial \mathcal{L}}{\partial z^{(\ell)}}.
$$

For the output layer $L$, this comes directly from the loss. Then for each earlier layer:

$$
\delta^{(\ell)} = \left((W^{(\ell+1)})^T \delta^{(\ell+1)}\right) \odot g'^{(\ell)}(z^{(\ell)}),
$$

where $\odot$ is element-wise multiplication. The parameter gradients are then:

$$
\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \delta^{(\ell)} (a^{(\ell-1)})^T,
\qquad
\frac{\partial \mathcal{L}}{\partial b^{(\ell)}} = \delta^{(\ell)}.
$$

### 7.3 Worked example: a tiny two-layer network

Consider a network with one hidden layer and one output — regression with MSE loss:

- Input: $x \in \mathbb{R}^d$
- Hidden: $h = \sigma(W_1 x + b_1)$
- Output: $\hat{y} = W_2 h + b_2$ (scalar)
- Loss: $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$

**Forward pass:**
$$
z_1 = W_1 x + b_1,\quad h = \sigma(z_1),\quad \hat{y} = W_2 h + b_2,\quad \mathcal{L} = \tfrac{1}{2}(\hat{y} - y)^2.
$$

**Backward pass:**
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \hat{y}} &= \hat{y} - y \\[4pt]
\frac{\partial \mathcal{L}}{\partial W_2} &= (\hat{y} - y) \cdot h^T \\[4pt]
\frac{\partial \mathcal{L}}{\partial b_2} &= \hat{y} - y \\[4pt]
\frac{\partial \mathcal{L}}{\partial h} &= W_2^T (\hat{y} - y) \\[4pt]
\frac{\partial \mathcal{L}}{\partial z_1} &= \frac{\partial \mathcal{L}}{\partial h} \odot \sigma'(z_1) \\[4pt]
\frac{\partial \mathcal{L}}{\partial W_1} &= \frac{\partial \mathcal{L}}{\partial z_1} \cdot x^T \\[4pt]
\frac{\partial \mathcal{L}}{\partial b_1} &= \frac{\partial \mathcal{L}}{\partial z_1}.
\end{aligned}
$$

This is the entire algorithm. Every deep learning framework does exactly this, automatically, on whatever computation graph you define. You should be able to do it by hand for a 2–3 layer network; it is a common interview question.

### 7.4 The computation graph perspective

Every operation in a neural network is a node in a directed acyclic graph. Given the loss at the root:

1. Traverse the graph forward to compute all values.
2. Traverse the graph backward, at each node computing the gradient of the loss with respect to that node's output, using gradients already computed for its consumers.

This is exactly what PyTorch's `autograd` and TensorFlow's `GradientTape` do.

### 7.5 Gradient checking

How do you know your backprop is correct? The finite-difference check:

$$
\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon},
$$

with $\epsilon \approx 10^{-5}$ to $10^{-7}$. The central difference is $O(\epsilon^2)$ accurate. Compare your analytical gradient to the numerical one; they should match to within $10^{-5}$ to $10^{-7}$ in relative error. If not, there is a bug.

Gradient checking is a debugging tool, not something you run in production. Do it once for a tiny network using a few random data points.

### 7.6 Backpropagation traps

| Trap | Symptom | Fix |
|---|---|---|
| Vanishing gradients (sigmoid/tanh deep nets) | Gradients of early layers approach zero; network stops learning. | ReLU, proper initialisation, batch norm, skip connections. |
| Exploding gradients | Gradients become NaN; loss diverges. | Gradient clipping, proper initialisation, lower learning rate. |
| Forgetting to zero gradients | Gradients accumulate across batches. | Always call `optimizer.zero_grad()` (PyTorch) or equivalent. |
| In-place modification of tensors needed for backward | Runtime error about modifying saved variables. | Use `.clone()` before modifying; avoid in-place ops on leaf tensors. |
| Numerical instability (softmax + cross-entropy) | NaN loss. | Use the fused `CrossEntropyLoss` (PyTorch) or `log_softmax + nll_loss` — never compute softmax then log manually. |
| Dead ReLUs | Large fractions of neurons output zero for all inputs. | Lower learning rate; use Leaky ReLU or ELU; check initialisation. |

---

## 8. Putting it all together: the training loop

Here is the complete mental model of neural network training:

```
for epoch in range(num_epochs):
    shuffle training data
    for each mini-batch (X_batch, y_batch):
        # Forward pass
        y_pred = model(X_batch)
        loss   = loss_fn(y_pred, y_batch)        # objective function
        
        # Backward pass
        optimizer.zero_grad()                    # clear old gradients
        loss.backward()                          # backprop — compute gradients
        
        # Update
        optimizer.step()                         # gradient descent step
    
    # Evaluate
    val_loss = evaluate(model, val_loader)
    if val_loss is best so far: save checkpoint
    if val_loss hasn't improved for K epochs: stop (early stopping)
```

That loop, plus the architecture and the data, is the heart of modern deep learning. Everything else — the fancy architectures, the clever loss functions, the training tricks — is a variation on this template.

---

## 9. Interview practice

### 9.1 The response pattern

For ML interview questions, answer in this order:

1. **Frame the problem.** "This is a [classification / regression / density estimation] problem. The quantity of interest is..."
2. **State assumptions.** "I will assume the data is i.i.d. and the relationship is learnable from the given features."
3. **Choose the model and loss.** "I will use [model] with [loss], because..."
4. **Describe optimisation.** "I will train with [optimiser], batch size $B$, learning rate $\eta$, for $E$ epochs with [schedule]."
5. **Name failure modes.** "This can fail if [overfitting / vanishing gradients / non-i.i.d. data / distribution shift / ...]"
6. **Say how you would validate.** "Hold-out test set; cross-validation for hyperparameters; check calibration and residuals."

### 9.2 Bayes' theorem question

**Question.** A disease affects 1% of the population. A test is 95% accurate (95% sensitivity and 95% specificity). If you test positive, what is the probability you actually have the disease?

**Answer.** Apply Bayes:

$$
\Pr(\text{Disease} \mid +) = \frac{\Pr(+ \mid \text{Disease})\,\Pr(\text{Disease})}{\Pr(+)} = \frac{0.95 \times 0.01}{(0.95 \times 0.01) + (0.05 \times 0.99)} \approx 16.1\%.
$$

Despite a "95% accurate" test, only ~16% of positives are true positives. This is the base-rate fallacy — always ask for the prevalence.

### 9.3 Gradient descent question

**Question.** Explain gradient descent to a non-technical stakeholder.

**Answer.** "Imagine you are standing on a mountain in thick fog and want to reach the bottom. You cannot see the whole landscape, but you can feel the slope under your feet. At each step, you take a small step in the steepest downhill direction, then re-feel the slope. That is gradient descent. The step size—the learning rate—matters: too large and you jump over the valley; too small and the descent takes forever. We have smart ways to adjust the step size automatically."

### 9.4 Derivation question: cross-entropy gradient

**Question.** Derive the gradient of binary cross-entropy loss with a sigmoid output.

**Answer.** Let $\hat{y} = \sigma(z)$, $z = w^T x + b$:

$$
\mathcal{L} = -[y\log\hat{y} + (1-y)\log(1-\hat{y})].
$$

We know $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$.

$$
\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}
= \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y})
= \hat{y} - y.
$$

This cancellation is elegant and is *why* we pair sigmoid with cross-entropy, not MSE. Then:

$$
\frac{\partial \mathcal{L}}{\partial w} = (\hat{y} - y)\,x,
\qquad
\frac{\partial \mathcal{L}}{\partial b} = \hat{y} - y.
$$

### 9.5 Design question: "How would you build a fraud detection model?"

**Strong answer.** "Frame it as binary classification with extreme class imbalance. The loss function should reflect business costs: false negatives (missed fraud) are much more expensive than false positives (incorrectly blocked transactions). Use gradient boosted trees as a strong baseline — they handle tabular features, missing values, and interactions well. For deep learning, consider an embedding layer for high-cardinality categorical features (merchant ID, card ID). Evaluate with precision-recall curves, not ROC — ROC is misleading under severe imbalance. Validate chronologically, not randomly, to catch temporal drift. Monitor feature distributions and prediction drift in production."

### 9.6 Calculation question: chain rule

**Question.** If $\mathcal{L} = (wx + b - y)^2$, compute $\frac{\partial \mathcal{L}}{\partial w}$.

**Answer.** Let $\hat{y} = wx + b$.

$$
\frac{\partial \mathcal{L}}{\partial w}
= 2(\hat{y} - y) \cdot \frac{\partial \hat{y}}{\partial w}
= 2(wx + b - y) \cdot x.
$$

### 9.7 Concept question: "Why does batch normalisation help?"

**Answer.** It reduces **internal covariate shift** — the change in the distribution of layer inputs as earlier layers update. More practically: (1) it allows higher learning rates by preventing activations from drifting into saturated regimes; (2) it acts as a mild regulariser via the noise in the mini-batch statistics; (3) it reduces sensitivity to initialisation. The original "internal covariate shift" explanation has been challenged by later work showing that the real benefit is smoother optimisation landscapes — batch norm makes the loss surface more Lipschitz, so gradients are more predictive.

### 9.8 Coding question: "Implement a decision tree split from scratch"

**Answer outline.**
```python
def best_split(X, y):
    best_gain = -float('inf')
    best_feature, best_threshold = None, None
    n = len(y)
    parent_impurity = gini(y)

    for j in range(X.shape[1]):
        thresholds = np.unique(X[:, j])
        for s in thresholds:
            left  = y[X[:, j] <= s]
            right = y[X[:, j] > s]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = parent_impurity - (
                len(left)/n  * gini(left) +
                len(right)/n * gini(right)
            )
            if gain > best_gain:
                best_gain = gain
                best_feature, best_threshold = j, s

    return best_feature, best_threshold
```

Then explain: this is $O(N \cdot d)$ for sorted features, growing a full tree is $O(N \cdot d \cdot \text{depth})$, and the greedy approach does not guarantee global optimality.

### 9.9 Trick question: "If your training loss is zero, is your model good?"

**Answer.** Almost certainly not. You have memorised the training data — overfitting. The validation loss is likely high. You have learned noise, not signal. You need regularisation: more data, simpler model, weight decay, dropout, early stopping, or data augmentation. In the limit, a model with more parameters than training points can achieve exactly zero training loss on any dataset (for suitably flexible models); this says nothing about generalisation.

### 9.10 Architecture question: "How many layers and units should I use?"

**Answer.** Start as simple as possible. One or two hidden layers often suffice for tabular data. For images, start with a known architecture (ResNet-18/34/50). For text, start with a small transformer or even a bigram baseline. Capacity should be set by cross-validation: increase width/depth until validation loss stops improving, then regularise. Do not optimise architecture before you have a solid data pipeline, loss, and evaluation framework. The best architecture in the world cannot fix bad data or a misaligned loss.

---

## 10. Cheat sheet

| Concept | Remember |
|---|---|
| Bayes' theorem | $\Pr(\theta \mid \mathcal{D}) \propto \Pr(\mathcal{D} \mid \theta)\,\Pr(\theta)$ |
| MLE | Maximise $\Pr(\mathcal{D} \mid \theta)$; equivalent to minimising negative log-likelihood. |
| Cross-entropy | The log loss — minimising it = maximising likelihood for categorical data. |
| Gradient descent | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ |
| SGD | Use one or a mini-batch of examples; unbiased but noisy gradient estimate. |
| Momentum | $v \leftarrow \beta v + \eta \nabla \mathcal{L}$; smooths and accelerates. |
| Adam | Adaptive step sizes from first and second moment estimates; default optimiser. |
| Batch norm | Normalise features per mini-batch: stabilises, regularises, speeds up. |
| Backpropagation | Chain rule applied to the computation graph, backward from loss to parameters. |
| Vanishing gradient | Gradients → 0 in early layers; use ReLU, skip connections, proper init. |
| Decision tree | Greedy, axis-aligned splits maximising impurity reduction. |
| Random forest | Bagged trees with random feature subsets; variance reduction. |
| Overfitting | Low training error, high validation error. Fix: data, regularisation, simpler model. |
| Epoch vs iteration | Epoch = one pass through the full dataset. One epoch = $\lceil N/B \rceil$ iterations. |
| Universal approximation | One hidden layer can approximate any continuous function — but may need exponentially many units. |

---

## 11. Final modelling checklist

Before deploying an ML model, be able to answer all of these:

- What is the model trying to estimate — a conditional probability, an expectation, a density?
- Is the loss function aligned with the business objective? What does getting it wrong cost?
- Is the data split chronologically (not randomly) when time matters?
- Has the model been validated on a truly held-out test set, used exactly once?
- Are the features available at prediction time? No future leakage?
- Is the model calibrated? Does a predicted probability of 0.8 actually mean 80% empirical frequency?
- Does the training pipeline handle missing values, outliers, and distribution shift?
- Has the model been compared to a simple, interpretable baseline?
- Are gradients checked for custom layers? Are numerical issues (NaN/Inf) monitored?
- Can you explain the model's predictions for the most important decisions it makes?
- Is there a plan for monitoring and retraining in production?

If you can explain these choices, work through the chain rule, and diagnose when training goes wrong, you have the ML fluency that most technical interviews and real-world projects demand.
