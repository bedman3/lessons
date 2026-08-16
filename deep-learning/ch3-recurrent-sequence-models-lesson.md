# Chapter 3 — Recurrent & Sequence Models

Sequence models must represent ordered information of variable length. Recurrent networks do this by updating a hidden state one step at a time. Their explicit memory makes causality natural, but long gradient paths make training difficult.

## 3.1 The recurrent state

A basic RNN is

$$
h_t=\phi(W_xx_t+W_hh_{t-1}+b),
$$

$$
\widehat y_t=g(W_yh_t+c).
$$

The same parameters are reused across time. $h_t$ summarizes the prefix $x_1,\ldots,x_t$ under a fixed-dimensional bottleneck.

Unrolling through time converts recurrence into a deep network with shared weights. This enables ordinary backpropagation but creates a path length proportional to sequence length.

## 3.2 Backpropagation through time

If loss depends on many times,

$$
L=\sum_tL_t,
$$

the gradient to an earlier state contains products of recurrent Jacobians:

$$
\frac{\partial L}{\partial h_t}
\supset
\frac{\partial L_T}{\partial h_T}
\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}.
$$

Repeated singular values below one make gradients vanish; above one make them explode. Gradient clipping controls explosions. Orthogonal initialization, normalization, gated cells, and shorter gradient paths help stability.

Truncated BPTT backpropagates only a fixed number of steps. It reduces memory and computation but biases gradients against dependencies longer than the truncation window.

## 3.3 Long short-term memory

An LSTM maintains cell state $c_t$ with gated additive updates:

$$
f_t=\sigma(W_f[x_t,h_{t-1}]+b_f),
$$

$$
i_t=\sigma(W_i[x_t,h_{t-1}]+b_i),
$$

$$
\widetilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c),
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot\widetilde c_t,
$$

$$
o_t=\sigma(W_o[x_t,h_{t-1}]+b_o),\qquad
h_t=o_t\odot\tanh(c_t).
$$

The forget gate preserves or erases memory; the input gate writes; the output gate exposes. The additive cell path can carry gradients better than repeated nonlinear replacement.

## 3.4 GRUs

A gated recurrent unit combines cell and hidden state with reset and update gates. It has fewer parameters than an LSTM and often similar performance. Neither architecture guarantees long-term memory: gates can saturate, optimization can fail, and the finite state remains a bottleneck.

Choose by evidence under the real sequence length and compute budget, not by a universal hierarchy.

## 3.5 Sequence task structures

- Many-to-one: classify a document or forecast from a history.
- One-to-many: generate a sequence from one context.
- Many-to-many aligned: label every token or time step.
- Encoder–decoder: map an input sequence to a differently sized output sequence.

Bidirectional recurrence uses past and future context and is appropriate for offline labelling. It leaks future information in causal forecasting or generation.

## 3.6 Teacher forcing and exposure bias

In autoregressive training, teacher forcing feeds the true previous token $y_{t-1}$ while predicting $y_t$. At inference, the model receives its own previous prediction. Errors can move the state into contexts absent during training—exposure bias.

Scheduled sampling mixes true and generated inputs but changes the training objective and can be statistically inconsistent. Sequence-level objectives, robust data, beam/search methods, and explicit evaluation under free-running generation are alternatives.

## 3.7 Padding, masking, and packed sequences

Batches pad shorter sequences. A mask must prevent padded positions from contributing to loss or state. If lengths are $\ell_i$, a masked loss is

$$
L=\frac{\sum_{i,t}\mathbf1\{t\le\ell_i\}L_{i,t}}
{\sum_i\ell_i}.
$$

Normalizing by batch size rather than valid tokens changes weighting with sequence length. Packed or bucketed sequences reduce wasted computation.

Hidden state should be reset at true sequence boundaries. Accidentally carrying state across unrelated entities creates leakage.

## 3.8 Encoder–decoder and attention

A recurrent encoder compressing an entire input into one final vector creates an information bottleneck. Attention lets the decoder form a weighted combination of all encoder states at each output time:

$$
c_t=\sum_s\alpha_{t,s}h_s.
$$

This shortens paths between distant positions and was the bridge to Transformers, which remove recurrence from the main sequence computation.

## 3.9 Time-series use

For multivariate time series, specify:

- sampling and missingness;
- causal feature availability;
- state reset across assets/episodes;
- normalization fitted on prior data;
- forecast horizon and overlapping labels;
- whether irregular intervals carry information.

Continuous-time, state-space, temporal-convolution, or Transformer models may fit long/irregular sequences better. RNN state can be efficient for streaming because inference updates in $O(1)$ per step relative to history length.

## 3.10 Worked shape example

For input dimension 20 and LSTM hidden size 64, each of four gates uses weights from the concatenated $20+64$ inputs to 64 outputs plus bias. Parameter count is

$$
4[(20+64)64+64]=21{,}760.
$$

A batch of 32 sequences of length 100 produces hidden tensor shape $32\times100\times64$ if all time outputs are retained.

## 3.11 Failure modes

- Carrying hidden state across unrelated samples.
- Using bidirectional recurrence for a causal forecast.
- Including padded tokens in loss or normalization.
- Treating gradient clipping as evidence that long dependencies are learned.
- Evaluating only with teacher-forced inputs.
- Choosing truncation length shorter than the hypothesized mechanism without acknowledging bias.
- Randomly splitting overlapping time windows.

## 3.12 Knowledge checks

1. Why do recurrent gradients vanish or explode?
2. What feature of LSTM cell updates improves gradient flow?
3. What bias does truncated BPTT introduce?
4. Why is teacher-forced validation incomplete for generation?
5. When is bidirectional recurrence invalid?

### Solution outlines

1. They contain long products of recurrent Jacobians.
2. An additive, gated cell path instead of complete nonlinear replacement at every step.
3. It ignores parameter effects mediated through dependencies beyond the truncation horizon.
4. Inference conditions on model-generated history, whose errors alter future contexts.
5. When predictions must not use observations after the decision time.

## 3.13 What to retain

- Recurrence encodes order through a shared state update.
- BPTT turns temporal depth into gradient products.
- Gates create controlled additive memory but do not guarantee long context.
- Masks, boundaries, and causality are part of model correctness.
- Streaming efficiency is a continuing strength of recurrent/state-space models.

Next: [Chapter 4 — Attention & Transformers](ch4-attention-transformers-viewer.html).
