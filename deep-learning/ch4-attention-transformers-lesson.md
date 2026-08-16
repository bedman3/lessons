# Chapter 4 — Attention & Transformers

Attention lets every position retrieve a content-dependent mixture of other positions. Transformers build sequence models from this operation, residual paths, normalization, and position-wise nonlinear layers. They shorten dependency paths and parallelize training, while making memory grow quickly with context length.

## 4.1 Queries, keys, and values

For token representations $X\in\mathbb R^{L\times d_{model}}$, project

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

For one head with dimension $d_k$,

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V.
$$

A query expresses what a position seeks, a key what each position offers for matching, and a value the content retrieved. These are learned projections, not fixed semantic roles.

## 4.2 Why scale dot products

If query and key components have variance near one, their dot product has variance proportional to $d_k$. Large logits push softmax toward saturation and tiny gradients. Dividing by $\sqrt{d_k}$ keeps scale roughly stable as dimension grows.

Softmax produces nonnegative weights summing to one per query. Attention output is a data-dependent weighted average of values.

## 4.3 Masks

Add $-\infty$ to disallowed logits before softmax.

- Padding mask hides nonexistent tokens.
- Causal mask hides future positions $j>i$.
- Structured masks restrict attention by locality or graph.

Masking after softmax is wrong unless probabilities are renormalized. A one-position shift error in a causal mask creates direct target leakage.

## 4.4 Multi-head attention

Split projections into $H$ heads:

$$
\operatorname{MHA}(X)
=\operatorname{Concat}(head_1,\ldots,head_H)W_O.
$$

Each head has separate projections and can represent different similarity patterns. Multiple heads increase representational subspaces, but observed attention maps are not guaranteed explanations of model reasoning.

Typically $d_k=d_{model}/H$, so total projection parameter scale remains $O(d_{model}^2)$.

## 4.5 Self- and cross-attention

Self-attention draws $Q,K,V$ from one sequence. Cross-attention draws queries from a decoder/current stream and keys/values from an encoded source or external memory.

Encoder blocks use bidirectional self-attention. Decoder-only language models use causal self-attention. Encoder–decoder models combine bidirectional source encoding, causal target attention, and cross-attention.

## 4.6 Position information

Attention alone is permutation-equivariant: reordering tokens reorders outputs. Position must be injected.

Sinusoidal encodings add fixed sine/cosine functions at multiple frequencies. Learned embeddings memorize positions within a trained range. Relative position biases score token distances.

Rotary position embeddings (RoPE) rotate query/key coordinates by position-dependent angles. The dot product then depends naturally on relative displacement. Extrapolation beyond training length is still not automatic; frequency scaling and training distribution matter.

## 4.7 Transformer block

A pre-normalized block is schematically

$$
X'=X+\operatorname{Attention}(\operatorname{LN}(X)),
$$

$$
Y=X'+\operatorname{MLP}(\operatorname{LN}(X')).
$$

The MLP independently transforms each position, often expanding dimension by about 4× and using GELU or a gated activation such as SwiGLU. Attention mixes positions; the MLP mixes and transforms channels.

Residual paths preserve information and gradients. Dropout or stochastic depth may regularize depending on scale and data.

## 4.8 Worked shape example

Let $B=8$, $L=128$, $d_{model}=512$, and $H=8$, so $d_k=64$.

- $Q,K,V$: $8\times8\times128\times64$ after head reshaping.
- Attention logits: $8\times8\times128\times128$.
- Weighted values recombine to $8\times128\times512$.

The logit tensor grows as $BL^2H$, which drives memory cost for long sequences.

## 4.9 Complexity and efficient kernels

Standard attention costs roughly $O(L^2d)$ compute and $O(L^2)$ score memory. FlashAttention reorganizes exact attention computation into hardware-efficient tiles, avoiding materializing the full score matrix in slow memory. It changes the algorithm's memory traffic, not the mathematical attention result.

Sparse, local, low-rank, linear-attention, recurrence, and state-space approaches change the approximation or architecture to reduce long-context cost. Their quality depends on which long-range interactions the task needs.

## 4.10 Autoregressive inference and KV cache

During generation, prior keys and values do not change. Cache them so each new token computes only its query and new key/value. This avoids recomputing the full prefix, but cache memory grows approximately as

$$
O(L\times\text{layers}\times H_{kv}\times d_k).
$$

Multi-query attention shares one key/value head across query heads. Grouped-query attention uses fewer KV heads than query heads. Both reduce cache bandwidth and memory, potentially trading some capacity.

Batching improves throughput but competes for KV-cache memory. Latency, throughput, context length, and memory are coupled system choices.

## 4.11 Training objective and generation

Causal language modelling minimizes

$$
-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

Teacher forcing parallelizes training because all next-token targets are known. Generation is sequential and feeds sampled tokens back into context.

Greedy, beam, temperature, top-$k$, and nucleus sampling change output distribution without changing learned probabilities. Decoding cannot recover knowledge or reasoning absent from the model and context.

## 4.12 What attention does not guarantee

Long context does not mean every token is used reliably. Models can attend to distractors, lose position/count information, or rely on superficial correlations. Attention weights are intermediate routing coefficients, not calibrated feature importance.

Transformers also inherit data bias, leakage, distribution shift, and objective mismatch. Next-token likelihood rewards plausible continuation, not factual verification.

## 4.13 Failure modes

- Omitting $1/\sqrt{d_k}$ and saturating softmax.
- Applying causal or padding masks at the wrong stage.
- Treating attention maps as causal explanations.
- Assuming positional encodings extrapolate indefinitely.
- Calling FlashAttention an approximation when it computes exact attention.
- Ignoring KV-cache memory in serving estimates.
- Equating larger context with trustworthy retrieval or reasoning.

## 4.14 Knowledge checks

1. Derive attention tensor shapes for given batch, length, heads, and head dimension.
2. Why is dot-product attention scaled?
3. What distinct roles do attention and the position-wise MLP play?
4. Why does KV caching help inference but consume memory?
5. Contrast MHA, MQA, and GQA.

### Solution outlines

1. $Q,K,V$ reshape to $B\times H\times L\times d_k$; scores to $B\times H\times L\times L$.
2. To keep logit variance and softmax gradients stable as $d_k$ grows.
3. Attention mixes information across positions; the MLP transforms channels within each position.
4. Past K/V are reused rather than recomputed, but must be stored for every layer and token.
5. MHA has per-query-head KV; MQA shares one KV head; GQA shares KV within groups.

## 4.15 What to retain

- Attention is content-dependent retrieval through query–key similarity and value mixing.
- Masks and positional structure define permissible information flow.
- Transformer blocks alternate token mixing and channel transformation with residual paths.
- Training parallelizes; autoregressive inference remains sequential and cache-bound.
- Architectural capacity does not guarantee faithful use of context or factual outputs.

Next: [Chapter 5 — Autoencoders, Contrastive Learning & Embeddings](ch5-autoencoders-contrastive-embeddings-viewer.html).
