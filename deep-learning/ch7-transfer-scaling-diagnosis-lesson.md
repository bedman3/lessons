# Chapter 7 — Transfer, Scaling & Failure Diagnosis

Modern deep learning rarely begins from random weights. Models are pretrained on broad objectives, adapted to narrower tasks, compressed or distributed for deployment, and monitored under shift. The central question is no longer only “Can this architecture fit?” but “Which knowledge transfers, at what cost, and how do we know it still works?”

## 7.1 Why pretraining transfers

Pretraining learns reusable statistical structure from a large source distribution. Examples include supervised classification, masked prediction, contrastive learning, autoregressive next-token prediction, and multimodal alignment.

Transfer succeeds when source representations contain features useful for the target and target optimization can adapt without destroying them. It weakens under domain, modality, label, language, temporal, or resolution shift.

Large pretraining data do not eliminate the need for point-in-time and contamination checks in target evaluation.

## 7.2 Frozen features versus full fine-tuning

- **Linear probe:** freeze backbone, train one linear head; diagnoses accessible features.
- **Partial fine-tuning:** unfreeze later blocks or normalization parameters.
- **Full fine-tuning:** update all weights; maximum flexibility and cost.

Frozen features reduce variance and compute when target data are scarce. Full fine-tuning can adapt deeper representations but risks overfitting and catastrophic forgetting.

A common schedule trains the head first, then gradually unfreezes with smaller learning rates for pretrained layers.

## 7.3 Catastrophic forgetting

Target gradients can overwrite capabilities learned during pretraining. Symptoms include target improvement alongside sharp loss on general or safety evaluations.

Controls include lower learning rates, regularization toward original weights, rehearsal/replay data, multi-task mixtures, adapters, and early stopping. The appropriate control depends on which original behaviours must be preserved.

Evaluation must include both target and retained-capability sets.

## 7.4 Parameter-efficient adaptation

Adapters insert small trainable modules while freezing the base model. LoRA represents a weight update as low rank:

$$
W'=W+\Delta W,
\qquad
\Delta W=BA,
$$

where $A\in\mathbb R^{r\times d_{in}}$, $B\in\mathbb R^{d_{out}\times r}$, and $r$ is small. Trainable parameters fall from $d_{in}d_{out}$ to

$$
r(d_{in}+d_{out}).
$$

LoRA assumes the useful update lies near a low-rank subspace. Rank, target modules, scaling, dropout, and data matter. QLoRA stores the frozen base in low precision while training higher-precision adapters, reducing memory but adding quantization considerations.

## 7.5 Instruction and preference post-training

Supervised fine-tuning trains on demonstrations using the same conditional likelihood machinery as pretraining, but changes the data distribution toward instructions and desired responses.

Preference training uses comparisons among outputs. RLHF commonly fits a reward model and optimizes a policy while penalizing departure from a reference. Direct preference optimization rewrites a class of preference objectives directly in terms of policy/reference log-probability ratios.

These methods optimize observed preferences, not a complete definition of truth or safety. Reward misspecification, annotator disagreement, sycophancy, distribution shift, and capability regressions require independent evaluation.

## 7.6 Data quality and mixtures

Scaling poor or duplicated data can reinforce noise and memorization. Important controls include deduplication, contamination audits, source weighting, licensing, language/domain balance, quality filters, and temporal cutoffs.

Mixture weights determine what the model spends capacity learning. Upsampling a small domain improves exposure but increases overfitting and memorization risk. Data curriculum and sequence length also change optimization.

Report token/example counts after filtering and deduplication, not only raw collection size.

## 7.7 Scaling laws

Empirical loss often follows approximate power laws:

$$
L(N,D,C)\approx L_\infty
+aN^{-\alpha}+bD^{-\beta}+cC^{-\gamma},
$$

where $N$ is parameters, $D$ data, and $C$ compute under a chosen regime. Exponents are empirical and domain-dependent.

Compute-optimal training balances model size and tokens; an oversized undertrained model can be worse than a smaller model trained on more data. Scaling curves support budget planning, not guarantees about downstream reasoning, safety, or rare tasks.

Emergent-looking threshold behaviour can arise from smooth underlying loss combined with nonlinear evaluation metrics.

## 7.8 Distributed training

- Data parallelism replicates model and splits batches; gradients are synchronized.
- Tensor parallelism splits large matrix operations across devices.
- Pipeline parallelism splits layers and schedules microbatches.
- Sharded data parallelism partitions parameters, gradients, and optimizer states.

Communication, memory, utilization, and numerical reproducibility trade off. Increasing device count can reduce efficiency if synchronization or pipeline bubbles dominate.

Gradient accumulation increases effective batch size without storing all examples at once. Mixed precision accelerates compute but may need loss scaling and careful treatment of reductions.

## 7.9 Memory and compute accounting

Training memory includes weights, gradients, optimizer states, activations, temporary buffers, and communication. Adam states can exceed raw weight memory. Activation checkpointing recomputes forward activations during backward pass to trade compute for memory.

For Transformers, sequence length raises attention activation cost quadratically. Parameter count alone is therefore an incomplete resource estimate.

Mixture-of-experts routes each token to a subset of feed-forward experts, increasing parameter capacity without proportional per-token compute. Load balancing, communication, expert collapse, and routing stability become new failure modes.

## 7.10 Inference optimization

- Batching improves throughput but may increase latency.
- Quantization reduces memory bandwidth and compute, with layer/task-dependent error.
- Distillation trains a smaller student to match teacher outputs or representations.
- Pruning removes parameters or structures, but unstructured sparsity may not yield hardware speed.
- KV caching avoids prefix recomputation in autoregressive Transformers.
- Speculative decoding uses a smaller draft model and verifies multiple candidate tokens with the target model while preserving the target distribution under the algorithm's conditions.

Measure end-to-end latency, throughput, memory, energy, and quality on target hardware. FLOPs alone can mispredict performance.

## 7.11 Quantization intuition

Map floating values to discrete levels:

$$
q=\operatorname{round}(x/s)+z,
$$

with scale $s$ and zero point $z$. Per-channel scales better match heterogeneous weight ranges than one global scale.

Outliers, activation ranges, accumulated error, and sensitive layers determine quality. Post-training quantization is cheap; quantization-aware training simulates quantization during learning to adapt weights.

Always evaluate rare, long-context, calibration, and safety behaviours—not only average benchmark accuracy.

## 7.12 Distillation

A student can minimize a mixture of hard-label loss and teacher-distribution divergence. With temperature $T$,

$$
p_i^{(T)}=\frac{e^{z_i/T}}{\sum_je^{z_j/T}}.
$$

Higher temperature exposes relative probabilities among non-max classes—the teacher's “dark knowledge.” Distillation can transfer errors, biases, and overconfidence as well as useful structure.

Intermediate-feature or sequence-level distillation may better preserve behaviour for structured outputs.

## 7.13 Evaluation under adaptation

Use separate suites for:

- target-task performance and calibration;
- retained general capabilities;
- domain and temporal shift;
- rare and adversarial cases;
- subgroup performance;
- memorization and contamination;
- latency, memory, throughput, and cost;
- robustness to prompt/input formatting and sequence length.

Benchmark selection is part of training if repeatedly used to guide changes. Keep final or rolling prospective evaluations genuinely untouched.

## 7.14 Systematic failure diagnosis

| Symptom | Likely layer to inspect first | Evidence |
|---|---|---|
| Cannot overfit tiny batch | code, loss, labels, optimizer | hand loss, gradient check, update norms |
| Train improves, validation worsens | capacity, leakage-free sample size, shift | learning curves, simpler baseline, resamples |
| Both losses high | target ambiguity, features, optimization | baseline gap, label audit, capacity sweep |
| Fine-tuning destroys old skills | forgetting, learning rate, data mixture | retained-suite curve, weight/update norms |
| Good offline, poor serving | parity, quantization, decoding, latency | golden replay, layer error, request traces |
| Long-context failure | positional range, retrieval, distraction | length sweep, controlled needle tests |
| Confident wrong outputs | objective mismatch, calibration, missing evidence | proper scores, abstention, retrieval checks |

Change one layer at a time and preserve the failing example as a regression test.

## 7.15 Failure modes

- Fine-tuning every weight when a frozen probe already solves the task.
- Reporting adapter parameter count while ignoring base-model inference cost.
- Treating preference optimization as factuality training.
- Extrapolating scaling laws beyond their measured regime.
- Adding devices without measuring communication and utilization.
- Quantizing from one average benchmark and missing rare regressions.
- Distilling a teacher without auditing its failure modes.
- Reusing public benchmarks until they become training feedback.

## 7.16 Knowledge checks

1. When should frozen features outperform full fine-tuning?
2. Derive LoRA parameter savings for a $d_{out}\times d_{in}$ matrix.
3. Why can a larger model be undertrained under fixed compute?
4. What does speculative decoding preserve?
5. Why must post-training evaluation include retained capabilities?

### Solution outlines

1. When target data are small and pretrained features align well, reducing estimation variance and forgetting.
2. Dense update has $d_{out}d_{in}$ parameters; rank-$r$ factors have $r(d_{out}+d_{in})$.
3. Parameters receive too few data/token updates relative to capacity; a smaller model can use the compute more effectively.
4. The target model's output distribution, while a draft model proposes candidates for faster verification.
5. Adaptation can improve the target while catastrophically forgetting or misaligning other required behaviour.

## 7.17 Course synthesis

Deep learning architecture is an interaction among:

1. inductive bias and representation;
2. information flow and gradient flow;
3. objective and data construction;
4. scale, optimization, and compute systems;
5. adaptation and retained capability;
6. inference constraints and approximation;
7. evaluation under real distribution shift.

Return to the [Deep Learning Architectures in Practice contents](index.html).
