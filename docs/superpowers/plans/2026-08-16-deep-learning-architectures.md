# Deep Learning Architectures in Practice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a seven-chapter theory-and-practice course covering representation learning, CNNs, sequence models, Transformers, embedding objectives, generative models, transfer, scaling, and failure diagnosis.

**Architecture:** Add `deep-learning/` using canonical Markdown lessons and matching shared-renderer viewers. The existing ML/DL foundations lesson remains the concise overview; this course derives architectural biases and training behaviour in depth.

**Tech Stack:** Markdown, MathJax 3, static HTML/CSS, shared ES-module viewer, Python/Node structural tests.

**Spec:** `docs/superpowers/specs/2026-08-16-advanced-curriculum-expansion-design.md`

## Global Constraints

- Target upper-undergraduate through practitioner or early-researcher depth.
- Explain tensor shapes, objectives, gradient flow, inductive biases, computational complexity, and failure modes.
- Link probabilistic objectives to the foundations and generalization to Applied ML.
- Include derivations, worked shape/parameter examples, practical diagnostics, exercises, and solutions.
- Cover modern Transformer and generative concepts without turning the course into product prompting guidance.

---

### Task 1: Representation learning and convolution

**Files:**
- Create: `deep-learning/ch1-representations-optimization-lesson.md`
- Create: `deep-learning/ch2-convolutional-networks-lesson.md`

- [ ] **Step 1:** Cover learned representations, computational graphs, backpropagation, initialization, normalization, residual paths, regularization, optimization, and gradient diagnosis.
- [ ] **Step 2:** Cover convolution, locality, weight sharing, equivariance, receptive fields, pooling/stride/dilation, channels, residual CNNs, vision transfer, and architectural failure modes.
- [ ] **Step 3:** Validate Markdown and commit.

### Task 2: Sequences and Transformers

**Files:**
- Create: `deep-learning/ch3-recurrent-sequence-models-lesson.md`
- Create: `deep-learning/ch4-attention-transformers-lesson.md`

- [ ] **Step 1:** Derive RNN recurrence and BPTT, vanishing/exploding gradients, LSTM/GRU gating, teacher forcing, masking, sequence objectives, and state-space intuition.
- [ ] **Step 2:** Derive scaled attention, multi-head projections, masking, positional methods including RoPE, encoder/decoder forms, KV cache, GQA/MQA, complexity, and Transformer failure modes.
- [ ] **Step 3:** Validate Markdown and commit.

### Task 3: Embeddings and generative families

**Files:**
- Create: `deep-learning/ch5-autoencoders-contrastive-embeddings-lesson.md`
- Create: `deep-learning/ch6-generative-models-lesson.md`

- [ ] **Step 1:** Cover autoencoders, bottlenecks, denoising, metric learning, contrastive objectives, negatives, collapse, embedding evaluation, and retrieval geometry.
- [ ] **Step 2:** Cover autoregressive likelihood, latent-variable models, ELBO/VAEs, adversarial objectives/GANs, score matching, diffusion forward/reverse processes, sampling, guidance, and trade-offs.
- [ ] **Step 3:** Validate Markdown and commit.

### Task 4: Transfer, scaling, and diagnosis

**Files:**
- Create: `deep-learning/ch7-transfer-scaling-diagnosis-lesson.md`

- [ ] **Step 1:** Cover pretraining, fine-tuning, freezing, adapters/LoRA, domain shift, data quality, scaling laws, distributed/inference trade-offs, quantization/distillation, evaluation, and systematic failure diagnosis.
- [ ] **Step 2:** Validate all seven chapters and commit.

### Task 5: Shell, integration, and publication

**Files:**
- Create: `deep-learning/index.html`
- Create: seven matching `deep-learning/*-viewer.html` pages
- Modify: `index.html`
- Modify: `README.md`

- [ ] **Step 1:** Add course index, prerequisites, seven chapter cards, and complete viewer navigation.
- [ ] **Step 2:** Integrate the completed course into the public site and learning paths.
- [ ] **Step 3:** Run structural, root-link, renderer, unittest, and whitespace checks.
- [ ] **Step 4:** Inspect representative pages over local HTTP at desktop and narrow widths; verify MathJax, code/tables, navigation, overflow, and browser logs.
- [ ] **Step 5:** Commit and push to `origin/master`.
