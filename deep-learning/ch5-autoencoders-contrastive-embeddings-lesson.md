# Chapter 5 — Autoencoders, Contrastive Learning & Embeddings

Representation learning can be trained without task labels by asking the model to reconstruct, compare, or predict structure in the data. The resulting embeddings become reusable coordinates for retrieval, clustering, transfer, anomaly detection, and downstream prediction.

An embedding is useful only relative to the similarity notion induced by its training objective and data.

## 5.1 Autoencoders

An encoder maps input to latent code,

$$
z=f_\theta(x),
$$

and a decoder reconstructs

$$
\widehat x=g_\phi(z).
$$

Training minimizes reconstruction loss such as

$$
\min_{\theta,\phi}E[\|x-g_\phi(f_\theta(x))\|^2].
$$

If encoder and decoder have unrestricted capacity, they can learn the identity without useful abstraction. Bottlenecks and regularization determine what structure must be retained.

## 5.2 Undercomplete, sparse, and denoising objectives

An undercomplete autoencoder uses latent dimension smaller than input. A linear undercomplete autoencoder with squared error spans the same principal subspace as PCA under appropriate conditions.

Sparse autoencoders penalize latent activation, allowing many dimensions but few active per example. Denoising autoencoders corrupt input $\widetilde x\sim q(\widetilde x\mid x)$ and reconstruct clean $x$:

$$
E[\|x-g(f(\widetilde x))\|^2].
$$

This encourages local stability and prevents trivial copying. The corruption process defines which variations should be ignored.

## 5.3 Reconstruction is not semantics

Pixel-level MSE emphasizes colour and precise location, not necessarily object identity. A model can reconstruct common background well while ignoring a rare task-relevant feature.

Evaluate representations on downstream probes, retrieval, clustering stability, perturbations, and subgroup behaviour. Low reconstruction error alone does not prove useful latent geometry.

## 5.4 Metric learning

Metric learning trains distances so related examples are close and unrelated examples far apart. For embeddings $z_a,z_p,z_n$, triplet loss is

$$
L=\max(0,d(z_a,z_p)-d(z_a,z_n)+m).
$$

The margin $m$ enforces relative separation. Hard negatives provide strong learning signal but mislabeled or false negatives can damage geometry.

Siamese networks share encoder weights across examples, ensuring comparisons use one representation function.

## 5.5 Contrastive learning

For positive pair $(i,j)$ and normalized embeddings, an InfoNCE-style loss is

$$
L_i=-\log
\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
{\sum_{k\ne i}\exp(\operatorname{sim}(z_i,z_k)/\tau)}.
$$

Temperature $\tau$ controls concentration. Lower values focus on hardest competitors and can amplify noise.

The objective learns invariance to transformations used to create positive views while preserving information that distinguishes negatives.

## 5.6 Positive-pair design

For images, crops and colour changes may preserve identity. For text, nearby spans or paired modalities may be positive. For finance, two augmentations of one series must not erase economically relevant timing or fabricate invariance to scale/regime.

Augmentation is an inductive-bias declaration. If positives collapse examples that the downstream task must distinguish, no architecture can recover the lost distinction.

## 5.7 Negatives and false negatives

Large batches or memory banks provide more negatives. But two semantically related examples treated as negatives create repulsion in the wrong direction. Debiased losses, supervised contrastive labels, hard-negative filtering, or positive mining can help.

In-batch negatives also couple examples: duplicate entities or temporally adjacent observations can change the effective objective and create leakage across train/validation construction.

## 5.8 Representation collapse

If all inputs map to one vector, representation has collapsed. Negative-based objectives resist collapse through competition. Non-contrastive methods can avoid it using asymmetric predictors, stop-gradient, momentum target networks, variance/covariance regularization, or redundancy reduction.

Monitor per-dimension variance, covariance spectrum, effective rank, and nearest-neighbour diversity—not loss alone.

## 5.9 Geometry and normalization

With $L_2$-normalized embeddings,

$$
\|z_i-z_j\|^2=2-2z_i^Tz_j,
$$

so Euclidean distance and cosine similarity induce the same ordering. Without normalization, vector norm can encode confidence, frequency, or nuisance scale.

High-dimensional embeddings can be anisotropic or exhibit hubness: a few points become nearest neighbours of many queries. Whitening, centring, better objectives, or domain-specific indexing may help.

## 5.10 Retrieval systems

Dense retrieval embeds queries and documents, then finds nearest vectors. A second-stage reranker can model richer query–document interaction.

Evaluate:

- recall@k: whether relevant items appear in top $k$;
- precision@k;
- mean reciprocal rank;
- nDCG for graded relevance;
- latency, index memory, and update freshness;
- performance by query type and subgroup.

Retrieval quality depends on corpus construction, relevance labels, chunking, and negatives as much as encoder architecture.

## 5.11 Linear probes and transfer

A frozen linear probe tests whether target information is linearly accessible. Full fine-tuning tests adaptability but confounds representation quality with optimization and capacity.

Compare nearest-neighbour, linear probe, shallow fine-tuning, and full fine-tuning under identical data splits. High probe accuracy can still rely on spurious features.

## 5.12 Failure modes

- Calling low reconstruction loss evidence of semantic understanding.
- Using augmentations that remove target-relevant information.
- Treating all in-batch examples as true negatives.
- Mining hard negatives from the validation/test corpus.
- Ignoring collapse because training loss decreases.
- Comparing cosine and Euclidean retrieval without checking normalization.
- Evaluating retrieval with incomplete relevance labels as though unjudged means irrelevant.

## 5.13 Knowledge checks

1. When does a linear autoencoder recover a PCA subspace?
2. What invariance does a contrastive model learn?
3. How does temperature affect InfoNCE competition?
4. Why can hard negatives be harmful?
5. What diagnostics reveal representation collapse?

### Solution outlines

1. Undercomplete linear encoder/decoder with squared reconstruction and appropriate optimization.
2. Invariance to transformations or pairing rules used to define positive views.
3. Lower temperature sharpens softmax and emphasizes close competitors.
4. They may be mislabeled false negatives or dominated by noise/outliers.
5. Low per-dimension variance, low effective rank, high covariance, and repeated nearest neighbours.

## 5.14 What to retain

- Self-supervised objectives define which information embeddings preserve.
- Bottlenecks, corruption, positives, and negatives are inductive biases.
- Reconstruction quality and semantic utility are different.
- Geometry, normalization, and corpus construction determine retrieval behaviour.
- Representation evaluation needs downstream and structural diagnostics.

Next: [Chapter 6 — Generative Models](ch6-generative-models-viewer.html).
