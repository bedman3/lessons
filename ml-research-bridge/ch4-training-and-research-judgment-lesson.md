# Chapter 4 — Training, Diagnosis & Research Judgment

Training is an optimization procedure inside a research experiment. A low training loss says the optimizer found parameters that fit the training data under the chosen objective. It does not establish useful future predictions. This chapter connects the familiar fit/transform/predict loop to training diagnostics, uncertainty, and decisions about whether more complexity is justified.

## 4.1 The training loop as an explicit contract

For a supervised model, training repeatedly performs four operations: take a batch of inputs and labels, compute predictions, evaluate a loss, and update parameters using gradients. In PyTorch, tensors carry shape and device information, and autograd records the operations needed to calculate derivatives. A typical loop calls `model.train()`, clears gradients, evaluates the model, computes loss, calls `loss.backward()`, and then `optimizer.step()`.

During evaluation, call `model.eval()` and disable gradient recording with `torch.no_grad()` or inference mode. This matters for layers such as dropout and batch normalization, whose behavior differs between training and evaluation. Evaluation mode does not automatically prevent leakage: the data split, target timing, and fitted preprocessing remain the researcher’s responsibility.

Keep transformations that learn state fitted on the training window only. For tabular models, this usually means a scikit-learn `Pipeline` containing imputation, scaling, and the estimator. For a neural model, save the training-only preprocessing state alongside model weights and apply the same operations to later data. A tensor conversion is not a substitute for a point-in-time audit.

## 4.2 Read learning curves before changing architecture

Track training and validation loss by epoch or iteration. If both remain high, suspect underfitting, a weak representation, poor optimization, a target problem, or a coding error. If training loss falls while validation loss worsens, the model is fitting details that do not transfer; reduce capacity, strengthen regularization, or gather better evidence before adding features. If both improve but validation is noisy across dates, more epochs may not solve instability.

Compare curves to simple references and use the same chronological blocks. A neural network should first beat a meaningful baseline on the objective that motivated it. Track more than a single best epoch: record the chosen stopping rule and preserve predictions by date. Repeatedly choosing the epoch with the best original holdout score turns that holdout into validation data.

### Worked example: identify the next diagnostic

Consider synthetic losses after 20 epochs:

| Model | Training MAE | Validation MAE |
|---|---:|---:|
| Training-mean baseline | 5.0 | 5.1 |
| Ridge | 4.7 | 4.9 |
| Neural network | 2.1 | 5.4 |

The network fits the training rows much more closely but does worse on the later block than ridge. The evidence supports an overfit diagnosis for this design, not a claim that neural networks are always unsuitable. First check split and label integrity; then reduce capacity or regularize and compare across multiple development blocks. The 5.4 is not a P&L estimate.

## 4.3 Debug from small to large

When a model behaves strangely, use a diagnostic ladder:

1. Hand-check a few rows, feature units, target construction, and timestamps.
2. Reproduce the baseline and a tiny loss calculation independently.
3. Overfit a tiny training subset. If the model cannot, inspect tensor shapes, labels, gradients, loss, and optimizer setup.
4. Compare train and validation curves on one fixed split.
5. Repeat on the prespecified temporal blocks and examine per-date errors.
6. Profile runtime and memory only after the experiment is correct.

Tiny-subset overfit tests implementation capacity; it is deliberately not a generalization test. Likewise, a successful GPU run proves that computation completed, not that the target was valid or the model useful.

## 4.4 Treat uncertainty as part of the result

Financial labels are noisy, observations share dates, and regimes change. Report the number of dates and observations, date-level metric distribution, and matched model-minus-baseline differences. Show block results and an uncertainty interval using a dependence-aware procedure such as contiguous date-block resampling. State the block choice and its rationale. If results vary materially by period, that variation is itself a finding.

Avoid interpreting small positive averages as strong evidence when date blocks disagree or the effective number of independent periods is low. A confidence interval cannot account for every source of uncertainty, including vendor revisions, unrecorded research trials, or a misspecified cost model. Keep those limitations visible.

## 4.5 Decide whether a result deserves more work

Before a run, write down what evidence would change the next decision. Examples: a linear model consistently improves the primary loss across development blocks; a tree improvement recurs across periods; or an apparent gain disappears under plausible execution costs. Record all material variants tried, including targets, feature sets, parameter choices, thresholds, and cost assumptions. Many attempts create a winner’s-curse problem even when each individual comparison looks reasonable.

The first end-to-end project is a learning exercise, roughly 50–90 focused hours depending on data and review. It can demonstrate an auditable workflow and sound reasoning. It does not establish professional ML experience, production expertise, or qualification for a job whose requirements exceed the project’s evidence.

### Optional extension: a small PyTorch comparison

After the CPU baselines and temporal evaluation work, fit a small multilayer perceptron to the same training window and compare it on exactly the same later dates. Use CPU first, set seeds while recognizing that deterministic behavior is not guaranteed across all operations, and record package versions, configuration, and runtime. Keep preprocessing state train-only. Plot both training and validation loss; compare date-level errors and uncertainty rather than declaring victory from one aggregate number.

GPU acceleration, distributed training, mixed precision, and large architectures are future extensions. They require a demonstrated computational bottleneck, enough data and memory, appropriate profiling, and a reproducibility plan. A CPU tabular experiment does not imply hands-on GPU or distributed-systems experience.

## 4.6 Practice and answer key

**Exercise A.** Training loss falls each epoch, while validation loss reaches a minimum and then rises. What should be inspected or changed, and which data must remain untouched?

**Exercise B.** A model improves pooled MAE by 0.2 bp, but loses in six of ten date blocks and the trial was selected after testing 40 variants. What can you conclude?

**Answer key.** A: Verify split, labels, and train-only transforms; then consider an early-stopping rule chosen within development/validation or lower capacity/stronger regularization. Preserve the original holdout for the frozen confirmation. B: The pooled gain is weak and unstable evidence, especially after extensive selection. Record the 40 trials, compare matched per-date differences and uncertainty, inspect costs and regimes, and avoid calling it a profitable signal. If the holdout helped select the variant, it is no longer untouched.

**Checkpoint rubric (0–2 each):** train/eval modes distinguished; curves interpreted correctly; debugging begins with data and tiny checks; uncertainty reflects dates and dependence; trial history and holdout status reported honestly. Score 8/10 with no zero on validation or selection before presenting a final result.

## 4.7 Failure diagnosis and next steps

If training loss does not move, test the model on a tiny batch and inspect shapes, gradients, learning rate, target scale, and optimizer updates. If validation improves only on one period, inspect per-date errors, data changes, and the number of variants tried. If GPU time dominates, profile preprocessing, transfer, and batch size before changing hardware. If a model’s edge vanishes after costs, report that result without relabeling gross score as net performance.

The bridge labs are intentionally CPU-first. From the repository root, run `python ml-research-bridge/lab.py --output results.json` for the scikit-learn baselines and chronological split; the optional neural comparison is in [PyTorch lab](torch_lab.py) when available. Start with [Chapter 1 — Diagnostic](ch1-diagnostic-lesson.md), then carry a single experiment through [Chapter 2 — Baselines](ch2-baselines-lesson.md) and [Chapter 3 — Honest Validation](ch3-honest-validation-lesson.md). For deeper foundations, continue selectively into [Applied ML — Debugging & Production](../applied-ml/ch6-debugging-production-lesson.md) and [Deep Learning — Representations & Optimization](../deep-learning/ch1-representations-optimization-lesson.md).

## References

- PyTorch, [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html), [saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html), and [performance tuning guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html).
- scikit-learn, [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) and [common pitfalls and leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- This lesson’s next steps: [Applied ML Chapter 6](../applied-ml/ch6-debugging-production-lesson.md), [ML Quant Chapter 6 — Non-stationarity & Robustness](../ml-quant/ch6-nonstationarity-robustness-lesson.md), and [Deep Learning Chapter 1](../deep-learning/ch1-representations-optimization-lesson.md).
