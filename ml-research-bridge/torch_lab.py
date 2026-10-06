"""Optional CPU-only PyTorch exercise using the lab's exact temporal split."""
from __future__ import annotations
import argparse
import json
import platform
import time

import numpy as np
import torch
from torch import nn

from lab import chronological_split, make_synthetic_data, scale_with_train_only, _metrics


def run_torch_experiment(seed: int = 19, n_samples: int = 600, epochs: int = 120) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    horizon = 5
    split = chronological_split(make_synthetic_data(seed, n_samples, horizon),
                                int(.55*n_samples), int(.78*n_samples), horizon)
    scaled = scale_with_train_only(split)
    x_train = torch.tensor(scaled.train, dtype=torch.float32)
    y_train = torch.tensor(split.train.target[:, None], dtype=torch.float32)
    x_val = torch.tensor(scaled.validation, dtype=torch.float32)
    y_val = torch.tensor(split.validation.target, dtype=torch.float32).numpy()
    x_test = torch.tensor(scaled.test, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(x_train.shape[1], 12), nn.Tanh(), nn.Linear(12, 1))
    optimizer = torch.optim.Adam(net.parameters(), lr=.02, weight_decay=.01)
    loss_fn = nn.MSELoss()
    start = time.perf_counter()
    loss_checkpoints = []
    for _ in range(epochs):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(net(x_train), y_train)
        loss.backward()
        optimizer.step()
        if (_ == 0) or (_ + 1) % 10 == 0 or _ + 1 == epochs:
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(x_val), torch.tensor(y_val[:, None], dtype=torch.float32))
            loss_checkpoints.append({"epoch": _ + 1, "train_mse": float(loss.detach()),
                                     "validation_mse": float(val_loss)})
            net.train()
    training_seconds = time.perf_counter() - start
    net.eval()
    with torch.no_grad():
        val_pred = net(x_val).squeeze(1).numpy()
    # A single prespecified validation check: use MLP only if it beats zero MAE.
    use_mlp = _metrics(y_val, val_pred)["mae"] < _metrics(y_val, np.zeros_like(y_val))["mae"]
    test_start = time.perf_counter()
    with torch.no_grad():
        prediction = net(x_test).squeeze(1).numpy() if use_mlp else np.zeros(len(x_test))
    inference_seconds = time.perf_counter() - test_start
    return {
        "seed": seed, "n_samples": n_samples, "epochs": epochs,
        "split_sizes": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "selected": "small_mlp" if use_mlp else "zero_baseline",
        "validation_mlp_metrics": _metrics(y_val, val_pred),
        "validation_zero_metrics": _metrics(y_val, np.zeros_like(y_val)),
        "loss_checkpoints": loss_checkpoints,
        "test_metrics": _metrics(split.test.target, prediction), "test_evaluation_count": 1,
        "training_seconds_cpu": training_seconds, "test_inference_seconds_cpu": inference_seconds,
        "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(),
        "interpretation": "Synthetic prediction exercise only; timing is local and not a hardware comparison.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--output")
    a = p.parse_args()
    result = run_torch_experiment(a.seed, a.n_samples, a.epochs)
    text = json.dumps(result, indent=2, sort_keys=True)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
